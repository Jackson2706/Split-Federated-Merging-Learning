import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ConfigLoader
from data import get_dataset
from models import get_model
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)


class HeteroServerAdapter(nn.Module):
    def __init__(self, server_model, wide_channels, server_expected_channels=64):
        super().__init__()
        self.decoder = nn.Conv2d(wide_channels, server_expected_channels, kernel_size=3, padding=1)
        self.server_model = server_model

    def forward(self, x):
        return self.server_model(self.decoder(x))


def extract_narrow_model(wide_model, narrow_channels):
    narrow_state = copy.deepcopy(wide_model.state_dict())
    keys = list(narrow_state.keys())
    narrow_state[keys[-2]] = narrow_state[keys[-2]][:narrow_channels]
    narrow_state[keys[-1]] = narrow_state[keys[-1]][:narrow_channels]
    return narrow_state


def aggregate_hetero(global_wide_weights, local_updates, narrow_channels):
    update_acc = {k: torch.zeros_like(v) for k, v in global_wide_weights.items()}
    count_acc = {k: torch.zeros_like(v) for k, v in global_wide_weights.items()}
    keys = list(global_wide_weights.keys())
    weight_key, bias_key = keys[-2], keys[-1]

    for local_w in local_updates:
        is_narrow = local_w[weight_key].shape[0] == narrow_channels
        for k in local_w:
            if k in [weight_key, bias_key] and is_narrow:
                update_acc[k][:narrow_channels] += local_w[k]
                count_acc[k][:narrow_channels] += 1
            else:
                update_acc[k] += local_w[k]
                count_acc[k] += 1

    avg_weights = copy.deepcopy(global_wide_weights)
    for k in avg_weights:
        mask = count_acc[k] > 0
        avg_weights[k][mask] = update_acc[k][mask] / count_acc[k][mask]
    return avg_weights


def bdks_loss(pred_wide, pred_narrow, label, alpha=1.0):
    loss_task = nn.CrossEntropyLoss()(pred_wide, label)
    loss_n2w = F.kl_div(F.log_softmax(pred_wide, dim=1), F.softmax(pred_narrow, dim=1), reduction="batchmean")
    loss_w2n = F.kl_div(F.log_softmax(pred_narrow, dim=1), F.softmax(pred_wide, dim=1), reduction="batchmean")
    return loss_task + alpha * loss_n2w + loss_w2n


def run(cfg_path: str):
    start_time = time.time()
    config_loader = ConfigLoader(cfg_path)
    config = config_loader.get_config()

    WIDE_CHANNELS = config.get("wide_channels", 76)
    NARROW_CHANNELS = config.get("narrow_channels", 4)
    SERVER_EXPECTED_CHANNELS = config.get("server_in_channels", 64)
    HIGH_END_RATIO = config.get("high_end_ratio", 0.3)
    BDKS_ALPHA = config.get("bdks_alpha", 1.0)

    print(f"HeteroSFL: Wide={WIDE_CHANNELS}, Narrow={NARROW_CHANNELS}, Ratio={HIGH_END_RATIO}")
    config["client_out_channels"] = WIDE_CHANNELS

    device = torch.device("cuda" if config["is_gpu"] else "cpu")
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)
    model_tuple = get_model(config["model"], config["dataset"])

    client_model_wide = model_tuple[0](config).to(device)
    main_server_model = HeteroServerAdapter(
        model_tuple[1](config).to(device), WIDE_CHANNELS, SERVER_EXPECTED_CHANNELS
    ).to(device)

    num_users = config["num_users"]
    num_high = int(num_users * HIGH_END_RATIO)
    client_types = ["high"] * num_high + ["low"] * (num_users - num_high)
    np.random.shuffle(client_types)

    comm_cost_dict = {"upload_MB": 0, "download_MB": 0}
    best_f1 = 0.0

    for epoch in tqdm(range(config["epochs"])):
        idxs_users = np.random.choice(range(num_users), max(int(config["frac"] * num_users), 1), replace=False)
        local_weights, epoch_losses = [], []
        server_optimizer = SGD(main_server_model.parameters(), lr=config["lr"], momentum=config["momentum"])

        for idx in idxs_users:
            client_type = client_types[idx]
            local_client_model = copy.deepcopy(client_model_wide)

            if client_type == "low":
                narrow_state = extract_narrow_model(client_model_wide, NARROW_CHANNELS)
                last_layer = list(local_client_model.children())[-1]
                if isinstance(last_layer, nn.Conv2d):
                    last_layer.out_channels = NARROW_CHANNELS
                    last_layer.weight = nn.Parameter(narrow_state[list(narrow_state.keys())[-2]])
                    last_layer.bias = nn.Parameter(narrow_state[list(narrow_state.keys())[-1]])

            local_client_model.to(device).train()
            client_optimizer = SGD(local_client_model.parameters(), lr=config["lr"], momentum=config["momentum"])
            loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), batch_size=config["local_bs"], shuffle=True)
            client_losses = []

            for _ in range(config["local_ep"]):
                for image, label in loader:
                    server_optimizer.zero_grad()
                    client_optimizer.zero_grad()
                    image, label = image.to(device), label.to(device)
                    activation = local_client_model(image)

                    if client_type == "high":
                        act_wide = activation.clone().detach().requires_grad_(True)
                        act_narrow_sim = act_wide.clone()
                        act_narrow_sim[:, NARROW_CHANNELS:] = 0
                        pred_wide = main_server_model(act_wide)
                        pred_narrow = main_server_model(act_narrow_sim)
                        loss = bdks_loss(pred_wide, pred_narrow, label, BDKS_ALPHA)
                        loss.backward()
                        activation_grad = act_wide.grad
                    else:
                        act_narrow = activation.clone().detach().requires_grad_(True)
                        padding = torch.zeros(act_narrow.shape[0], WIDE_CHANNELS - NARROW_CHANNELS,
                                              act_narrow.shape[2], act_narrow.shape[3]).to(device)
                        act_input = torch.cat([act_narrow, padding], dim=1)
                        pred = main_server_model(act_input)
                        loss = nn.CrossEntropyLoss()(pred, label)
                        loss.backward()
                        activation_grad = act_narrow.grad

                    server_optimizer.step()
                    activation.backward(activation_grad)
                    client_optimizer.step()
                    client_losses.append(loss.item())
                    comm_cost_dict["upload_MB"] += activation.numel() * 4 / (1024**2)

            local_weights.append(copy.deepcopy(local_client_model.state_dict()))
            epoch_losses.append(np.mean(client_losses))
            del local_client_model
            torch.cuda.empty_cache()

        client_model_wide.load_state_dict(aggregate_hetero(client_model_wide.state_dict(), local_weights, NARROW_CHANNELS))

        if (epoch + 1) % config["print_every"] == 0:
            client_model_wide.eval()
            main_server_model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for image, label in DataLoader(valid_dataset, batch_size=config["local_bs"], shuffle=False):
                    image, label = image.to(device), label.to(device)
                    out = main_server_model(client_model_wide(image))
                    _, predicted = torch.max(out.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

            eval_f1 = f1_score(all_labels, all_preds, average="macro")
            print(f"Epoch {epoch+1}: F1={eval_f1:.4f}  Acc={accuracy_score(all_labels, all_preds):.4f}")
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                print(f" -> New Best F1: {best_f1:.4f}")
            client_model_wide.train()
            main_server_model.train()

    # Final test
    client_model_wide.eval()
    main_server_model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for image, label in DataLoader(test_dataset, batch_size=config["local_bs"], shuffle=False):
            image, label = image.to(device), label.to(device)
            out = main_server_model(client_model_wide(image))
            _, predicted = torch.max(out.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(label.cpu().numpy())

    final_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"\nFinal Test F1: {final_f1:.4f}  Acc: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"Total Upload: {comm_cost_dict['upload_MB']:.2f} MB")
    print("Total Run Time: {:.2f}s".format(time.time() - start_time))
