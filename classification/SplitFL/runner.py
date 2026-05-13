import copy
import gc
import json
import os
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from config import ConfigLoader
from data import get_dataset
from FedServer import get_strategy
from models import get_model
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)


def get_weight_size_mb(weights):
    return sum(torch.numel(v) for v in weights.values()) * 4 / (1024**2)


def estimate_gradient_size_mb(model, input_shape, device):
    model = model.to(device).eval()
    with torch.no_grad():
        output = model(torch.randn(*input_shape).to(device))
    return (output.numel() * output.element_size()) / (1024**2)


def run(cfg_path: str):
    start_time = time.time()
    config_loader = ConfigLoader(cfg_path)
    config = config_loader.get_config()
    print("Method: {}".format(config["strategy"]))

    logger = SummaryWriter("./logs")
    if config["is_gpu"]:
        torch.cuda.set_device(config["gpu"])
    device = torch.device("cuda") if config["is_gpu"] else torch.device("cpu")

    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)
    model = get_model(config["model"], config["dataset"])
    client_model_abs, main_server_model, merge_model = (
        model[0]().to(device), model[1](config).to(device), model[2](config).to(device)
    )

    strategy = get_strategy(config["strategy"])(config)
    training_loss, eval_f1_scores = [], []
    round_cpu_usages, round_ram_usages, round_gpu_usages = [], [], []
    comm_cost_dict = {
        "client_upload_smashed_MB": 0, "client_model_upload_MB": 0,
        "client_model_download_MB": 0, "cloud_download_grad_MB": 0,
    }
    criterion = nn.NLLLoss().to(device)
    best_f1 = 0.0
    best_model_weights = None

    for epoch in tqdm(range(config["epochs"])):
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(range(config["num_users"]), m, replace=False)

        local_weights, user_losses_per_epoch = [], []
        server_optimizer = SGD(main_server_model.parameters(), lr=config["lr"], momentum=config["momentum"])
        round_cpu_per_client, round_ram_per_client, round_gpu_per_client = [], [], []

        for idx in idxs_users:
            client_model = copy.deepcopy(client_model_abs).to(device)
            client_optimizer = SGD(client_model.parameters(), lr=config["lr"], momentum=config["momentum"])
            client_model.train()

            loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), batch_size=config["local_bs"], shuffle=True)
            local_cpu_usages, local_ram_usages, local_gpu_usages = [], [], []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            user_losses_per_iter = []
            for _ in range(config["local_ep"]):
                local_cpu_usages.append(psutil.cpu_percent(interval=None))
                local_ram_usages.append(psutil.virtual_memory().percent)
                if config["is_gpu"]:
                    local_gpu_usages.append(torch.cuda.memory_allocated(device=device) / 1024**2)

                losses = []
                for image, label in loader:
                    server_optimizer.zero_grad()
                    client_optimizer.zero_grad()
                    image, label = image.to(device), label.to(device)
                    activation = client_model(image)
                    predict = main_server_model(activation)
                    loss = criterion(predict, label)
                    loss.backward()
                    server_optimizer.step()
                    client_optimizer.step()
                    losses.append(loss.item())
                    comm_cost_dict["client_upload_smashed_MB"] += (activation.numel() + label.numel()) * 4 / (1024**2)
                    comm_cost_dict["cloud_download_grad_MB"] += estimate_gradient_size_mb(main_server_model, activation.shape, device)
                user_losses_per_iter.append(np.mean(losses))

            round_cpu_per_client.append(np.mean(local_cpu_usages))
            round_ram_per_client.append(np.mean(local_ram_usages))
            if config["is_gpu"]:
                round_gpu_per_client.append(np.mean(local_gpu_usages))

            user_losses_per_epoch.append(np.mean(user_losses_per_iter))
            comm_cost_dict["client_model_upload_MB"] += get_weight_size_mb(client_model.state_dict())
            local_weights.append(copy.deepcopy(client_model.state_dict()))

        training_loss.append(np.mean(user_losses_per_epoch))
        client_model_abs.load_state_dict(strategy.aggregate(None, None, local_weights))
        comm_cost_dict["client_model_download_MB"] += get_weight_size_mb(client_model_abs.state_dict()) * config["num_users"]

        merge_model.load_weight(copy.deepcopy(client_model_abs.state_dict()), copy.deepcopy(main_server_model.state_dict()))
        merge_model.to(device).eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for image, label in DataLoader(valid_dataset, batch_size=config["local_bs"], shuffle=False):
                image, label = image.to(device), label.to(device)
                outputs = merge_model(image)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        eval_f1 = f1_score(all_labels, all_preds, average="macro")
        eval_f1_scores.append(eval_f1)
        round_cpu_usages.append(np.mean(round_cpu_per_client))
        round_ram_usages.append(np.mean(round_ram_per_client))
        round_gpu_usages.append(np.mean(round_gpu_per_client) if config["is_gpu"] else 0)

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_model_weights = copy.deepcopy(merge_model.state_dict())
            print(f"New Best F1: {best_f1:.4f} at Epoch {epoch+1}")

        if wandb is not None and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": training_loss[-1],
                "f1": eval_f1,
                "best_f1": best_f1,
                "avg_cpu_pct": round_cpu_usages[-1],
                "avg_ram_pct": round_ram_usages[-1],
                "avg_gpu_ram_MB": round_gpu_usages[-1],
                **{k: v for k, v in comm_cost_dict.items()},
            })

        if (epoch + 1) % config["print_every"] == 0:
            print(f"Epoch {epoch+1}: Loss={training_loss[-1]:.4f}  F1={eval_f1:.4f}")
        for k, v in comm_cost_dict.items():
            print(f"  {k}: {v:.2f} MB")

    # Final test
    merge_model.load_state_dict(best_model_weights)
    merge_model.to(device).eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for image, label in DataLoader(test_dataset, batch_size=1, shuffle=False):
            image, label = image.to(device), label.to(device)
            _, predicted = torch.max(merge_model(image).data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(label.cpu().numpy())

    test_f1 = f1_score(test_labels, test_preds, average="macro")
    total_time = time.time() - start_time
    print(f"\nFinal Test F1: {test_f1*100:.2f}%")
    print("Total Run Time: {:.2f}s".format(total_time))

    if wandb is not None and wandb.run is not None:
        wandb.summary["test_f1"] = test_f1
        wandb.summary["best_f1"] = best_f1
        wandb.summary["total_time_s"] = total_time

    out_dir = os.path.join(os.path.dirname(__file__), "Figure", "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"SplitFL_{config['dataset']}_iid:{config['iid']}_{config['model']}_{config['num_users']}users.json"), "w") as f:
        json.dump({
            "avg_cpu_percent": round_cpu_usages, "avg_ram_percent": round_ram_usages,
            "avg_gpu_memory_MB": round_gpu_usages, "train_accuracy": eval_f1_scores,
            "train_loss": training_loss, "final_test_f1": test_f1,
        }, f, indent=4)
