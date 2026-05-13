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
from clients import test_inference, DiceFocalLoss
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
        image, mask = self.dataset[self.idxs[item]]
        return image.clone(), mask.clone()


class HeteroServerAdapter(nn.Module):
    """Wraps server model with a decoder that maps wide channels -> expected channels."""

    def __init__(self, server_model, wide_channels, server_expected_channels=512):
        super().__init__()
        self.decoder = nn.Conv2d(wide_channels, server_expected_channels, kernel_size=3, padding=1)
        self.server_model = server_model

    def forward(self, x):
        return self.server_model(self.decoder(x))


def extract_narrow_model(wide_model, narrow_channels):
    narrow_state = copy.deepcopy(wide_model.state_dict())
    keys = list(narrow_state.keys())
    # bl_encoder weight and bias are the last two keys of the client model
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


def run(cfg_path: str):
    start_time = time.time()
    config_loader = ConfigLoader(cfg_path)
    config = config_loader.get_config()

    WIDE_CHANNELS = config.get("wide_channels", 76)
    NARROW_CHANNELS = config.get("narrow_channels", 4)
    SERVER_EXPECTED_CHANNELS = config.get("server_in_channels", 512)
    HIGH_END_RATIO = config.get("high_end_ratio", 0.3)

    print(f"HeteroSFL (Seg): Wide={WIDE_CHANNELS}, Narrow={NARROW_CHANNELS}, Ratio={HIGH_END_RATIO}")
    config["client_out_channels"] = WIDE_CHANNELS

    device = torch.device("cuda" if config["is_gpu"] else "cpu")
    train_dataset, test_dataset, user_groups = get_dataset(config)
    model_tuple = get_model(config["model"], config["dataset"])

    client_model_wide = model_tuple[0](config).to(device)
    server_model_raw = model_tuple[1](config).to(device)
    main_server_model = HeteroServerAdapter(
        server_model_raw, WIDE_CHANNELS, SERVER_EXPECTED_CHANNELS
    ).to(device)

    num_users = config["num_users"]
    num_high = int(num_users * HIGH_END_RATIO)
    client_types = ["high"] * num_high + ["low"] * (num_users - num_high)
    np.random.shuffle(client_types)

    criterion = DiceFocalLoss().to(device)
    comm_cost_dict = {"upload_MB": 0, "download_MB": 0}
    best_iou = 0.0

    for epoch in tqdm(range(config["epochs"])):
        idxs_users = np.random.choice(
            range(num_users), max(int(config["frac"] * num_users), 1), replace=False
        )
        local_weights, epoch_losses = [], []
        server_optimizer = SGD(
            main_server_model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )

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
            client_optimizer = SGD(
                local_client_model.parameters(), lr=config["lr"], momentum=config["momentum"]
            )
            loader = DataLoader(
                DatasetSplit(train_dataset, user_groups[idx]),
                batch_size=config["local_bs"], shuffle=True,
            )
            client_losses = []

            for _ in range(config["local_ep"]):
                for image, mask in loader:
                    server_optimizer.zero_grad()
                    client_optimizer.zero_grad()
                    image, mask = image.to(device), mask.to(device)
                    activation = local_client_model(image)

                    if client_type == "high":
                        act_wide = activation.clone().detach().requires_grad_(True)
                        # Simulate narrow by zeroing out channels beyond NARROW
                        act_narrow_sim = act_wide.clone()
                        act_narrow_sim[:, NARROW_CHANNELS:] = 0
                        pred_wide = main_server_model(act_wide)
                        pred_narrow = main_server_model(act_narrow_sim)
                        # Segmentation: use DiceFocalLoss for both, average
                        loss_wide = criterion(pred_wide, mask)
                        loss_narrow = criterion(pred_narrow, mask)
                        loss = loss_wide + 0.5 * loss_narrow
                        loss.backward()
                        activation_grad = act_wide.grad
                    else:
                        act_narrow = activation.clone().detach().requires_grad_(True)
                        padding = torch.zeros(
                            act_narrow.shape[0],
                            WIDE_CHANNELS - NARROW_CHANNELS,
                            act_narrow.shape[2],
                            act_narrow.shape[3],
                        ).to(device)
                        act_input = torch.cat([act_narrow, padding], dim=1)
                        pred = main_server_model(act_input)
                        loss = criterion(pred, mask)
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

        client_model_wide.load_state_dict(
            aggregate_hetero(client_model_wide.state_dict(), local_weights, NARROW_CHANNELS)
        )

        if (epoch + 1) % config["print_every"] == 0:
            # Build merged model for evaluation
            eval_model = copy.deepcopy(main_server_model)
            eval_client = copy.deepcopy(client_model_wide)

            class _EvalWrapper(nn.Module):
                def __init__(self, client, server):
                    super().__init__()
                    self.client = client
                    self.server = server

                def forward(self, x):
                    return self.server(self.client(x))

            eval_wrapper = _EvalWrapper(eval_client, eval_model).to(device)
            eval_iou, eval_dice, eval_loss = test_inference(config, eval_wrapper, test_dataset)
            print(f"Epoch {epoch+1}: IoU={eval_iou:.4f}  Dice={eval_dice:.4f}")

            if eval_iou > best_iou:
                best_iou = eval_iou
                print(f" -> New Best IoU: {best_iou:.4f}")

            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "iou": eval_iou,
                    "dice": eval_dice,
                    "best_iou": best_iou,
                    "train_loss": np.mean(epoch_losses),
                    **{k: v for k, v in comm_cost_dict.items()},
                })

            del eval_wrapper, eval_client, eval_model

    # Final test
    class _FinalWrapper(nn.Module):
        def __init__(self, client, server):
            super().__init__()
            self.client = client
            self.server = server

        def forward(self, x):
            return self.server(self.client(x))

    final_model = _FinalWrapper(client_model_wide, main_server_model).to(device)
    final_iou, final_dice, final_loss = test_inference(config, final_model, test_dataset)
    total_time = time.time() - start_time

    print(f"\nFinal Test IoU: {final_iou:.4f}  Dice: {final_dice:.4f}")
    print(f"Total Upload: {comm_cost_dict['upload_MB']:.2f} MB")
    print(f"Total Run Time: {total_time:.2f}s")

    if wandb is not None and wandb.run is not None:
        wandb.summary["test_iou"] = final_iou
        wandb.summary["test_dice"] = final_dice
        wandb.summary["best_iou"] = best_iou
        wandb.summary["total_time_s"] = total_time

    out_dir = os.path.join(os.path.dirname(__file__), "Figure", "data")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"HeteroSFL_{config['dataset']}_iid:{config['iid']}_{config['model']}_{config['num_users']}users.json"
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump({
            "test_iou": final_iou, "test_dice": final_dice,
            "best_iou": best_iou, "total_time": total_time,
            "comm_cost": comm_cost_dict,
        }, f, indent=4)
