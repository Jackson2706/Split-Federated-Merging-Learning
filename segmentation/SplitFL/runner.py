import copy
import gc
import json
import os
import time

import numpy as np
import psutil
import torch
from ehsfp.communication import add_communication, new_communication_tracker
from segmentation.HeteroSFL.clients import DiceFocalLoss, compute_iou_and_dice
from config import ConfigLoader
from data import get_dataset
from FedServer import get_strategy
from models import get_model
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
        return image.clone(), label.clone()


def evaluate_segmentation(model, dataset, batch_size, device):
    total_iou, total_dice, total_samples = 0.0, 0.0, 0
    model.to(device).eval()
    with torch.no_grad():
        for image, mask in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            image, mask = image.to(device), mask.to(device)
            prediction = model(image)
            iou, dice = compute_iou_and_dice(prediction, mask)
            size = image.size(0)
            total_iou += iou * size
            total_dice += dice * size
            total_samples += size
    return total_iou / total_samples, total_dice / total_samples


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
    training_loss, eval_iou_scores, eval_dice_scores = [], [], []
    round_cpu_usages, round_ram_usages, round_gpu_usages = [], [], []
    comm_cost_dict = new_communication_tracker()
    criterion = DiceFocalLoss().to(device)
    best_dice = -1.0
    best_model_weights = None

    for epoch in tqdm(range(config["epochs"])):
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(range(config["num_users"]), m, replace=False)
        local_weights, user_losses_per_epoch = [], []
        server_optimizer = SGD(main_server_model.parameters(), lr=config["lr"], momentum=config["momentum"])
        round_cpu_per_client, round_ram_per_client, round_gpu_per_client = [], [], []

        for idx in idxs_users:
            client_model = copy.deepcopy(client_model_abs).to(device)
            add_communication(comm_cost_dict, "server_to_client_MB", payload=client_model.state_dict())
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
                for image, mask in loader:
                    server_optimizer.zero_grad()
                    client_optimizer.zero_grad()
                    image, mask = image.to(device), mask.to(device)
                    activation = client_model(image)
                    server_activation = activation.detach().requires_grad_(True)
                    predict = main_server_model(server_activation)
                    loss = criterion(predict, mask)
                    loss.backward()
                    activation.backward(server_activation.grad)
                    server_optimizer.step()
                    client_optimizer.step()
                    losses.append(loss.item())
                    add_communication(comm_cost_dict, "client_to_server_MB", payload=(activation, mask))
                    add_communication(comm_cost_dict, "server_to_client_MB", payload=server_activation.grad)
                user_losses_per_iter.append(np.mean(losses))

            round_cpu_per_client.append(np.mean(local_cpu_usages))
            round_ram_per_client.append(np.mean(local_ram_usages))
            if config["is_gpu"]:
                round_gpu_per_client.append(np.mean(local_gpu_usages))
            user_losses_per_epoch.append(np.mean(user_losses_per_iter))
            local_state = copy.deepcopy(client_model.state_dict())
            add_communication(comm_cost_dict, "client_to_server_MB", payload=local_state)
            local_weights.append(local_state)

        training_loss.append(np.mean(user_losses_per_epoch))
        client_model_abs.load_state_dict(strategy.aggregate(None, None, local_weights))

        merge_model.load_weight(copy.deepcopy(client_model_abs.state_dict()), copy.deepcopy(main_server_model.state_dict()))
        eval_iou, eval_dice = evaluate_segmentation(
            merge_model, valid_dataset, config["local_bs"], device
        )
        eval_iou_scores.append(eval_iou)
        eval_dice_scores.append(eval_dice)
        round_cpu_usages.append(np.mean(round_cpu_per_client))
        round_ram_usages.append(np.mean(round_ram_per_client))
        round_gpu_usages.append(np.mean(round_gpu_per_client) if config["is_gpu"] else 0)

        if eval_dice > best_dice:
            best_dice = eval_dice
            best_model_weights = copy.deepcopy(merge_model.state_dict())
            print(f"New Best Dice: {best_dice:.4f} at Epoch {epoch+1}")

        if wandb is not None and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": training_loss[-1],
                "iou": eval_iou,
                "dice": eval_dice,
                "best_dice": best_dice,
                "avg_cpu_pct": round_cpu_usages[-1],
                "avg_ram_pct": round_ram_usages[-1],
                "avg_gpu_ram_MB": round_gpu_usages[-1],
                **{k: v for k, v in comm_cost_dict.items()},
            })

        if (epoch + 1) % config["print_every"] == 0:
            print(
                f"Epoch {epoch+1}: Loss={training_loss[-1]:.4f}  "
                f"IoU={eval_iou:.4f}  Dice={eval_dice:.4f}"
            )
        for k, v in comm_cost_dict.items():
            print(f"  {k}: {v:.2f} MB")

    merge_model.load_state_dict(best_model_weights)
    merge_model.to(device).eval()
    test_iou, test_dice = evaluate_segmentation(merge_model, test_dataset, 1, device)
    total_time = time.time() - start_time
    print(f"\nFinal Test IoU: {test_iou*100:.2f}%  Dice: {test_dice*100:.2f}%")
    print("Total Run Time: {:.2f}s".format(total_time))

    if wandb is not None and wandb.run is not None:
        wandb.summary["test_iou"] = test_iou
        wandb.summary["test_dice"] = test_dice
        wandb.summary["best_dice"] = best_dice
        wandb.summary["total_time_s"] = total_time

    out_dir = os.path.join(os.path.dirname(__file__), "Figure", "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"SplitFL_{config['dataset']}_iid:{config['iid']}_{config['model']}_{config['num_users']}users.json"), "w") as f:
        json.dump({
            "avg_cpu_percent": round_cpu_usages, "avg_ram_percent": round_ram_usages,
            "avg_gpu_memory_MB": round_gpu_usages,
            "train_iou": eval_iou_scores, "train_dice": eval_dice_scores,
            "train_loss": training_loss, "final_test_iou": test_iou,
            "final_test_dice": test_dice,
            "total_comm_MB": comm_cost_dict["total_comm_MB"],
            "comm_report": comm_cost_dict,
        }, f, indent=4)
