import copy
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
from ehsfp.communication import add_communication, mb_of, new_communication_tracker
from clients import get_client_update_strategy, test_inference
from config import ConfigLoader
from data import get_dataset
from models import get_model
from server import get_strategy
from segmentation.training_metrics import BestSegmentationMetrics
from tensorboardX import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


def get_weight_size_mb(weights):
    return sum(torch.numel(v) for v in weights.values()) * 4 / (1024**2)


def run(cfg_path: str):
    start_time = time.time()
    torch.cuda.memory._record_memory_history(max_entries=100000)

    config_loader = ConfigLoader(cfg_path)
    config = config_loader.get_config()
    print("Method: {}".format(config["strategy"]))

    logger = SummaryWriter("./logs")
    if config["is_gpu"]:
        torch.cuda.set_device(config["gpu"])
    device = torch.device("cuda") if config["is_gpu"] else torch.device("cpu")

    train_dataset, test_dataset, user_groups = get_dataset(config)
    global_model = get_model(config["model"], config["dataset"])(config).to(device)
    global_model.train()

    strategy = get_strategy(config["strategy"])(config)
    global_weights = global_model.state_dict()

    training_loss, train_accuracy = [], []
    client_cpu_list, client_time_list, client_ram_list, client_gpu_ram_list = [], [], [], []
    print_every = 2
    comm_cost_dict = new_communication_tracker()
    best = BestSegmentationMetrics()
    best_weights = None

    for epoch in tqdm(range(config["epochs"])):
        torch.cuda.empty_cache()
        global_model.train()
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(range(config["num_users"]), m, replace=False)

        local_weights, local_losses, local_updates = [], [], []
        client_cpu_usages, client_compute_times, client_ram_usages, client_gpu_ram_usage = [], [], [], []

        for idx in idxs_users:
            torch.cuda.empty_cache()
            t0 = time.time()
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            torch.cuda.reset_peak_memory_stats()

            local_update = get_client_update_strategy(config["strategy"])(
                args=config, dataset=train_dataset, idxs=user_groups[idx], logger=logger
            )
            w, loss = local_update.update_weights(model=copy.deepcopy(global_model), global_round=epoch)

            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            cpu_after = psutil.cpu_percent(interval=None)
            client_compute_times.append(time.time() - t0)
            client_cpu_usages.append((cpu_before + cpu_after) / 2)
            client_ram_usages.append(mem_after - mem_before)
            client_gpu_ram_usage.append(torch.cuda.max_memory_allocated(device) / (1024**2))

            upload_mb = mb_of(w) + (8 / (1024**2) if config["strategy"] == "fednova" else 0)
            add_communication(comm_cost_dict, "client_to_server_MB", mb=upload_mb)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_updates.append((copy.deepcopy(w), copy.deepcopy(loss)))
            del w, loss

        training_loss.append(sum(local_losses) / len(local_losses))
        avg_time = sum(client_compute_times) / len(client_compute_times)
        avg_cpu = sum(client_cpu_usages) / len(client_cpu_usages)
        avg_ram = sum(client_ram_usages) / len(client_ram_usages) if sum(client_ram_usages) > 0 else 0
        avg_gpu_ram = sum(client_gpu_ram_usage) / len(client_gpu_ram_usage)
        print(f"Round {epoch+1}: time={avg_time:.2f}s  cpu={avg_cpu:.1f}%  ram={avg_ram:.1f}MB  gpu={avg_gpu_ram:.1f}MB")

        client_time_list.append(avg_time)
        client_cpu_list.append(avg_cpu)
        client_ram_list.append(avg_ram)
        client_gpu_ram_list.append(avg_gpu_ram)

        global_weights, _ = strategy.aggregate(local_updates, global_weights, local_weights)
        global_model.load_state_dict(global_weights)
        add_communication(comm_cost_dict, "server_to_client_MB", payload=global_model.state_dict(), copies=len(idxs_users))

        global_model.eval()
        test_iou, test_dice, test_loss = test_inference(args=config, model=global_model, test_dataset=test_dataset)
        train_accuracy.append(test_iou)
        if best.update(test_iou, test_dice, epoch + 1):
            best_weights = copy.deepcopy(global_model.state_dict())
            print(f" -> New Best IoU: {best.iou:.4f}")

        if wandb is not None and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": training_loss[-1],
                "iou": test_iou,
                "dice": test_dice,
                "best_iou": best.iou,
                "avg_client_time_s": avg_time,
                "avg_client_cpu_pct": avg_cpu,
                "avg_client_ram_MB": avg_ram,
                "avg_client_gpu_ram_MB": avg_gpu_ram,
                **{k: v for k, v in comm_cost_dict.items()},
            })

        if (epoch + 1) % print_every == 0:
            print(f"Avg Stats after {epoch+1} rounds: Loss={np.mean(np.array(training_loss)):.4f}  IoU={100*train_accuracy[-1]:.2f}%")
        for k, v in comm_cost_dict.items():
            print(f"  {k}: {v:.2f} MB")

    last_iou, last_dice = test_iou, test_dice
    if best_weights is not None:
        global_model.load_state_dict(best_weights)
    test_iou, test_dice, test_loss = test_inference(args=config, model=global_model, test_dataset=test_dataset)
    total_time = time.time() - start_time
    print(f"\nResults after {config['epochs']} global rounds:")
    print("|---- Best Validation IoU: {:.2f}%  Dice: {:.2f}%  Round: {}".format(100 * best.iou, 100 * best.dice, best.round))
    print("|---- Best-checkpoint Test IoU: {:.2f}%  Test Dice: {:.2f}%".format(100 * test_iou, 100 * test_dice))
    print("Total Run Time: {:.4f}s".format(total_time))

    if wandb is not None and wandb.run is not None:
        wandb.summary["test_iou"] = test_iou
        wandb.summary["test_dice"] = test_dice
        wandb.summary["best_iou"] = best.iou
        wandb.summary["best_dice"] = best.dice
        wandb.summary["best_round"] = best.round
        wandb.summary["total_time_s"] = total_time

    out_dir = os.path.join(os.path.dirname(__file__), "Figure", "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{config['strategy']}_{config['dataset']}_iid:{config['iid']}_{config['model']}_{config['num_users']}users.json"), "w") as f:
        json.dump({
            "train_loss": training_loss, "train_accuracy": train_accuracy,
            "client_time_list": client_time_list, "client_cpu_list": client_cpu_list,
            "client_ram_list": client_ram_list, "client_gpu_ram_list": client_gpu_ram_list,
            "test_iou": test_iou, "test_dice": test_dice,
            "best_iou": best.iou, "best_dice": best.dice, "best_round": best.round,
            "last_iou": last_iou, "last_dice": last_dice,
            "total_comm_MB": comm_cost_dict["total_comm_MB"],
            "comm_report": comm_cost_dict,
        }, f, indent=4)
