import copy
import json
import os
import time
import warnings

import numpy as np
import psutil
import torch
from clients import FedAvgClient, test_inference
from config import ConfigLoader
from data import get_dataset
from hierarchy import HierarchicalFL
from models import get_model
from tensorboardX import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

warnings.filterwarnings("ignore")


def run(cfg_path: str):
    start_time = time.time()
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

    global_weights = global_model.state_dict()
    hierarchical_fl = HierarchicalFL(config, global_weights, global_model, test_dataset)
    hierarchical_fl.print_structure()

    train_loss, train_accuracy = [], []
    client_cpu_list, client_time_list, client_ram_list, client_gpu_ram_list = [], [], [], []
    print_every = 2

    for epoch in tqdm(range(config["epochs"])):
        local_weights, local_losses = {}, []
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(range(config["num_users"]), m, replace=False)

        client_cpu_usages, client_compute_times, client_ram_usages, client_gpu_ram_usage = [], [], [], []

        for idx in idxs_users:
            t0 = time.time()
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            local_update = FedAvgClient(args=config, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            client_model, server_idx = hierarchical_fl.get_model_for_client(idx, config.get("download", True))
            w, loss = local_update.update_weights(model=client_model, global_round=epoch)

            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            cpu_after = psutil.cpu_percent(interval=None)
            client_compute_times.append(time.time() - t0)
            client_cpu_usages.append((cpu_before + cpu_after) / 2)
            client_ram_usages.append(mem_after - mem_before)
            client_gpu_ram_usage.append(torch.cuda.max_memory_allocated(device) / (1024**2))

            # Keep accumulated client weights on CPU so frac=1.0 (200 clients)
            # does not pile 200 state_dicts onto the GPU (-> CUDA OOM).
            w_cpu = {k: v.detach().to("cpu") for k, v in w.items()}
            if server_idx in local_weights:
                local_weights[server_idx].append(w_cpu)
            else:
                local_weights[server_idx] = [w_cpu]
            local_losses.append(loss)
            del w
            torch.cuda.empty_cache()

        avg_time = sum(client_compute_times) / len(client_compute_times)
        avg_cpu = sum(client_cpu_usages) / len(client_cpu_usages)
        avg_ram = sum(client_ram_usages) / len(client_ram_usages) if sum(client_ram_usages) > 0 else 0
        avg_gpu_ram = sum(client_gpu_ram_usage) / len(client_gpu_ram_usage)
        print(f"Round {epoch+1}: time={avg_time:.2f}s  cpu={avg_cpu:.1f}%  ram={avg_ram:.1f}MB  gpu={avg_gpu_ram:.1f}MB")

        client_time_list.append(avg_time)
        client_cpu_list.append(avg_cpu)
        client_ram_list.append(avg_ram)
        client_gpu_ram_list.append(avg_gpu_ram)

        hierarchical_fl.upload_client_weights(local_weights)
        if config["management"]:
            hierarchical_fl.manage_models_top_down()

        train_loss.append(sum(local_losses) / len(local_losses))
        global_model.eval()
        test_acc, test_loss = test_inference(config, global_model, test_dataset)
        train_accuracy.append(test_acc)

        if wandb is not None and wandb.run is not None:
            log_data = {
                "epoch": epoch + 1,
                "train_loss": train_loss[-1],
                "f1": train_accuracy[-1],
                "avg_client_time_s": avg_time,
                "avg_client_cpu_pct": avg_cpu,
                "avg_client_ram_MB": avg_ram,
                "avg_client_gpu_ram_MB": avg_gpu_ram,
            }
            log_data.update(hierarchical_fl.get_communication_status())
            wandb.log(log_data)

        if (epoch + 1) % print_every == 0:
            print(f"Avg Training Stats after {epoch+1} rounds:")
            print(f"  Loss: {np.mean(np.array(train_loss)):.4f}  F1: {100*train_accuracy[-1]:.2f}%")

        for k, v in hierarchical_fl.get_communication_status().items():
            print(f"  {k}: {v:.2f} MB")

    test_acc, test_loss = test_inference(config, global_model, test_dataset)
    total_time = time.time() - start_time
    print(f"\nResults after {config['epochs']} global rounds:")
    print("|---- Avg Train Acc: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Acc: {:.2f}%".format(100 * test_acc))
    print("Total Run Time: {:.4f}s".format(total_time))

    if wandb is not None and wandb.run is not None:
        wandb.summary["test_f1"] = test_acc
        wandb.summary["total_time_s"] = total_time

    out = {
        "train_loss": train_loss, "train_accuracy": train_accuracy,
        "client_time_list": client_time_list, "client_cpu_list": client_cpu_list,
        "client_ram_list": client_ram_list, "client_gpu_ram_list": client_gpu_ram_list,
        "test_accuracy": test_acc, "test_loss": test_loss,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "Figure", "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"HierFL_{config['dataset']}_iid:{config['iid']}_{config['model']}_{config['num_users']}users.json"), "w") as f:
        json.dump(out, f, indent=4)
