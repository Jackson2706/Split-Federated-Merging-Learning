import argparse
import copy
import os
import sys
import time

import numpy as np
import psutil
import torch
from clients import get_client_update_strategy, test_inference
from config import ConfigLoader
from data import get_dataset
from models import get_model
from server import get_strategy
from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_weight_size_mb(weights):
    return sum(torch.numel(v) for v in weights.values()) * 4 / (1024**2)


def main():
    start_time = time.time()
    torch.cuda.memory._record_memory_history(max_entries=100000)

    parser = argparse.ArgumentParser(description="Run with config file")
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()

    config_loader = ConfigLoader(args.cfg)
    config = config_loader.get_config()
    print("Baseline: {}".format(config["strategy"]))
    if config["verbose"]:
        print("✅ Loaded Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")
    logger = SummaryWriter("./logs")
    if config["is_gpu"]:
        torch.cuda.set_device(config["gpu"])
    device = torch.device("cuda") if config["is_gpu"] else "cpu"

    train_dataset, test_dataset, user_groups = get_dataset(config)
    global_model = get_model(config["model"], config["dataset"])(config)

    global_model = global_model.to(device)
    global_model.train()
    if config["verbose"]:
        print(global_model)

    strategy = get_strategy(config["strategy"])(config)
    # copy weights
    global_weights = global_model.state_dict()

    # Training
    training_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = config["print_every"]
    val_loss_pre, counter = 0, 0
    train_loss, train_accuracy = [], []
    client_cpu_list = []
    client_time_list = []
    client_ram_list = []
    client_gpu_ram_list = []
    print_every = 2
    comm_cost_dict = {
        "client_model_upload_MB": 0,
        "client_model_download_MB": 0,
    }
    for epoch in tqdm(range(config["epochs"])):
        torch.cuda.empty_cache()
        if config["verbose"]:
            print(f"\n | Global Training Round: {epoch+1} |\n")
        global_model.train()
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(
            range(config["num_users"]), m, replace=False
        )

        local_weights, local_losses, local_updates = [], [], []

        client_cpu_usages = []
        client_compute_times = []
        client_ram_usages = []
        client_gpu_ram_usage = []
        for idx in idxs_users:
            torch.cuda.empty_cache()
            start_time = time.time()
            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = psutil.Process(os.getpid()).memory_info().rss / (
                1024**2
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            local_update = get_client_update_strategy(config["strategy"])(
                args=config,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
            )
            w, loss = local_update.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch
            )

            mem_after = psutil.Process(os.getpid()).memory_info().rss / (
                1024**2
            )
            end_time = time.time()
            cpu_after = psutil.cpu_percent(interval=None)
            mem_gpu_used = torch.cuda.max_memory_allocated(device) / (1024**2)
            # Metrics per client
            elapsed_time = end_time - start_time
            avg_cpu = (cpu_before + cpu_after) / 2
            mem_used = mem_after - mem_before
            client_compute_times.append(elapsed_time)
            client_cpu_usages.append(avg_cpu)
            client_ram_usages.append(mem_used)
            client_gpu_ram_usage.append(mem_gpu_used)
            if config["strategy"] == "fednova":
                comm_cost_dict["client_model_upload_MB"] += get_weight_size_mb(
                    w
                ) + sys.getsizeof(loss) / (1024**2)
            else:
                comm_cost_dict["client_model_upload_MB"] += get_weight_size_mb(
                    w
                )
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_updates.append((copy.deepcopy(w), copy.deepcopy(loss)))
            del w, loss
        loss_avg = sum(local_losses) / len(local_losses)
        training_loss.append(loss_avg)
        avg_time = sum(client_compute_times) / len(client_compute_times)
        avg_cpu = sum(client_cpu_usages) / len(client_cpu_usages)
        avg_ram = (
            sum(client_ram_usages) / len(client_ram_usages)
            if sum(client_ram_usages) > 0
            else 0
        )
        avg_gpu_ram = sum(client_gpu_ram_usage) / len(client_gpu_ram_usage)
        print(f"Round {epoch+1} Metrics:")
        print(f"  ⏱ Avg Time/Client: {avg_time:.2f}s")
        print(f"  💻 Avg CPU/Client: {avg_cpu:.2f}%")
        print(f"  💻 Avg RAM: {avg_ram:.2f} MB")
        print(f"  💻 Avg GPU RAM: {avg_gpu_ram:.2f} MB")

        client_time_list.append(avg_time)
        client_cpu_list.append(avg_cpu)
        client_ram_list.append(avg_ram)
        client_gpu_ram_list.append(avg_gpu_ram)
        # update global weights
        global_weights, global_comm = strategy.aggregate(
            local_updates, global_weights, local_weights
        )
        # update global weights
        global_model.load_state_dict(global_weights)

        # Calculate training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        cpu_start = psutil.cpu_percent()

        comm_cost_dict["client_model_download_MB"] += (
            get_weight_size_mb(global_model.state_dict()) * config["num_users"]
        )
        test_iou, test_dice, test_loss = test_inference(
            args=config, model=global_model, test_dataset=test_dataset
        )
        train_accuracy.append(test_iou)

        # print global training loss after every i rounds
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Loss : {np.mean(np.array(training_loss))}")
            print("Train IoU: {:.2f}% \n".format(100 * train_accuracy[-1]))

        for k, v in comm_cost_dict.items():
            print(f"{k}: {v:.2f} MB")

    test_iou, test_dice, test_loss = test_inference(
        args=config, model=global_model, test_dataset=test_dataset
    )
    print(f' \n Results after {config["epochs"]} global rounds of training:')
    print("|---- Avg Train IoU: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test IoU: {:.2f}%".format(100 * test_iou))
    print("|---- Test Dice: {:.2f}%".format(100 * test_dice))

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
    import json

    filtered_output = {
        "train_loss": training_loss,
        "train_accuracy": train_accuracy,
        "client_time_list": client_time_list,
        "client_cpu_list": client_cpu_list,
        "client_ram_list": client_ram_list,
        "client_gpu_ram_list": client_gpu_ram_list,
    }
    with open(
        f'/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/Figure/data/{config["strategy"]}_{config["dataset"]}_iid:{config["iid"]}_{config["model"]}_{config["num_users"]} users.json',
        "w",
    ) as f:
        json.dump(filtered_output, f, indent=4)


if __name__ == "__main__":
    main()
