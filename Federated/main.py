import argparse
import copy
import os
import pickle
import time
import psutil
import pandas as pd

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from clients import get_client_update_strategy, test_inference
from config import ConfigLoader
from data import get_dataset
from models import get_model
from server import get_strategy


def main():
    start_time = time.time()
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
    model = get_model(config["model"], config["dataset"])
    if config["model"] == "cnn":
        global_model = model(config)
    elif config["model"] == "mlp":
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in image_size:
            len_in *= x
        global_model = model(
            dim_in=len_in, dim_hidden=64, dim_out=config["num_classes"]
        )

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
    metrics = {
        "client_cpu": [], # Store CPU utilization for each round
        'global_cpu': [], # Store CPU utilization for global model
        "server_comm": []
    }

    for epoch in tqdm(range(config["epochs"])):
        
        if config["verbose"]:
            print(f"\n | Global Training Round: {epoch+1} |\n")
        global_model.train()
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(
            range(config["num_users"]), m, replace=False
        )

        local_weights, local_losses, local_updates = [], [], []
        
        # Start CPU monitoring
        cpu_start = psutil.cpu_percent()
        
        for idx in idxs_users:
            local_update = get_client_update_strategy(config["strategy"])(
                args=config,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
            )
            w, loss = local_update.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch
            )
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_updates.append((copy.deepcopy(w), copy.deepcopy(loss)))

        # End CPU monitoring and calculate utilization
        cpu_end = psutil.cpu_percent()
        client_cpu_util = round((cpu_start + cpu_end) / 2, 2)

        # Store average CPU utilization for this round
        metrics['client_cpu'].append(client_cpu_util)

        # Compute CPU utilization by global model
        cpu_start = psutil.cpu_percent()

        # update global weights
        global_weights, global_comm = strategy.aggregate(
            local_updates, global_weights, local_weights
        )
        # update global weights
        global_model.load_state_dict(global_weights)

        cpu_end = psutil.cpu_percent()
        global_cpu_util = round((cpu_start + cpu_end) / 2, 2)

        # Append communication & computation overhead
        metrics['global_cpu'].append(global_cpu_util)
        metrics['server_comm'].append(global_comm)

        loss_avg = sum(local_losses) / len(local_losses)
        training_loss.append(loss_avg)

        # Calculate training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(config["num_users"]):
            local_update = get_client_update_strategy(config["strategy"])(
                args=config,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger,
            )
            acc, loss = local_update.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every i rounds
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Loss : {np.mean(np.array(training_loss))}")
            print("Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1]))

    test_acc, test_loss = test_inference(
        args=config, model=global_model, test_dataset=test_dataset
    )
    print(f' \n Results after {config["epochs"]} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Save results including CPU utilization
    file_name = (
        "./save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl".format(
            config["strategy"],
            config["dataset"],
            config["model"],
            config["epochs"],
            config["frac"],
            config["iid"],
            config["local_ep"],
            config["local_bs"],
        )
    )
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump([training_loss, train_accuracy, metrics['client_cpu']], f)

    # Save all metrics to CSV
    metrics_data = {
        'round': range(len(metrics['client_cpu'])),
        'client_cpu_util_percent': metrics['client_cpu'],
        'global_cpu_util_percent': metrics['global_cpu'],
        'server_comm_bytes': metrics['server_comm']
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = (
        "./save/cpu_metrics/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_cpu.csv".format(
            config["strategy"],
            config["dataset"],
            config["model"],
            config["epochs"],
            config["frac"],
            config["iid"],
            config["local_ep"],
            config["local_bs"],
        )
    )
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nCPU utilization data saved to: {metrics_csv_path}")

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    # Plot Loss curve
    plt.figure()
    plt.title("Training Loss vs Communication rounds")
    plt.plot(range(len(training_loss)), training_loss, color="r")
    plt.ylabel("Training loss")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "./save/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png".format(
            config["strategy"],
            config["dataset"],
            config["model"],
            config["epochs"],
            config["frac"],
            config["iid"],
            config["local_ep"],
            config["local_bs"],
        )
    )
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title("Average Accuracy vs Communication rounds")
    plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
    plt.ylabel("Average Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "./save/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png".format(
            config["strategy"],
            config["dataset"],
            config["model"],
            config["epochs"],
            config["frac"],
            config["iid"],
            config["local_ep"],
            config["local_bs"],
        )
    )

    # Plot CPU utilization
    plt.figure()
    plt.title("CPU Utilization vs Communication rounds")
    plt.plot(range(len(metrics['client_cpu'])), metrics['client_cpu'], color="b")
    plt.ylabel("CPU Utilization (%)")
    plt.xlabel("Communication Rounds")
    plt.savefig(
        "./save/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_cpu.png".format(
            config["strategy"],
            config["dataset"],
            config["model"],
            config["epochs"],
            config["frac"],
            config["iid"],
            config["local_ep"],
            config["local_bs"],
        )
    )


if __name__ == "__main__":
    main()