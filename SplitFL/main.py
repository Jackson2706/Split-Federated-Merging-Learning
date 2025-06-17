import argparse
import copy
import os
import pickle
import time

import numpy as np
import pandas as pd
import psutil
import torch
from config import ConfigLoader
from data import get_dataset
from FedServer import get_strategy
from models import get_model
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)


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
        client_model, main_server_model, merge_model = (
            model[0](),
            model[1](config),
            model[2](config),
        )
    elif config["model"] == "mlp":
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in image_size:
            len_in *= x
        global_model = model(
            dim_in=len_in, dim_hidden=64, dim_out=config["num_classes"]
        )

    client_model, main_server_model, merge_model = (
        client_model.to(device),
        main_server_model.to(device),
        merge_model.to(device),
    )
    if config["verbose"]:
        print(global_model)
    strategy = get_strategy(config["strategy"])(config)
    # Training
    training_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = config["print_every"]
    val_loss_pre, counter = 0, 0
    client_cpu_utils = []  # Store CPU utilization for each round

    for epoch in tqdm(range(config["epochs"])):

        if config["verbose"]:
            print(f"\n | Global Training Round: {epoch+1} |\n")
        # global_model.train()
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(
            range(config["num_users"]), m, replace=False
        )

        local_weights, local_losses, local_updates = [], [], []

        # Start CPU monitoring
        start_time = time.time()
        criterion = nn.NLLLoss().to(device)
        client_optimizer = SGD(
            client_model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
        )
        server_optimizer = SGD(
            main_server_model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
        )
        user_losses_per_epoch = []
        for idx in idxs_users:
            user_losses_per_iter = []
            for iter in range(config["local_ep"]):
                local_train_dataset = DatasetSplit(
                    dataset=train_dataset, idxs=user_groups[idx]
                )
                local_train_loader = DataLoader(
                    dataset=local_train_dataset,
                    batch_size=config["local_bs"],
                    shuffle=True,
                )
                # print(len(local_train_loader.dataset))
                losses = []
                for image, label in local_train_loader:
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
                user_losses_per_iter.append(sum(losses) / len(losses))
            user_losses_per_epoch.append(
                sum(user_losses_per_iter) / len(user_losses_per_iter)
            )
            local_weights.append(copy.deepcopy(client_model.state_dict()))
        training_loss.append(
            sum(user_losses_per_epoch) / len(user_losses_per_epoch)
        )
        aggregated_local_model = strategy.aggregate(None, None, local_weights)
        client_model.load_state_dict(aggregated_local_model)

        # End CPU monitoring and calculate utilization
        end_time = time.time()
        interval = end_time - start_time
        round_cpu_util = psutil.cpu_percent(interval=interval)

        # Store average CPU utilization for this round
        client_cpu_utils.append(round_cpu_util)

        merge_model.load_weight(
            aggregated_local_model, main_server_model.state_dict()
        )
        list_acc = []
        for idx in range(config["num_users"]):
            local_train_dataset = DatasetSplit(
                dataset=train_dataset, idxs=user_groups[idx]
            )
            local_train_loader = DataLoader(
                dataset=local_train_dataset,
                batch_size=config["local_bs"],
                shuffle=False,
            )
            merge_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for image, label in local_train_loader:
                    image, label = image.to(device), label.to(device)
                    outputs = merge_model(image)
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            train_accuracy.append(float(correct) / total)

        # print global training loss after every i rounds
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Loss : {np.mean(np.array(training_loss))}")
            print("Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1]))


    test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                shuffle=False,
    )
    merge_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            outputs = merge_model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    test_acc = correct/total
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
        pickle.dump([training_loss, train_accuracy, client_cpu_utils], f)

    # Save CPU utilization data to CSV
    cpu_data = {
        "round": range(len(client_cpu_utils)),
        "cpu_utilization": client_cpu_utils,
    }
    cpu_df = pd.DataFrame(cpu_data)
    cpu_csv_path = "./save/cpu_metrics/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_cpu.csv".format(
        config["strategy"],
        config["dataset"],
        config["model"],
        config["epochs"],
        config["frac"],
        config["iid"],
        config["local_ep"],
        config["local_bs"],
    )
    os.makedirs(os.path.dirname(cpu_csv_path), exist_ok=True)
    cpu_df.to_csv(cpu_csv_path, index=False)
    print(f"\nCPU utilization data saved to: {cpu_csv_path}")

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
    plt.plot(range(len(client_cpu_utils)), client_cpu_utils, color="b")
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
