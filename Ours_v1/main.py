import argparse
import copy
import pickle
import time

import numpy as np
import psutil
import torch
from config import ConfigLoader
from data import get_dataset
from hierarchy import HierarchicalFL
from models import get_model
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)
    client_model, egde_model, cloud_model = get_model(config["model"], config["dataset"])
    if config["model"] == "cnn":
        client_model, egde_model, cloud_model = client_model(), egde_model(), cloud_model(config)
    elif config["model"] == "mlp":
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = model(
            dim_in=len_in, dim_hidden=64, dim_out=config["num_classes"]
        )
    hierachical_fl = HierarchicalFL(
        args = config, 
        client_model=client_model,
        client_weights=client_model.state_dict(),
        edge_model=egde_model,
        edge_weights=egde_model.state_dict(),
        cloud_model=cloud_model,
        cloud_weight=cloud_model.state_dict(),
        test_dataset = test_dataset
    )
    hierachical_fl.print_structure()
    
    output = hierachical_fl.train_end_to_end(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        user_groups=user_groups,
        config=config, 
        epochs=config["epochs"]
    )
    train_loss = output["train_loss"]
    train_accuracy = output["train_accuracy"]
    best_model = output["best_weight"]
    client_time_list = output["client_time_list"]
    client_ram_list = output["client_ram"]
    client_gpu_ram_list = output["client_gpu_ram"]
    best_model = best_model.to(device)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            out = best_model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    test_acc = correct / total
    print(f' \n Results after {config["epochs"]} global rounds of training:')
    print("|---- Avg Train F1 Score: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test F1 Score: {:.2f}%".format(100*test_acc))
    # PLOTTING (optional)
    import os

    import matplotlib.pyplot as plt
    os.makedirs('./save', exist_ok=True)

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot([config["t2"] * (i + 1) for i in range(len(train_loss))], train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/Oursv1_{}_{}_loss.png'.
                format(config["dataset"], config["epochs"]))
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average F1 Score vs Communication Rounds')
    plt.plot([config['t2'] * (i + 1) for i in range(len(train_accuracy))], train_accuracy, color='k')
    plt.ylabel('Average F1 Score')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/Oursv1_{}_{}_f1.png'.
                format(config["dataset"], config["epochs"]))
    
    plt.figure()
    plt.title('Average training time in each rounds')
    plt.plot(range(len(client_time_list)), client_time_list, color='k')
    plt.ylabel('Average Training Time')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/Oursv1_{}_{}_training_time.png'.
                format(config["dataset"], config["epochs"]))
    
    plt.figure()
    plt.title('Average CPU usage in each rounds')
    plt.plot(range(len(client_time_list)), client_time_list, color='k')
    plt.ylabel('Average CPU Usage')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/Oursv1_{}_{}_cpu_usage.png'.
                format(config["dataset"], config["epochs"]))
    
    plt.figure()
    plt.title('Average RAM Usage in each rounds')
    plt.plot(range(len(client_ram_list)), client_ram_list, color='k')
    plt.ylabel('Average RAM Usage')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/Oursv1_{}_{}_ram_usage.png'.
                format(config["dataset"], config["epochs"]))
    
    plt.figure()
    plt.title('Average GPU RAM Usage in each rounds')
    plt.plot(range(len(client_gpu_ram_list)), client_gpu_ram_list, color='k')
    plt.ylabel('Average GPU RAM Usage')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/Oursv1_{}_{}_gpu_ram_usage.png'.
                format(config["dataset"], config["epochs"]))
if __name__ == "__main__":
    main()