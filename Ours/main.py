import argparse
import time
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import numpy as np
from config import ConfigLoader
from data import get_dataset
from models import get_model
from hierarchy import HierarchicalFL
from clients import FedAvgClient
import copy
from clients import test_inference
import pickle
import psutil
from torch.utils.data import Dataset, DataLoader

    
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
    
    hierachical_fl.train_end_to_end(
        train_dataset=train_dataset,
        user_groups=user_groups,
        config=config, 
        epochs=config["epochs"]
    )

if __name__ == "__main__":
    main()