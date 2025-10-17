import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from config import ConfigLoader
from data import get_dataset
from hierarchy import HierarchicalFL
from models import get_model
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader




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

    train_dataset, test_dataset, user_groups = get_dataset(
        config
    )

    client_model, egde_model, cloud_model = get_model(
        config["model"], config["dataset"]
    )
    client_model, egde_model, cloud_model = (
        client_model(),
        egde_model(),
        cloud_model(config),
    )

    hierachical_fl = HierarchicalFL(
        args=config,
        client_model=client_model,
        client_weights=client_model.state_dict(),
        edge_model=egde_model,
        edge_weights=egde_model.state_dict(),
        cloud_model=cloud_model,
        cloud_weight=cloud_model.state_dict(),
        test_dataset=test_dataset,
    )
    hierachical_fl.print_structure()

    output = hierachical_fl.train_end_to_end(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        user_groups=user_groups,
        config=config,
        epochs=config["epochs"],
    )
    import json

    exclude_keys = ["best_weight"]  # ví dụ các key bạn muốn bỏ
    filtered_output = {k: v for k, v in output.items() if k not in exclude_keys}
    json_path = f"/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/Figure/data/OursV1_no_Supcon_{config['dataset']}_iid:{config['iid']}_{config['model']}_{config['num_users']} users_t1:{config['t1']}_t2:{config['t2']}.json"
    os.makedirs(
        "/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure",
        exist_ok=True,
    )
    with open(json_path, "w") as f:
        json.dump(filtered_output, f, indent=4)

    train_loss = output["train_loss"]
    train_accuracy = output["train_accuracy"]
    best_model = output["best_weight"]
    checkpoint_path = f"./{config['dataset']}_iid:{config['iid']}_{config['model']}_{config['num_users']} users_t1:{config['t1']}_t2:{config['t2']}.pt"
    torch.save(best_model, checkpoint_path)
    client_time_list = output["client_time_list"]
    client_ram_list = output["client_ram"]
    client_gpu_ram_list = output["client_gpu_ram"]
    best_model = best_model.to(device)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    from sklearn.metrics import f1_score

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = best_model(data)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute macro-F1 (recommended for imbalanced classes)
    f1 = f1_score(
        all_targets, all_preds, average="macro"
    )  # or 'micro', 'weighted' as needed

    print(f"\n Results after {config['epochs']} global rounds of training:")
    print("|---- Avg Train F1 Score: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test F1 Score: {:.2f}%".format(100 * f1))


if __name__ == "__main__":
    main()
