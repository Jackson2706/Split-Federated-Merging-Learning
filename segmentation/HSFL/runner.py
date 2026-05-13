import json
import os
import time

import torch
from config import ConfigLoader
from data import get_dataset
from hierarchy import HierarchicalFL
from models import get_model
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


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
    client_model, edge_model, cloud_model = get_model(config["model"], config["dataset"])
    client_model, edge_model, cloud_model = (
        client_model(), edge_model(), cloud_model(config)
    )

    hierarchical_fl = HierarchicalFL(
        args=config,
        client_model=client_model,
        client_weights=client_model.state_dict(),
        edge_model=edge_model,
        edge_weights=edge_model.state_dict(),
        cloud_model=cloud_model,
        cloud_weight=cloud_model.state_dict(),
        test_dataset=test_dataset,
    )
    hierarchical_fl.print_structure()

    output = hierarchical_fl.train_end_to_end(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        user_groups=user_groups,
        config=config,
        epochs=config["epochs"],
    )

    filtered_output = {k: v for k, v in output.items() if k != "best_weight"}
    filename = (
        f"HSFL_{config['dataset']}_iid:{config['iid']}_{config['model']}_"
        f"{config['num_users']}users_t1:{config['t1']}_t2:{config['t2']}.json"
    )
    out_dir = os.path.join(os.path.dirname(__file__), "Figure", "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(filtered_output, f, indent=4)

    best_model = output["best_weight"].to(device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = best_model(data).argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"\nResults after {config['epochs']} global rounds:")
    print("|---- Avg Train F1: {:.2f}%".format(100 * output["train_accuracy"][-1]))
    print("|---- Test F1: {:.2f}%".format(100 * f1))
    print("Total Run Time: {:.4f}s".format(time.time() - start_time))
