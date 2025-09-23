import argparse

import yaml

from src.baseline_factory import get_baseline
from src.dataset.mnist_dataset import MNISTDataset
from src.dataset.cifar100_dataset import CIFAR100Dataset
from src.models.model_factory import get_model


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config["experiment"]["dataset"]== "MNIST":
        dataset = MNISTDataset(config)
    elif config["experiment"]["dataset"] == "CIFAR100":
        dataset = CIFAR100Dataset(config)
    clients, test_dataset = dataset.prepare()

    # Global model
    model_name = config["experiment"]["model"]
    global_model = get_model(model_name)

    # Edge config
    try:
        num_edge = config["experiment"]["edges"] 
    except:
        num_edge=None
    # Get baseline cloud class
    baseline_cfg = get_baseline(config["experiment"]["baseline"])
    CloudClass = baseline_cfg["cloud"]

    cloud = CloudClass(
        global_model=global_model, 
        clients=clients, 
        edges=num_edge, 
        config=config, 
        test_dataset=test_dataset
    )
    cloud.run(config["experiment"]["rounds"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg)