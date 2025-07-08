import argparse
import copy
import os
import time
import gc
import json

import numpy as np
import pandas as pd
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
import psutil
from sklearn.metrics import f1_score


class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)


def get_weight_size_mb(weights):
    return sum(torch.numel(v) for v in weights.values()) * 4 / (1024**2)


def estimate_gradient_size_MB(model, input_shape, device="cpu"):
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    numel = output.numel()
    element_size = output.element_size()
    return (numel * element_size) / (1024**2)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run with config file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    config_loader = ConfigLoader(args.cfg)
    config = config_loader.get_config()
    print("Baseline: {}".format(config["strategy"]))

    if config["verbose"]:
        print("\n✅ Loaded Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

    logger = SummaryWriter("./logs")
    if config["is_gpu"]:
        torch.cuda.set_device(config["gpu"])
    device = torch.device("cuda") if config["is_gpu"] else torch.device("cpu")

    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(config)
    model = get_model(config["model"], config["dataset"])

    if config["model"] == "cnn":
        client_model_abs, main_server_model, merge_model = model[0](), model[1](config), model[2](config)
    elif config["model"] == "mlp":
        img_size = train_dataset[0][0].shape
        len_in = np.prod(img_size)
        client_model_abs = model(dim_in=len_in, dim_hidden=64, dim_out=config["num_classes"])
        main_server_model = copy.deepcopy(client_model_abs)
        merge_model = copy.deepcopy(client_model_abs)

    client_model_abs = client_model_abs.to(device)
    main_server_model = main_server_model.to(device)
    merge_model = merge_model.to(device)

    strategy = get_strategy(config["strategy"])(config)
    print_every = config["print_every"]

    training_loss, eval_losses, eval_f1_scores = [], [], []
    round_cpu_usages, round_ram_usages, round_gpu_usages = [], [], []

    comm_cost_dict = {
        "client_upload_smashed_MB": 0,
        "client_model_upload_MB": 0,
        "client_model_download_MB": 0,
        "cloud_download_grad_MB": 0,
    }
    best_f1 = 0.0
    for epoch in tqdm(range(config["epochs"])):
        if config["verbose"]:
            print(f"\n | Global Training Round: {epoch+1} |\n")

        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(range(config["num_users"]), m, replace=False)

        local_weights, user_losses_per_epoch = [], []
        criterion = nn.NLLLoss().to(device)
        server_optimizer = SGD(main_server_model.parameters(), lr=config["lr"], momentum=config["momentum"])

        round_cpu_per_client, round_ram_per_client, round_gpu_per_client = [], [], []

        for idx in idxs_users:
            client_model = copy.deepcopy(client_model_abs).to(device)
            client_optimizer = SGD(client_model.parameters(), lr=config["lr"], momentum=config["momentum"])
            client_model.train()

            local_train_dataset = DatasetSplit(train_dataset, user_groups[idx])
            local_train_loader = DataLoader(local_train_dataset, batch_size=config["local_bs"], shuffle=True)

            local_cpu_usages, local_ram_usages, local_gpu_usages = [], [], []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            user_losses_per_iter = []
            for _ in range(config["local_ep"]):
                local_cpu_usages.append(psutil.cpu_percent(interval=None))
                local_ram_usages.append(psutil.virtual_memory().percent)
                if config["is_gpu"]:
                    local_gpu_usages.append(torch.cuda.memory_allocated(device=device) / 1024**2)

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

                    comm_cost_dict["client_upload_smashed_MB"] += (activation.numel() + label.numel()) * 4 / (1024**2)
                    comm_cost_dict["cloud_download_grad_MB"] += estimate_gradient_size_MB(main_server_model, activation.shape, device)

                user_losses_per_iter.append(np.mean(losses))

            round_cpu_per_client.append(np.mean(local_cpu_usages))
            round_ram_per_client.append(np.mean(local_ram_usages))
            if config["is_gpu"]:
                round_gpu_per_client.append(np.mean(local_gpu_usages))

            user_losses_per_epoch.append(np.mean(user_losses_per_iter))
            comm_cost_dict["client_model_upload_MB"] += get_weight_size_mb(client_model.state_dict())
            local_weights.append(copy.deepcopy(client_model.state_dict()))

        training_loss.append(np.mean(user_losses_per_epoch))
        aggregated_local_model = strategy.aggregate(None, None, local_weights)
        client_model_abs.load_state_dict(aggregated_local_model)

        comm_cost_dict["client_model_download_MB"] += get_weight_size_mb(client_model_abs.state_dict()) * config["num_users"]

        # Evaluation
        all_preds, all_labels = [], []
        total_loss, total_samples = 0.0, 0
        merge_model.load_weight(copy.deepcopy(client_model_abs.state_dict()), copy.deepcopy(main_server_model.state_dict()))
        merge_model.to(device)
        merge_model.eval()

        with torch.no_grad():
            eval_loader = DataLoader(valid_dataset, batch_size=config["local_bs"], shuffle=False)
            for image, label in eval_loader:
                image, label = image.to(device), label.to(device)
                outputs = merge_model(image)
                loss = criterion(outputs, label)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                total_loss += loss.item() * label.size(0)
                total_samples += label.size(0)

        eval_f1 = f1_score(all_labels, all_preds, average="macro")
        
        round_cpu_usages.append(np.mean(round_cpu_per_client))
        round_ram_usages.append(np.mean(round_ram_per_client))
        round_gpu_usages.append(np.mean(round_gpu_per_client) if config["is_gpu"] else 0)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}: Train Loss {training_loss[-1]:.4f}, Eval F1 {eval_f1:.4f}")
        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_model_weights = copy.deepcopy(merge_model.state_dict())
            print(f"New Best F1 Score: {best_f1:.4f} at Epoch {epoch+1}")

        with torch.no_grad():
            eval_loader = DataLoader(valid_dataset, batch_size=config["local_bs"], shuffle=False)
            for image, label in eval_loader:
                image, label = image.to(device), label.to(device)
                outputs = merge_model(image)
                loss = criterion(outputs, label)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                total_loss += loss.item() * label.size(0)
                total_samples += label.size(0)

        eval_f1_scores.append(f1_score(all_labels, all_preds, average="macro"))
        print(f"F1 Score: {eval_f1_scores[-1]:.4f}, Loss: {total_loss / total_samples:.4f}")
        for k,v in comm_cost_dict.items():
            print(f"{k}: {v:.2f} MB")
    # Final Test
    test_preds, test_labels, test_loss = [], [], 0.0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    merge_model.load_state_dict(best_model_weights)
    merge_model.to(device)
    merge_model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            outputs = merge_model(image)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(label.cpu().numpy())
            test_loss += loss.item()

    test_f1 = f1_score(test_labels, test_preds, average="macro")
    avg_test_loss = test_loss / len(test_loader)
    print(f"\n✅ Final Results: Test F1 {test_f1*100:.2f}%, Test Loss {avg_test_loss:.4f}")

    metrics_dict = {
        "avg_cpu_percent": round_cpu_usages,
        "avg_ram_percent": round_ram_usages,
        "avg_gpu_memory_MB": round_gpu_usages,
        "train_accuracy": eval_f1_scores,
        "train_loss": training_loss,
        "final_test_f1": test_f1,
        "final_test_loss": avg_test_loss,
    }

    json_path = f"/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/{config['dataset']}_SplitFed_{config['num_users']}_{config['epochs']}_{config['local_ep']}_output.json"
    os.makedirs("/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"\n📦 Metrics saved to JSON: {json_path}")
    print("⏱ Total Run Time: {:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()