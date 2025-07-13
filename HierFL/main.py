import argparse
import copy
import os
import pickle
import time
import warnings

import numpy as np
import psutil
import torch
from clients import FedAvgClient, test_inference
from config import ConfigLoader
from data import get_dataset
from fvcore.nn import FlopCountAnalysis
from hierarchy import HierarchicalFL
from models import get_model
from tensorboardX import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_model_size(model):
    # Assumes model is in float32
    param_size = sum(p.numel() for p in model.parameters())
    bytes_size = param_size * 4  # float32 = 4 bytes
    mb_size = bytes_size / (1024 ** 2)
    return mb_size

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
    global_model = get_model(config["model"], config["dataset"])(config)

    global_model = global_model.to(device)
    global_model.train()
    if config["verbose"]:
        print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Create Hierarchical FL
    hierarchical_fl = HierarchicalFL(
        config, global_weights, global_model, test_dataset
    )
    hierarchical_fl.print_structure()
    # Training
    train_loss, train_accuracy = [], []
    client_cpu_list = [] 
    client_time_list = []
    client_ram_list = []
    client_gpu_ram_list = []
    print_every = 2

    for epoch in tqdm(range(config["epochs"])):
        local_weights, local_losses = {}, []
        if config["verbose"]:
            print(f"\n | Global Training Round: {epoch+1} |\n")
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(
            range(config["num_users"]), m, replace=False
        )

        # Start CPU monitoring
        client_cpu_usages = []
        client_compute_times = []
        client_ram_usages = []
        client_gpu_ram_usage = []
        for idx in idxs_users:
            start_time = time.time()
            cpu_before = psutil.cpu_percent(interval=None)
            import os
            mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            local_update = FedAvgClient(
                args=config, dataset=train_dataset, idxs=user_groups[idx], logger=logger
            )
            client_model, server_idx = hierarchical_fl.get_model_for_client(
                idx, config["download"]
            )
            w, loss = local_update.update_weights(
                model=client_model, global_round=epoch
            )

            # End time and CPU
            mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
            end_time = time.time()
            cpu_after = psutil.cpu_percent(interval=None)
            mem_gpu_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            # Metrics per client
            elapsed_time = end_time - start_time
            avg_cpu = (cpu_before + cpu_after) / 2
            mem_used = mem_after - mem_before
            client_compute_times.append(elapsed_time)
            client_cpu_usages.append(avg_cpu)
            client_ram_usages.append(mem_used)
            client_gpu_ram_usage.append(mem_gpu_used)

            # prepare local weights for uploading weights
            if server_idx in local_weights.keys():
                local_weights[server_idx].append((copy.deepcopy(w)))
            else:
                local_weights[server_idx] = [copy.deepcopy(w)]

            local_losses.append(copy.deepcopy(loss))

        # Update & store CPU utilization
        avg_time = sum(client_compute_times) / len(client_compute_times)
        avg_cpu = sum(client_cpu_usages) / len(client_cpu_usages)
        avg_ram = sum(client_ram_usages) / len(client_ram_usages) if sum(client_ram_usages) > 0 else 0
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
        

        # update system weights
        hierarchical_fl.upload_client_weights(local_weights)      

        # Top - down model management
        if config["management"]:
            if config["verbose"]:
                print("Management is activated")
            hierarchical_fl.manage_models_top_down()
        # compute the loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        test_acc, test_loss = test_inference(config, global_model, test_dataset)
        train_accuracy.append(test_acc)
        
        # # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train F1 Score: {:.2f}% \n'.format(100*train_accuracy[-1]))

        for k, v in hierarchical_fl.get_communication_status().items():
            print(f"{k}: {v:.2f} MB")

    # Test inference after completion of training
    test_acc, test_loss = test_inference(config, global_model, test_dataset)
    
    print(f' \n Results after {config["epochs"]} global rounds of training:')
    print("|---- Avg Train F1 Score: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test F1 Score: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(config["dataset"], config["model"], config["epochs"], config["frac"], config["iid"],
               config["local_ep"], config["local_bs"])
    
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    
    import json
    filtered_output = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "client_time_list": client_time_list,
        "client_cpu_list": client_cpu_list,
        "client_ram_list": client_ram_list,
        "client_gpu_ram_list": client_gpu_ram_list,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
    }
    with open(f'/home/jackson/Desktop/Split-Federated-Merging-Learning/Figure/data/HierFL_{config["dataset"]}_iid:{config["iid"]}_{config["model"]}_{config["num_users"]} users.json', 'w') as f:
        json.dump(filtered_output, f, indent=4)

   
if __name__ == "__main__":
    main()