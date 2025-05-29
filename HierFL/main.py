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
        for x in img_size:
            len_in *= x
        global_model = model(
            dim_in=len_in, dim_hidden=64, dim_out=config["num_classes"]
        )

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
    print_every = 2

    for epoch in tqdm(range(config["epochs"])):
        local_weights, local_losses = {}, []
        if config["verbose"]:
            print(f"\n | Global Training Round: {epoch+1} |\n")
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(
            range(config["num_users"]), m, replace=False
        )

        for idx in idxs_users:
            local_update = FedAvgClient(
                args=config, dataset=train_dataset, idxs=user_groups[idx], logger=logger
            )
            client_model, server_idx = hierarchical_fl.get_model_for_client(
                idx, config["download"]
            )
            w, loss = local_update.update_weights(
                model=client_model, global_round=epoch
            )

            # prepare local weights for uploading weights
            if server_idx in local_weights.keys():
                local_weights[server_idx].append((copy.deepcopy(w)))
            else:
                local_weights[server_idx] = [copy.deepcopy(w)]

            local_losses.append(copy.deepcopy(loss))

        # update system weights
        hierarchical_fl.upload_client_weights(local_weights)

            # Top - down model management
        if config["management"]:
            print("Management is activated")
            hierarchical_fl.manage_models_top_down()
        # compute the loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(config["num_users"]):
            local_model = FedAvgClient(args=config, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            user_model, _ = hierarchical_fl.get_model_for_client(idx, config["download"])
            acc, loss = local_model.inference(model=user_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        #
        # # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(config, global_model, test_dataset)
    
    print(f' \n Results after {config["epochs"]} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(config["dataset"], config["model"], config["epochs"], config["frac"], config["iid"],
               config["local_ep"], config["local_bs"])
    
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib.pyplot as plt

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/hierFed_{}_{}_loss.png'.
                format(config["dataset"], config["epochs"]))
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/hierFed_{}_{}_acc.png'.
                format(config["dataset"], config["epochs"]))
    
if __name__ == "__main__":
    main()