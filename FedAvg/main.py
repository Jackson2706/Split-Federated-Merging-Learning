import argparse
import yaml
import os
import torch
from data import get_dataset
from models import get_model
from tqdm import tqdm
import numpy as np
from update import LocalUpdate, test_inference
from tensorboardX import SummaryWriter
import copy
import pickle
import time

from strategies import get_strategy
def load_config(cfg_path):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run with config file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_config(args.cfg)

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
    elif config["model"] =="mlp":
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in image_size:
            len_in *= x
        global_model = model(dim_in=len_in, dim_hidden = 64, dim_out = config["num_classes"])

    global_model = global_model.to(device)
    global_model.train()
    print(global_model)

    strategy = get_strategy(config["strategy"])
    # copy weights
    global_weights = global_model.state_dict()

    # Training
    training_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(config["epochs"])):
        local_weights, local_losses = [], []
        print(f"\n | Global Training Round: {epoch+1} |\n")
        global_model.train()
        m = max(int(config["frac"] * config["num_users"]), 1)
        idxs_users = np.random.choice(range(config["num_users"]), m, replace=False)

        for idx in idxs_users:
            local_update = LocalUpdate(args=config, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss = local_update.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch
            )
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = strategy(local_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        training_loss.append(loss_avg)

        # Calculate training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for idx in range(config["num_users"]):
            local_update = LocalUpdate(
                args=config, dataset=train_dataset, idxs=user_groups[idx], logger=logger
            )
            acc, loss = local_update.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        #print global training loss after every i rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(training_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    test_acc, test_loss = test_inference(args=config, model=global_model, test_dataset=test_dataset)
    print(f' \n Results after {config["epochs"]} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(config["dataset"], config["model"], config["epochs"], config["frac"], config["iid"],
               config["local_ep"], config["local_bs"])
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        pickle.dump([training_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(training_loss)), training_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(config["dataset"], config["model"], config["epochs"], config["frac"],
                       config["iid"], config["local_ep"], config["local_bs"]))
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(config["dataset"], config["model"], config["epochs"], config["frac"],
                       config["iid"], config["local_ep"], config["local_bs"]))
if __name__ == "__main__":
    main()
