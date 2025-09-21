import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(model, config):
    opt_name = config["training"].get("optimizer", "SGD").lower()
    lr = config["training"].get("lr", 0.01)

    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def get_loss(config):
    loss_name = config["training"].get("loss", "CrossEntropy").lower()

    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")