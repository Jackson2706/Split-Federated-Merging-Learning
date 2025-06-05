from collections import OrderedDict
from typing import Dict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar

from model import Net, test, train


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_classes)-> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.num_classes = num_classes

        self.model = Net(num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        parameters_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in parameters_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [ val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # copy parameters sent by the server to the loclal model)
        self.set_parameters(parameters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        train(self.model, self.trainloader, optimizer, epochs, self.device)

        return self.get_parameters(config), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # copy parameters sent by the server to the local model
        self.set_parameters(parameters)

        # evaluate the model on the validation set
        loss, accuracy = test(self.model, self.valloader, self.device)

        return loss, len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, validationloaders, num_classes):
    def client_fn(cid:  str):
        # Create a Flower client
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=validationloaders[int(cid)],
            num_classes=num_classes
        )
    
    return client_fn