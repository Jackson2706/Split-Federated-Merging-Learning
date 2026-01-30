import copy

import torch
from torch import nn

from .Client import Client


class FedNovaClient(Client):
    def update_weights(self, model, global_round):
        model.train()
        criterion = nn.NLLLoss().to(self.device)
        model = model.to(self.device)
        local_steps = 0
        global_weights = copy.deepcopy(model.state_dict())
    
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args["lr"], momentum=self.args["momentum"]) \
            if self.args["optimizer"] == 'sgd' else torch.optim.Adam(model.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
        avg_loss = 0.0
        model = model.to(self.device)
        for _ in range(self.args["local_ep"]):
            torch.cuda.empty_cache()
            for images, labels in self.trainloader:
                torch.cuda.empty_cache()
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                self.logger.add_scalar('loss', loss.item())
                avg_loss += loss.item()
                local_steps += 1

        avg_loss /= local_steps

        updated_weights = model.state_dict()

        # Compute pseudo-gradient (normalized update)
        delta_weights = copy.deepcopy(updated_weights)
        for key in delta_weights.keys():
            delta_weights[key] = (delta_weights[key] - global_weights[key]) / local_steps

        return delta_weights, avg_loss
