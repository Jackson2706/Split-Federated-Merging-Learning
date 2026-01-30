import copy

import torch
from torch import nn

from .Client import Client


class FedProxClient(Client):
    def update_weights(self, model, global_round):
        model.train()
        model = model.to(self.device)
        global_weights = copy.deepcopy(model.state_dict())
        criterion = nn.NLLLoss().to(self.device)
        optimizer = (
            torch.optim.SGD(
                model.parameters(), lr=self.args["lr"], momentum=self.args["momentum"]
            )
            if self.args["optimizer"] == "sgd"
            else torch.optim.Adam(
                model.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"]
            )
        )

        mu = self.args["mu"]
        epoch_loss = []
        for _ in range(self.args["local_ep"]):
            torch.cuda.empty_cache()
            batch_loss = []
            for images, labels in self.trainloader:
                torch.cuda.empty_cache()
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                prox_term = sum(
                    torch.norm(param - global_weights[name]) ** 2
                    for name, param in model.named_parameters()
                )
                loss += (mu / 2) * prox_term
                loss.backward()
                optimizer.step()
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
