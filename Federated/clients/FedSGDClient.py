from .Client import Client
import torch
from torch import nn

class FedSGDClient(Client):
    def update_weights(self, model, global_round):
        model.train()
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

        # Single batch update
        images, labels = next(iter(self.trainloader))
        images, labels = images.to(self.device), labels.to(self.device)

        model.zero_grad()
        log_probs = model(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        self.logger.add_scalar("loss", loss.item())
        return model.state_dict(), loss.item()
