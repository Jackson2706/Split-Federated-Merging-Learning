import torch
from torch import nn

from .Client import Client


class FedAvgClient(Client):
    def update_weights(self, model, global_round):
        model.train()
        criterion = nn.NLLLoss().to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args["lr"], momentum=self.args["momentum"]) \
            if self.args["optimizer"] == 'sgd' else torch.optim.Adam(model.parameters(), lr=self.args["lr"], weight_decay=1e-4)

        epoch_loss = []
        for _ in range(self.args["local_ep"]):
            batch_loss = []
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
