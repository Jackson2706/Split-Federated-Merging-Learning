import torch
from torch import nn

from .Client import Client
from tqdm import tqdm
from .DiceFocalLoss import DiceFocalLoss


class FedAvgClient(Client):
    def update_weights(self, model, global_round):
        model.train()
        criterion = DiceFocalLoss().to(self.device)
        optimizer = (
            torch.optim.SGD(
                model.parameters(),
                lr=self.args["lr"],
                momentum=self.args["momentum"],
            )
            if self.args["optimizer"] == "sgd"
            else torch.optim.Adam(
                model.parameters(), lr=self.args["lr"], weight_decay=1e-4
            )
        )

        epoch_loss = []
        model = model.to(self.device)
        for _ in range(self.args["local_ep"]):
            torch.cuda.empty_cache()
            batch_loss = []
            for images, masks in self.trainloader:
                torch.cuda.empty_cache()
                images, masks = images.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                del images, masks, outputs
                torch.cuda.empty_cache()
                loss.backward()
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
                del loss
                torch.cuda.empty_cache()
                optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
