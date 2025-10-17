from .base_client import BaseClient


class SplitFedClient(BaseClient):
    def __init__(self, client_id, data_loader, model=None, config=None)
        super().__init__(client_id, data_loader, model, config)
        self.logger = Logger(log_dir=f"logs/HierFedAvg", name=f"client_{client_id}")
    
    def local_train(self, epochs, lr):
        pass

    def send_update(self):
        """Send local weights to cloud"""
        return copy.deepcopy(self.model.state_dict())
    
    