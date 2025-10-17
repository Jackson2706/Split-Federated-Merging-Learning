import time
import torch
import torch.nn.functional as F
import copy
from src.client.base_client import BaseClient
from src.utils.logger import Logger


class HFedProtoGenCLClient(BaseClient):
    def __init__(self, client_id, data_loader, model=None, config=None, tau=0.1, alpha=0.1, beta=0.1):
        """
        Client for Hierarchical Federated Prototypical Learning with Contrastive Loss.
        
        Args:
            client_id (int): unique identifier
            data_loader (DataLoader): client local dataset
            model (nn.Module): local model
            config (dict): configuration dict
            tau (float): temperature for contrastive loss
            alpha (float): weight for contrastive loss
            beta (float): weight for alignment loss (local vs global prototypes)
        """
        super().__init__(client_id, data_loader, model, config)
        self.logger = Logger(log_dir=f"logs/HFedProtoGenCL", name=f"client_{client_id}")
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.align_loss_at_edge=0
        # filled when edge server sends prototypes
        self.global_protos = None
        self.label_to_mean = None  

    # ---------------------------------------------------------
    # Loss functions
    # ---------------------------------------------------------
    def prototype_contrastive_loss(self, z, labels, prototypes):
        """Compute prototype-based contrastive loss in vectorized form"""
        device = z.device
        proto_labels = sorted(prototypes.keys())
        proto_matrix = torch.stack([prototypes[l] for l in proto_labels], dim=0)  # (C, D)

        # Cosine similarity between each sample and all prototypes
        sim = F.cosine_similarity(z.unsqueeze(1), proto_matrix.unsqueeze(0), dim=2)  # (B, C)
        sim = sim / self.tau
        sim_exp = torch.exp(sim)

        # Positive prototype indices
        label_to_index = {l: i for i, l in enumerate(proto_labels)}
        pos_idx = torch.tensor([label_to_index[l.item()] for l in labels], device=device)
        sim_pos = sim_exp[torch.arange(z.size(0), device=device), pos_idx]

        # Sum over all prototypes
        sim_all = sim_exp.sum(dim=1)

        loss = -torch.log(sim_pos / sim_all)
        return loss.mean()

    def prototype_alignment_loss(self, local_protos, global_protos, device=None):
        """
        Compute alignment loss between local and global prototypes.
        Cosine distance is used here.
        """
        if global_protos is None or len(global_protos) == 0:
            return torch.tensor(0.0, device=device or "cpu")

        device = device or next(iter(local_protos.values())).device
        loss = 0.0
        count = 0

        for label, local_vec in local_protos.items():
            if label in global_protos:
                gv = global_protos[label].to(device)
                lv = local_vec.to(device)
                sim = F.cosine_similarity(lv.unsqueeze(0), gv.unsqueeze(0))
                loss += (1 - sim)  # smaller when aligned
                count += 1

        if count > 0:
            loss = loss / count
        return loss

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    def local_train(self, epochs=1, lr=0.01):
        """Perform local training on client dataset with CE + contrastive + alignment losses"""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.model.train()
        self.model.to(self.device)

        for e in range(epochs):
            epoch_start = time.time()
            total_loss, correct, total = 0.0, 0, 0
            label_to_vectors = {}

            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # Forward
                out = self.model(x)
                prototype_vectors = self.model.forward_prototype(x)  # (B, D)

                # Compute batch-level prototypes (mean per label in this batch)
                batch_prototypes = {}
                for i in range(y.size(0)):
                    label = y[i].item()
                    vec = prototype_vectors[i]
                    if label not in batch_prototypes:
                        batch_prototypes[label] = []
                    batch_prototypes[label].append(vec)
                batch_prototypes = {label: torch.stack(vecs).mean(dim=0)
                                    for label, vecs in batch_prototypes.items()}

                # Losses
                ce_loss = criterion(out, y)
                contrastive_loss = self.prototype_contrastive_loss(prototype_vectors, y, batch_prototypes)
                align_loss = self.prototype_alignment_loss(batch_prototypes, self.global_protos, device=self.device)

                loss =  ce_loss \
                       + contrastive_loss \
                       +  align_loss \
                       + self.align_loss_at_edge

                # Backward
                loss.backward()
                optimizer.step()

                # Metrics
                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                # Collect local prototypes
                for label, vec in batch_prototypes.items():
                    if label not in label_to_vectors:
                        label_to_vectors[label] = []
                    label_to_vectors[label].append(vec.detach())

            # Epoch-level prototypes
            self.label_to_mean = {label: torch.stack(vecs).mean(dim=0)
                                  for label, vecs in label_to_vectors.items()}

            avg_loss = total_loss / total
            acc = 100.0 * correct / total
            self.logger.log(step=f"epoch_{e+1}", loss=avg_loss, acc=acc, step_start=epoch_start)

    # ---------------------------------------------------------
    # Communication
    # ---------------------------------------------------------
    def send_update(self):
        """Send local prototypes to edge"""
        return copy.deepcopy(self.label_to_mean)

    def send_smashed_data(self, batch):
        """Not used in FedAvg (SplitNN only)"""
        return None

    def receive_gradients(self, grad):
        """Not used in FedAvg (SplitNN only)"""
        return None