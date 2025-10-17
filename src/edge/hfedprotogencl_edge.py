import torch
import torch.nn.functional as F
from src.edge.base_edge import BaseEdge
from src.utils.logger import Logger
import copy

def filter_outliers(client_protos, threshold=0.7, device='cpu'):
    filtered_list = []
    for protos in client_protos:
        filtered = {}
        for label, proto in protos.items():
            proto = proto.to(device)
            # Collect all prototypes for this label
            all_protos_label = [p[label].to(device) for p in client_protos if label in p]
            stacked = torch.stack(all_protos_label)
            center = stacked.mean(dim=0, keepdim=True)
            sim = F.cosine_similarity(proto.unsqueeze(0), center, dim=1)
            if sim.item() > threshold:
                filtered[label] = proto
        filtered_list.append(filtered)
    return filtered_list

def contrastive_final_prototypes(client_protos, temperature=0.1, device='cpu'):
    class_to_protos = {}
    for protos in client_protos:
        for label, proto in protos.items():
            class_to_protos.setdefault(label, []).append(proto.to(device))

    final_protos = {}
    for label, protos in class_to_protos.items():
        if len(protos) == 1:
            final_protos[label] = protos[0]
        else:
            proto_stack = torch.stack(protos)
            proto_stack = F.normalize(proto_stack, dim=1)  # normalize for cosine similarity
            sim_matrix = proto_stack @ proto_stack.T
            sim_matrix = sim_matrix / temperature
            sim_sum = sim_matrix.sum(dim=1) - 1  # exclude self-similarity
            best_idx = sim_sum.argmax().item()
            final_protos[label] = proto_stack[best_idx]

    return final_protos

def prototype_alignment_loss(client_protos, region_protos, device='cpu'):
    """
    Compute alignment loss between client prototypes and regional prototypes.
    Each client prototype is encouraged to align with its regional class prototype.
    
    Args:
        client_protos: list of dicts {class_label: proto_tensor}
        region_protos: dict {class_label: proto_tensor} (aggregated at edge)
        device: 'cpu' or 'cuda'

    Returns:
        torch.Tensor: scalar alignment loss
    """
    losses = []
    for protos in client_protos:
        for label, proto in protos.items():
            if label in region_protos:
                proto = F.normalize(proto.to(device), dim=0)
                region = F.normalize(region_protos[label].to(device), dim=0)
                # Contrastive-style loss: maximize cosine similarity
                sim = F.cosine_similarity(proto.unsqueeze(0), region.unsqueeze(0))
                loss = 1 - sim  # encourage similarity → smaller loss if aligned
                losses.append(loss)
    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.mean(torch.stack(losses))


class HFedProtoGenCLEdge(BaseEdge):
    """
    Edge server for Hierarchical Federated Learning with contrastive prototype aggregation.
    """
    def __init__(self, edge_id, clients, config):
        super().__init__(edge_id, clients, config)
        self.logger = Logger(log_dir=f"logs/HFedProtoGenCL", name=f"edge_{edge_id}")
        self.region_proto = None
        self.last_loss = None

    def aggregate(self, updates):
        """
        Aggregate client prototypes using contrastive-style selection
        and compute alignment loss.
        """
        # Step 1: Filter out outliers
        filtered_protos = filter_outliers(updates, threshold=0.5, device=self.device)
        # Step 2: Select final prototypes using contrastive similarity
        self.region_proto = contrastive_final_prototypes(filtered_protos, temperature=0.1, device=self.device)
        # Step 3: Compute alignment loss
        self.last_loss = prototype_alignment_loss(filtered_protos, self.region_proto, device=self.device)

    def send_update(self):
        """Send deep copy of regional prototypes to cloud"""
        return copy.deepcopy(self.region_proto)

    def send_alignment_loss(self):
        """Send last computed alignment loss (e.g., for logging/monitoring)"""
        return self.last_loss

    def send_smashed_data(self):
        """Placeholder for split learning activations"""
        pass
