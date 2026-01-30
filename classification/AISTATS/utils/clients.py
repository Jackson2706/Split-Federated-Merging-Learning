import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Dict, List

# ==============================
# 🔹 Supervised Contrastive Loss
# ==============================
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, D] embeddings (normalized recommended)
            labels:   [N] ground truth labels
        """
        device = features.device
        labels = labels.contiguous().view(-1, 1)   # [N, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]

        # Normalize embeddings
        features = F.normalize(features, dim=1)

        # Cosine similarity matrix [N, N]
        sim = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Exclude self-comparisons
        mask_self = torch.eye(labels.shape[0], device=device)
        mask = mask * (1 - mask_self)

        # Log-softmax over similarities
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss = average over batch
        loss = -mean_log_prob_pos.mean()
        return loss


# ==================================
# 🔹 Compute Prototypes per Class
# ==================================
def compute_prototypes(features: torch.Tensor, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
    """
    Compute local prototypes (mean embedding) per class.

    Args:
        features: [N, D] embeddings
        labels:   [N] class labels

    Returns:
        dict: {class_id: prototype tensor [D]}
    """
    prototypes: Dict[int, torch.Tensor] = {}
    unique_labels = torch.unique(labels)

    for l in unique_labels:
        mask = (labels == l)
        if mask.any():
            class_feat = features[mask]
            prototypes[int(l.item())] = class_feat.mean(dim=0)

    return prototypes


# ==================================
# 🔹 Client Training Step
# ==================================
def client_step(model: nn.Module,
                loader: torch.utils.data.DataLoader,
                device: torch.device,
                supcon_loss_fn: SupConLoss,
                optimizer: torch.optim.Optimizer,
                proj_head: nn.Module = None):
    """
    Run local training step for one client:
    - Extract embeddings
    - Compute contrastive loss
    - Compute local prototypes

    Args:
        model: local model
        loader: DataLoader with local data
        device: torch device
        supcon_loss_fn: supervised contrastive loss instance
        proj_head: optional projection head (MLP)

    Returns:
        loss (torch.Tensor), local_prototype (dict[class_id, prototype])
    """
    all_features: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    model.to(device)
    model.eval()  # no gradients unless you wrap in optimizer.step()

    for img, label in loader:
        optimizer.zero_grad()
        img, label = img.to(device).float(), label.to(device)

        # Forward → feature map [B, C, H, W]
        feat_map = model(img)

        # Global average pooling → [B, C]
        feat_vec = torch.mean(feat_map, dim=(2, 3))

        # Optional: projection head for contrastive embeddings
        if proj_head is not None:
            feat_vec = proj_head(feat_vec)

        all_features.append(feat_vec)
        all_labels.append(label)

    # Concatenate across batches
    all_features = torch.cat(all_features, dim=0)   # [N, D]
    all_labels = torch.cat(all_labels, dim=0)       # [N]

    # Compute supervised contrastive loss
    loss = supcon_loss_fn(all_features, all_labels)
    loss.backward()
    # Compute prototypes per class
    local_prototype = compute_prototypes(all_features.detach(), all_labels)

    logger.info(f"Client step completed → Loss: {loss.item():.4f}, "
                f"Num classes: {len(local_prototype)}")

    return model, local_prototype
