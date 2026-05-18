"""
E-HSFP loss functions.

Provides the Prototype Replay Consistency (PRC) loss and
dropout consistency loss for episodic training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, Optional


def prototype_replay_consistency_loss(
    current_protos: Dict[int, torch.Tensor],
    current_stds: Dict[int, torch.Tensor],
    memory_protos: Dict[int, torch.Tensor],
    memory_stds: Dict[int, torch.Tensor],
    representation_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Prototype Replay Consistency loss.

    For each class present in both current and memory prototypes:
    - Sample synthetic features from current distribution
    - Sample synthetic features from memory distribution
    - Pass both through the same representation module
    - Minimize L2 distance between representations

    L_prc = mean(|| h(z_current) - h(z_memory) ||_2^2)

    Returns zero tensor if no shared classes or early rounds.
    """
    shared_classes = set(current_protos.keys()) & set(memory_protos.keys())
    if not shared_classes:
        if device is not None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.tensor(0.0, requires_grad=True)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True) if device else torch.tensor(0.0, requires_grad=True)
    count = 0

    for cls_id in shared_classes:
        cur_mu = current_protos[cls_id]
        cur_sigma = current_stds[cls_id]
        mem_mu = memory_protos[cls_id].to(cur_mu.device)
        mem_sigma = memory_stds[cls_id].to(cur_sigma.device)

        # Sample from current distribution: z = mu + sigma * epsilon
        eps_cur = torch.randn(num_samples, *cur_mu.shape, device=cur_mu.device, dtype=cur_mu.dtype)
        z_current = cur_mu.unsqueeze(0) + cur_sigma.unsqueeze(0) * eps_cur

        # Sample from memory distribution
        eps_mem = torch.randn(num_samples, *mem_mu.shape, device=mem_mu.device, dtype=mem_mu.dtype)
        z_memory = mem_mu.unsqueeze(0) + mem_sigma.unsqueeze(0) * eps_mem

        # Pass through representation function
        h_current = representation_fn(z_current)
        h_memory = representation_fn(z_memory)

        # L2 distance
        loss = F.mse_loss(h_current, h_memory)
        total_loss = total_loss + loss
        count += 1

    if count > 0:
        total_loss = total_loss / count

    return total_loss


def dropout_consistency_loss(
    logits_full: torch.Tensor,
    logits_dropped: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between predictions from full and dropped prototypes.

    L_drop = KL(p(y|z_full) || p(y|z_drop))

    Both inputs are logits [N, C].
    """
    p_full = F.log_softmax(logits_full, dim=-1)
    p_drop = F.softmax(logits_dropped, dim=-1)
    return F.kl_div(p_full, p_drop, reduction="batchmean")
