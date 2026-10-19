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
    runtime_counters=None,
    max_replay_batch: int = 128,
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
    if runtime_counters is not None:
        runtime_counters.increment("prc.calls")
        runtime_counters.increment("prc.shared_classes", len(shared_classes))
    if not shared_classes:
        if device is not None:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.tensor(0.0, requires_grad=True)

    # A per-class model call made CIFAR-100 execute 200 tiny CUDA forwards for
    # every minibatch.  At proxy scale that can stall in the CUDA/BLAS worker
    # pool (observed as futex_wait).  Batch classes exactly as reliability does.
    class_ids = sorted(shared_classes)
    compute_device = (
        torch.device(device) if device is not None
        else current_protos[class_ids[0]].device
    )
    # Prototype statistics are replay data, not trainable graph inputs.  In
    # particular, never retain a graph from prototype extraction/aggregation.
    cur_mu = torch.stack([current_protos[c].detach().to(compute_device) for c in class_ids])
    cur_sigma = torch.stack([current_stds[c].detach().to(compute_device) for c in class_ids])
    mem_mu = torch.stack([memory_protos[c].detach().to(compute_device) for c in class_ids])
    mem_sigma = torch.stack([memory_stds[c].detach().to(compute_device) for c in class_ids])

    # Keep at least one Monte Carlo pair for every shared class, but cap the
    # number of pairs retained by the representation graph.  Without this cap,
    # CIFAR-100's default creates two 1,600-example edge-model graphs in every
    # SSL minibatch and the CUDA autograd engine can stall in backward.  This is
    # the same PRC estimator with fewer draws at large class counts; gradients
    # still flow through both representation forwards into model parameters.
    if max_replay_batch <= 0:
        raise ValueError("max_replay_batch must be positive")
    samples_per_class = min(
        num_samples,
        max(1, max_replay_batch // len(class_ids)),
    )

    sample_shape = (len(class_ids), samples_per_class, *cur_mu.shape[1:])
    z_current = cur_mu.unsqueeze(1) + cur_sigma.unsqueeze(1) * torch.randn(
        sample_shape, device=compute_device, dtype=cur_mu.dtype
    )
    z_memory = mem_mu.unsqueeze(1) + mem_sigma.unsqueeze(1) * torch.randn(
        sample_shape, device=compute_device, dtype=mem_mu.dtype
    )

    # Flatten class and replay dimensions for two model calls total.  A global
    # mean is identical to the old mean of equal-sized per-class MSE losses.
    flat_shape = (-1, *cur_mu.shape[1:])
    h_current = representation_fn(z_current.reshape(flat_shape))
    h_memory = representation_fn(z_memory.reshape(flat_shape))
    total_loss = F.mse_loss(h_current, h_memory)

    if runtime_counters is not None:
        # Keep instrumentation asynchronous.  Calling .item() here serialized
        # every CUDA PRC forward with the host (hundreds of synchronizations per
        # proxy round), which can present as a futex hang.  RuntimeCounters
        # materializes these detached scalars once at a phase snapshot.
        scalar_loss = total_loss.detach()
        runtime_counters.increment(
            "prc.nonzero_loss_calls", (scalar_loss != 0).to(torch.int64)
        )
        runtime_counters.increment("prc.loss_sum", scalar_loss)
        runtime_counters.increment("prc.loss_observations")

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
