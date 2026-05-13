"""
Federated aggregation utilities.

Provides functions for aggregating model weights across
clients and edge servers (e.g., FedAvg).
"""

import torch


def average_state_dicts(state_dicts):
    """
    Compute the element-wise average of a list of state_dicts (FedAvg).

    Args:
        state_dicts: List of OrderedDict (model state_dicts).

    Returns:
        Averaged state_dict.
    """
    if not state_dicts:
        return {}
    avg_dict = {}
    device = next(iter(state_dicts[0].values())).device
    for key in state_dicts[0].keys():
        tensors = [d[key].to(device) for d in state_dicts]
        avg_dict[key] = sum(tensors) / len(tensors)
    return avg_dict
