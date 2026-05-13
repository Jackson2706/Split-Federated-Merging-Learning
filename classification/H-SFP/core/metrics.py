"""
Communication and model size metrics.

Provides utilities for measuring model sizes and communication
costs in the hierarchical federated learning pipeline.
"""


def get_model_size_MB(state_dict):
    """
    Calculate model size in MB from a state_dict.
    Assumes float32 (4 bytes per parameter).

    Args:
        state_dict: Model state_dict.

    Returns:
        Size in megabytes (float).
    """
    return sum(param.numel() for param in state_dict.values()) * 4 / 1e6


def get_proto_dist_size_MB(proto_dist_tuple):
    """
    Calculate the size in MB of a (proto_dict, dist_dict) tuple.

    Args:
        proto_dist_tuple: Tuple of (prototypes_dict, distributions_dict),
                          or None.

    Returns:
        Size in megabytes (float).
    """
    if proto_dist_tuple is None:
        return 0.0

    proto_dict, dist_dict = proto_dist_tuple
    total_bytes = 0

    for tensor in proto_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    for tensor in dist_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024**2)
