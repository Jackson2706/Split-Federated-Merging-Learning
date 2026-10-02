"""MiSFP: Mixture-Preserving Prototyping for Hierarchical Split-Federated Learning.

Shared library (task-agnostic). See README_MiSFP.md for the design.
"""

from .compat import between_source_dispersion
from .config import (MISFP_DEFAULTS, VARIANT_LABELS, VARIANT_PRESETS, get_misfp_config,
                     public_view)
from .fit import FitConfig, fit_class_mixture, kmeans, mixture_log_prob, stream_moments
from .merge import (CompressionReport, collapse_baseline_average, compress_classes, merge_cost,
                    merge_pair, pool_two, w2_squared)
from .packets import (ClassMixture, FeatureSpace, FeatureSpaceMismatch, MixturePacket,
                      MixtureTransport, component_nbytes, deserialize_packet, packet_nbytes,
                      pool_packets, serialize_packet, transmit)
from .sampling import derive_seed, make_generator, sample_component, sample_mixture, synthesize

__all__ = [
    "between_source_dispersion", "MISFP_DEFAULTS", "VARIANT_LABELS", "VARIANT_PRESETS",
    "get_misfp_config", "public_view", "FitConfig", "fit_class_mixture", "kmeans",
    "mixture_log_prob", "stream_moments", "CompressionReport", "collapse_baseline_average",
    "compress_classes", "merge_cost", "merge_pair", "pool_two", "w2_squared", "ClassMixture",
    "FeatureSpace", "FeatureSpaceMismatch", "MixturePacket", "MixtureTransport",
    "component_nbytes", "deserialize_packet", "packet_nbytes", "pool_packets",
    "serialize_packet", "transmit", "derive_seed", "make_generator", "sample_component",
    "sample_mixture", "synthesize",
]
