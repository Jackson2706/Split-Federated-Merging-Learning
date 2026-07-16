"""
E-HSFP: Episodic Hierarchical Split-Federated Prototyping.

A serverless-aware extension of H-SFP that adds:
- Episodic prototype memory with replay
- Learnable reliability-aware aggregation
- Prototype replay consistency loss
- Serverless prototype dropout regularization
- Lightweight serverless episode simulation
- Optional residual prototype generator
"""

__version__ = "0.1.0"

from ehsfp.config import get_ehsfp_config, EHSFP_DEFAULTS, ABLATION_PRESETS
from ehsfp.memory import (
    PrototypeRecord,
    EpisodicPrototypeMemory,
    mix_current_and_memory,
)
from ehsfp.reliability import (
    PrototypeReliabilityNetwork,
    build_reliability_features,
    compute_heuristic_reliability,
)
from ehsfp.aggregation import (
    aggregate_proto_dicts,
    reliability_weighted_aggregate,
    train_reliability_bootstrap,
)
from ehsfp.losses import (
    prototype_replay_consistency_loss,
    dropout_consistency_loss,
)
from ehsfp.dropout import PrototypeDropout
from ehsfp.serverless_sim import (
    ServerlessEpisode,
    ServerlessInvocationRecord,
    ServerlessMetricsTracker,
)
from ehsfp.generator import ResidualPrototypeGenerator
from ehsfp.metrics_logger import EHSFPMetricsLogger
from ehsfp.integrity import (
    RuntimeCounters,
    architecture_manifest,
    partition_hash,
    prepare_run_dir,
    resolved_config_hash,
    validate_checkpoint_config_hash,
)
from ehsfp.prototype_space import (
    PROTOTYPE_SPACES,
    COSINE_SPACES,
    CosineClassifier,
    center_features,
    center_source_outputs,
    derive_global_feature_mean,
    derive_whitening_transform,
    recenter_memory,
    validate_prototype_space,
    whiten_features,
    whiten_source_outputs,
)
from ehsfp.client_objective import (
    CLIENT_OBJECTIVES,
    compose_client_objective_loss,
    local_supervised_contrastive_loss,
    validate_client_objective,
)

__all__ = [
    "get_ehsfp_config",
    "EHSFP_DEFAULTS",
    "ABLATION_PRESETS",
    "PrototypeRecord",
    "EpisodicPrototypeMemory",
    "mix_current_and_memory",
    "PrototypeReliabilityNetwork",
    "build_reliability_features",
    "compute_heuristic_reliability",
    "reliability_weighted_aggregate",
    "aggregate_proto_dicts",
    "train_reliability_bootstrap",
    "prototype_replay_consistency_loss",
    "dropout_consistency_loss",
    "PrototypeDropout",
    "ServerlessEpisode",
    "ServerlessInvocationRecord",
    "ServerlessMetricsTracker",
    "ResidualPrototypeGenerator",
    "EHSFPMetricsLogger",
    "RuntimeCounters",
    "architecture_manifest",
    "partition_hash",
    "prepare_run_dir",
    "resolved_config_hash",
    "validate_checkpoint_config_hash",
    "PROTOTYPE_SPACES",
    "CosineClassifier",
    "COSINE_SPACES",
    "center_features",
    "center_source_outputs",
    "derive_global_feature_mean",
    "derive_whitening_transform",
    "recenter_memory",
    "validate_prototype_space",
    "whiten_features",
    "whiten_source_outputs",
    "CLIENT_OBJECTIVES",
    "compose_client_objective_loss",
    "local_supervised_contrastive_loss",
    "validate_client_objective",
]
