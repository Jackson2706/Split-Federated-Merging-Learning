"""
camera_ready: shared utilities for the ECCV camera-ready experiments of H-SFP.

All functionality here is additive and is consumed either by standalone
experiment scripts (scripts/camera_ready/) or by minimal, flag-gated hooks in the
existing method code. Importing this package never changes default H-SFP behavior.

Modules
-------
partition          : Dirichlet and two-level (edge/client) Dirichlet partitioning.
synthesis          : prototype-synthesis covariance modes + comm-cost accounting.
feature_distance   : RBF-MMD between real and synthesized features.
lstat              : empirical Lipschitz sensitivity of prototype statistics.
profiling          : CUDA-synchronized phase timers.
inversion          : optimization-based feature-inversion exposure analysis.
latex              : booktabs LaTeX table generation.
io_utils           : result paths, JSON/CSV writers, mean +/- std aggregation.
"""

__version__ = "0.1.0"
