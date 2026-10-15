Implemented FULL-60 aggregation across all six methods using held-out test/validation semantics. No GPU was used, and unrelated changes were preserved.

Per-method top-1 mean ± sample std across three seeds:

- SplitFL: 60.04 ± 1.42%
- Federated: 56.59 ± 2.19%
- HierFL: 54.84 ± 3.25%
- HSFL: 57.32 ± 1.25%
- HeteroSFL: 38.45 ± 4.59%
- H-SFP: 11.49 ± 0.25%

CSV: [fair_comparison_cifar100_full60.csv](/home/jackson/Desktop/Split-Federated-Merging-Learning/results/fair_comparison_cifar100_full60.csv)

Updated aggregator: [aggregate.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/aggregate.py)