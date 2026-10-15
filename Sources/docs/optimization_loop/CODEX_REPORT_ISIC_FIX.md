Implemented the ISIC-2018 segmentation fixes locally without changing training hyperparameters or running GPU workloads.

Double-sigmoid evaluation:

- [Federated test.py:59](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/clients/test.py:59) — removed `torch.sigmoid`; directly thresholds model probabilities at `> 0.5`.
- [Federated Client.py:111](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/clients/Client.py:111) — fixed the additional client-validation path.
- [HierFL test.py:59](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/clients/test.py:59) — removed redundant sigmoid.
- [HeteroSFL test.py:41](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HeteroSFL/clients/test.py:41) — verified it already thresholds probabilities directly.
- [H-SFP test.py:40](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/clients/test.py:40) — verified it already thresholds probabilities directly.

Crash fixes:

- HeteroSFL root cause: ambiguous `from config import ConfigLoader` failed because its `config/` lacks `__init__.py`.
- [HeteroSFL runner.py:10](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HeteroSFL/runner.py:10) now imports `ConfigLoader` explicitly from `config.config_loader`.
- H-SFP root cause: plan runs requested `plan_isic2018_h-sfp.yaml`, but only the unhyphenated `plan_isic2018_hsfp.yaml` existed.
- Added [plan_isic2018_h-sfp.yaml:1](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_isic2018_h-sfp.yaml:1) as a compatible proxy pointing to the existing H-SFP ISIC configuration.

Identifier pairing:

- [Federated ISIC.py:51](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/data/utils/ISIC.py:51)
- [HierFL ISIC.py:51](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/data/utils/ISIC.py:51)
- [HeteroSFL ISIC.py:51](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HeteroSFL/data/utils/ISIC.py:51)
- [H-SFP ISIC.py:52](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/data/utils/ISIC.py:52)

Each loader now indexes images and masks by the ISIC identifier, strips the mask `_segmentation` suffix, validates exact identifier-set equality, and constructs both lists in sorted identifier order.

Verification passed:

- `git diff --check`
- Full segmentation `compileall`
- HeteroSFL runner import
- H-SFP proxy/base resolution
- Synthetic metric test across all four methods: expected meaningful `IoU=0.333333`, `Dice=0.500000`

Existing unrelated uncommitted changes were preserved.