Implemented the H-SFP segmentation fix locally, preserving all existing uncommitted changes.

- [`_run_validation()` lines 705–714](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/hierarchy.py:705): now builds the FedAvg global client model from cached trained clients, with the client-0 fallback only when none are cached.
- [`_train_decoder_phase()` lines 758–767](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/hierarchy.py:758): applies the identical construction for decoder training features.
- Edge-model selection and all other logic remain unchanged.

Verification:

- Both inserted blocks are exact AST matches for the proven pattern in `classification/H-SFP/hierarchy.py`.
- `python -m py_compile segmentation/H-SFP/hierarchy.py`: passed.
- `python -m unittest discover -s tests`: **27 tests passed**.
- No GPU or training run performed.