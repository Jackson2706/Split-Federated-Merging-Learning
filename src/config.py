import os

class SystemConfig:
    SEED               = 8386
    CONFIG_PATH        = os.path.join('..', 'configs', 'config.yaml')
    DATA_PATH          = os.path.join('..', 'data')
    BATCH_SIZE         = 32
    NUM_TIERS          = 3
    ENTITIES_PER_TIER  = [5, 1]
