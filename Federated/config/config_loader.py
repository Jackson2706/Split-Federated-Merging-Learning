import os
import yaml

class ConfigLoader:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.config = self._load_and_merge(cfg_path)

    def get_config(self):
        return self.config

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _load_and_merge(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        cfg = self._load_yaml(path)

        # Check for inheritance
        if 'base' in cfg:
            base_path = os.path.join(os.path.dirname(path), cfg['base'])
            base_cfg = self._load_and_merge(base_path)
            cfg.pop('base')  # remove base key
            return self._merge_dicts(base_cfg, cfg)
        return cfg

    def _merge_dicts(self, base, override):
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = self._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
