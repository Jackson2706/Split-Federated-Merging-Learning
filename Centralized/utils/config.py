import os
from typing import Any, Dict

import torch
import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override configuration with base configuration"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

class Config:
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file
        Args:
            config_path: Path to YAML configuration file
        """
        # Load base config
        base_config_path = os.path.join(os.path.dirname(config_path), 'base_config.yaml')
        self.config = load_yaml(base_config_path)
        
        # Load and merge specific config
        if config_path != base_config_path:
            specific_config = load_yaml(config_path)
            self.config = merge_configs(self.config, specific_config)
        
        # Add CUDA availability
        self.config['cuda'] = torch.cuda.is_available()
    
    def __getattr__(self, name: str) -> Any:
        """Allow accessing config values as attributes"""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Configuration has no attribute '{name}'")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        self.config.update(updates)
    
    def save(self, save_path: str) -> None:
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    @property
    def device(self) -> torch.device:
        """Get PyTorch device based on configuration"""
        return torch.device(f'cuda:{self.gpu}' if self.cuda else 'cpu') 