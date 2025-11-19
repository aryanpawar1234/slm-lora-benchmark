"""Configuration loader for YAML files"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_all_configs(configs_dir: str = "configs") -> Dict[str, Any]:
    """
    Load all configuration files from configs directory
    
    Args:
        configs_dir: Directory containing config files
        
    Returns:
        Merged configuration dictionary
    """
    config_files = {
        'datasets': 'datasets.yaml',
        'model': 'model_config.yaml',
        'lora': 'lora_config.yaml',
        'training': 'training_config.yaml'
    }
    
    all_configs = {}
    for key, filename in config_files.items():
        filepath = os.path.join(configs_dir, filename)
        if os.path.exists(filepath):
            all_configs[key] = load_config(filepath)
        else:
            print(f"Warning: Config file not found: {filepath}")
    
    return all_configs


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration
    """
    merged = {}
    for config in configs:
        merged.update(config)
    return merged


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Config saved to: {save_path}")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "training.learning_rate")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that all required keys exist in config
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key paths
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_keys = []
    
    for key in required_keys:
        if get_config_value(config, key) is None:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return True


class ConfigManager:
    """Configuration manager with caching and validation"""
    
    def __init__(self, configs_dir: str = "configs"):
        self.configs_dir = Path(configs_dir)
        self._cache = {}
    
    def load(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load configuration with caching"""
        if use_cache and config_name in self._cache:
            return self._cache[config_name]
        
        config_path = self.configs_dir / f"{config_name}.yaml"
        config = load_config(str(config_path))
        
        if use_cache:
            self._cache[config_name] = config
        
        return config
    
    def load_all(self) -> Dict[str, Any]:
        """Load all configurations"""
        return load_all_configs(str(self.configs_dir))
    
    def clear_cache(self):
        """Clear configuration cache"""
        self._cache.clear()