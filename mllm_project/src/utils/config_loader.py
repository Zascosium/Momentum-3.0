"""
Configuration management utilities for the MLLM project.
Handles loading, validation, and merging of configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
import json

logger = logging.getLogger(__name__)

@dataclass
class ConfigPaths:
    """Configuration file paths."""
    config_dir: str = "config"
    training_config: str = "training_config.yaml"
    model_config: str = "model_config.yaml"
    data_config: str = "data_config.yaml"
    
class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigLoader:
    """
    Configuration loader and validator for MLLM training.
    
    Supports YAML configuration files with environment variable substitution,
    configuration merging, and validation.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir or "config")
        self.configs = {}
        self._validate_config_dir()
    
    def _validate_config_dir(self) -> None:
        """Validate that configuration directory exists."""
        if not self.config_dir.exists():
            raise ConfigurationError(f"Configuration directory {self.config_dir} does not exist")
    
    def load_config(self, config_name: str) -> DictConfig:
        """
        Load a single configuration file.
        
        Args:
            config_name: Name of the configuration file (with or without .yaml extension)
            
        Returns:
            Loaded configuration as OmegaConf DictConfig
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
            
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file {config_path} does not exist")
        
        try:
            # Load with OmegaConf for advanced features
            config = OmegaConf.load(config_path)
            
            # Resolve environment variables and interpolations
            config = OmegaConf.create(self._resolve_env_vars(OmegaConf.to_yaml(config)))
            
            # Cache the loaded config
            self.configs[config_name.replace('.yaml', '')] = config
            
            logger.info(f"Successfully loaded configuration: {config_name}")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration {config_name}: {str(e)}")
    
    def load_all_configs(self) -> Dict[str, DictConfig]:
        """
        Load all standard configuration files.
        
        Returns:
            Dictionary of loaded configurations
        """
        paths = ConfigPaths()
        config_files = [
            paths.training_config,
            paths.model_config, 
            paths.data_config
        ]
        
        all_configs = {}
        for config_file in config_files:
            config_name = config_file.replace('.yaml', '')
            all_configs[config_name] = self.load_config(config_file)
        
        return all_configs
    
    def merge_configs(self, *configs: DictConfig, override_config: Optional[Dict] = None) -> DictConfig:
        """
        Merge multiple configurations with optional overrides.
        
        Args:
            *configs: Configuration objects to merge
            override_config: Additional overrides to apply
            
        Returns:
            Merged configuration
        """
        merged = OmegaConf.create({})
        
        # Merge all configs in order
        for config in configs:
            merged = OmegaConf.merge(merged, config)
        
        # Apply overrides if provided
        if override_config:
            override_config = OmegaConf.create(override_config)
            merged = OmegaConf.merge(merged, override_config)
        
        return merged
    
    def _resolve_env_vars(self, yaml_content: str) -> str:
        """
        Resolve environment variables in YAML content.
        
        Args:
            yaml_content: YAML content as string
            
        Returns:
            YAML content with resolved environment variables
        """
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern to match ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        resolved_content = re.sub(pattern, replace_env_var, yaml_content)
        
        return resolved_content
    
    def validate_config(self, config: DictConfig, schema: Optional[Dict] = None) -> bool:
        """
        Validate configuration against a schema.
        
        Args:
            config: Configuration to validate
            schema: Optional schema for validation
            
        Returns:
            True if validation passes
        """
        # Basic validation - check for required fields
        required_fields = {
            'training_config': ['training', 'optimizer', 'scheduler'],
            'model_config': ['model', 'time_series_encoder', 'text_decoder'],
            'data_config': ['dataset', 'domains', 'splits']
        }
        
        config_type = self._detect_config_type(config)
        if config_type in required_fields:
            for field in required_fields[config_type]:
                if field not in config:
                    raise ConfigurationError(f"Missing required field '{field}' in {config_type}")
        
        # Additional custom validation can be added here
        return True
    
    def _detect_config_type(self, config: DictConfig) -> str:
        """Detect the type of configuration based on its structure."""
        if 'training' in config:
            return 'training_config'
        elif 'time_series_encoder' in config:
            return 'model_config'
        elif 'dataset' in config:
            return 'data_config'
        else:
            return 'unknown'
    
    def save_config(self, config: DictConfig, filepath: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            OmegaConf.save(config, f)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def get_config_summary(self, config: DictConfig) -> str:
        """
        Generate a human-readable summary of the configuration.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Configuration summary as string
        """
        summary_lines = []
        
        def _summarize_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    summary_lines.append(f"{prefix}{key}:")
                    _summarize_dict(value, prefix + "  ")
                else:
                    summary_lines.append(f"{prefix}{key}: {value}")
        
        _summarize_dict(config)
        return "\n".join(summary_lines)
    
    def update_config_from_env(self, config: DictConfig, env_prefix: str = "MLLM_") -> DictConfig:
        """
        Update configuration from environment variables.
        
        Args:
            config: Base configuration
            env_prefix: Prefix for environment variables
            
        Returns:
            Updated configuration
        """
        env_updates = {}
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert MLLM_TRAINING_BATCH_SIZE to training.batch_size
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                
                # Try to convert to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                except:
                    pass  # Keep as string if conversion fails
                
                # Set nested key
                OmegaConf.set(config, config_key, value)
        
        return config

def load_config_for_training(
    config_dir: Optional[str] = None,
    overrides: Optional[Dict] = None,
    env_prefix: str = "MLLM_"
) -> DictConfig:
    """
    Convenience function to load and merge all configurations for training.
    
    Args:
        config_dir: Configuration directory
        overrides: Configuration overrides
        env_prefix: Environment variable prefix
        
    Returns:
        Complete merged configuration
    """
    loader = ConfigLoader(config_dir)
    
    # Load all configurations
    configs = loader.load_all_configs()
    
    # Merge configurations
    merged_config = loader.merge_configs(
        configs['training_config'],
        configs['model_config'],
        configs['data_config'],
        override_config=overrides
    )
    
    # Update from environment variables
    merged_config = loader.update_config_from_env(merged_config, env_prefix)
    
    # Validate merged configuration
    loader.validate_config(merged_config)
    
    return merged_config

def save_experiment_config(config: DictConfig, experiment_dir: str) -> None:
    """
    Save experiment configuration for reproducibility.
    
    Args:
        config: Configuration to save
        experiment_dir: Experiment directory
    """
    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Save as YAML
    config_path = experiment_path / "experiment_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)
    
    # Save as JSON for easy parsing
    json_path = experiment_path / "experiment_config.json"
    with open(json_path, 'w') as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=2)
    
    logger.info(f"Experiment configuration saved to {experiment_dir}")

# Example usage
if __name__ == "__main__":
    # Basic usage example
    loader = ConfigLoader("../config")
    
    # Load individual config
    training_config = loader.load_config("training_config")
    print("Training config loaded successfully")
    
    # Load all configs and merge
    complete_config = load_config_for_training(
        config_dir="../config",
        overrides={"training.batch_size": 32}
    )
    
    print("Complete configuration loaded and merged")
    print(loader.get_config_summary(complete_config))