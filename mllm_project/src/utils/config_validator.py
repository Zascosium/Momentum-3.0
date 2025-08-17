"""
Configuration validation utilities for multimodal LLM.
Validates configuration compatibility and completeness.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass

class ConfigValidator:
    """
    Comprehensive configuration validator for multimodal LLM training.
    """
    
    def __init__(self):
        """Initialize the validator with required configuration schemas."""
        self.required_sections = {
            'model': ['name'],
            'time_series_encoder': ['embedding_dim'],
            'text_decoder': ['embedding_dim', 'model_name'],
            'training': ['epochs', 'batch_size'],
            'data': ['data_dir']
        }
        
        self.optional_sections = {
            'projection': ['hidden_dims', 'activation', 'dropout'],
            'cross_attention': ['hidden_size', 'num_heads'],
            'fusion': ['strategy'],
            'loss': ['text_generation_weight', 'alignment_loss_weight'],
            'optimizer': ['name', 'learning_rate'],
            'scheduler': ['name'],
            'mixed_precision': ['enabled'],
            'distributed': ['backend'],
            'checkpointing': ['save_top_k', 'monitor'],
            'validation': ['val_check_interval'],
            'logging': ['level']
        }
        
        self.valid_fusion_strategies = ['early', 'late', 'cross_attention', 'hybrid']
        self.valid_optimizers = ['adamw', 'adam', 'sgd']
        self.valid_schedulers = ['cosine_with_warmup', 'linear_with_warmup', 'step', 'exponential', 'none']
    
    def validate_complete_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate complete configuration for training.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check required sections
            errors.extend(self._validate_required_sections(config))
            
            # Check dimension compatibility
            errors.extend(self._validate_dimension_compatibility(config))
            
            # Check fusion strategy
            errors.extend(self._validate_fusion_strategy(config))
            
            # Check training parameters
            errors.extend(self._validate_training_parameters(config))
            
            # Check optimizer and scheduler
            errors.extend(self._validate_optimization_config(config))
            
            # Check mixed precision compatibility
            errors.extend(self._validate_mixed_precision(config))
            
            # Check data paths
            errors.extend(self._validate_data_paths(config))
            
            # Check model compatibility
            errors.extend(self._validate_model_compatibility(config))
            
        except Exception as e:
            errors.append(f"Configuration validation failed with exception: {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_required_sections(self, config: Dict[str, Any]) -> List[str]:
        """Validate that all required sections and fields are present."""
        errors = []
        
        for section, required_fields in self.required_sections.items():
            if section not in config:
                errors.append(f"Missing required section: {section}")
                continue
            
            section_config = config[section]
            if not isinstance(section_config, dict):
                errors.append(f"Section '{section}' must be a dictionary")
                continue
            
            for field in required_fields:
                if field not in section_config:
                    errors.append(f"Missing required field '{field}' in section '{section}'")
        
        return errors
    
    def _validate_dimension_compatibility(self, config: Dict[str, Any]) -> List[str]:
        """Validate that encoder and decoder dimensions are compatible."""
        errors = []
        
        try:
            ts_dim = config.get('time_series_encoder', {}).get('embedding_dim')
            text_dim = config.get('text_decoder', {}).get('embedding_dim')
            
            if ts_dim is None or text_dim is None:
                return errors  # Will be caught by required sections validation
            
            # Check if dimensions are reasonable
            if ts_dim <= 0 or text_dim <= 0:
                errors.append("Embedding dimensions must be positive integers")
            
            if ts_dim > 4096 or text_dim > 4096:
                errors.append("Embedding dimensions seem unreasonably large (>4096)")
            
            # Check fusion strategy compatibility
            fusion_strategy = config.get('fusion', {}).get('strategy', 'cross_attention')
            
            if fusion_strategy == 'early':
                # Early fusion concatenates dimensions
                total_dim = ts_dim + text_dim
                if total_dim > 8192:
                    errors.append(f"Early fusion total dimension {total_dim} may be too large")
            
            elif fusion_strategy in ['cross_attention', 'hybrid']:
                # Cross attention requires projection layers
                projection_config = config.get('projection', {})
                if not projection_config:
                    logger.warning("Cross-attention fusion recommended with projection configuration")
            
        except Exception as e:
            errors.append(f"Error validating dimensions: {e}")
        
        return errors
    
    def _validate_fusion_strategy(self, config: Dict[str, Any]) -> List[str]:
        """Validate fusion strategy configuration."""
        errors = []
        
        fusion_config = config.get('fusion', {})
        strategy = fusion_config.get('strategy', 'cross_attention')
        
        if strategy not in self.valid_fusion_strategies:
            errors.append(f"Invalid fusion strategy '{strategy}'. Valid options: {self.valid_fusion_strategies}")
        
        # Strategy-specific validation
        if strategy == 'cross_attention':
            cross_attn_config = config.get('cross_attention', {})
            if not cross_attn_config.get('hidden_size'):
                errors.append("Cross-attention fusion requires 'hidden_size' in cross_attention config")
            
            num_heads = cross_attn_config.get('num_heads', 8)
            hidden_size = cross_attn_config.get('hidden_size', 512)
            
            if hidden_size % num_heads != 0:
                errors.append(f"Cross-attention hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        elif strategy == 'early':
            # Early fusion might need projection
            ts_dim = config.get('time_series_encoder', {}).get('embedding_dim', 0)
            text_dim = config.get('text_decoder', {}).get('embedding_dim', 0)
            
            if ts_dim != text_dim:
                projection_config = config.get('projection', {})
                if not projection_config:
                    logger.warning("Different encoder/decoder dimensions with early fusion - consider adding projection config")
        
        return errors
    
    def _validate_training_parameters(self, config: Dict[str, Any]) -> List[str]:
        """Validate training parameters."""
        errors = []
        
        training_config = config.get('training', {})
        
        # Validate epochs
        epochs = training_config.get('epochs', 1)
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append("Training epochs must be a positive integer")
        elif epochs > 1000:
            logger.warning(f"Training for {epochs} epochs seems excessive")
        
        # Validate batch size
        batch_size = training_config.get('batch_size', 1)
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append("Batch size must be a positive integer")
        elif batch_size > 128:
            logger.warning(f"Large batch size ({batch_size}) may require significant memory")
        
        # Validate gradient accumulation
        grad_accum = training_config.get('gradient_accumulation_steps', 1)
        if not isinstance(grad_accum, int) or grad_accum <= 0:
            errors.append("Gradient accumulation steps must be a positive integer")
        
        effective_batch_size = batch_size * grad_accum
        if effective_batch_size > 512:
            logger.warning(f"Large effective batch size ({effective_batch_size}) may affect training dynamics")
        
        # Validate gradient clipping
        max_grad_norm = training_config.get('max_grad_norm', 1.0)
        if not isinstance(max_grad_norm, (int, float)) or max_grad_norm <= 0:
            errors.append("Max gradient norm must be a positive number")
        
        return errors
    
    def _validate_optimization_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate optimizer and scheduler configuration."""
        errors = []
        
        # Validate optimizer
        optimizer_config = config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adamw').lower()
        
        if optimizer_name not in self.valid_optimizers:
            errors.append(f"Invalid optimizer '{optimizer_name}'. Valid options: {self.valid_optimizers}")
        
        learning_rate = optimizer_config.get('learning_rate', 5e-5)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            errors.append("Learning rate must be a positive number")
        elif learning_rate > 0.1:
            logger.warning(f"High learning rate ({learning_rate}) may cause training instability")
        
        # Validate scheduler
        scheduler_config = config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine_with_warmup').lower()
        
        if scheduler_name not in self.valid_schedulers:
            errors.append(f"Invalid scheduler '{scheduler_name}'. Valid options: {self.valid_schedulers}")
        
        # Warmup validation
        if 'warmup' in scheduler_name:
            warmup_steps = scheduler_config.get('warmup_steps', 1000)
            if not isinstance(warmup_steps, int) or warmup_steps < 0:
                errors.append("Warmup steps must be a non-negative integer")
        
        return errors
    
    def _validate_mixed_precision(self, config: Dict[str, Any]) -> List[str]:
        """Validate mixed precision configuration."""
        errors = []
        
        mp_config = config.get('mixed_precision', {})
        
        if mp_config.get('enabled', False):
            fp16 = mp_config.get('fp16', False)
            bf16 = mp_config.get('bf16', False)
            
            if fp16 and bf16:
                errors.append("Cannot enable both fp16 and bf16 simultaneously")
            
            if not fp16 and not bf16:
                logger.warning("Mixed precision enabled but neither fp16 nor bf16 specified")
            
            # Check loss scaling
            if fp16:
                loss_scale = mp_config.get('loss_scale', 0)
                if not isinstance(loss_scale, (int, float)) or loss_scale < 0:
                    errors.append("Loss scale must be non-negative")
        
        return errors
    
    def _validate_data_paths(self, config: Dict[str, Any]) -> List[str]:
        """Validate data paths and directories."""
        errors = []
        
        data_config = config.get('data', {})
        
        # Check data directory
        data_dir = data_config.get('data_dir')
        if data_dir:
            data_path = Path(data_dir)
            # Don't validate existence for DBFS paths or remote paths
            if not data_dir.startswith(('/dbfs', 'http', 's3://', 'gs://', 'azure://')):
                if not data_path.exists():
                    logger.warning(f"Data directory does not exist: {data_dir}")
        
        # Check cache directory
        cache_dir = data_config.get('paths', {}).get('cache_dir')
        if cache_dir:
            cache_path = Path(cache_dir)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create cache directory {cache_dir}: {e}")
        
        return errors
    
    def _validate_model_compatibility(self, config: Dict[str, Any]) -> List[str]:
        """Validate model-specific configurations."""
        errors = []
        
        # Check MOMENT encoder config
        ts_config = config.get('time_series_encoder', {})
        moment_config = ts_config.get('moment_config', {})
        
        if moment_config:
            patch_len = moment_config.get('patch_len', 8)
            stride = moment_config.get('stride', 8)
            
            if not isinstance(patch_len, int) or patch_len <= 0:
                errors.append("MOMENT patch_len must be a positive integer")
            
            if not isinstance(stride, int) or stride <= 0:
                errors.append("MOMENT stride must be a positive integer")
            
            if stride > patch_len:
                logger.warning(f"MOMENT stride ({stride}) > patch_len ({patch_len}) may cause gaps")
        
        # Check text decoder config
        text_config = config.get('text_decoder', {})
        model_name = text_config.get('model_name', '')
        
        if model_name and not any(valid in model_name.lower() for valid in ['gpt', 'llama', 'opt', 'bloom']):
            logger.warning(f"Unfamiliar text model: {model_name}")
        
        vocab_size = text_config.get('vocab_size')
        if vocab_size and (not isinstance(vocab_size, int) or vocab_size <= 0):
            errors.append("Vocabulary size must be a positive integer")
        
        return errors
    
    def validate_databricks_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration for Databricks environment.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        try:
            # Check checkpoint paths
            checkpointing_config = config.get('checkpointing', {})
            checkpoint_dir = checkpointing_config.get('dirpath', '')
            
            if checkpoint_dir and not checkpoint_dir.startswith('/dbfs'):
                warnings.append(f"Checkpoint path '{checkpoint_dir}' should use DBFS path for Databricks")
            
            # Check distributed training
            distributed_config = config.get('distributed', {})
            backend = distributed_config.get('backend', 'nccl')
            
            if backend != 'nccl':
                warnings.append("NCCL backend recommended for Databricks GPU clusters")
            
            # Check batch size for cluster resources
            batch_size = config.get('training', {}).get('batch_size', 16)
            if batch_size > 64:
                warnings.append("Large batch sizes may require high-memory Databricks clusters")
            
            # Check MLflow integration
            if 'mlflow' not in str(config):
                warnings.append("MLflow integration not configured for Databricks")
            
        except Exception as e:
            warnings.append(f"Databricks validation failed: {e}")
        
        return len(warnings) == 0, warnings
    
    def fix_common_issues(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically fix common configuration issues.
        
        Args:
            config: Original configuration
            
        Returns:
            Fixed configuration
        """
        fixed_config = config.copy()
        
        # Set default values for missing optional fields
        if 'projection' not in fixed_config:
            fixed_config['projection'] = {
                'hidden_dims': [768],
                'activation': 'gelu',
                'dropout': 0.1
            }
        
        if 'cross_attention' not in fixed_config:
            fixed_config['cross_attention'] = {
                'hidden_size': 512,
                'num_heads': 8,
                'dropout': 0.1
            }
        
        if 'fusion' not in fixed_config:
            fixed_config['fusion'] = {
                'strategy': 'cross_attention',
                'temperature': 0.1
            }
        
        # Fix dimension compatibility
        ts_dim = fixed_config.get('time_series_encoder', {}).get('embedding_dim')
        text_dim = fixed_config.get('text_decoder', {}).get('embedding_dim')
        
        if ts_dim and text_dim and ts_dim != text_dim:
            # Add projection to align dimensions
            fixed_config['projection']['input_dim'] = ts_dim
            fixed_config['projection']['output_dim'] = text_dim
        
        # Fix cross-attention dimensions
        cross_attn = fixed_config.get('cross_attention', {})
        if cross_attn.get('hidden_size') and cross_attn.get('num_heads'):
            hidden_size = cross_attn['hidden_size']
            num_heads = cross_attn['num_heads']
            
            if hidden_size % num_heads != 0:
                # Round to nearest compatible size
                fixed_hidden_size = (hidden_size // num_heads) * num_heads
                fixed_config['cross_attention']['hidden_size'] = fixed_hidden_size
                logger.warning(f"Adjusted cross-attention hidden_size from {hidden_size} to {fixed_hidden_size}")
        
        # Fix mixed precision conflicts
        mp_config = fixed_config.get('mixed_precision', {})
        if mp_config.get('fp16') and mp_config.get('bf16'):
            # Prefer bf16 for stability
            fixed_config['mixed_precision']['fp16'] = False
            logger.warning("Disabled fp16 in favor of bf16 for mixed precision")
        
        return fixed_config


def validate_config_file(config_path: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (is_valid, errors, loaded_config)
    """
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            return False, [f"Configuration file not found: {config_path}"], {}
        
        # Load configuration
        with open(config_file, 'r') as f:
            if config_file.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                config = json.load(f)
            else:
                return False, [f"Unsupported config file format: {config_file.suffix}"], {}
        
        # Validate configuration
        validator = ConfigValidator()
        is_valid, errors = validator.validate_complete_config(config)
        
        return is_valid, errors, config
        
    except Exception as e:
        return False, [f"Failed to load configuration: {e}"], {}


def validate_and_fix_config(config: Dict[str, Any], 
                           databricks: bool = False) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate and automatically fix configuration issues.
    
    Args:
        config: Configuration dictionary
        databricks: Whether to validate for Databricks environment
        
    Returns:
        Tuple of (is_valid, warnings, fixed_config)
    """
    validator = ConfigValidator()
    
    # Fix common issues first
    fixed_config = validator.fix_common_issues(config)
    
    # Validate fixed configuration
    is_valid, errors = validator.validate_complete_config(fixed_config)
    
    # Additional Databricks validation if requested
    if databricks:
        db_valid, db_warnings = validator.validate_databricks_config(fixed_config)
        errors.extend(db_warnings)
        is_valid = is_valid and db_valid
    
    return is_valid, errors, fixed_config


# Example usage
if __name__ == "__main__":
    # Test configuration validation
    test_config = {
        'model': {'name': 'multimodal_llm'},
        'time_series_encoder': {
            'embedding_dim': 512,
            'model_name': 'AutonLab/MOMENT-1-large'
        },
        'text_decoder': {
            'embedding_dim': 1024,
            'model_name': 'gpt2-medium'
        },
        'training': {
            'epochs': 10,
            'batch_size': 16
        },
        'data': {
            'data_dir': '/path/to/data'
        }
    }
    
    validator = ConfigValidator()
    is_valid, errors = validator.validate_complete_config(test_config)
    
    print(f"Configuration valid: {is_valid}")
    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  - {error}")
    
    # Test auto-fix
    fixed_config = validator.fix_common_issues(test_config)
    print(f"\nFixed config added {len(fixed_config) - len(test_config)} sections")