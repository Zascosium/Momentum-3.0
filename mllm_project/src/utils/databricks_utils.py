"""
Databricks utilities for MLLM training.
Handles Databricks-specific environment setup and configuration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)

class DatabricksEnvironment:
    """
    Databricks environment detection and configuration.
    """
    
    def __init__(self):
        """Initialize Databricks environment."""
        self._is_databricks = self._detect_databricks()
        self._cluster_info = None
        self._workspace_url = None
        self._dbfs_root = '/dbfs'
    
    def _detect_databricks(self) -> bool:
        """Detect if running on Databricks."""
        return (
            'DATABRICKS_RUNTIME_VERSION' in os.environ or
            'DATABRICKS_HOST' in os.environ or
            'DB_WORKSPACE_URL' in os.environ or
            os.path.exists('/databricks')
        )
    
    def is_databricks(self) -> bool:
        """Check if running on Databricks."""
        return self._is_databricks
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information."""
        if self._cluster_info is None:
            self._cluster_info = self._detect_cluster_info()
        return self._cluster_info
    
    def _detect_cluster_info(self) -> Dict[str, Any]:
        """Detect cluster information."""
        info = {
            'is_databricks': self.is_databricks(),
            'runtime_version': os.environ.get('DATABRICKS_RUNTIME_VERSION', 'unknown'),
            'cluster_id': os.environ.get('DATABRICKS_CLUSTER_ID', 'unknown'),
            'is_driver': os.environ.get('DATABRICKS_IS_DRIVER', 'false').lower() == 'true',
            'node_type': os.environ.get('DATABRICKS_NODE_TYPE', 'unknown'),
            'num_workers': int(os.environ.get('DATABRICKS_NUM_WORKERS', '0')),
            'gpu_enabled': os.environ.get('DATABRICKS_GPU_ENABLED', 'false').lower() == 'true'
        }
        
        return info
    
    def get_workspace_url(self) -> Optional[str]:
        """Get workspace URL if available."""
        if self._workspace_url is None:
            self._workspace_url = self._detect_workspace_url()
        return self._workspace_url
    
    def _detect_workspace_url(self) -> Optional[str]:
        """Detect workspace URL from environment."""
        url_sources = [
            os.environ.get('DATABRICKS_HOST'),
            os.environ.get('DATABRICKS_WORKSPACE_URL'),
            os.environ.get('DB_WORKSPACE_URL'),
        ]
        
        for url in url_sources:
            if url and url.startswith('https://'):
                return url
        
        return None
    
    def normalize_path(self, path: Union[str, Path], ensure_dbfs: bool = True) -> str:
        """Normalize path for Databricks environment."""
        path_str = str(path)
        
        if not self.is_databricks():
            return path_str
        
        if path_str.startswith('/dbfs/'):
            return path_str
        elif path_str.startswith('dbfs:/'):
            return path_str.replace('dbfs:/', '/dbfs/')
        elif path_str.startswith('/'):
            if ensure_dbfs and not path_str.startswith('/databricks'):
                return f'/dbfs{path_str}'
            return path_str
        else:
            if ensure_dbfs:
                return f'/dbfs/FileStore/mllm/{path_str}'
            return path_str
    
    def create_databricks_paths(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Databricks-compatible paths from configuration."""
        if not self.is_databricks():
            return base_config
        
        config = base_config.copy()
        
        # Update data paths
        if 'data' in config:
            data_config = config['data']
            if 'data_dir' in data_config:
                data_config['data_dir'] = self.normalize_path(data_config['data_dir'])
            
            if 'paths' in data_config:
                paths = data_config['paths']
                for key in ['cache_dir', 'temp_dir', 'output_dir']:
                    if key in paths:
                        paths[key] = self.normalize_path(paths[key])
        
        # Update checkpoint paths
        if 'checkpointing' in config:
            checkpoint_config = config['checkpointing']
            if 'dirpath' in checkpoint_config:
                checkpoint_config['dirpath'] = self.normalize_path(
                    checkpoint_config['dirpath'], 
                    ensure_dbfs=True
                )
        
        # Update logging paths
        if 'logging' in config and 'log_dir' in config['logging']:
            config['logging']['log_dir'] = self.normalize_path(
                config['logging']['log_dir']
            )
        
        # Update model save paths
        if 'model' in config and 'save_dir' in config['model']:
            config['model']['save_dir'] = self.normalize_path(
                config['model']['save_dir']
            )
        
        return config
    
    def setup_databricks_environment(self) -> Dict[str, Any]:
        """Setup environment variables and configurations for Databricks."""
        setup_results = {
            'is_databricks': self.is_databricks(),
            'setup_actions': [],
            'warnings': [],
            'errors': []
        }
        
        if not self.is_databricks():
            setup_results['warnings'].append('Not running on Databricks - skipping setup')
            return setup_results
        
        try:
            # Set up DBFS root if not set
            if not os.environ.get('DBFS_ROOT'):
                os.environ['DBFS_ROOT'] = self._dbfs_root
                setup_results['setup_actions'].append('Set DBFS_ROOT environment variable')
            
            # Ensure required directories exist
            required_dirs = [
                '/dbfs/FileStore/mllm',
                '/dbfs/FileStore/mllm/checkpoints',
                '/dbfs/FileStore/mllm/logs',
                '/dbfs/FileStore/mllm/models',
                '/dbfs/FileStore/mllm/data'
            ]
            
            for dir_path in required_dirs:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    setup_results['setup_actions'].append(f'Created directory: {dir_path}')
                except Exception as e:
                    setup_results['warnings'].append(f'Could not create directory {dir_path}: {e}')
            
            # Set up MLflow tracking URI for Databricks
            if not os.environ.get('MLFLOW_TRACKING_URI'):
                os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
                setup_results['setup_actions'].append('Set MLflow tracking URI to Databricks')
            
            # Configure distributed training settings
            cluster_info = self.get_cluster_info()
            if cluster_info.get('is_driver'):
                setup_results['setup_actions'].append('Detected driver node - ready for distributed training')
            
            # Set memory and performance optimizations
            self._setup_performance_optimizations(setup_results)
            
        except Exception as e:
            setup_results['errors'].append(f'Environment setup failed: {e}')
            logger.error(f'Databricks environment setup failed: {e}')
        
        return setup_results
    
    def _setup_performance_optimizations(self, setup_results: Dict[str, Any]) -> None:
        """Setup performance optimizations for Databricks."""
        try:
            # Set CUDA settings if available
            if self._has_gpu():
                os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
                setup_results['setup_actions'].append('Configured GPU settings')
            
            # Set PyTorch settings
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
            
            # Set parallelism settings
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            os.environ.setdefault('OMP_NUM_THREADS', str(min(cpu_count, 8)))
            
            setup_results['setup_actions'].append('Applied performance optimizations')
            
        except Exception as e:
            setup_results['warnings'].append(f'Performance optimization setup failed: {e}')
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_databricks_config_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides specific to Databricks environment."""
        if not self.is_databricks():
            return {}
        
        cluster_info = self.get_cluster_info()
        
        overrides = {
            'distributed': {
                'backend': 'nccl' if self._has_gpu() else 'gloo',
                'find_unused_parameters': False,
                'gradient_as_bucket_view': True,
                'broadcast_buffers': False
            },
            'mixed_precision': {
                'enabled': True,
                'fp16': False,
                'bf16': True if self._supports_bf16() else False
            },
            'checkpointing': {
                'dirpath': '/dbfs/FileStore/mllm/checkpoints',
                'save_top_k': 3,
                'monitor': 'val_loss',
                'mode': 'min'
            },
            'logging': {
                'log_dir': '/dbfs/FileStore/mllm/logs',
                'level': 'INFO'
            },
            'memory': {
                'pin_memory': True,
                'non_blocking': True,
                'persistent_workers': True,
                'prefetch_factor': 2
            }
        }
        
        # Adjust batch size based on available memory
        if self._has_gpu():
            try:
                import torch
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory >= 16:
                    recommended_batch_size = 32
                elif gpu_memory >= 8:
                    recommended_batch_size = 16
                else:
                    recommended_batch_size = 8
                
                overrides['training'] = {
                    'batch_size': recommended_batch_size,
                    'gradient_accumulation_steps': max(1, 64 // recommended_batch_size)
                }
            except Exception:
                pass
        
        return overrides
    
    def _supports_bf16(self) -> bool:
        """Check if the environment supports bfloat16."""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except (ImportError, AttributeError):
            return False
    
    def validate_databricks_setup(self) -> Tuple[bool, List[str]]:
        """Validate Databricks setup and configuration."""
        issues = []
        
        if not self.is_databricks():
            return True, []
        
        # Check DBFS access
        try:
            test_path = '/dbfs/FileStore/mllm/.test_write'
            with open(test_path, 'w') as f:
                f.write('test')
            os.remove(test_path)
        except Exception as e:
            issues.append(f'DBFS write access failed: {e}')
        
        # Check MLflow access
        try:
            import mlflow
            mlflow.set_tracking_uri('databricks')
            mlflow.get_tracking_uri()
        except Exception as e:
            issues.append(f'MLflow access failed: {e}')
        
        # Check distributed training prerequisites
        cluster_info = self.get_cluster_info()
        if not cluster_info.get('is_driver', False):
            issues.append('Not running on driver node - distributed training may not work')
        
        # Check GPU availability if expected
        if self._has_gpu():
            try:
                import torch
                torch.cuda.device_count()
            except Exception as e:
                issues.append(f'GPU access failed: {e}')
        
        is_valid = len(issues) == 0
        return is_valid, issues


def get_databricks_environment() -> DatabricksEnvironment:
    """Get Databricks environment instance (singleton pattern)."""
    if not hasattr(get_databricks_environment, '_instance'):
        get_databricks_environment._instance = DatabricksEnvironment()
    return get_databricks_environment._instance


def setup_databricks_logging() -> None:
    """Setup logging configuration optimized for Databricks."""
    env = get_databricks_environment()
    
    if env.is_databricks():
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Setup file logging to DBFS
        log_dir = '/dbfs/FileStore/mllm/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        import logging.handlers
        
        # Create rotating file handler
        log_file = os.path.join(log_dir, 'mllm_training.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logger.info(f'Databricks logging configured - logs saved to {log_file}')


def configure_for_databricks(config: Dict[str, Any]) -> Dict[str, Any]:
    """Configure settings for optimal Databricks performance."""
    env = get_databricks_environment()
    
    if not env.is_databricks():
        return config
    
    # Setup environment
    setup_results = env.setup_databricks_environment()
    
    if setup_results['errors']:
        logger.error(f"Databricks setup errors: {setup_results['errors']}")
    
    if setup_results['warnings']:
        for warning in setup_results['warnings']:
            logger.warning(warning)
    
    # Apply path normalization
    config = env.create_databricks_paths(config)
    
    # Apply Databricks-specific overrides
    databricks_overrides = env.get_databricks_config_overrides()
    
    # Merge overrides into config
    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    config = deep_merge(config, databricks_overrides)
    
    # Validate setup
    is_valid, issues = env.validate_databricks_setup()
    if not is_valid:
        logger.warning(f"Databricks validation issues: {issues}")
    
    logger.info("Configuration optimized for Databricks environment")
    return config


# Example usage
if __name__ == "__main__":
    # Test Databricks environment detection
    env = get_databricks_environment()
    
    print(f"Is Databricks: {env.is_databricks()}")
    print(f"Cluster info: {env.get_cluster_info()}")
    print(f"Workspace URL: {env.get_workspace_url()}")
    
    # Test path normalization
    test_paths = [
        "/tmp/data",
        "./models",
        "/dbfs/FileStore/data",
        "dbfs:/mnt/data"
    ]
    
    for path in test_paths:
        normalized = env.normalize_path(path)
        print(f"{path} -> {normalized}")
    
    # Test configuration
    test_config = {
        'data': {'data_dir': './data'},
        'checkpointing': {'dirpath': './checkpoints'}
    }
    
    configured = configure_for_databricks(test_config)
    print(f"Configured: {configured}")