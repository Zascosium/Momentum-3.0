"""
Databricks environment utilities for MLLM project.
Handles environment detection, path management, and Databricks-specific configurations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import socket
import subprocess
import json

logger = logging.getLogger(__name__)

class DatabricksEnvironment:
    """
    Utility class for managing Databricks environment detection and configuration.
    """
    
    def __init__(self):
        """Initialize Databricks environment detector."""
        self._is_databricks = None
        self._cluster_info = None
        self._workspace_url = None
        self._dbfs_root = "/dbfs"
        
    def is_databricks(self) -> bool:
        """
        Detect if running in Databricks environment.
        
        Returns:
            True if running in Databricks
        """
        if self._is_databricks is None:
            self._is_databricks = self._detect_databricks()
        return self._is_databricks
    
    def _detect_databricks(self) -> bool:
        """Detect Databricks environment using multiple indicators."""
        indicators = [
            # Environment variable indicators
            os.environ.get('DATABRICKS_RUNTIME_VERSION') is not None,
            os.environ.get('SPARK_HOME') is not None and 'databricks' in os.environ.get('SPARK_HOME', '').lower(),
            os.environ.get('DB_HOME') is not None,
            
            # File system indicators
            os.path.exists('/databricks'),
            os.path.exists('/dbfs'),
            
            # Hostname indicators
            'driver' in socket.gethostname().lower() or 'worker' in socket.gethostname().lower(),
        ]
        
        # If any strong indicator is true, we're likely on Databricks
        return any(indicators)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get cluster information if available.
        
        Returns:
            Dictionary with cluster information
        """
        if self._cluster_info is None:
            self._cluster_info = self._extract_cluster_info()
        return self._cluster_info
    
    def _extract_cluster_info(self) -> Dict[str, Any]:
        """Extract cluster information from environment."""
        info = {
            'runtime_version': os.environ.get('DATABRICKS_RUNTIME_VERSION', 'unknown'),
            'cluster_id': os.environ.get('DB_CLUSTER_ID', 'unknown'),
            'is_driver': os.environ.get('DB_IS_DRIVER', 'false').lower() == 'true',
            'spark_version': os.environ.get('SPARK_VERSION', 'unknown'),
            'hostname': socket.gethostname(),
        }
        
        # Try to get additional cluster info
        try:
            # Check for cluster configuration file
            cluster_config_paths = [
                '/databricks/common/conf/cluster.conf',
                '/databricks/driver/conf/cluster-driver.conf'
            ]
            
            for config_path in cluster_config_paths:
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config_content = f.read()
                            # Parse basic cluster info from config
                            # This is environment-dependent and may vary
                            info['config_file'] = config_path
                            break
                    except Exception as e:
                        logger.debug(f"Could not read cluster config {config_path}: {e}")
        
        except Exception as e:
            logger.debug(f"Could not extract detailed cluster info: {e}")
        
        return info
    
    def get_workspace_url(self) -> Optional[str]:
        """
        Get workspace URL if available.
        
        Returns:
            Workspace URL or None
        """
        if self._workspace_url is None:
            self._workspace_url = self._detect_workspace_url()
        return self._workspace_url
    
    def _detect_workspace_url(self) -> Optional[str]:
        """Detect workspace URL from environment."""
        # Try different environment variables
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
        """
        Normalize path for Databricks environment.
        
        Args:
            path: Path to normalize
            ensure_dbfs: Whether to ensure path uses DBFS format
            
        Returns:
            Normalized path string
        """
        path_str = str(path)
        
        if not self.is_databricks():
            # Not on Databricks, return as-is
            return path_str
        
        # Handle different path formats
        if path_str.startswith('/dbfs/'):
            # Already DBFS format
            return path_str
        elif path_str.startswith('dbfs:/'):
            # Convert dbfs:/ to /dbfs/
            return path_str.replace('dbfs:/', '/dbfs/')
        elif path_str.startswith('/'):
            # Absolute path - add DBFS prefix if required
            if ensure_dbfs and not path_str.startswith('/databricks'):
                return f'/dbfs{path_str}'
            return path_str
        else:
            # Relative path - make it DBFS relative if required
            if ensure_dbfs:
                return f'/dbfs/FileStore/mllm/{path_str}'
            return path_str
    
    def create_databricks_paths(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Databricks-compatible paths from configuration.
        
        Args:
            base_config: Base configuration dictionary
            
        Returns:
            Configuration with Databricks-compatible paths
        """
        if not self.is_databricks():
            return base_config
        
        config = base_config.copy()
        
        # Update data paths
        if 'data' in config:
            data_config = config['data']
            if 'data_dir' in data_config:
                data_config['data_dir'] = self.normalize_path(data_config['data_dir'])
            
            # Update cache and temp directories
            if 'paths' in data_config:
                paths = data_config['paths']
                for key in ['cache_dir', 'temp_dir', 'output_dir']:
                    if key in paths:
                        paths[key] = self.normalize_path(paths[key])
        
        # Update checkpoint paths
        if 'checkpointing' in config:
            checkpoint_config = config['checkpointing']
            if 'dirpath' in checkpoint_config:
                checkpoint_config['dirpath'] = self.normalize_path(\n                    checkpoint_config['dirpath'], \n                    ensure_dbfs=True\n                )\n        \n        # Update logging paths\n        if 'logging' in config and 'log_dir' in config['logging']:\n            config['logging']['log_dir'] = self.normalize_path(\n                config['logging']['log_dir']\n            )\n        \n        # Update model save paths\n        if 'model' in config and 'save_dir' in config['model']:\n            config['model']['save_dir'] = self.normalize_path(\n                config['model']['save_dir']\n            )\n        \n        return config\n    \n    def setup_databricks_environment(self) -> Dict[str, Any]:\n        \"\"\"\n        Setup environment variables and configurations for Databricks.\n        \n        Returns:\n            Dictionary with environment setup results\n        \"\"\"\n        setup_results = {\n            'is_databricks': self.is_databricks(),\n            'setup_actions': [],\n            'warnings': [],\n            'errors': []\n        }\n        \n        if not self.is_databricks():\n            setup_results['warnings'].append('Not running on Databricks - skipping setup')\n            return setup_results\n        \n        try:\n            # Set up DBFS root if not set\n            if not os.environ.get('DBFS_ROOT'):\n                os.environ['DBFS_ROOT'] = self._dbfs_root\n                setup_results['setup_actions'].append('Set DBFS_ROOT environment variable')\n            \n            # Ensure required directories exist\n            required_dirs = [\n                '/dbfs/FileStore/mllm',\n                '/dbfs/FileStore/mllm/checkpoints',\n                '/dbfs/FileStore/mllm/logs',\n                '/dbfs/FileStore/mllm/models',\n                '/dbfs/FileStore/mllm/data'\n            ]\n            \n            for dir_path in required_dirs:\n                try:\n                    os.makedirs(dir_path, exist_ok=True)\n                    setup_results['setup_actions'].append(f'Created directory: {dir_path}')\n                except Exception as e:\n                    setup_results['warnings'].append(f'Could not create directory {dir_path}: {e}')\n            \n            # Set up MLflow tracking URI for Databricks\n            if not os.environ.get('MLFLOW_TRACKING_URI'):\n                # Use Databricks MLflow tracking\n                os.environ['MLFLOW_TRACKING_URI'] = 'databricks'\n                setup_results['setup_actions'].append('Set MLflow tracking URI to Databricks')\n            \n            # Configure distributed training settings\n            cluster_info = self.get_cluster_info()\n            if cluster_info.get('is_driver'):\n                setup_results['setup_actions'].append('Detected driver node - ready for distributed training')\n            \n            # Set memory and performance optimizations\n            self._setup_performance_optimizations(setup_results)\n            \n        except Exception as e:\n            setup_results['errors'].append(f'Environment setup failed: {e}')\n            logger.error(f'Databricks environment setup failed: {e}')\n        \n        return setup_results\n    \n    def _setup_performance_optimizations(self, setup_results: Dict[str, Any]) -> None:\n        \"\"\"Setup performance optimizations for Databricks.\"\"\"\n        try:\n            # Set CUDA settings if available\n            if self._has_gpu():\n                os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')\n                setup_results['setup_actions'].append('Configured GPU settings')\n            \n            # Set PyTorch settings\n            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')\n            \n            # Set parallelism settings\n            import multiprocessing\n            cpu_count = multiprocessing.cpu_count()\n            os.environ.setdefault('OMP_NUM_THREADS', str(min(cpu_count, 8)))\n            \n            setup_results['setup_actions'].append('Applied performance optimizations')\n            \n        except Exception as e:\n            setup_results['warnings'].append(f'Performance optimization setup failed: {e}')\n    \n    def _has_gpu(self) -> bool:\n        \"\"\"Check if GPU is available.\"\"\"\n        try:\n            import torch\n            return torch.cuda.is_available()\n        except ImportError:\n            return False\n    \n    def get_databricks_config_overrides(self) -> Dict[str, Any]:\n        \"\"\"\n        Get configuration overrides specific to Databricks environment.\n        \n        Returns:\n            Dictionary with Databricks-specific configuration overrides\n        \"\"\"\n        if not self.is_databricks():\n            return {}\n        \n        cluster_info = self.get_cluster_info()\n        \n        overrides = {\n            'distributed': {\n                'backend': 'nccl' if self._has_gpu() else 'gloo',\n                'find_unused_parameters': False,\n                'gradient_as_bucket_view': True,\n                'broadcast_buffers': False\n            },\n            'mixed_precision': {\n                'enabled': True,\n                'fp16': False,  # Prefer bf16 on modern hardware\n                'bf16': True if self._supports_bf16() else False\n            },\n            'checkpointing': {\n                'dirpath': '/dbfs/FileStore/mllm/checkpoints',\n                'save_top_k': 3,\n                'monitor': 'val_loss',\n                'mode': 'min'\n            },\n            'logging': {\n                'log_dir': '/dbfs/FileStore/mllm/logs',\n                'level': 'INFO'\n            },\n            'memory': {\n                'pin_memory': True,\n                'non_blocking': True,\n                'persistent_workers': True,\n                'prefetch_factor': 2\n            }\n        }\n        \n        # Adjust batch size based on available memory\n        if self._has_gpu():\n            try:\n                import torch\n                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB\n                if gpu_memory >= 16:\n                    recommended_batch_size = 32\n                elif gpu_memory >= 8:\n                    recommended_batch_size = 16\n                else:\n                    recommended_batch_size = 8\n                \n                overrides['training'] = {\n                    'batch_size': recommended_batch_size,\n                    'gradient_accumulation_steps': max(1, 64 // recommended_batch_size)\n                }\n            except Exception:\n                pass\n        \n        return overrides\n    \n    def _supports_bf16(self) -> bool:\n        \"\"\"Check if the environment supports bfloat16.\"\"\"\n        try:\n            import torch\n            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()\n        except (ImportError, AttributeError):\n            return False\n    \n    def validate_databricks_setup(self) -> Tuple[bool, List[str]]:\n        \"\"\"\n        Validate Databricks setup and configuration.\n        \n        Returns:\n            Tuple of (is_valid, list_of_issues)\n        \"\"\"\n        issues = []\n        \n        if not self.is_databricks():\n            return True, []  # Not on Databricks, no validation needed\n        \n        # Check DBFS access\n        try:\n            test_path = '/dbfs/FileStore/mllm/.test_write'\n            with open(test_path, 'w') as f:\n                f.write('test')\n            os.remove(test_path)\n        except Exception as e:\n            issues.append(f'DBFS write access failed: {e}')\n        \n        # Check MLflow access\n        try:\n            import mlflow\n            mlflow.set_tracking_uri('databricks')\n            # Try to access MLflow\n            mlflow.get_tracking_uri()\n        except Exception as e:\n            issues.append(f'MLflow access failed: {e}')\n        \n        # Check distributed training prerequisites\n        cluster_info = self.get_cluster_info()\n        if not cluster_info.get('is_driver', False):\n            issues.append('Not running on driver node - distributed training may not work')\n        \n        # Check GPU availability if expected\n        if self._has_gpu():\n            try:\n                import torch\n                torch.cuda.device_count()\n            except Exception as e:\n                issues.append(f'GPU access failed: {e}')\n        \n        is_valid = len(issues) == 0\n        return is_valid, issues


def get_databricks_environment() -> DatabricksEnvironment:\n    \"\"\"\n    Get Databricks environment instance (singleton pattern).\n    \n    Returns:\n        DatabricksEnvironment instance\n    \"\"\"\n    if not hasattr(get_databricks_environment, '_instance'):\n        get_databricks_environment._instance = DatabricksEnvironment()\n    return get_databricks_environment._instance\n\n\ndef setup_databricks_logging() -> None:\n    \"\"\"\n    Setup logging configuration optimized for Databricks.\n    \"\"\"\n    env = get_databricks_environment()\n    \n    if env.is_databricks():\n        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n        \n        # Setup file logging to DBFS\n        log_dir = '/dbfs/FileStore/mllm/logs'\n        os.makedirs(log_dir, exist_ok=True)\n        \n        import logging.handlers\n        \n        # Create rotating file handler\n        log_file = os.path.join(log_dir, 'mllm_training.log')\n        file_handler = logging.handlers.RotatingFileHandler(\n            log_file, maxBytes=10*1024*1024, backupCount=5\n        )\n        file_handler.setFormatter(logging.Formatter(log_format))\n        \n        # Setup console handler\n        console_handler = logging.StreamHandler()\n        console_handler.setFormatter(logging.Formatter(log_format))\n        \n        # Configure root logger\n        root_logger = logging.getLogger()\n        root_logger.setLevel(logging.INFO)\n        root_logger.addHandler(file_handler)\n        root_logger.addHandler(console_handler)\n        \n        logger.info(f'Databricks logging configured - logs saved to {log_file}')\n\n\ndef configure_for_databricks(config: Dict[str, Any]) -> Dict[str, Any]:\n    \"\"\"\n    Configure settings for optimal Databricks performance.\n    \n    Args:\n        config: Base configuration dictionary\n        \n    Returns:\n        Databricks-optimized configuration\n    \"\"\"\n    env = get_databricks_environment()\n    \n    if not env.is_databricks():\n        return config\n    \n    # Setup environment\n    setup_results = env.setup_databricks_environment()\n    \n    if setup_results['errors']:\n        logger.error(f\"Databricks setup errors: {setup_results['errors']}\")\n    \n    if setup_results['warnings']:\n        for warning in setup_results['warnings']:\n            logger.warning(warning)\n    \n    # Apply path normalization\n    config = env.create_databricks_paths(config)\n    \n    # Apply Databricks-specific overrides\n    databricks_overrides = env.get_databricks_config_overrides()\n    \n    # Merge overrides into config\n    def deep_merge(base: Dict, override: Dict) -> Dict:\n        result = base.copy()\n        for key, value in override.items():\n            if key in result and isinstance(result[key], dict) and isinstance(value, dict):\n                result[key] = deep_merge(result[key], value)\n            else:\n                result[key] = value\n        return result\n    \n    config = deep_merge(config, databricks_overrides)\n    \n    # Validate setup\n    is_valid, issues = env.validate_databricks_setup()\n    if not is_valid:\n        logger.warning(f\"Databricks validation issues: {issues}\")\n    \n    logger.info(\"Configuration optimized for Databricks environment\")\n    return config\n\n\n# Example usage\nif __name__ == \"__main__\":\n    # Test Databricks environment detection\n    env = get_databricks_environment()\n    \n    print(f\"Is Databricks: {env.is_databricks()}\")\n    print(f\"Cluster info: {env.get_cluster_info()}\")\n    print(f\"Workspace URL: {env.get_workspace_url()}\")\n    \n    # Test path normalization\n    test_paths = [\n        \"/tmp/data\",\n        \"./models\",\n        \"/dbfs/FileStore/data\",\n        \"dbfs:/mnt/data\"\n    ]\n    \n    for path in test_paths:\n        normalized = env.normalize_path(path)\n        print(f\"{path} -> {normalized}\")\n    \n    # Test configuration\n    test_config = {\n        'data': {'data_dir': './data'},\n        'checkpointing': {'dirpath': './checkpoints'}\n    }\n    \n    configured = configure_for_databricks(test_config)\n    print(f\"Configured: {configured}\")\n