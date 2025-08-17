# 🚀 Databricks Production Training Guide
## Multimodal LLM Training Pipeline

This guide walks you through setting up and running production-level training for your multimodal LLM in Databricks workspace.

---

## 📋 Prerequisites

### Databricks Cluster Requirements
- **Runtime**: Databricks Runtime 13.3 LTS ML or higher
- **Node Type**: GPU-enabled instances (recommended: `g5.xlarge` or higher)
- **Workers**: 2-4 worker nodes for distributed training
- **Driver**: Same as worker nodes for memory consistency

### Required Libraries (Install on Cluster)
```bash
# Core ML libraries
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Databricks integration
mlflow>=2.4.0
databricks-cli>=0.17.0

# Optional: MOMENT model (if available)
# momentfm  # Install if you have access to MOMENT package
```

---

## 🏗️ Step 1: Workspace Setup

### 1.1 Upload Project Files
```python
# In Databricks notebook cell
%sh
# Create project directory structure
mkdir -p /databricks/driver/mllm_project
cd /databricks/driver/mllm_project

# If uploading via UI, ensure this structure:
# mllm_project/
# ├── src/
# │   ├── models/
# │   ├── data/
# │   ├── training/
# │   ├── utils/
# │   └── pipelines/
# ├── config/
# ├── notebooks/
# └── data/
```

### 1.2 Verify Environment
```python
# Run this in a Databricks notebook
%python
import os
import sys
from pathlib import Path

# Add project to Python path
project_root = "/databricks/driver/mllm_project"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Verify Databricks environment
print(f"Databricks Runtime: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Not detected')}")
print(f"Cluster ID: {os.environ.get('DB_CLUSTER_ID', 'Not detected')}")
print(f"Is Driver: {os.environ.get('DB_IS_DRIVER', 'false')}")

# Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

---

## ⚙️ Step 2: Configuration Setup

### 2.1 Create Production Configuration
```python
# Create production config file
%python
import yaml
import os

# Production training configuration
production_config = {
    'model': {
        'name': 'multimodal_llm_production',
        'version': '1.0'
    },
    'time_series_encoder': {
        'model_name': 'AutonLab/MOMENT-1-large',
        'embedding_dim': 512,
        'max_sequence_length': 1024,
        'freeze_encoder': False,
        'moment_config': {
            'n_channels': 3,  # Adjust based on your data
            'patch_len': 16,
            'stride': 16,
            'normalize': True,
            'use_revin': True
        }
    },
    'text_decoder': {
        'model_name': 'microsoft/DialoGPT-medium',
        'embedding_dim': 512,
        'vocab_size': 50257,
        'max_position_embeddings': 1024,
        'freeze_decoder': False
    },
    'projection': {
        'hidden_dims': [512, 512],
        'activation': 'gelu',
        'dropout': 0.1
    },
    'cross_attention': {
        'hidden_size': 512,
        'num_heads': 8,
        'dropout': 0.1
    },
    'fusion': {
        'strategy': 'cross_attention',
        'temperature': 0.1
    },
    'training': {
        'epochs': 10,
        'batch_size': 8,  # Adjust based on GPU memory
        'gradient_accumulation_steps': 4,  # Effective batch size = 32
        'max_grad_norm': 1.0,
        'warmup_steps': 500,
        'save_steps': 1000,
        'eval_steps': 500,
        'logging_steps': 100,
        'early_stopping': {
            'patience': 3,
            'min_delta': 0.001,
            'metric': 'val_loss',
            'mode': 'min'
        }
    },
    'optimizer': {
        'name': 'adamw',
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8
    },
    'scheduler': {
        'name': 'cosine_with_warmup',
        'warmup_ratio': 0.1,
        'num_cycles': 0.5
    },
    'mixed_precision': {
        'enabled': True,
        'fp16': False,
        'bf16': True,  # Better stability on modern GPUs
        'loss_scale': 0
    },
    'distributed': {
        'backend': 'nccl',
        'find_unused_parameters': False,
        'gradient_as_bucket_view': True
    },
    'checkpointing': {
        'save_top_k': 3,
        'monitor': 'val_loss',
        'mode': 'min',
        'save_last': True,
        'dirpath': '/dbfs/FileStore/mllm/checkpoints',
        'filename': 'epoch_{epoch:02d}-val_loss_{val_loss:.3f}'
    },
    'data': {
        'data_dir': '/dbfs/FileStore/mllm/data',
        'domains': {
            'included': ['finance', 'weather', 'energy']  # Adjust for your domains
        },
        'splits': {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1
        },
        'max_sequence_length': 512,
        'preprocessing': {
            'normalize_timeseries': True,
            'handle_missing_values': True,
            'augmentation_enabled': False  # Disable for production stability
        }
    },
    'loss': {
        'text_generation_weight': 1.0,
        'time_series_reconstruction_weight': 0.1,
        'alignment_loss_weight': 0.05,
        'label_smoothing': 0.1
    },
    'validation': {
        'val_check_interval': 0.25,
        'limit_val_batches': 1.0,
        'num_sanity_val_steps': 2
    },
    'logging': {
        'level': 'INFO',
        'log_every_n_steps': 50,
        'log_model_summary': True
    },
    'memory': {
        'pin_memory': True,
        'non_blocking': True,
        'persistent_workers': True,
        'prefetch_factor': 2
    },
    'environment': {
        'auto_detect_databricks': True,
        'setup_logging': True,
        'validate_setup': True
    }
}

# Save configuration
os.makedirs('/dbfs/FileStore/mllm/config', exist_ok=True)
with open('/dbfs/FileStore/mllm/config/production_config.yaml', 'w') as f:
    yaml.dump(production_config, f, default_flow_style=False, indent=2)

print("✅ Production configuration saved to /dbfs/FileStore/mllm/config/")
```

### 2.2 Validate Configuration
```python
# Validate the configuration
%python
from src.utils.config_validator import validate_and_fix_config
import yaml

# Load and validate config
with open('/dbfs/FileStore/mllm/config/production_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Validate for Databricks
is_valid, warnings, fixed_config = validate_and_fix_config(
    config=config,
    databricks=True
)

print(f"Configuration Valid: {is_valid}")
if warnings:
    print("\n⚠️ Warnings:")
    for warning in warnings:
        print(f"  - {warning}")

# Save fixed config if needed
if fixed_config != config:
    with open('/dbfs/FileStore/mllm/config/production_config_fixed.yaml', 'w') as f:
        yaml.dump(fixed_config, f, default_flow_style=False, indent=2)
    print("✅ Fixed configuration saved")
```

---

## 📊 Step 3: Data Preparation

### 3.1 Upload and Organize Data
```python
# Create data directory structure
%python
import os

# Create directory structure
data_dirs = [
    '/dbfs/FileStore/mllm/data/raw',
    '/dbfs/FileStore/mllm/data/processed',
    '/dbfs/FileStore/mllm/data/splits'
]

for dir_path in data_dirs:
    os.makedirs(dir_path, exist_ok=True)

print("✅ Data directories created")
print("Upload your data files to:")
print("  - Time series data: /dbfs/FileStore/mllm/data/raw/timeseries/")
print("  - Text data: /dbfs/FileStore/mllm/data/raw/text/")
```

### 3.2 Data Preprocessing Script
```python
# Create data preprocessing notebook cell
%python
from src.data.data_loader import MultimodalDataModule
from src.utils.databricks_utils import get_databricks_environment
import yaml

# Load configuration
with open('/dbfs/FileStore/mllm/config/production_config_fixed.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup Databricks environment
db_env = get_databricks_environment()
if db_env.is_databricks():
    print("✅ Databricks environment detected")
    config = db_env.create_databricks_paths(config)

# Initialize data module
data_module = MultimodalDataModule(config)

# Prepare data
print("🔄 Preparing data...")
data_module.setup('fit')

# Get data statistics
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

print(f"✅ Data prepared:")
print(f"  - Training samples: {len(train_loader.dataset)}")
print(f"  - Validation samples: {len(val_loader.dataset)}")
print(f"  - Batch size: {train_loader.batch_size}")
print(f"  - Total training batches: {len(train_loader)}")
```

---

## 🏋️ Step 4: Model Initialization

### 4.1 Initialize Model with Configuration Validation
```python
# Initialize the multimodal model
%python
from src.models.multimodal_model import MultimodalLLM
from src.utils.databricks_utils import configure_for_databricks
import yaml
import torch

# Load and configure for Databricks
with open('/dbfs/FileStore/mllm/config/production_config_fixed.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Apply Databricks optimizations
config = configure_for_databricks(config)

# Initialize model
print("🔄 Initializing multimodal model...")
try:
    model = MultimodalLLM(config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get model summary
    memory_stats = model.get_memory_usage()
    
    print("✅ Model initialized successfully")
    print(f"  - Device: {device}")
    print(f"  - Total parameters: {memory_stats['total_parameters']:,}")
    print(f"  - Trainable parameters: {memory_stats['trainable_parameters']:,}")
    print(f"  - Parameter memory: {memory_stats['parameter_memory_mb']:.1f} MB")
    
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    raise
```

---

## 🚀 Step 5: Training Setup

### 5.1 Create Training Notebook
```python
# Setup MLflow experiment
%python
import mlflow
from src.utils.mlflow_utils import setup_databricks_mlflow, log_system_info

# Setup MLflow for production
experiment_name = "/Users/your_username/MLLM_Production_Training"  # Adjust username
success = setup_databricks_mlflow(
    experiment_name=experiment_name,
    run_name=f"production_run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
)

if success:
    print("✅ MLflow experiment setup successful")
    log_system_info()
else:
    print("⚠️ MLflow setup failed - training will continue without logging")
```

### 5.2 Initialize Trainer
```python
# Create production trainer
%python
from src.training.trainer import create_trainer
import pandas as pd

# Create trainer with production settings
print("🔄 Initializing trainer...")
try:
    trainer = create_trainer(
        config_path='/dbfs/FileStore/mllm/config/production_config_fixed.yaml',
        databricks_optimized=True
    )
    
    print("✅ Trainer initialized successfully")
    print(f"  - Training device: {trainer.device}")
    print(f"  - Mixed precision: {trainer.precision_mode}")
    print(f"  - Distributed training: {trainer.is_distributed}")
    print(f"  - World size: {trainer.world_size}")
    
except Exception as e:
    print(f"❌ Trainer initialization failed: {e}")
    raise
```

---

## 🎯 Step 6: Production Training

### 6.1 Pre-training Validation
```python
# Run pre-training checks
%python
print("🔍 Running pre-training validation...")

# Check data loaders
try:
    # Test data loading
    train_batch = next(iter(trainer.train_dataloader))
    val_batch = next(iter(trainer.val_dataloader))
    
    print("✅ Data loaders working correctly")
    print(f"  - Train batch keys: {list(train_batch.keys())}")
    print(f"  - Batch shapes: {[(k, v.shape) for k, v in train_batch.items() if hasattr(v, 'shape')]}")
    
    # Test model forward pass
    with torch.no_grad():
        test_output = trainer.model(**train_batch)
    
    print("✅ Model forward pass successful")
    print(f"  - Output keys: {list(test_output.keys()) if hasattr(test_output, 'keys') else 'tensor'}")
    
except Exception as e:
    print(f"❌ Pre-training validation failed: {e}")
    raise
```

### 6.2 Start Training
```python
# Begin production training
%python
import time
from datetime import datetime

print("🚀 Starting production training...")
print(f"Training started at: {datetime.now()}")

# Start training
try:
    training_start_time = time.time()
    
    # Run training
    training_summary = trainer.fit()
    
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    print("🎉 Training completed successfully!")
    print(f"Training duration: {training_duration/3600:.2f} hours")
    print(f"Epochs completed: {training_summary['epochs_completed']}")
    
    # Print final metrics
    if 'final_train_metrics' in training_summary:
        print("\n📊 Final Training Metrics:")
        for metric, value in training_summary['final_train_metrics'].items():
            print(f"  - {metric}: {value:.4f}")
    
    if 'final_val_metrics' in training_summary:
        print("\n📊 Final Validation Metrics:")
        for metric, value in training_summary['final_val_metrics'].items():
            print(f"  - {metric}: {value:.4f}")

except Exception as e:
    print(f"❌ Training failed: {e}")
    # Save emergency checkpoint
    try:
        trainer._save_checkpoint('emergency_checkpoint')
        print("💾 Emergency checkpoint saved")
    except:
        pass
    raise
```

---

## 📊 Step 7: Monitoring and Validation

### 7.1 Monitor Training Progress
```python
# Monitor training in real-time (run in separate cell)
%python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Get current experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment:
    # Get runs from current experiment
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if not runs_df.empty:
        # Plot training metrics
        latest_run = runs_df.iloc[0]
        
        print(f"Latest run: {latest_run['run_name']}")
        print(f"Status: {latest_run['status']}")
        
        # Show metrics if available
        metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
        if metric_cols:
            print("\nCurrent Metrics:")
            for col in metric_cols:
                metric_name = col.replace('metrics.', '')
                value = latest_run[col]
                if pd.notna(value):
                    print(f"  - {metric_name}: {value:.4f}")
```

### 7.2 Model Evaluation
```python
# Evaluate final model
%python
print("🔍 Evaluating final model...")

if hasattr(trainer, 'test_dataloader') and trainer.test_dataloader is not None:
    # Run test evaluation
    test_metrics = trainer.test()
    
    print("📊 Test Results:")
    for metric, value in test_metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Log test metrics to MLflow
    if mlflow.active_run():
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"final_test_{metric}", value)
else:
    print("ℹ️ No test data available for evaluation")
```

---

## 💾 Step 8: Model Deployment Preparation

### 8.1 Model Registration
```python
# Register best model
%python
from src.utils.mlflow_utils import log_model_with_databricks

# Get the best model from checkpoints
best_checkpoint_path = trainer.checkpointing_config.get('dirpath', '/dbfs/FileStore/mllm/checkpoints')
print(f"Best model saved at: {best_checkpoint_path}/final_model")

# Register model in MLflow Model Registry
if mlflow.active_run():
    try:
        # Log final model with comprehensive metadata
        model_logged = log_model_with_databricks(
            model=trainer.model,
            model_name="multimodal_llm_production",
            register_model=True
        )
        
        if model_logged:
            print("✅ Model registered successfully in MLflow Model Registry")
        else:
            print("⚠️ Model logging had issues - check manually")
            
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
```

### 8.2 Create Inference Package
```python
# Prepare model for inference
%python
import os
import shutil
import json

# Create inference package
inference_dir = '/dbfs/FileStore/mllm/inference_package'
os.makedirs(inference_dir, exist_ok=True)

# Copy model files
final_model_dir = f"{trainer.checkpointing_config.get('dirpath', '/dbfs/FileStore/mllm/checkpoints')}/final_model"
if os.path.exists(final_model_dir):
    shutil.copytree(final_model_dir, f"{inference_dir}/model", dirs_exist_ok=True)

# Copy configuration
shutil.copy('/dbfs/FileStore/mllm/config/production_config_fixed.yaml', 
           f"{inference_dir}/config.yaml")

# Create inference metadata
inference_metadata = {
    "model_version": "1.0.0",
    "training_date": datetime.now().isoformat(),
    "training_duration_hours": training_duration / 3600 if 'training_duration' in locals() else None,
    "final_metrics": training_summary.get('final_val_metrics', {}) if 'training_summary' in locals() else {},
    "model_parameters": trainer.model.count_parameters(),
    "databricks_runtime": os.environ.get('DATABRICKS_RUNTIME_VERSION'),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
}

with open(f"{inference_dir}/metadata.json", 'w') as f:
    json.dump(inference_metadata, f, indent=2)

print(f"✅ Inference package created at: {inference_dir}")
```

---

## 🔧 Step 9: Production Checklist

### 9.1 Validation Checklist
```python
# Final production validation
%python
print("📋 Production Validation Checklist:")

checks = []

# Check 1: Model files exist
model_dir = f"{trainer.checkpointing_config.get('dirpath', '/dbfs/FileStore/mllm/checkpoints')}/final_model"
checks.append(("Model files saved", os.path.exists(model_dir)))

# Check 2: Configuration validated
checks.append(("Configuration validated", is_valid))

# Check 3: Training completed
checks.append(("Training completed", 'training_summary' in locals()))

# Check 4: Model registered
registered = False
if mlflow.active_run():
    try:
        client = mlflow.MlflowClient()
        models = client.search_registered_models("name='multimodal_llm_production'")
        registered = len(models) > 0
    except:
        pass
checks.append(("Model registered in MLflow", registered))

# Check 5: Performance metrics acceptable
acceptable_performance = False
if 'training_summary' in locals() and 'final_val_metrics' in training_summary:
    val_loss = training_summary['final_val_metrics'].get('val_loss', float('inf'))
    acceptable_performance = val_loss < 3.0  # Adjust threshold as needed
checks.append(("Performance acceptable", acceptable_performance))

# Print results
for check_name, passed in checks:
    status = "✅" if passed else "❌"
    print(f"  {status} {check_name}")

all_passed = all(passed for _, passed in checks)
print(f"\n{'🎉 All checks passed! Ready for production.' if all_passed else '⚠️ Some checks failed. Review before deployment.'}")
```

---

## 📚 Step 10: Documentation and Handoff

### 10.1 Generate Training Report
```python
# Generate comprehensive training report
%python
from src.utils.mlflow_utils import MLflowExperimentManager

if 'experiment' in locals() and experiment:
    try:
        # Create experiment manager
        exp_manager = MLflowExperimentManager(experiment_name)
        
        # Generate report
        report = exp_manager.create_experiment_report(
            output_path='/dbfs/FileStore/mllm/training_report.md'
        )
        
        print("✅ Training report generated at: /dbfs/FileStore/mllm/training_report.md")
        
    except Exception as e:
        print(f"⚠️ Report generation failed: {e}")
```

### 10.2 Cleanup and Finalization
```python
# Clean up resources
%python

# End MLflow run
if mlflow.active_run():
    mlflow.end_run()
    print("✅ MLflow run ended")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ GPU memory cleared")

print("\n🎉 Production training pipeline completed successfully!")
print("\n📋 Next Steps:")
print("1. Review training report and metrics")
print("2. Test inference with the deployed model")
print("3. Set up monitoring for production usage")
print("4. Create automated retraining pipeline if needed")
```

---

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

**Issue**: Out of memory errors
```python
# Solution: Reduce batch size and increase gradient accumulation
config['training']['batch_size'] = 4
config['training']['gradient_accumulation_steps'] = 8
```

**Issue**: Slow training
```python
# Solution: Enable optimizations
config['memory']['persistent_workers'] = True
config['memory']['pin_memory'] = True
config['distributed']['gradient_as_bucket_view'] = True
```

**Issue**: Model convergence problems
```python
# Solution: Adjust learning rate and warmup
config['optimizer']['learning_rate'] = 1e-5
config['training']['warmup_steps'] = 1000
```

**Issue**: DBFS path errors
```python
# Solution: Use environment variables and validation
from src.utils.databricks_utils import get_databricks_environment
db_env = get_databricks_environment()
config = db_env.create_databricks_paths(config)
```

---

## 📊 Production Monitoring Setup

After training, set up monitoring:

1. **Model Performance**: Track inference latency and accuracy
2. **Resource Usage**: Monitor GPU/CPU utilization
3. **Data Drift**: Monitor input data distribution changes
4. **Error Rates**: Track inference failures and recovery

---

This guide provides a complete production-ready training pipeline with comprehensive error handling, monitoring, and Databricks optimization. Follow each step carefully and adjust configurations based on your specific data and requirements.
