#!/usr/bin/env python3
"""
Databricks CLI for MLLM Training
==============================

Simplified CLI that works reliably in Databricks environments.
"""

import click
import sys
import os
from pathlib import Path
import logging
import yaml

# Setup paths
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

@click.group()
def cli():
    """Databricks-compatible MLLM CLI"""
    pass

@cli.command()
@click.option('--epochs', type=int, default=5, help='Number of training epochs')
@click.option('--batch-size', type=int, default=16, help='Training batch size')
@click.option('--mixed-precision', is_flag=True, help='Enable mixed precision training')
@click.option('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
def train(epochs, batch_size, mixed_precision, checkpoint_dir):
    """Train the multimodal model - Databricks compatible"""
    
    print("🚀 Databricks Training Started")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}")
    
    try:
        # Load config
        config_dir = project_root / 'config'
        config = {}
        
        for config_file in ['training_config.yaml', 'model_config.yaml', 'data_config.yaml']:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
                print(f"✅ Loaded {config_file}")
        
        # Update config with CLI options
        config.setdefault('training', {})['epochs'] = epochs
        config.setdefault('training', {})['batch_size'] = batch_size
        config.setdefault('mixed_precision', {})['enabled'] = mixed_precision
        
        # Fix checkpoint directory for local/non-Databricks environments
        if 'checkpointing' in config and config['checkpointing'].get('dirpath', '').startswith('/dbfs'):
            config['checkpointing']['dirpath'] = checkpoint_dir
        
        # Disable time series reconstruction loss for testing with mock components
        config.setdefault('loss', {})['time_series_reconstruction_weight'] = 0.0
        config.setdefault('loss', {})['alignment_loss_weight'] = 0.0
        
        # Import patched training pipeline
        import importlib.util
        pipeline_file = src_path / 'pipelines' / 'training_pipeline_databricks.py'
        
        if pipeline_file.exists():
            spec = importlib.util.spec_from_file_location("training_pipeline_databricks", pipeline_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TrainingPipeline = module.TrainingPipeline
        else:
            raise ImportError("Patched training pipeline not found. Run fix_databricks_training.py first.")
        
        # Create and run pipeline
        pipeline = TrainingPipeline(
            config=config,
            checkpoint_dir=checkpoint_dir,
            experiment_name="databricks_training"
        )
        
        result = pipeline.run(verbose=True)
        
        print("🎉 Training completed successfully!")
        print(f"Results: {result}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    cli()
