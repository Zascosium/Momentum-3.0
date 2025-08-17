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

@cli.command()
@click.option('--model-path', default='./checkpoints/final_model', help='Path to trained model')
@click.option('--interactive', is_flag=True, help='Run interactive demo')
@click.option('--batch-size', type=int, default=5, help='Number of samples for batch demo')
@click.option('--performance-test', is_flag=True, help='Run performance test')
@click.option('--iterations', type=int, default=10, help='Number of iterations for performance test')
def demo(model_path, interactive, batch_size, performance_test, iterations):
    """Run model demonstration - Databricks compatible"""
    
    print("🚀 Databricks Demo Started")
    print(f"Model path: {model_path}")
    
    try:
        # Load config
        config_dir = project_root / 'config'
        config = {}
        
        for config_file in ['model_config.yaml', 'data_config.yaml']:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
                print(f"✅ Loaded {config_file}")
        
        # Import patched demo pipeline
        import importlib.util
        pipeline_file = src_path / 'pipelines' / 'demo_pipeline_databricks.py'
        
        if pipeline_file.exists():
            spec = importlib.util.spec_from_file_location("demo_pipeline_databricks", pipeline_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            DemoPipeline = module.DemoPipeline
        else:
            raise ImportError("Demo pipeline not found.")
        
        # Create demo pipeline
        pipeline = DemoPipeline(
            model_path=model_path,
            config=config
        )
        
        # Run appropriate demo mode
        if interactive:
            result = pipeline.run_interactive_demo()
        elif performance_test:
            result = pipeline.run_performance_test(num_iterations=iterations)
        else:
            result = pipeline.run_batch_demo(num_samples=batch_size)
        
        print("🎉 Demo completed successfully!")
        print(f"Results: {result}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    cli()
