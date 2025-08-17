#!/usr/bin/env python3
"""
Simple Databricks Training Script
================================

This bypasses all CLI import issues and directly loads the training pipeline.
Use this if you're still getting "MultimodalLLM model not available" errors.
"""

import sys
import os
from pathlib import Path
import importlib.util

def load_module_directly(module_name, file_path):
    """Load a module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None

def run_training_direct(epochs=5, batch_size=16, mixed_precision=True):
    """Run training directly without CLI"""
    print("🚀 DIRECT DATABRICKS TRAINING")
    print("=" * 40)
    
    # Get current directory
    current_dir = Path.cwd()
    src_dir = current_dir / 'src'
    
    print(f"Working directory: {current_dir}")
    print(f"Src directory exists: {src_dir.exists()}")
    
    if not src_dir.exists():
        print("❌ src directory not found. Make sure you're in the project root.")
        return False
    
    # Add to path
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(current_dir))
    
    try:
        # Step 1: Load config
        print("\nStep 1: Loading configuration...")
        config_module = load_module_directly(
            "config_loader", 
            src_dir / "utils" / "config_loader.py"
        )
        
        if config_module:
            config = config_module.load_config_for_training("config")
            print("✅ Configuration loaded")
        else:
            # Fallback minimal config
            config = {
                'model': {'name': 'multimodal_llm'},
                'training': {'epochs': epochs, 'batch_size': batch_size},
                'data': {'data_dir': './data'},
                'mixed_precision': {'enabled': mixed_precision}
            }
            print("⚠️ Using fallback configuration")
        
        # Update config with parameters
        config['training']['epochs'] = epochs
        config['training']['batch_size'] = batch_size
        config['mixed_precision']['enabled'] = mixed_precision
        
        # Step 2: Load training pipeline directly
        print("\nStep 2: Loading training pipeline...")
        pipeline_module = load_module_directly(
            "training_pipeline",
            src_dir / "pipelines" / "training_pipeline.py"
        )
        
        if not pipeline_module:
            print("❌ Failed to load training pipeline module")
            return False
        
        TrainingPipeline = pipeline_module.TrainingPipeline
        print("✅ TrainingPipeline class loaded")
        
        # Step 3: Create pipeline instance
        print("\nStep 3: Creating training pipeline...")
        pipeline = TrainingPipeline(
            config=config,
            checkpoint_dir="./databricks_checkpoints",
            experiment_name="databricks_training"
        )
        print("✅ Training pipeline created successfully")
        
        # Step 4: Run training
        print(f"\nStep 4: Starting training (epochs={epochs}, batch_size={batch_size})...")
        print("Note: This will download models and start actual training.")
        print("Press Ctrl+C to stop if needed.")
        
        training_summary = pipeline.run(verbose=True)
        print("✅ Training completed successfully!")
        print(f"Summary: {training_summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 DATABRICKS DIRECT TRAINING")
    print("This script bypasses CLI import issues")
    print("")
    
    # You can modify these parameters
    EPOCHS = 2
    BATCH_SIZE = 8
    MIXED_PRECISION = True
    
    print(f"Training parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Mixed precision: {MIXED_PRECISION}")
    print("")
    
    success = run_training_direct(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE, 
        mixed_precision=MIXED_PRECISION
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("You can now use the CLI normally.")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        print("\nTroubleshooting steps:")
        print("1. Ensure all project files are uploaded to Databricks")
        print("2. Run: python databricks_train_debug.py")
        print("3. Check that you have sufficient cluster resources")
        print("4. Install missing dependencies: pip install torch mlflow transformers")