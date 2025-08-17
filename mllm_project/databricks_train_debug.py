#!/usr/bin/env python3
"""
Databricks Training Debug Script
===============================

This script specifically debugs the training pipeline import issues in Databricks.
Run this script in your Databricks environment to diagnose the exact problem.
"""

import sys
import os
import logging
from pathlib import Path
import traceback

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_training_import():
    """Debug the training pipeline import step by step"""
    print("🔍 DATABRICKS TRAINING IMPORT DEBUG")
    print("=" * 50)
    
    # Check environment
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        print(f"✅ Databricks detected: {os.environ['DATABRICKS_RUNTIME_VERSION']}")
    else:
        print("⚠️ Not running on Databricks")
    
    print(f"sys.path (first 5): {sys.path[:5]}")
    
    # Setup paths
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    
    print(f"Project root: {project_root}")
    print(f"Src path exists: {src_path.exists()}")
    
    if src_path.exists():
        print(f"Src contents: {list(src_path.iterdir())}")
    
    # Add paths
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(project_root))
    
    print("\n🔧 STEP-BY-STEP IMPORT TESTING")
    print("=" * 50)
    
    # Step 1: Test basic torch import
    print("\nStep 1: Testing PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"❌ PyTorch failed: {e}")
        return False
    
    # Step 2: Test MLflow import
    print("\nStep 2: Testing MLflow...")
    try:
        import mlflow
        print(f"✅ MLflow: {mlflow.__version__}")
    except Exception as e:
        print(f"❌ MLflow failed: {e}")
        return False
    
    # Step 3: Test MultimodalLLM import directly
    print("\nStep 3: Testing MultimodalLLM import...")
    try:
        # Try method 1: Direct file import
        import importlib.util
        model_file = src_path / "models" / "multimodal_model.py"
        print(f"Model file exists: {model_file.exists()}")
        
        if model_file.exists():
            spec = importlib.util.spec_from_file_location("multimodal_model", model_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                print("Loading multimodal_model module...")
                spec.loader.exec_module(module)
                MultimodalLLM = module.MultimodalLLM
                print(f"✅ MultimodalLLM loaded: {MultimodalLLM}")
            else:
                print("❌ Could not create module spec")
                return False
        else:
            print("❌ Model file not found")
            return False
    except Exception as e:
        print(f"❌ MultimodalLLM import failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: Test MultimodalTrainer import
    print("\nStep 4: Testing MultimodalTrainer import...")
    try:
        trainer_file = src_path / "training" / "trainer.py"
        print(f"Trainer file exists: {trainer_file.exists()}")
        
        if trainer_file.exists():
            spec = importlib.util.spec_from_file_location("trainer", trainer_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                print("Loading trainer module...")
                spec.loader.exec_module(module)
                MultimodalTrainer = module.MultimodalTrainer
                print(f"✅ MultimodalTrainer loaded: {MultimodalTrainer}")
            else:
                print("❌ Could not create trainer spec")
                return False
        else:
            print("❌ Trainer file not found")
            return False
    except Exception as e:
        print(f"❌ MultimodalTrainer import failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Test TrainingPipeline import
    print("\nStep 5: Testing TrainingPipeline import...")
    try:
        pipeline_file = src_path / "pipelines" / "training_pipeline.py"
        print(f"Training pipeline file exists: {pipeline_file.exists()}")
        
        if pipeline_file.exists():
            spec = importlib.util.spec_from_file_location("training_pipeline", pipeline_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                print("Loading training_pipeline module...")
                spec.loader.exec_module(module)
                TrainingPipeline = module.TrainingPipeline
                print(f"✅ TrainingPipeline loaded: {TrainingPipeline}")
            else:
                print("❌ Could not create training pipeline spec")
                return False
        else:
            print("❌ Training pipeline file not found")
            return False
    except Exception as e:
        print(f"❌ TrainingPipeline import failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 6: Test CLI import mechanism
    print("\nStep 6: Testing CLI import mechanism...")
    try:
        # Simulate CLI import
        sys.path.insert(0, str(src_path))
        
        # Test the exact import method used by CLI
        from src.pipelines.training_pipeline import TrainingPipeline as CLITrainingPipeline
        print(f"✅ CLI TrainingPipeline import: {CLITrainingPipeline}")
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        traceback.print_exc()
        
        # Try alternative
        try:
            print("Trying alternative CLI import...")
            import importlib
            training_module = importlib.import_module('src.pipelines.training_pipeline')
            CLITrainingPipeline = training_module.TrainingPipeline
            print(f"✅ Alternative CLI import: {CLITrainingPipeline}")
        except Exception as e2:
            print(f"❌ Alternative CLI import also failed: {e2}")
            return False
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    return True

def test_training_pipeline_creation():
    """Test actually creating a TrainingPipeline instance"""
    print("\n🏗️ TESTING TRAINING PIPELINE CREATION")
    print("=" * 50)
    
    try:
        # Create a minimal config
        config = {
            'model': {'name': 'test'},
            'training': {'epochs': 1, 'batch_size': 2},
            'data': {'data_dir': './data'}
        }
        
        from src.pipelines.training_pipeline import TrainingPipeline
        
        print("Creating TrainingPipeline instance...")
        pipeline = TrainingPipeline(
            config=config,
            checkpoint_dir='./test_checkpoints',
            experiment_name='test_experiment'
        )
        print(f"✅ TrainingPipeline created: {pipeline}")
        return True
        
    except Exception as e:
        print(f"❌ TrainingPipeline creation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Databricks Training Debug")
    print("This script will help diagnose the 'MultimodalLLM model not available' error")
    print("")
    
    # Test imports
    import_success = debug_training_import()
    
    if import_success:
        # Test pipeline creation
        creation_success = test_training_pipeline_creation()
        
        if creation_success:
            print("\n✅ DIAGNOSIS: Everything working correctly!")
            print("Try running: python cli.py train --epochs 1 --batch-size 2")
        else:
            print("\n⚠️ DIAGNOSIS: Import works but pipeline creation fails")
            print("Check model dependencies and config files")
    else:
        print("\n❌ DIAGNOSIS: Import issues detected")
        print("Check the detailed error messages above")
    
    print("\n📋 NEXT STEPS FOR DATABRICKS:")
    print("1. Save this output and check which step failed")
    print("2. Ensure all files are uploaded to Databricks workspace")
    print("3. Check file permissions and paths")
    print("4. Install missing dependencies if any")