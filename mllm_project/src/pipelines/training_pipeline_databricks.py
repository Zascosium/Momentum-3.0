"""
Training Pipeline - Databricks Compatible Version

This module implements the training pipeline with enhanced Databricks compatibility.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
import time
from datetime import datetime
import numpy as np

# Databricks compatibility setup
current_file_dir = Path(__file__).parent
project_root = current_file_dir.parent.parent
src_dir = project_root / 'src'

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

# Core dependencies with fallbacks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import real components
from models.multimodal_model import MultimodalLLM
from training.trainer import MultimodalTrainer

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Pipeline for model training with MLflow tracking - Databricks Compatible
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: str,
                 experiment_name: str = "multimodal_llm_training",
                 use_wandb: bool = False):
        """
        Initialize the training pipeline with enhanced error handling.
        """
        print("🚀 Initializing Training Pipeline (Databricks Compatible)")
        
        # Check critical dependencies
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training. Install with: pip install torch")
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for training. Install with: pip install mlflow")
        
        print("✅ Core dependencies available")
        
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.mlflow_manager = None
        
        print(f"✅ Training pipeline initialized")
        print(f"   Device: {self.device}")
        print(f"   Checkpoint dir: {self.checkpoint_dir}")
        print(f"   Experiment: {experiment_name}")
    
    def run(self, resume_from: Optional[str] = None, verbose: bool = False):
        """
        Run the complete training pipeline.
        """
        try:
            print("🚀 Starting training pipeline...")
            
            # Setup MLflow
            print("📊 Setting up MLflow...")
            if mlflow:
                # Check if running on Databricks and format experiment name accordingly
                import os
                if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
                    # Use Databricks workspace path format
                    experiment_name = f"/Users/tarek.buerner@porsche.de/{self.experiment_name}"
                else:
                    # Use regular name for local environments
                    experiment_name = self.experiment_name
                
                mlflow.set_experiment(experiment_name)
                mlflow.start_run()
                print("✅ MLflow experiment started")
            
            # Initialize model
            print("🏗️ Initializing model...")
            self.model = MultimodalLLM(self.config)
            self.model = self.model.to(self.device)
            print(f"✅ Model initialized on {self.device}")
            
            # Initialize trainer  
            print("🏃 Initializing trainer...")
            self.trainer = MultimodalTrainer(
                model=self.model,
                config=self.config
            )
            print("✅ Trainer initialized")
            
            # Run training
            print("🎯 Starting training...")
            training_summary = self.trainer.train()
            print("✅ Training completed!")
            
            # Save results
            if mlflow and mlflow.active_run():
                mlflow.log_metrics(training_summary.get('final_metrics', {}))
                mlflow.end_run()
            
            return training_summary
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            if mlflow and mlflow.active_run():
                mlflow.end_run(status='FAILED')
            raise
