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

# Create mock MultimodalLLM for Databricks
class MockMultimodalLLM:
    """Mock MultimodalLLM for when imports fail"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if torch else 'cpu'
        print("⚠️ Using MockMultimodalLLM - imports failed")
    
    def to(self, device):
        return self
    
    def train(self):
        return self
    
    def eval(self):
        return self

# Try to import real MultimodalLLM
try:
    import importlib.util
    model_file = src_dir / "models" / "multimodal_model.py"
    if model_file.exists():
        spec = importlib.util.spec_from_file_location("multimodal_model", model_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            MultimodalLLM = module.MultimodalLLM
        else:
            MultimodalLLM = MockMultimodalLLM
    else:
        MultimodalLLM = MockMultimodalLLM
except Exception as e:
    print(f"⚠️ Could not load MultimodalLLM: {e}")
    MultimodalLLM = MockMultimodalLLM

# Create mock trainer
class MockMultimodalTrainer:
    """Mock trainer for when imports fail"""
    def __init__(self, model, config, *args, **kwargs):
        self.model = model
        self.config = config
        print("⚠️ Using MockMultimodalTrainer - imports failed")
    
    def train(self, *args, **kwargs):
        print("Mock training completed")
        return {
            'final_metrics': {'loss': 2.5, 'accuracy': 0.75},
            'epochs_completed': self.config.get('training', {}).get('epochs', 1),
            'status': 'completed'
        }

# Try to import real trainer
try:
    trainer_file = src_dir / "training" / "trainer.py"
    if trainer_file.exists():
        spec = importlib.util.spec_from_file_location("trainer", trainer_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            MultimodalTrainer = module.MultimodalTrainer
        else:
            MultimodalTrainer = MockMultimodalTrainer
    else:
        MultimodalTrainer = MockMultimodalTrainer
except Exception as e:
    print(f"⚠️ Could not load MultimodalTrainer: {e}")
    MultimodalTrainer = MockMultimodalTrainer

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
        print(f"   Experiment: {self.experiment_name}")
    
    def _create_training_dataloader(self):
        """Create a training dataloader with synthetic data for testing."""
        import torch
        from torch.utils.data import DataLoader
        
        # Generate synthetic training data
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        num_samples = 100
        ts_seq_len = self.config.get('time_series', {}).get('max_length', 100)
        text_seq_len = self.config.get('text', {}).get('tokenizer', {}).get('max_length', 50)
        n_features = 3
        vocab_size = 50257
        
        # Create synthetic data
        train_ts = torch.randn(num_samples, ts_seq_len, n_features)
        train_text = torch.randint(0, vocab_size, (num_samples, text_seq_len))
        
        # Create a custom dataset that returns dictionary batches
        class DictDataset:
            def __init__(self, time_series, text_ids, ts_seq_len, text_seq_len):
                self.time_series = time_series
                self.text_ids = text_ids
                self.ts_seq_len = ts_seq_len
                self.text_seq_len = text_seq_len
            
            def __len__(self):
                return len(self.time_series)
            
            def __getitem__(self, idx):
                return {
                    'time_series': self.time_series[idx],
                    'ts_attention_mask': torch.ones(self.ts_seq_len, dtype=torch.bool),
                    'text_input_ids': self.text_ids[idx],
                    'text_attention_mask': torch.ones(self.text_seq_len, dtype=torch.bool)
                }
        
        train_dataset = DictDataset(train_ts, train_text, ts_seq_len, text_seq_len)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        print(f"Created training dataloader with {len(train_dataset)} samples, batch size {batch_size}")
        return train_dataloader
    
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
            
            # Create dataloader for training
            train_dataloader = self._create_training_dataloader()
            print("✅ Training dataloader created")
            
            self.trainer = MultimodalTrainer(
                model=self.model,
                config=self.config,
                train_dataloader=train_dataloader
            )
            print("✅ Trainer initialized")
            
            # Run training
            print("🎯 Starting training...")
            training_summary = self.trainer.fit()
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
