"""
Training Pipeline

This module implements the training pipeline that corresponds to
notebook 02_model_training.py functionality.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
import time
from datetime import datetime
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np

from ..models.multimodal_model import MultimodalLLM
from ..training.trainer import MultimodalTrainer
from ..training.callbacks import (
    EarlyStoppingCallback, CheckpointCallback,
    ProgressCallback, MLflowCallback
)
from ..utils.mlflow_utils import MLflowExperimentManager
from ..utils.visualization import TrainingVisualizer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Pipeline for model training with MLflow tracking.
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: str,
                 experiment_name: str = "multimodal_llm_training",
                 use_wandb: bool = False):
        """
        Initialize the training pipeline.
        
        Args:
            config: Training configuration dictionary
            checkpoint_dir: Directory for model checkpoints
            experiment_name: MLflow experiment name
            use_wandb: Whether to use Weights & Biases
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.mlflow_manager = None
        
        # Setup visualizer
        self.plots_dir = self.checkpoint_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.visualizer = TrainingVisualizer(str(self.plots_dir))
        
    def run(self, resume_from: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            resume_from: Path to checkpoint to resume from
            verbose: Enable verbose logging
            
        Returns:
            Dictionary containing training summary
        """
        logger.info("Starting training pipeline...")
        
        # Initialize MLflow
        self._setup_mlflow()
        
        try:
            # Step 1: Setup data loaders
            logger.info("Step 1: Setting up data loaders...")
            train_loader, val_loader = self._setup_data_loaders()
            
            # Step 2: Initialize model
            logger.info("Step 2: Initializing model...")
            self._initialize_model(resume_from)
            
            # Step 3: Setup trainer
            logger.info("Step 3: Setting up trainer...")
            self._setup_trainer(train_loader, val_loader)
            
            # Step 4: Train model
            logger.info("Step 4: Training model...")
            training_summary = self._train_model(verbose)
            
            # Step 5: Save final model
            logger.info("Step 5: Saving final model...")
            self._save_final_model(training_summary)
            
            # Step 6: Generate training report
            logger.info("Step 6: Generating training report...")
            self._generate_training_report(training_summary)
            
            # End MLflow run
            mlflow.end_run()
            
            logger.info("Training pipeline completed successfully")
            return training_summary
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            mlflow.log_param("training_status", "failed")
            mlflow.log_param("error_message", str(e))
            mlflow.end_run()
            raise
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        logger.info(f"Setting up MLflow experiment: {self.experiment_name}")
        
        self.mlflow_manager = MLflowExperimentManager(
            experiment_name=self.experiment_name,
            tags={
                "project": "multimodal_llm",
                "framework": "pytorch",
                "device": str(self.device)
            }
        )
        
        run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_run = self.mlflow_manager.start_run(
            run_name=run_name,
            tags={
                "training_date": datetime.now().isoformat(),
                "config_version": self.config.get('version', 'unknown')
            }
        )
        
        # Log configuration
        self.mlflow_manager.log_config(self.config)
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders."""
        batch_size = self.config['training']['batch_size']
        
        # For demonstration, create synthetic data loaders
        # In production, use actual DataModule
        from torch.utils.data import TensorDataset
        
        # Synthetic data
        num_train_samples = 1000
        num_val_samples = 200
        ts_seq_len = self.config['time_series']['max_length']
        text_seq_len = self.config['text']['tokenizer']['max_length']
        n_features = 3
        vocab_size = 50257
        
        # Training data
        train_ts = torch.randn(num_train_samples, ts_seq_len, n_features)
        train_text = torch.randint(0, vocab_size, (num_train_samples, text_seq_len))
        train_dataset = TensorDataset(train_ts, train_text)
        
        # Validation data  
        val_ts = torch.randn(num_val_samples, ts_seq_len, n_features)
        val_text = torch.randint(0, vocab_size, (num_val_samples, text_seq_len))
        val_dataset = TensorDataset(val_ts, val_text)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Log dataset info
        dataset_info = {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'batch_size': batch_size,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader)
        }
        
        for key, value in dataset_info.items():
            mlflow.log_param(key, value)
        
        logger.info(f"Data loaders created: {dataset_info}")
        
        return train_loader, val_loader
    
    def _initialize_model(self, resume_from: Optional[str] = None):
        """Initialize or load model."""
        logger.info("Initializing model...")
        
        # Create model
        self.model = MultimodalLLM(self.config)
        self.model.to(self.device)
        
        # Load checkpoint if resuming
        if resume_from:
            logger.info(f"Loading checkpoint from {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint loaded successfully")
        
        # Log model info
        model_stats = self.model.get_memory_usage()
        mlflow.log_param("total_parameters", model_stats['total_parameters'])
        mlflow.log_param("trainable_parameters", model_stats['trainable_parameters'])
        mlflow.log_param("model_memory_mb", model_stats['parameter_memory_mb'])
        
        logger.info(f"Model initialized: {model_stats}")
    
    def _setup_trainer(self, train_loader: DataLoader, val_loader: DataLoader):
        """Setup trainer with callbacks."""
        logger.info("Setting up trainer...")
        
        # Create trainer
        self.trainer = MultimodalTrainer(
            model=self.model,
            config=self.config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=self.device
        )
        
        # Add callbacks
        callbacks = []
        
        # Early stopping
        if self.config['training'].get('early_stopping', {}).get('enabled', True):
            callbacks.append(EarlyStoppingCallback(
                patience=self.config['training']['early_stopping'].get('patience', 5),
                min_delta=self.config['training']['early_stopping'].get('min_delta', 0.001)
            ))
        
        # Checkpointing
        callbacks.append(CheckpointCallback(
            checkpoint_dir=str(self.checkpoint_dir),
            save_frequency=self.config['training'].get('checkpoint_frequency', 1)
        ))
        
        # Progress tracking
        callbacks.append(ProgressCallback())
        
        # MLflow tracking
        callbacks.append(MLflowCallback())
        
        self.trainer.callbacks = callbacks
        
        logger.info(f"Trainer setup with {len(callbacks)} callbacks")
    
    def _train_model(self, verbose: bool = False) -> Dict[str, Any]:
        """Execute model training."""
        logger.info("Starting model training...")
        
        training_start_time = time.time()
        
        # Training loop
        epochs = self.config['training']['epochs']
        best_val_loss = float('inf')
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(epoch, verbose)
            training_history['train_losses'].append(train_loss)
            training_history['train_metrics'].append(train_metrics)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(epoch)
            training_history['val_losses'].append(val_loss)
            training_history['val_metrics'].append(val_metrics)
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value, step=epoch)
            
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            
            # Check early stopping
            if self._check_early_stopping(training_history['val_losses']):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Update learning rate
            if self.trainer.scheduler:
                self.trainer.scheduler.step(val_loss)
                training_history['learning_rates'].append(
                    self.trainer.optimizer.param_groups[0]['lr']
                )
        
        training_time = time.time() - training_start_time
        
        # Create training summary
        training_summary = {
            'epochs_completed': epoch + 1,
            'total_training_time': training_time,
            'best_val_loss': best_val_loss,
            'final_train_loss': training_history['train_losses'][-1],
            'final_val_loss': training_history['val_losses'][-1],
            'final_train_metrics': training_history['train_metrics'][-1] if training_history['train_metrics'] else {},
            'final_val_metrics': training_history['val_metrics'][-1] if training_history['val_metrics'] else {},
            'training_history': training_history
        }
        
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        
        return training_summary
    
    def _train_epoch(self, epoch: int, verbose: bool = False) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'accuracy': [], 'perplexity': []}
        
        progress_bar = tqdm(self.trainer.train_dataloader, desc=f"Training Epoch {epoch + 1}",
                           disable=not verbose)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                # Synthetic data format
                time_series, text_ids = batch
                batch = {
                    'time_series': time_series.to(self.device),
                    'ts_attention_mask': torch.ones_like(time_series[:, :, 0], dtype=torch.bool).to(self.device),
                    'text_input_ids': text_ids.to(self.device),
                    'text_attention_mask': torch.ones_like(text_ids, dtype=torch.bool).to(self.device)
                }
            else:
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            
            # Forward pass
            outputs = self.model(
                time_series=batch['time_series'],
                ts_attention_mask=batch['ts_attention_mask'],
                text_input_ids=batch['text_input_ids'],
                text_attention_mask=batch['text_attention_mask'],
                labels=batch['text_input_ids']
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer.max_grad_norm)
            self.trainer.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            
            # Calculate metrics
            with torch.no_grad():
                perplexity = torch.exp(loss).item()
                epoch_metrics['perplexity'].append(perplexity)
                
                # Simple accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy = (predictions == batch['text_input_ids']).float().mean().item()
                epoch_metrics['accuracy'].append(accuracy)
            
            if verbose:
                progress_bar.set_postfix({'loss': loss.item(), 'ppl': perplexity})
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = []
        val_metrics = {'accuracy': [], 'perplexity': []}
        
        with torch.no_grad():
            for batch in self.trainer.val_dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    time_series, text_ids = batch
                    batch = {
                        'time_series': time_series.to(self.device),
                        'ts_attention_mask': torch.ones_like(time_series[:, :, 0], dtype=torch.bool).to(self.device),
                        'text_input_ids': text_ids.to(self.device),
                        'text_attention_mask': torch.ones_like(text_ids, dtype=torch.bool).to(self.device)
                    }
                else:
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    time_series=batch['time_series'],
                    ts_attention_mask=batch['ts_attention_mask'],
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    labels=batch['text_input_ids']
                )
                
                val_losses.append(outputs.loss.item())
                
                # Calculate metrics
                perplexity = torch.exp(outputs.loss).item()
                val_metrics['perplexity'].append(perplexity)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy = (predictions == batch['text_input_ids']).float().mean().item()
                val_metrics['accuracy'].append(accuracy)
        
        avg_loss = np.mean(val_losses)
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _check_early_stopping(self, val_losses: List[float]) -> bool:
        """Check if early stopping should be triggered."""
        if len(val_losses) < 2:
            return False
        
        patience = self.config['training'].get('early_stopping', {}).get('patience', 5)
        min_delta = self.config['training'].get('early_stopping', {}).get('min_delta', 0.001)
        
        if len(val_losses) > patience:
            recent_losses = val_losses[-patience:]
            best_recent = min(recent_losses)
            
            # Check if no improvement
            if all(loss > best_recent - min_delta for loss in recent_losses[-patience+1:]):
                return True
        
        return False
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self, training_summary: Dict[str, Any]):
        """Save final trained model."""
        final_model_path = self.checkpoint_dir / 'final_model.pt'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_summary': training_summary
        }, final_model_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(
            self.model,
            "model",
            registered_model_name="multimodal_llm"
        )
        
        logger.info(f"Final model saved: {final_model_path}")
    
    def _generate_training_report(self, training_summary: Dict[str, Any]):
        """Generate training report with visualizations."""
        # Create training curves
        history = training_summary['training_history']
        
        if history['train_losses'] and history['val_losses']:
            fig = self.visualizer.plot_training_curves(
                train_losses=history['train_losses'],
                val_losses=history['val_losses'],
                train_metrics=history.get('train_metrics', []),
                val_metrics=history.get('val_metrics', []),
                save_path=str(self.plots_dir / 'training_curves.png')
            )
            
            # Log plot to MLflow
            mlflow.log_artifact(str(self.plots_dir / 'training_curves.png'))
        
        # Save training summary
        summary_path = self.checkpoint_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        mlflow.log_artifact(str(summary_path))
        
        logger.info(f"Training report generated: {summary_path}")
