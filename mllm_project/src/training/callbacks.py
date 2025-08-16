"""
Training callbacks for monitoring and controlling training process.
Implements early stopping, checkpointing, and logging callbacks.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import torch
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_epoch_start(self, trainer, epoch: int):
        """Called at the start of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Called at the end of each epoch.
        
        Returns:
            True if training should be stopped, False otherwise
        """
        pass
    
    def on_batch_start(self, trainer, batch_idx: int):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Called at the end of each batch."""
        pass


class CallbackManager:
    """Manages multiple callbacks during training."""
    
    def __init__(self):
        """Initialize callback manager."""
        self.callbacks = []
    
    def add_callback(self, callback: Callback):
        """Add a callback to the manager."""
        self.callbacks.append(callback)
        logger.info(f"Added callback: {callback.__class__.__name__}")
    
    def remove_callback(self, callback: Callback):
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Removed callback: {callback.__class__.__name__}")
    
    def on_epoch_start(self, trainer, epoch: int):
        """Call on_epoch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Call on_epoch_end for all callbacks.
        
        Returns:
            True if any callback requests training to stop
        """
        should_stop = False
        for callback in self.callbacks:
            if callback.on_epoch_end(trainer, epoch, metrics):
                should_stop = True
        return should_stop
    
    def on_batch_start(self, trainer, batch_idx: int):
        """Call on_batch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_start(trainer, batch_idx)
    
    def on_batch_end(self, trainer, batch_idx: int, loss: float):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss)


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback to prevent overfitting.
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        metric: str = 'val_loss',
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            verbose: Whether to log early stopping events
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = None
        
        # Determine comparison function
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'")
    
    def on_epoch_start(self, trainer, epoch: int):
        """Called at epoch start."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should be stopped based on metric improvement.
        
        Returns:
            True if training should be stopped
        """
        if self.metric not in metrics:
            if self.verbose:
                logger.warning(f"Metric '{self.metric}' not found in metrics")
            return False
        
        current_metric = metrics[self.metric]
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return False
        
        if self.monitor_op(current_metric - self.min_delta, self.best_metric):
            self.best_metric = current_metric
            self.wait = 0
            if self.verbose:
                logger.info(f"Epoch {epoch}: {self.metric} improved to {current_metric:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                logger.info(f"Epoch {epoch}: {self.metric} did not improve from {self.best_metric:.6f}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch} "
                               f"(patience: {self.patience}, best {self.metric}: {self.best_metric:.6f})")
                return True
        
        return False


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints during training.
    Can save best models based on a metric or save at regular intervals.
    """
    
    def __init__(
        self,
        save_dir: str = '/dbfs/mllm/checkpoints',
        save_top_k: int = 3,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_last: bool = True,
        filename_template: str = 'epoch_{epoch:02d}-{monitor:.2f}',
        verbose: bool = True
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints
            save_top_k: Number of best checkpoints to keep (0 = save all)
            monitor: Metric to monitor for best checkpoints
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            save_last: Whether to save the last checkpoint
            filename_template: Template for checkpoint filenames
            verbose: Whether to log checkpoint events
        """
        self.save_dir = Path(save_dir)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.filename_template = filename_template
        self.verbose = verbose
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best checkpoints
        self.best_checkpoints = []  # List of (metric_value, filepath) tuples
        self.best_metric = None
        
        # Determine comparison function
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode {mode} is unknown, please use 'min' or 'max'")
    
    def on_epoch_start(self, trainer, epoch: int):
        """Called at epoch start."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Save checkpoint if conditions are met.
        
        Returns:
            False (never stops training)
        """
        # Always save last checkpoint if requested
        if self.save_last:
            last_path = self.save_dir / 'last.pt'
            self._save_checkpoint(trainer, last_path, epoch)
        
        # Save best checkpoints based on monitored metric
        if self.monitor in metrics:
            current_metric = metrics[self.monitor]
            
            # Check if this is a new best
            if self.best_metric is None or self.monitor_op(current_metric, self.best_metric):
                self.best_metric = current_metric
                
                # Create filename
                filename = self._format_filename(epoch, metrics)
                filepath = self.save_dir / f"{filename}.pt"
                
                # Save checkpoint
                self._save_checkpoint(trainer, filepath, epoch)
                
                # Add to best checkpoints list
                self.best_checkpoints.append((current_metric, filepath))
                
                # Sort checkpoints by metric value
                if self.mode == 'min':
                    self.best_checkpoints.sort(key=lambda x: x[0])
                else:
                    self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
                
                # Remove excess checkpoints
                if self.save_top_k > 0 and len(self.best_checkpoints) > self.save_top_k:
                    _, old_filepath = self.best_checkpoints.pop()
                    if old_filepath.exists():
                        old_filepath.unlink()
                        if self.verbose:
                            logger.info(f"Removed old checkpoint: {old_filepath}")
                
                if self.verbose:
                    logger.info(f"Saved new best checkpoint: {filepath} "
                               f"({self.monitor}={current_metric:.6f})")
        
        return False
    
    def _format_filename(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Format checkpoint filename using template."""
        format_dict = {'epoch': epoch}
        
        # Add metrics to format dict
        for key, value in metrics.items():
            # Replace dots and spaces in metric names for valid filenames
            clean_key = key.replace('.', '_').replace(' ', '_')
            format_dict[clean_key] = value
        
        # Replace monitor placeholder
        if self.monitor in metrics:
            format_dict['monitor'] = metrics[self.monitor]
        
        try:
            filename = self.filename_template.format(**format_dict)
        except KeyError as e:
            logger.warning(f"Error formatting filename: {e}. Using default format.")
            filename = f"epoch_{epoch:02d}"
        
        return filename
    
    def _save_checkpoint(self, trainer, filepath: Path, epoch: int):
        """Save model checkpoint."""
        # Get model state dict
        if hasattr(trainer.model, 'module'):
            model_state_dict = trainer.model.module.state_dict()
        else:
            model_state_dict = trainer.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'scaler_state_dict': trainer.scaler.state_dict() if hasattr(trainer, 'scaler') else None,
            'config': trainer.config,
            'best_metric': getattr(self, 'best_metric', None)
        }
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            logger.info(f"Checkpoint saved: {filepath}")


class LearningRateMonitorCallback(Callback):
    """
    Callback for monitoring and logging learning rate during training.
    """
    
    def __init__(self, log_momentum: bool = False):
        """
        Initialize learning rate monitor.
        
        Args:
            log_momentum: Whether to also log momentum values
        """
        self.log_momentum = log_momentum
        self.lr_history = []
        self.momentum_history = []
    
    def on_epoch_start(self, trainer, epoch: int):
        """Log learning rate at epoch start."""
        current_lr = trainer.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        logger.info(f"Epoch {epoch}: Learning rate = {current_lr:.2e}")
        
        # Log momentum if requested
        if self.log_momentum and 'momentum' in trainer.optimizer.param_groups[0]:
            current_momentum = trainer.optimizer.param_groups[0]['momentum']
            self.momentum_history.append(current_momentum)
            logger.info(f"Epoch {epoch}: Momentum = {current_momentum:.4f}")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Called at epoch end (no action needed)."""
        return False


class ProgressCallback(Callback):
    """
    Callback for logging training progress and statistics.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize progress callback.
        
        Args:
            log_every_n_epochs: Log detailed progress every N epochs
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.start_time = None
        self.epoch_times = []
    
    def on_epoch_start(self, trainer, epoch: int):
        """Record epoch start time."""
        self.start_time = time.time()
        
        if epoch % self.log_every_n_epochs == 0:
            logger.info(f"Starting epoch {epoch}/{trainer.training_config.get('epochs', 'unknown')}")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Log epoch completion and statistics."""
        if self.start_time is not None:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            
            if epoch % self.log_every_n_epochs == 0:
                # Calculate statistics
                avg_epoch_time = np.mean(self.epoch_times)
                total_epochs = trainer.training_config.get('epochs', epoch + 1)
                remaining_epochs = total_epochs - (epoch + 1)
                estimated_remaining_time = avg_epoch_time * remaining_epochs
                
                # Format time
                def format_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    seconds = int(seconds % 60)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # Log progress
                logger.info(f"Epoch {epoch} completed in {format_time(epoch_time)}")
                logger.info(f"Average epoch time: {format_time(avg_epoch_time)}")
                logger.info(f"Estimated remaining time: {format_time(estimated_remaining_time)}")
                
                # Log key metrics
                key_metrics = ['loss', 'val_loss', 'accuracy', 'perplexity']
                for metric in key_metrics:
                    if metric in metrics:
                        logger.info(f"{metric}: {metrics[metric]:.6f}")
        
        return False


class GradientNormCallback(Callback):
    """
    Callback for monitoring gradient norms during training.
    Helps detect gradient explosion or vanishing problems.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize gradient norm callback.
        
        Args:
            log_every_n_epochs: Log gradient norms every N epochs
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.grad_norms = []
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Compute and log gradient norms."""
        if epoch % self.log_every_n_epochs == 0:
            total_norm = 0.0
            param_count = 0
            
            model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.grad_norms.append(total_norm)
                
                logger.info(f"Epoch {epoch}: Gradient norm = {total_norm:.6f}")
                
                # Check for gradient explosion
                if total_norm > 100:
                    logger.warning(f"Large gradient norm detected: {total_norm:.2f}")
                elif total_norm < 1e-6:
                    logger.warning(f"Very small gradient norm detected: {total_norm:.2e}")
        
        return False


class MLflowCallback(Callback):
    """
    Callback for logging metrics and artifacts to MLflow.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize MLflow callback.
        
        Args:
            log_every_n_epochs: Log to MLflow every N epochs
        """
        self.log_every_n_epochs = log_every_n_epochs
        
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow_available = True
        except ImportError:
            logger.warning("MLflow not available, metrics will not be logged")
            self.mlflow_available = False
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Log metrics to MLflow."""
        if not self.mlflow_available or epoch % self.log_every_n_epochs != 0:
            return False
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            self.mlflow.log_metric(metric_name, metric_value, step=epoch)
        
        # Log learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        self.mlflow.log_metric('learning_rate', current_lr, step=epoch)
        
        return False


class MemoryMonitorCallback(Callback):
    """
    Callback for monitoring GPU memory usage during training.
    """
    
    def __init__(self, log_every_n_epochs: int = 5):
        """
        Initialize memory monitor callback.
        
        Args:
            log_every_n_epochs: Log memory usage every N epochs
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.memory_usage = []
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Log memory usage."""
        if epoch % self.log_every_n_epochs == 0 and torch.cuda.is_available():
            # Get GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            self.memory_usage.append({
                'epoch': epoch,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            })
            
            logger.info(f"Epoch {epoch}: GPU Memory - "
                       f"Allocated: {allocated:.2f}GB, "
                       f"Reserved: {reserved:.2f}GB, "
                       f"Max Allocated: {max_allocated:.2f}GB")
            
            # Check for potential memory issues
            if allocated > 30:  # More than 30GB
                logger.warning(f"High GPU memory usage: {allocated:.2f}GB")
        
        return False


# Factory function for creating common callback combinations
def create_default_callbacks(config: Dict[str, Any]) -> List[Callback]:
    """
    Create a default set of callbacks based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Early stopping
    early_stopping_config = config.get('training', {}).get('early_stopping', {})
    if early_stopping_config.get('patience', 0) > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=early_stopping_config.get('patience', 5),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            metric=early_stopping_config.get('metric', 'val_loss'),
            mode=early_stopping_config.get('mode', 'min')
        ))
    
    # Checkpointing
    checkpointing_config = config.get('checkpointing', {})
    callbacks.append(CheckpointCallback(
        save_dir=checkpointing_config.get('dirpath', '/dbfs/mllm/checkpoints'),
        save_top_k=checkpointing_config.get('save_top_k', 3),
        monitor=checkpointing_config.get('monitor', 'val_loss'),
        mode=checkpointing_config.get('mode', 'min'),
        save_last=checkpointing_config.get('save_last', True)
    ))
    
    # Learning rate monitoring
    callbacks.append(LearningRateMonitorCallback())
    
    # Progress logging
    callbacks.append(ProgressCallback(log_every_n_epochs=1))
    
    # MLflow logging
    callbacks.append(MLflowCallback(log_every_n_epochs=1))
    
    # Memory monitoring
    callbacks.append(MemoryMonitorCallback(log_every_n_epochs=5))
    
    return callbacks


# Example usage
if __name__ == "__main__":
    # Test callback system
    
    # Mock trainer class
    class MockTrainer:
        def __init__(self):
            self.model = torch.nn.Linear(10, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.scheduler = None
            self.global_step = 0
            self.config = {'training': {'epochs': 10}}
            self.training_config = {'epochs': 10}
    
    # Create callbacks
    callbacks = [
        EarlyStoppingCallback(patience=3, metric='val_loss'),
        CheckpointCallback(save_dir='/tmp/test_checkpoints'),
        ProgressCallback(),
        LearningRateMonitorCallback()
    ]
    
    # Create callback manager
    manager = CallbackManager()
    for callback in callbacks:
        manager.add_callback(callback)
    
    # Simulate training
    trainer = MockTrainer()
    
    for epoch in range(5):
        manager.on_epoch_start(trainer, epoch)
        
        # Mock metrics
        metrics = {
            'loss': 2.0 - epoch * 0.3,
            'val_loss': 1.8 - epoch * 0.2,
            'accuracy': 0.5 + epoch * 0.1
        }
        
        # Check if training should stop
        should_stop = manager.on_epoch_end(trainer, epoch, metrics)
        
        print(f"Epoch {epoch}: {metrics}")
        
        if should_stop:
            print("Training stopped by callback")
            break
    
    print("Callbacks implementation completed successfully!")