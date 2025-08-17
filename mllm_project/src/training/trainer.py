"""
Training infrastructure for multimodal LLM.
Handles distributed training, optimization, and monitoring on Databricks.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Databricks compatibility
if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
    current_path = Path(__file__).parent
    while current_path != current_path.parent:
        if (current_path / 'src').exists():
            sys.path.insert(0, str(current_path / 'src'))
            break
        current_path = current_path.parent

# Import project modules with fallbacks
try:
    from models.multimodal_model import MultimodalLLM
except (ImportError, ValueError):
    MultimodalLLM = None

try:
    from data.data_loader import MultimodalDataModule
except (ImportError, ValueError):
    try:
        from ..data.data_loader import MultimodalDataModule
    except (ImportError, ValueError):
        MultimodalDataModule = None

try:
    from training.losses import MultimodalLossFunction
except (ImportError, ValueError):
    try:
        from .losses import MultimodalLossFunction
    except (ImportError, ValueError):
        MultimodalLossFunction = None

try:
    from training.metrics import MetricsTracker
except (ImportError, ValueError):
    try:
        from .metrics import MetricsTracker
    except (ImportError, ValueError):
        class MetricsTracker:
            def __init__(self, *args, **kwargs): pass
            def update(self, *args, **kwargs): pass
            def compute(self): return {}

try:
    from training.callbacks import CallbackManager, EarlyStoppingCallback, CheckpointCallback
except (ImportError, ValueError):
    try:
        from .callbacks import CallbackManager, EarlyStoppingCallback, CheckpointCallback
    except (ImportError, ValueError):
        class CallbackManager:
            def __init__(self, *args, **kwargs): pass
        class EarlyStoppingCallback:
            def __init__(self, *args, **kwargs): pass
        class CheckpointCallback:
            def __init__(self, *args, **kwargs): pass

try:
    from utils.config_loader import load_config_for_training
except (ImportError, ValueError):
    try:
        from ..utils.config_loader import load_config_for_training
    except (ImportError, ValueError):
        def load_config_for_training(config_dir):
            import yaml
            config_path = Path(config_dir) / 'model_config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}

logger = logging.getLogger(__name__)

class MultimodalTrainer:
    """
    Comprehensive trainer for multimodal LLM with Databricks optimization.
    Supports distributed training, mixed precision, and MLflow integration.
    """
    
    def __init__(
        self,
        model: MultimodalLLM,
        config: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Multimodal LLM model
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            device: Training device
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.distributed_config = config.get('distributed', {})
        self.mixed_precision_config = config.get('mixed_precision', {})
        self.checkpointing_config = config.get('checkpointing', {})
        
        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Distributed training setup
        self.is_distributed = self._setup_distributed()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_main_process = self.local_rank == 0
        
        # Model setup
        self.model = model.to(self.device)
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.distributed_config.get('find_unused_parameters', False),
                gradient_as_bucket_view=self.distributed_config.get('gradient_as_bucket_view', True),
                broadcast_buffers=self.distributed_config.get('broadcast_buffers', False)
            )
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Optimization
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        
        # Mixed precision
        self.use_amp = self.mixed_precision_config.get('enabled', True)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.mixed_precision_config.get('fp16', False)
        )
        
        # Loss function and metrics
        self.loss_fn = MultimodalLossFunction(config.get('loss', {}))
        self.metrics_tracker = MetricsTracker(config.get('metrics', {}))
        
        # Callbacks
        self.callback_manager = self._setup_callbacks()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.training_logs = []
        
        # Gradient accumulation
        self.gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
        
        logger.info(f"Trainer initialized - Device: {self.device}, Distributed: {self.is_distributed}")
    
    def _setup_distributed(self) -> bool:
        """Setup distributed training if available."""
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            if world_size > 1:
                if not dist.is_initialized():
                    backend = self.distributed_config.get('backend', 'nccl')
                    dist.init_process_group(backend=backend)
                return True
        return False
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adamw').lower()
        
        # Get model parameters
        if hasattr(self.model, 'module'):
            model_params = self.model.module.parameters()
        else:
            model_params = self.model.parameters()
        
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                model_params,
                lr=optimizer_config.get('learning_rate', 5e-5),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                model_params,
                lr=optimizer_config.get('learning_rate', 5e-5),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                model_params,
                lr=optimizer_config.get('learning_rate', 5e-5),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine_with_warmup').lower()
        
        if scheduler_name == 'none':
            return None
        
        total_steps = len(self.train_dataloader) * self.training_config.get('epochs', 10)
        warmup_steps = scheduler_config.get('warmup_steps', 1000)
        
        if scheduler_name == 'cosine_with_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from transformers import get_cosine_schedule_with_warmup
            
            try:
                scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    num_cycles=scheduler_config.get('num_cycles', 0.5)
                )
            except ImportError:
                # Fallback to PyTorch CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps,
                    last_epoch=scheduler_config.get('last_epoch', -1)
                )
        
        elif scheduler_name == 'linear_with_warmup':
            from transformers import get_linear_schedule_with_warmup
            
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        
        elif scheduler_name == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.get('gamma', 0.95)
            )
        
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using none")
            return None
        
        return scheduler
    
    def _setup_callbacks(self) -> CallbackManager:
        """Setup training callbacks."""
        callback_manager = CallbackManager()
        
        # Early stopping
        early_stopping_config = self.training_config.get('early_stopping', {})
        if early_stopping_config.get('patience', 0) > 0:
            early_stopping = EarlyStoppingCallback(
                patience=early_stopping_config.get('patience', 5),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                metric=early_stopping_config.get('metric', 'val_loss'),
                mode=early_stopping_config.get('mode', 'min')
            )
            callback_manager.add_callback(early_stopping)
        
        # Checkpointing
        if self.is_main_process:
            checkpoint_callback = CheckpointCallback(
                save_dir=self.checkpointing_config.get('dirpath', '/dbfs/mllm/checkpoints'),
                save_top_k=self.checkpointing_config.get('save_top_k', 3),
                monitor=self.checkpointing_config.get('monitor', 'val_loss'),
                mode=self.checkpointing_config.get('mode', 'min'),
                save_last=self.checkpointing_config.get('save_last', True),
                filename_template=self.checkpointing_config.get('filename', 'epoch_{epoch:02d}-val_loss_{val_loss:.2f}')
            )
            callback_manager.add_callback(checkpoint_callback)
        
        return callback_manager
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        if self.is_main_process:
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        else:
            pbar = self.train_dataloader
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with gradient accumulation
            loss = self._training_step(batch, batch_idx)
            
            # Update metrics
            total_loss += loss
            num_batches += 1
            
            # Log progress
            if self.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Validation during training
            if (self.global_step + 1) % self.training_config.get('eval_steps', 500) == 0:
                if self.val_dataloader is not None:
                    val_metrics = self.validate()
                    self.model.train()  # Switch back to training mode
                    
                    # Log validation metrics
                    if self.is_main_process:
                        self._log_metrics(val_metrics, step=self.global_step, prefix='val')
            
            # Save checkpoint during training
            if (self.global_step + 1) % self.training_config.get('save_steps', 1000) == 0:
                if self.is_main_process:
                    self._save_checkpoint(f'step_{self.global_step}')
            
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_metrics['train_loss'] = total_loss / num_batches
        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return epoch_metrics
    
    def _training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        """Execute a single training step."""
        
        # Extract inputs
        time_series = batch.get('time_series')
        ts_attention_mask = batch.get('ts_attention_mask')
        text_input_ids = batch.get('text_input_ids')
        text_attention_mask = batch.get('text_attention_mask')
        
        # Create labels for language modeling (shift input_ids)
        labels = text_input_ids.clone()
        
        # Mixed precision forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast(enabled=self.mixed_precision_config.get('fp16', False)):
                outputs = self.model(
                    time_series=time_series,
                    ts_attention_mask=ts_attention_mask,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.gradient_accumulation_steps
        else:
            outputs = self.model(
                time_series=time_series,
                ts_attention_mask=ts_attention_mask,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                labels=labels
            )
            loss = outputs.loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_metrics = {}
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", disable=not self.is_main_process):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract inputs
                time_series = batch.get('time_series')
                ts_attention_mask = batch.get('ts_attention_mask')
                text_input_ids = batch.get('text_input_ids')
                text_attention_mask = batch.get('text_attention_mask')
                labels = text_input_ids.clone()
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision_config.get('fp16', False)):
                        outputs = self.model(
                            time_series=time_series,
                            ts_attention_mask=ts_attention_mask,
                            text_input_ids=text_input_ids,
                            text_attention_mask=text_attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        time_series=time_series,
                        ts_attention_mask=ts_attention_mask,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask,
                        labels=labels
                    )
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
                # Update metrics tracker
                self.metrics_tracker.update(outputs, batch, split='val')
        
        # Calculate validation metrics
        val_metrics['val_loss'] = total_loss / num_batches
        
        # Add computed metrics
        computed_metrics = self.metrics_tracker.compute(split='val')
        val_metrics.update(computed_metrics)
        
        return val_metrics
    
    def test(self) -> Dict[str, float]:
        """Run testing."""
        if self.test_dataloader is None:
            return {}
        
        self.model.eval()
        test_metrics = {}
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing", disable=not self.is_main_process):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract inputs
                time_series = batch.get('time_series')
                ts_attention_mask = batch.get('ts_attention_mask')
                text_input_ids = batch.get('text_input_ids')
                text_attention_mask = batch.get('text_attention_mask')
                labels = text_input_ids.clone()
                
                # Forward pass
                outputs = self.model(
                    time_series=time_series,
                    ts_attention_mask=ts_attention_mask,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
                # Update metrics tracker
                self.metrics_tracker.update(outputs, batch, split='test')
        
        # Calculate test metrics
        test_metrics['test_loss'] = total_loss / num_batches
        
        # Add computed metrics
        computed_metrics = self.metrics_tracker.compute(split='test')
        test_metrics.update(computed_metrics)
        
        return test_metrics
    
    def fit(self) -> Dict[str, Any]:
        """Main training loop."""
        start_time = time.time()
        
        # Initialize training
        if self.is_main_process:
            logger.info("Starting training...")
            self._log_training_info()
        
        # Training loop
        for epoch in range(self.training_config.get('epochs', 10)):
            self.current_epoch = epoch
            
            # Callbacks on epoch start
            self.callback_manager.on_epoch_start(self, epoch)
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_dataloader is not None:
                val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            if self.is_main_process:
                self._log_metrics(epoch_metrics, step=epoch, prefix='epoch')
            
            # Callbacks on epoch end
            early_stop = self.callback_manager.on_epoch_end(self, epoch, epoch_metrics)
            
            # Check early stopping
            if early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Final testing
        test_metrics = {}
        if self.test_dataloader is not None:
            test_metrics = self.test()
            if self.is_main_process:
                self._log_metrics(test_metrics, step=self.current_epoch, prefix='final')
        
        # Training summary
        total_time = time.time() - start_time
        training_summary = {
            'total_training_time': total_time,
            'epochs_completed': self.current_epoch + 1,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'final_test_metrics': test_metrics
        }
        
        if self.is_main_process:
            logger.info(f"Training completed in {total_time:.2f} seconds")
            self._save_final_model()
        
        return training_summary
    
    def _log_training_info(self) -> None:
        """Log training configuration and model info."""
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        
        memory_stats = model.get_memory_usage()
        
        training_info = {
            'model_parameters': memory_stats['total_parameters'],
            'trainable_parameters': memory_stats['trainable_parameters'],
            'parameter_memory_mb': memory_stats['parameter_memory_mb'],
            'device': str(self.device),
            'distributed': self.is_distributed,
            'world_size': self.world_size,
            'mixed_precision': self.use_amp,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'train_batch_size': self.train_dataloader.batch_size,
            'train_dataset_size': len(self.train_dataloader.dataset),
            'val_dataset_size': len(self.val_dataloader.dataset) if self.val_dataloader else 0
        }
        
        logger.info("Training Configuration:")
        for key, value in training_info.items():
            logger.info(f"  {key}: {value}")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = '') -> None:
        """Log metrics to MLflow."""
        if not self.is_main_process:
            return
        
        for metric_name, metric_value in metrics.items():
            full_metric_name = f"{prefix}/{metric_name}" if prefix else metric_name
            mlflow.log_metric(full_metric_name, metric_value, step=step)
    
    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save training checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint_dir = Path(self.checkpointing_config.get('dirpath', '/dbfs/mllm/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
        
        # Get model state dict
        if hasattr(self.model, 'module'):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'best_metric': self.best_metric
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self) -> None:
        """Save final trained model."""
        if not self.is_main_process:
            return
        
        save_dir = Path(self.checkpointing_config.get('dirpath', '/dbfs/mllm/checkpoints')) / 'final_model'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        
        model.save_pretrained(str(save_dir))
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "final_model")
        
        logger.info(f"Final model saved: {save_dir}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.use_amp and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")


# Factory function for creating trainer
def create_trainer(
    config_path: str,
    model_class: type = MultimodalLLM,
    data_module_class: type = MultimodalDataModule,
    resume_from_checkpoint: Optional[str] = None
) -> MultimodalTrainer:
    """
    Factory function to create a trainer with all components.
    
    Args:
        config_path: Path to configuration file
        model_class: Model class to instantiate
        data_module_class: Data module class to instantiate
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Configured trainer instance
    """
    # Load configuration
    config = load_config_for_training(config_path)
    
    # Create model
    model = model_class(config)
    
    # Create data module
    data_module = data_module_class(config)
    data_module.setup('fit')
    
    # Create data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader() if hasattr(data_module, 'test_dataloader') else None
    
    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader
    )
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint:
        trainer.load_checkpoint(resume_from_checkpoint)
    
    return trainer


# Example usage
if __name__ == "__main__":
    # Example training script
    config_path = "../config"
    
    # Create trainer
    trainer = create_trainer(config_path)
    
    # Start training
    training_summary = trainer.fit()
    
    print("Training completed!")
    print(f"Summary: {training_summary}")