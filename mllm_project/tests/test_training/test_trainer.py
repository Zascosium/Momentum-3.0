"""
Tests for the training module.

This module contains tests for the training infrastructure including
the trainer, metrics, losses, and callbacks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from training.trainer import MultimodalTrainer
from training.metrics import MetricsTracker, compute_bleu_score, compute_rouge_score
from training.losses import MultimodalLoss, ContrastiveLoss, AlignmentLoss
from training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


class TestMultimodalTrainer:
    """Test suite for MultimodalTrainer."""
    
    def test_trainer_initialization(self, test_config, mock_model, device):
        """Test trainer initialization."""
        trainer = MultimodalTrainer(
            model=mock_model,
            config=test_config,
            device=device
        )
        
        assert trainer.model == mock_model
        assert trainer.device == device
        assert trainer.config == test_config
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'scheduler')
        assert hasattr(trainer, 'criterion')
    
    def test_trainer_setup_optimizer(self, test_config, mock_model, device):
        """Test optimizer setup."""
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Test different optimizer types
        test_config['training']['optimizer'] = 'adamw'
        trainer._setup_optimizer()
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        
        test_config['training']['optimizer'] = 'adam'
        trainer._setup_optimizer()
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        
        test_config['training']['optimizer'] = 'sgd'
        trainer._setup_optimizer()
        assert isinstance(trainer.optimizer, torch.optim.SGD)
    
    def test_trainer_setup_scheduler(self, test_config, mock_model, device):
        """Test scheduler setup."""
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Test different scheduler types
        test_config['training']['scheduler'] = 'cosine'
        trainer._setup_scheduler(1000)  # total_steps
        assert trainer.scheduler is not None
        
        test_config['training']['scheduler'] = 'linear'
        trainer._setup_scheduler(1000)
        assert trainer.scheduler is not None
        
        test_config['training']['scheduler'] = 'constant'
        trainer._setup_scheduler(1000)
        assert trainer.scheduler is not None
    
    def test_training_step(self, test_config, mock_model, sample_batch, device):
        """Test single training step."""
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.5, requires_grad=True)
        mock_output.logits = torch.randn(8, 128, 50257)
        mock_model.return_value = mock_output
        
        # Move batch to device
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                sample_batch[key] = value.to(device)
        
        loss = trainer.training_step(sample_batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_validation_step(self, test_config, mock_model, sample_batch, device):
        """Test single validation step."""
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(1.8)
        mock_output.logits = torch.randn(8, 128, 50257)
        mock_model.return_value = mock_output
        
        # Move batch to device
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                sample_batch[key] = value.to(device)
        
        with torch.no_grad():
            metrics = trainer.validation_step(sample_batch)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    @patch('torch.utils.data.DataLoader')
    def test_train_epoch(self, mock_dataloader, test_config, mock_model, sample_batch, device):
        """Test training for one epoch."""
        # Setup mock dataloader
        mock_dataloader.return_value = [sample_batch] * 5  # 5 batches
        
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0, requires_grad=True)
        mock_output.logits = torch.randn(8, 128, 50257)
        mock_model.return_value = mock_output
        
        train_loader = mock_dataloader()
        epoch_metrics = trainer.train_epoch(train_loader, epoch=1)
        
        assert isinstance(epoch_metrics, dict)
        assert 'loss' in epoch_metrics
        assert epoch_metrics['loss'] > 0
    
    @patch('torch.utils.data.DataLoader')
    def test_validate_epoch(self, mock_dataloader, test_config, mock_model, sample_batch, device):
        """Test validation for one epoch."""
        # Setup mock dataloader
        mock_dataloader.return_value = [sample_batch] * 3  # 3 batches
        
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(1.5)
        mock_output.logits = torch.randn(8, 128, 50257)
        mock_model.return_value = mock_output
        
        val_loader = mock_dataloader()
        val_metrics = trainer.validate_epoch(val_loader, epoch=1)
        
        assert isinstance(val_metrics, dict)
        assert 'loss' in val_metrics
    
    def test_save_checkpoint(self, test_config, mock_model, device):
        """Test checkpoint saving."""
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = f"{temp_dir}/checkpoint.pt"
            
            trainer.save_checkpoint(checkpoint_path, epoch=5, best_metric=0.85)
            
            # Check that file exists
            assert Path(checkpoint_path).exists()
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert 'epoch' in checkpoint
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'best_metric' in checkpoint
            assert checkpoint['epoch'] == 5
            assert checkpoint['best_metric'] == 0.85
    
    def test_load_checkpoint(self, test_config, mock_model, device):
        """Test checkpoint loading."""
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = f"{temp_dir}/checkpoint.pt"
            
            # Save a checkpoint first
            trainer.save_checkpoint(checkpoint_path, epoch=10, best_metric=0.9)
            
            # Load checkpoint
            loaded_data = trainer.load_checkpoint(checkpoint_path)
            
            assert loaded_data['epoch'] == 10
            assert loaded_data['best_metric'] == 0.9


class TestMetricsTracker:
    """Test suite for MetricsTracker."""
    
    def test_metrics_tracker_initialization(self, test_config):
        """Test metrics tracker initialization."""
        metrics_config = test_config.get('metrics', {})
        tracker = MetricsTracker(metrics_config)
        
        assert hasattr(tracker, 'config')
        assert hasattr(tracker, 'metrics')
    
    def test_update_metrics(self, test_config):
        """Test metrics update."""
        tracker = MetricsTracker(test_config.get('metrics', {}))
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0)
        mock_output.logits = torch.randn(4, 64, 1000)
        
        # Mock batch
        mock_batch = {
            'text_input_ids': torch.randint(0, 1000, (4, 64)),
            'text_attention_mask': torch.ones(4, 64, dtype=torch.bool)
        }
        
        tracker.update(mock_output, mock_batch, split='train')
        
        # Check that metrics were recorded
        assert 'train' in tracker.metrics
        assert len(tracker.metrics['train']) > 0
    
    def test_compute_metrics(self, test_config):
        """Test metrics computation."""
        tracker = MetricsTracker(test_config.get('metrics', {}))
        
        # Add some dummy metrics
        tracker.metrics['train'] = [
            {'loss': 2.0, 'accuracy': 0.7},
            {'loss': 1.8, 'accuracy': 0.75},
            {'loss': 1.6, 'accuracy': 0.8}
        ]
        
        computed = tracker.compute('train')
        
        assert isinstance(computed, dict)
        assert 'loss' in computed
        assert computed['loss'] == pytest.approx(1.8, rel=1e-2)  # Average
    
    def test_reset_metrics(self, test_config):
        """Test metrics reset."""
        tracker = MetricsTracker(test_config.get('metrics', {}))
        
        # Add some metrics
        tracker.metrics['train'] = [{'loss': 2.0}]
        tracker.metrics['val'] = [{'loss': 1.5}]
        
        # Reset specific split
        tracker.reset('train')
        assert len(tracker.metrics['train']) == 0
        assert len(tracker.metrics['val']) == 1
        
        # Reset all
        tracker.reset()
        assert len(tracker.metrics['train']) == 0
        assert len(tracker.metrics['val']) == 0


class TestMultimodalLoss:
    """Test suite for MultimodalLoss."""
    
    def test_loss_initialization(self, test_config):
        """Test loss function initialization."""
        loss_fn = MultimodalLoss(test_config)
        
        assert hasattr(loss_fn, 'lm_loss_weight')
        assert hasattr(loss_fn, 'alignment_loss_weight')
        assert hasattr(loss_fn, 'contrastive_loss_weight')
    
    def test_language_modeling_loss(self, test_config):
        """Test language modeling loss computation."""
        loss_fn = MultimodalLoss(test_config)
        
        # Mock inputs
        logits = torch.randn(4, 64, 1000)
        labels = torch.randint(0, 1000, (4, 64))
        
        lm_loss = loss_fn.compute_lm_loss(logits, labels)
        
        assert isinstance(lm_loss, torch.Tensor)
        assert lm_loss.requires_grad
        assert lm_loss.item() > 0
    
    def test_alignment_loss(self, test_config):
        """Test alignment loss computation."""
        loss_fn = MultimodalLoss(test_config)
        
        # Mock embeddings
        ts_embeddings = torch.randn(4, 256, 512)
        text_embeddings = torch.randn(4, 64, 512)
        
        alignment_loss = loss_fn.compute_alignment_loss(ts_embeddings, text_embeddings)
        
        assert isinstance(alignment_loss, torch.Tensor)
        assert alignment_loss.requires_grad
    
    def test_contrastive_loss(self, test_config):
        """Test contrastive loss computation."""
        loss_fn = MultimodalLoss(test_config)
        
        # Mock embeddings
        ts_embeddings = torch.randn(4, 512)  # Pooled
        text_embeddings = torch.randn(4, 512)  # Pooled
        
        contrastive_loss = loss_fn.compute_contrastive_loss(ts_embeddings, text_embeddings)
        
        assert isinstance(contrastive_loss, torch.Tensor)
        assert contrastive_loss.requires_grad
    
    def test_total_loss_computation(self, test_config):
        """Test total loss computation."""
        loss_fn = MultimodalLoss(test_config)
        
        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.randn(4, 64, 1000)
        mock_output.ts_embeddings = torch.randn(4, 256, 512)
        mock_output.text_embeddings = torch.randn(4, 64, 512)
        
        # Mock batch
        mock_batch = {
            'text_input_ids': torch.randint(0, 1000, (4, 64)),
            'labels': torch.randint(0, 1000, (4, 64))
        }
        
        total_loss = loss_fn(mock_output, mock_batch)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert total_loss.item() > 0


class TestCallbacks:
    """Test suite for training callbacks."""
    
    def test_early_stopping_initialization(self):
        """Test early stopping callback initialization."""
        early_stop = EarlyStopping(patience=5, min_delta=0.01, mode='min')
        
        assert early_stop.patience == 5
        assert early_stop.min_delta == 0.01
        assert early_stop.mode == 'min'
        assert early_stop.best_metric is None
        assert early_stop.wait == 0
    
    def test_early_stopping_improvement(self):
        """Test early stopping with metric improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        # Simulate improving metrics
        assert not early_stop(2.0)  # First metric
        assert not early_stop(1.8)  # Improvement
        assert not early_stop(1.6)  # Improvement
        assert not early_stop(1.5)  # Improvement
        
        # Should not stop yet
        assert early_stop.wait == 0
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement."""
        early_stop = EarlyStopping(patience=2, min_delta=0.01, mode='min')
        
        # Simulate no improvement
        assert not early_stop(2.0)  # First metric
        assert not early_stop(2.1)  # No improvement (wait=1)
        assert not early_stop(2.05)  # No improvement (wait=2)
        assert early_stop(2.08)  # Should stop now (wait=3 > patience=2)
    
    def test_model_checkpoint_initialization(self):
        """Test model checkpoint callback initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                filepath=f"{temp_dir}/model.pt",
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            
            assert checkpoint.filepath == f"{temp_dir}/model.pt"
            assert checkpoint.monitor == 'val_loss'
            assert checkpoint.mode == 'min'
            assert checkpoint.save_best_only is True
    
    def test_model_checkpoint_save_best(self):
        """Test model checkpoint saving best model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                filepath=f"{temp_dir}/model.pt",
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            
            # Mock trainer
            mock_trainer = Mock()
            mock_trainer.save_checkpoint = Mock()
            
            # Simulate metrics
            metrics = {'val_loss': 1.5, 'val_accuracy': 0.8}
            checkpoint(mock_trainer, metrics, epoch=1)
            
            # Should save because it's the first/best
            mock_trainer.save_checkpoint.assert_called_once()
            
            # Reset mock
            mock_trainer.save_checkpoint.reset_mock()
            
            # Worse metric - should not save
            metrics = {'val_loss': 2.0, 'val_accuracy': 0.7}
            checkpoint(mock_trainer, metrics, epoch=2)
            
            mock_trainer.save_checkpoint.assert_not_called()
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduler callback."""
        # Create a simple model and optimizer for testing
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        lr_scheduler_callback = LearningRateScheduler(scheduler)
        
        # Initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step the scheduler
        lr_scheduler_callback(epoch=1)  # Should not change (step_size=2)
        assert optimizer.param_groups[0]['lr'] == initial_lr
        
        lr_scheduler_callback(epoch=2)  # Should change
        assert optimizer.param_groups[0]['lr'] == initial_lr * 0.5


class TestUtilityFunctions:
    """Test utility functions for training."""
    
    def test_compute_bleu_score(self):
        """Test BLEU score computation."""
        predictions = ["the cat sat on the mat", "hello world"]
        references = ["the cat is on the mat", "hello there world"]
        
        bleu_score = compute_bleu_score(predictions, references)
        
        assert isinstance(bleu_score, float)
        assert 0.0 <= bleu_score <= 1.0
    
    def test_compute_rouge_score(self):
        """Test ROUGE score computation."""
        predictions = ["the quick brown fox", "machine learning is great"]
        references = ["the quick red fox", "deep learning is amazing"]
        
        rouge_scores = compute_rouge_score(predictions, references)
        
        assert isinstance(rouge_scores, dict)
        assert 'rouge1' in rouge_scores
        assert 'rouge2' in rouge_scores
        assert 'rougeL' in rouge_scores
        
        for score in rouge_scores.values():
            assert 0.0 <= score <= 1.0


@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for training components."""
    
    @pytest.mark.slow
    @patch('torch.utils.data.DataLoader')
    def test_complete_training_loop(self, mock_dataloader, test_config, mock_model, 
                                  sample_batch, device):
        """Test complete training loop integration."""
        # Setup mock dataloader
        mock_dataloader.return_value = [sample_batch] * 3  # 3 batches per epoch
        
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0, requires_grad=True)
        mock_output.logits = torch.randn(8, 128, 50257)
        mock_model.return_value = mock_output
        
        train_loader = mock_dataloader()
        val_loader = mock_dataloader()
        
        # Add callbacks
        callbacks = [
            EarlyStopping(patience=2, min_delta=0.01),
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            callbacks.append(
                ModelCheckpoint(f"{temp_dir}/best_model.pt", monitor='val_loss', mode='min')
            )
            
            # Run training for a few epochs
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=3,
                callbacks=callbacks
            )
            
            assert isinstance(history, dict)
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) <= 3  # May stop early
    
    def test_training_with_mixed_precision(self, test_config, mock_model, sample_batch, device):
        """Test training with mixed precision."""
        # Skip if CUDA not available (AMP requires CUDA)
        if not torch.cuda.is_available():
            pytest.skip("Mixed precision requires CUDA")
        
        test_config['training']['mixed_precision'] = True
        trainer = MultimodalTrainer(mock_model, test_config, device)
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0, requires_grad=True).cuda()
        mock_output.logits = torch.randn(8, 128, 50257).cuda()
        mock_model.return_value = mock_output
        
        # Move batch to device
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                sample_batch[key] = value.cuda()
        
        # Should not raise error with AMP
        loss = trainer.training_step(sample_batch)
        assert isinstance(loss, torch.Tensor)
