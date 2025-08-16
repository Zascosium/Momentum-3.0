"""
Custom loss functions for multimodal training.
Implements various loss components for time series and text alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class MultimodalLossFunction(nn.Module):
    """
    Comprehensive loss function for multimodal training.
    Combines text generation, time series reconstruction, and alignment losses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multimodal loss function.
        
        Args:
            config: Loss configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.text_weight = config.get('text_generation_weight', 1.0)
        self.ts_weight = config.get('time_series_reconstruction_weight', 0.5)
        self.alignment_weight = config.get('alignment_loss_weight', 0.1)
        self.consistency_weight = config.get('consistency_loss_weight', 0.05)
        
        # Loss configuration
        self.label_smoothing = config.get('label_smoothing', 0.1)
        self.temperature = config.get('contrastive_temperature', 0.1)
        self.margin = config.get('triplet_margin', 0.5)
        
        # Initialize loss components
        self.text_loss_fn = self._create_text_loss()
        self.ts_loss_fn = self._create_ts_loss()
        self.alignment_loss_fn = self._create_alignment_loss()
        
        logger.info(f"Initialized MultimodalLossFunction with weights: "
                   f"text={self.text_weight}, ts={self.ts_weight}, "
                   f"alignment={self.alignment_weight}")
    
    def _create_text_loss(self) -> nn.Module:
        """Create text generation loss function."""
        if self.label_smoothing > 0:
            return nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
                reduction='mean'
            )
        else:
            return nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    
    def _create_ts_loss(self) -> nn.Module:
        """Create time series reconstruction loss function."""
        ts_loss_type = self.config.get('ts_loss_type', 'mse')
        
        if ts_loss_type == 'mse':
            return nn.MSELoss(reduction='mean')
        elif ts_loss_type == 'mae':
            return nn.L1Loss(reduction='mean')
        elif ts_loss_type == 'huber':
            return nn.HuberLoss(reduction='mean')
        else:
            logger.warning(f"Unknown TS loss type: {ts_loss_type}, using MSE")
            return nn.MSELoss(reduction='mean')
    
    def _create_alignment_loss(self) -> str:
        """Create alignment loss function type."""
        return self.config.get('alignment_loss_type', 'contrastive')
    
    def forward(
        self,
        model_outputs: Any,
        batch: Dict[str, torch.Tensor],
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total multimodal loss.
        
        Args:
            model_outputs: Model outputs containing logits and embeddings
            batch: Input batch dictionary
            return_components: Whether to return individual loss components
            
        Returns:
            Total loss or dictionary of loss components
        """
        losses = {}
        total_loss = 0.0
        
        # Text generation loss
        if hasattr(model_outputs, 'logits') and model_outputs.logits is not None:
            text_loss = self.compute_text_loss(model_outputs.logits, batch)
            losses['text_loss'] = text_loss
            total_loss += self.text_weight * text_loss
        
        # Time series reconstruction loss
        if (hasattr(model_outputs, 'ts_embeddings') and 
            model_outputs.ts_embeddings is not None and
            'time_series' in batch):
            ts_loss = self.compute_ts_reconstruction_loss(
                model_outputs.ts_embeddings, 
                batch['time_series']
            )
            losses['ts_reconstruction_loss'] = ts_loss
            total_loss += self.ts_weight * ts_loss
        
        # Alignment loss
        if (hasattr(model_outputs, 'ts_embeddings') and 
            hasattr(model_outputs, 'text_embeddings') and
            model_outputs.ts_embeddings is not None):
            alignment_loss = self.compute_alignment_loss(
                model_outputs.ts_embeddings,
                model_outputs.text_embeddings,
                batch
            )
            losses['alignment_loss'] = alignment_loss
            total_loss += self.alignment_weight * alignment_loss
        
        # Consistency loss
        if self.consistency_weight > 0:
            consistency_loss = self.compute_consistency_loss(model_outputs, batch)
            losses['consistency_loss'] = consistency_loss
            total_loss += self.consistency_weight * consistency_loss
        
        losses['total_loss'] = total_loss
        
        if return_components:
            return losses
        return total_loss
    
    def compute_text_loss(
        self, 
        logits: torch.Tensor, 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute text generation loss.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            batch: Input batch containing text_input_ids
            
        Returns:
            Text generation loss
        """
        if 'text_input_ids' not in batch:
            return torch.tensor(0.0, device=logits.device)
        
        labels = batch['text_input_ids'].clone()
        
        # Shift for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        text_loss = self.text_loss_fn(shift_logits, shift_labels)
        
        return text_loss
    
    def compute_ts_reconstruction_loss(
        self,
        ts_embeddings: torch.Tensor,
        original_ts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute time series reconstruction loss.
        
        Args:
            ts_embeddings: Time series embeddings
            original_ts: Original time series data
            
        Returns:
            Reconstruction loss
        """
        # For now, implement a simple reconstruction loss
        # In practice, you would need a reconstruction head
        
        # Use mean squared error between embedding statistics and original statistics
        ts_mean = original_ts.mean(dim=1, keepdim=True)  # [batch_size, 1, n_features]
        ts_std = original_ts.std(dim=1, keepdim=True)    # [batch_size, 1, n_features]
        
        # Get embedding statistics
        emb_mean = ts_embeddings.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
        emb_std = ts_embeddings.std(dim=1, keepdim=True)    # [batch_size, 1, embed_dim]
        
        # Project embeddings to original feature space for comparison
        # This is a simplified approach - in practice, use a proper reconstruction head
        if ts_embeddings.size(-1) != original_ts.size(-1):
            projection = nn.Linear(
                ts_embeddings.size(-1), 
                original_ts.size(-1)
            ).to(ts_embeddings.device)
            
            projected_mean = projection(emb_mean)
            projected_std = F.relu(projection(emb_std)) + 1e-8
        else:
            projected_mean = emb_mean
            projected_std = emb_std
        
        # MSE loss between statistics
        mean_loss = self.ts_loss_fn(projected_mean, ts_mean)
        std_loss = self.ts_loss_fn(projected_std, ts_std)
        
        return mean_loss + std_loss
    
    def compute_alignment_loss(
        self,
        ts_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cross-modal alignment loss.
        
        Args:
            ts_embeddings: Time series embeddings [batch_size, ts_seq_len, embed_dim]
            text_embeddings: Text embeddings [batch_size, text_seq_len, embed_dim]
            batch: Input batch
            
        Returns:
            Alignment loss
        """
        if self.alignment_loss_fn == 'contrastive':
            return self.compute_contrastive_loss(ts_embeddings, text_embeddings, batch)
        elif self.alignment_loss_fn == 'triplet':
            return self.compute_triplet_loss(ts_embeddings, text_embeddings, batch)
        elif self.alignment_loss_fn == 'cosine':
            return self.compute_cosine_loss(ts_embeddings, text_embeddings, batch)
        else:
            logger.warning(f"Unknown alignment loss: {self.alignment_loss_fn}")
            return torch.tensor(0.0, device=ts_embeddings.device)
    
    def compute_contrastive_loss(
        self,
        ts_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss."""
        # Pool embeddings to get single representations
        ts_repr = ts_embeddings.mean(dim=1)  # [batch_size, embed_dim]
        
        # For text embeddings, handle the case where they might be None
        if text_embeddings is not None:
            text_repr = text_embeddings.mean(dim=1)  # [batch_size, embed_dim]
        else:
            # Use text input embeddings if text_embeddings is None
            if 'text_input_ids' in batch:
                # Create a simple embedding lookup (placeholder)
                vocab_size = 50257  # GPT-2 vocab size
                embed_dim = ts_embeddings.size(-1)
                text_embedding_layer = nn.Embedding(vocab_size, embed_dim).to(ts_embeddings.device)
                text_embeds = text_embedding_layer(batch['text_input_ids'])
                text_repr = text_embeds.mean(dim=1)
            else:
                return torch.tensor(0.0, device=ts_embeddings.device)
        
        # L2 normalize
        ts_repr = F.normalize(ts_repr, p=2, dim=-1)
        text_repr = F.normalize(text_repr, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(ts_repr, text_repr.transpose(0, 1)) / self.temperature
        
        # Labels (diagonal should be positive pairs)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # InfoNCE loss
        loss_t2ts = F.cross_entropy(similarity, labels)
        loss_ts2t = F.cross_entropy(similarity.transpose(0, 1), labels)
        
        return (loss_t2ts + loss_ts2t) / 2
    
    def compute_triplet_loss(
        self,
        ts_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute triplet loss for alignment."""
        # Pool embeddings
        ts_repr = ts_embeddings.mean(dim=1)  # [batch_size, embed_dim]
        
        if text_embeddings is not None:
            text_repr = text_embeddings.mean(dim=1)
        else:
            return torch.tensor(0.0, device=ts_embeddings.device)
        
        batch_size = ts_repr.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=ts_embeddings.device)
        
        # Create positive and negative pairs
        positive_distances = F.pairwise_distance(ts_repr, text_repr, p=2)
        
        # Negative samples (circular shift)
        negative_text = torch.roll(text_repr, shifts=1, dims=0)
        negative_distances = F.pairwise_distance(ts_repr, negative_text, p=2)
        
        # Triplet loss
        triplet_loss = F.relu(positive_distances - negative_distances + self.margin)
        
        return triplet_loss.mean()
    
    def compute_cosine_loss(
        self,
        ts_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute cosine similarity loss."""
        # Pool embeddings
        ts_repr = ts_embeddings.mean(dim=1)
        
        if text_embeddings is not None:
            text_repr = text_embeddings.mean(dim=1)
        else:
            return torch.tensor(0.0, device=ts_embeddings.device)
        
        # Cosine similarity (should be high for aligned pairs)
        cosine_sim = F.cosine_similarity(ts_repr, text_repr, dim=-1)
        
        # Loss is 1 - cosine_similarity (minimize to maximize similarity)
        cosine_loss = 1 - cosine_sim.mean()
        
        return cosine_loss
    
    def compute_consistency_loss(
        self,
        model_outputs: Any,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute consistency loss to ensure model consistency.
        
        Args:
            model_outputs: Model outputs
            batch: Input batch
            
        Returns:
            Consistency loss
        """
        # Implement consistency regularization
        # For example, ensure similar inputs produce similar outputs
        
        consistency_loss = torch.tensor(0.0, device=next(iter(batch.values())).device)
        
        # Example: L2 regularization on embeddings
        if hasattr(model_outputs, 'ts_embeddings') and model_outputs.ts_embeddings is not None:
            ts_reg = torch.norm(model_outputs.ts_embeddings, p=2, dim=-1).mean()
            consistency_loss += 0.01 * ts_reg
        
        if hasattr(model_outputs, 'text_embeddings') and model_outputs.text_embeddings is not None:
            text_reg = torch.norm(model_outputs.text_embeddings, p=2, dim=-1).mean()
            consistency_loss += 0.01 * text_reg
        
        return consistency_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Target labels [batch_size]
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for sequence labeling tasks.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Model predictions [batch_size, seq_len, num_classes]
            targets: Target labels [batch_size, seq_len]
            
        Returns:
            Dice loss
        """
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=-1)
        
        # One-hot encode targets
        num_classes = inputs.size(-1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Flatten
        inputs = inputs.view(-1, num_classes)
        targets_one_hot = targets_one_hot.view(-1, num_classes)
        
        # Compute Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=0)
        dice_coeff = (2 * intersection + self.smooth) / (
            inputs.sum(dim=0) + targets_one_hot.sum(dim=0) + self.smooth
        )
        
        # Dice loss is 1 - Dice coefficient
        dice_loss = 1 - dice_coeff.mean()
        
        return dice_loss


# Factory function for creating loss functions
def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss function instance
    """
    loss_type = config.get('type', 'multimodal')
    
    if loss_type == 'multimodal':
        return MultimodalLossFunction(config)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(
            ignore_index=config.get('ignore_index', -100),
            label_smoothing=config.get('label_smoothing', 0.0),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('alpha', 1.0),
            gamma=config.get('gamma', 2.0),
            reduction=config.get('reduction', 'mean')
        )
    elif loss_type == 'dice':
        return DiceLoss(smooth=config.get('smooth', 1e-6))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test loss function
    config = {
        'text_generation_weight': 1.0,
        'time_series_reconstruction_weight': 0.5,
        'alignment_loss_weight': 0.1,
        'label_smoothing': 0.1,
        'contrastive_temperature': 0.1,
        'alignment_loss_type': 'contrastive'
    }
    
    loss_fn = MultimodalLossFunction(config)
    
    # Mock model outputs
    class MockOutputs:
        def __init__(self):
            self.logits = torch.randn(4, 32, 1000)  # [batch, seq_len, vocab]
            self.ts_embeddings = torch.randn(4, 16, 512)  # [batch, ts_seq, embed]
            self.text_embeddings = torch.randn(4, 32, 512)  # [batch, text_seq, embed]
    
    # Mock batch
    batch = {
        'text_input_ids': torch.randint(0, 1000, (4, 32)),
        'time_series': torch.randn(4, 128, 3)
    }
    
    outputs = MockOutputs()
    
    # Compute loss
    total_loss = loss_fn(outputs, batch)
    print(f"Total loss: {total_loss}")
    
    # Compute loss components
    loss_components = loss_fn(outputs, batch, return_components=True)
    print(f"Loss components: {loss_components}")
    
    print("Loss functions implementation completed successfully!")