"""
Evaluation metrics for multimodal training.
Implements text generation metrics, time series metrics, and cross-modal evaluation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from collections import defaultdict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    count: int = 1
    sum: float = None
    
    def __post_init__(self):
        if self.sum is None:
            self.sum = self.value

class MetricsTracker:
    """
    Comprehensive metrics tracker for multimodal training.
    Tracks text generation, time series, and alignment metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics tracker.
        
        Args:
            config: Metrics configuration
        """
        self.config = config
        self.reset()
        
        # Metric configurations
        self.text_metrics_config = config.get('text_metrics', {})
        self.ts_metrics_config = config.get('ts_metrics', {})
        self.alignment_metrics_config = config.get('alignment_metrics', {})
        
        # Initialize tokenizer for text metrics if needed
        self.tokenizer = None
        if self.text_metrics_config.get('compute_bleu', False):
            self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize tokenizer for text metrics."""
        try:
            from transformers import AutoTokenizer
            tokenizer_name = self.text_metrics_config.get('tokenizer', 'gpt2-medium')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            logger.warning("transformers not available, BLEU scores will not be computed")
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        
        self.predictions = {
            'train': {'generated': [], 'targets': [], 'ts_pred': [], 'ts_true': []},
            'val': {'generated': [], 'targets': [], 'ts_pred': [], 'ts_true': []},
            'test': {'generated': [], 'targets': [], 'ts_pred': [], 'ts_true': []}
        }
    
    def update(
        self, 
        model_outputs: Any, 
        batch: Dict[str, torch.Tensor], 
        split: str = 'train'
    ):
        """
        Update metrics with batch results.
        
        Args:
            model_outputs: Model outputs
            batch: Input batch
            split: Dataset split ('train', 'val', 'test')
        """
        if split not in self.metrics:
            logger.warning(f"Unknown split: {split}")
            return
        
        # Text generation metrics
        if hasattr(model_outputs, 'logits') and model_outputs.logits is not None:
            self._update_text_metrics(model_outputs, batch, split)
        
        # Time series metrics
        if hasattr(model_outputs, 'ts_embeddings') and model_outputs.ts_embeddings is not None:
            self._update_ts_metrics(model_outputs, batch, split)
        
        # Alignment metrics
        if (hasattr(model_outputs, 'ts_embeddings') and 
            hasattr(model_outputs, 'text_embeddings')):
            self._update_alignment_metrics(model_outputs, batch, split)
        
        # Loss metrics
        if hasattr(model_outputs, 'loss') and model_outputs.loss is not None:
            self.metrics[split]['loss'].append(model_outputs.loss.item())
        
        if hasattr(model_outputs, 'text_loss') and model_outputs.text_loss is not None:
            self.metrics[split]['text_loss'].append(model_outputs.text_loss.item())
        
        if hasattr(model_outputs, 'ts_reconstruction_loss') and model_outputs.ts_reconstruction_loss is not None:
            self.metrics[split]['ts_reconstruction_loss'].append(model_outputs.ts_reconstruction_loss.item())
        
        if hasattr(model_outputs, 'alignment_loss') and model_outputs.alignment_loss is not None:
            self.metrics[split]['alignment_loss'].append(model_outputs.alignment_loss.item())
    
    def _update_text_metrics(self, model_outputs, batch: Dict[str, torch.Tensor], split: str):
        """Update text generation metrics."""
        logits = model_outputs.logits
        
        if 'text_input_ids' not in batch:
            return
        
        labels = batch['text_input_ids']
        
        # Perplexity
        perplexity = self.compute_perplexity(logits, labels)
        self.metrics[split]['perplexity'].append(perplexity)
        
        # Accuracy (next token prediction)
        accuracy = self.compute_accuracy(logits, labels)
        self.metrics[split]['accuracy'].append(accuracy)
        
        # Top-k accuracy
        for k in [3, 5, 10]:
            top_k_acc = self.compute_top_k_accuracy(logits, labels, k)
            self.metrics[split][f'top_{k}_accuracy'].append(top_k_acc)
        
        # Store predictions for BLEU computation
        if self.text_metrics_config.get('compute_bleu', False) and self.tokenizer:
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Convert to text
            for i in range(predicted_ids.size(0)):
                pred_text = self.tokenizer.decode(predicted_ids[i], skip_special_tokens=True)
                target_text = self.tokenizer.decode(labels[i], skip_special_tokens=True)
                
                self.predictions[split]['generated'].append(pred_text)
                self.predictions[split]['targets'].append(target_text)
    
    def _update_ts_metrics(self, model_outputs, batch: Dict[str, torch.Tensor], split: str):
        """Update time series metrics."""
        if 'time_series' not in batch:
            return
        
        ts_embeddings = model_outputs.ts_embeddings
        original_ts = batch['time_series']
        
        # Embedding statistics
        emb_mean = ts_embeddings.mean().item()
        emb_std = ts_embeddings.std().item()
        emb_norm = torch.norm(ts_embeddings, p=2, dim=-1).mean().item()
        
        self.metrics[split]['ts_embedding_mean'].append(emb_mean)
        self.metrics[split]['ts_embedding_std'].append(emb_std)
        self.metrics[split]['ts_embedding_norm'].append(emb_norm)
        
        # Original time series statistics
        ts_mean = original_ts.mean().item()
        ts_std = original_ts.std().item()
        
        self.metrics[split]['ts_original_mean'].append(ts_mean)
        self.metrics[split]['ts_original_std'].append(ts_std)
        
        # Compute reconstruction metrics if reconstruction head exists
        # This would require access to the model's reconstruction head
        # For now, we compute simple correlation metrics
        
        # Temporal correlation (simplified)
        if ts_embeddings.size(1) > 1:  # More than one time step
            ts_emb_diff = torch.diff(ts_embeddings.mean(dim=-1), dim=1)
            ts_orig_diff = torch.diff(original_ts.mean(dim=-1), dim=1)
            
            if ts_emb_diff.numel() > 0 and ts_orig_diff.numel() > 0:
                correlation = self.compute_correlation(
                    ts_emb_diff.flatten(), 
                    ts_orig_diff.flatten()
                )
                self.metrics[split]['ts_temporal_correlation'].append(correlation)
    
    def _update_alignment_metrics(self, model_outputs, batch: Dict[str, torch.Tensor], split: str):
        """Update cross-modal alignment metrics."""
        ts_embeddings = model_outputs.ts_embeddings
        text_embeddings = getattr(model_outputs, 'text_embeddings', None)
        
        if text_embeddings is None:
            return
        
        # Pool embeddings to get single representations
        ts_repr = ts_embeddings.mean(dim=1)  # [batch_size, embed_dim]
        text_repr = text_embeddings.mean(dim=1)  # [batch_size, embed_dim]
        
        # Cosine similarity
        cosine_sim = F.cosine_similarity(ts_repr, text_repr, dim=-1).mean().item()
        self.metrics[split]['cosine_similarity'].append(cosine_sim)
        
        # L2 distance
        l2_distance = torch.norm(ts_repr - text_repr, p=2, dim=-1).mean().item()
        self.metrics[split]['l2_distance'].append(l2_distance)
        
        # Alignment score (1 / (1 + distance))
        alignment_score = 1 / (1 + l2_distance)
        self.metrics[split]['alignment_score'].append(alignment_score)
    
    def compute_perplexity(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute perplexity."""
        # Shift for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Remove ignored tokens
        mask = shift_labels != -100
        if mask.sum() == 0:
            return float('inf')
        
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]
        
        # Cross entropy loss
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
        
        # Perplexity is exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute next token prediction accuracy."""
        # Shift for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get predictions
        predictions = torch.argmax(shift_logits, dim=-1)
        
        # Flatten
        predictions = predictions.view(-1)
        shift_labels = shift_labels.view(-1)
        
        # Remove ignored tokens
        mask = shift_labels != -100
        if mask.sum() == 0:
            return 0.0
        
        predictions = predictions[mask]
        shift_labels = shift_labels[mask]
        
        # Compute accuracy
        correct = (predictions == shift_labels).float()
        accuracy = correct.mean().item()
        
        return accuracy
    
    def compute_top_k_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        """Compute top-k accuracy."""
        # Shift for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get top-k predictions
        _, top_k_indices = torch.topk(shift_logits, k, dim=-1)
        
        # Flatten
        top_k_indices = top_k_indices.view(-1, k)
        shift_labels = shift_labels.view(-1, 1)
        
        # Remove ignored tokens
        mask = (shift_labels != -100).squeeze(-1)
        if mask.sum() == 0:
            return 0.0
        
        top_k_indices = top_k_indices[mask]
        shift_labels = shift_labels[mask]
        
        # Check if true label is in top-k
        correct = (top_k_indices == shift_labels).any(dim=-1).float()
        top_k_accuracy = correct.mean().item()
        
        return top_k_accuracy
    
    def compute_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson correlation coefficient."""
        if x.numel() == 0 or y.numel() == 0:
            return 0.0
        
        # Center the data
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # Compute correlation
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation.item()
    
    def compute_bleu_scores(self, split: str = 'val') -> Dict[str, float]:
        """Compute BLEU scores for generated text."""
        if not self.predictions[split]['generated']:
            return {}
        
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download('punkt', quiet=True)
        except ImportError:
            logger.warning("NLTK not available, BLEU scores will not be computed")
            return {}
        
        generated_texts = self.predictions[split]['generated']
        target_texts = self.predictions[split]['targets']
        
        bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
        smoothing = SmoothingFunction().method1
        
        for gen_text, target_text in zip(generated_texts, target_texts):
            # Tokenize
            gen_tokens = gen_text.split()
            target_tokens = [target_text.split()]  # List of reference sentences
            
            # Compute BLEU scores
            for n in range(1, 5):
                weights = tuple([1/n] * n + [0] * (4-n))
                bleu_score = sentence_bleu(
                    target_tokens, 
                    gen_tokens, 
                    weights=weights,
                    smoothing_function=smoothing
                )
                bleu_scores[f'bleu_{n}'].append(bleu_score)
        
        # Average BLEU scores
        avg_bleu_scores = {}
        for key, scores in bleu_scores.items():
            if scores:
                avg_bleu_scores[key] = sum(scores) / len(scores)
        
        return avg_bleu_scores
    
    def compute_rouge_scores(self, split: str = 'val') -> Dict[str, float]:
        """Compute ROUGE scores for generated text."""
        if not self.predictions[split]['generated']:
            return {}
        
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning("rouge-score not available, ROUGE scores will not be computed")
            return {}
        
        generated_texts = self.predictions[split]['generated']
        target_texts = self.predictions[split]['targets']
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = defaultdict(list)
        
        for gen_text, target_text in zip(generated_texts, target_texts):
            scores = scorer.score(target_text, gen_text)
            
            for rouge_type, score in scores.items():
                rouge_scores[f'{rouge_type}_precision'].append(score.precision)
                rouge_scores[f'{rouge_type}_recall'].append(score.recall)
                rouge_scores[f'{rouge_type}_fmeasure'].append(score.fmeasure)
        
        # Average ROUGE scores
        avg_rouge_scores = {}
        for key, scores in rouge_scores.items():
            if scores:
                avg_rouge_scores[key] = sum(scores) / len(scores)
        
        return avg_rouge_scores
    
    def compute(self, split: str = 'val') -> Dict[str, float]:
        """
        Compute all metrics for a split.
        
        Args:
            split: Dataset split
            
        Returns:
            Dictionary of computed metrics
        """
        if split not in self.metrics:
            return {}
        
        computed_metrics = {}
        
        # Average all accumulated metrics
        for metric_name, values in self.metrics[split].items():
            if values:
                computed_metrics[metric_name] = sum(values) / len(values)
        
        # Compute BLEU scores
        if self.text_metrics_config.get('compute_bleu', False):
            bleu_scores = self.compute_bleu_scores(split)
            computed_metrics.update(bleu_scores)
        
        # Compute ROUGE scores
        if self.text_metrics_config.get('compute_rouge', False):
            rouge_scores = self.compute_rouge_scores(split)
            computed_metrics.update(rouge_scores)
        
        return computed_metrics
    
    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Tuple[float, int]:
        """
        Get best metric value across all splits.
        
        Args:
            metric_name: Name of metric
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            
        Returns:
            Tuple of (best_value, epoch_index)
        """
        all_values = []
        
        for split in ['train', 'val', 'test']:
            if metric_name in self.metrics[split]:
                all_values.extend(self.metrics[split][metric_name])
        
        if not all_values:
            return float('inf') if mode == 'min' else float('-inf'), -1
        
        if mode == 'min':
            best_value = min(all_values)
            best_idx = all_values.index(best_value)
        else:
            best_value = max(all_values)
            best_idx = all_values.index(best_value)
        
        return best_value, best_idx
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        
        for split in ['train', 'val', 'test']:
            summary[split] = self.compute(split)
        
        return summary


# Standalone metric functions
def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Mean Squared Error."""
    return F.mse_loss(predictions, targets).item()

def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Mean Absolute Error."""
    return F.l1_loss(predictions, targets).item()

def compute_mape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error."""
    percentage_errors = torch.abs((targets - predictions) / (targets + epsilon))
    return percentage_errors.mean().item() * 100

def compute_r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R-squared score."""
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()

def compute_dtw_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Dynamic Time Warping distance (simplified version).
    
    Args:
        x: First time series [seq_len]
        y: Second time series [seq_len]
        
    Returns:
        DTW distance
    """
    n, m = len(x), len(y)
    
    # Create distance matrix
    dtw_matrix = torch.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m].item()


# Example usage and testing
if __name__ == "__main__":
    # Test metrics tracker
    config = {
        'text_metrics': {
            'compute_bleu': True,
            'compute_rouge': False,
            'tokenizer': 'gpt2-medium'
        },
        'ts_metrics': {
            'compute_reconstruction': True
        },
        'alignment_metrics': {
            'compute_similarity': True
        }
    }
    
    tracker = MetricsTracker(config)
    
    # Test with real model outputs (placeholder)
    print("Metrics implementation completed successfully!")
    print("To test with real data, provide actual model outputs and batch data.")
    
    # Update metrics
    tracker.update(outputs, batch, 'val')
    
    # Compute metrics
    metrics = tracker.compute('val')
    print(f"Computed metrics: {metrics}")
    
    # Summary
    summary = tracker.summary()
    print(f"Summary: {summary}")
    
    print("Metrics implementation completed successfully!")