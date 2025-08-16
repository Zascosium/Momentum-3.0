"""
Visualization utilities for training monitoring and analysis.
Creates plots for loss curves, attention maps, and model performance.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """
    Comprehensive visualization utilities for training monitoring.
    """
    
    def __init__(self, save_dir: str = '/dbfs/mllm/plots'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib backend for headless environments
        import matplotlib
        matplotlib.use('Agg')
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Dict[str, List[float]] = None,
        val_metrics: Dict[str, List[float]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            train_losses: Training loss values
            val_losses: Validation loss values
            train_metrics: Additional training metrics
            val_metrics: Additional validation metrics
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Determine number of subplots
        n_metrics = 1  # Loss plot
        if train_metrics:
            n_metrics += len(train_metrics)
        
        # Create figure
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot loss curves
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            axes[0].plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot additional metrics
        if train_metrics and val_metrics:
            for i, metric_name in enumerate(train_metrics.keys(), 1):
                if i < len(axes):
                    train_values = train_metrics[metric_name]
                    val_values = val_metrics.get(metric_name, [])
                    
                    epochs_train = range(1, len(train_values) + 1)
                    axes[i].plot(epochs_train, train_values, 'b-', 
                               label=f'Training {metric_name}', linewidth=2)
                    
                    if val_values:
                        epochs_val = range(1, len(val_values) + 1)
                        axes[i].plot(epochs_val, val_values, 'r-', 
                                   label=f'Validation {metric_name}', linewidth=2)
                    
                    axes[i].set_xlabel('Epoch')
                    axes[i].set_ylabel(metric_name.replace('_', ' ').title())
                    axes[i].set_title(f'Training and Validation {metric_name.replace("_", " ").title()}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / 'training_curves.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
        
        return fig
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        input_tokens: List[str] = None,
        output_tokens: List[str] = None,
        head_idx: int = 0,
        layer_idx: int = -1,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot attention heatmap for cross-modal attention.
        
        Args:
            attention_weights: Attention weights tensor [batch, heads, seq_len, seq_len]
            input_tokens: Input token labels
            output_tokens: Output token labels
            head_idx: Which attention head to visualize
            layer_idx: Which layer to visualize (-1 for last)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Extract attention for specific head
        if len(attention_weights.shape) == 4:
            # [batch, heads, seq_len, seq_len]
            attn = attention_weights[0, head_idx].cpu().numpy()
        elif len(attention_weights.shape) == 3:
            # [heads, seq_len, seq_len]
            attn = attention_weights[head_idx].cpu().numpy()
        else:
            # [seq_len, seq_len]
            attn = attention_weights.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        im = ax.imshow(attn, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        if input_tokens:
            ax.set_xticks(range(len(input_tokens)))
            ax.set_xticklabels(input_tokens, rotation=45, ha='right')
        
        if output_tokens:
            ax.set_yticks(range(len(output_tokens)))
            ax.set_yticklabels(output_tokens)
        
        # Labels and title
        ax.set_xlabel('Input Tokens (Keys)')
        ax.set_ylabel('Output Tokens (Queries)')
        ax.set_title(f'Cross-Modal Attention Heatmap (Head {head_idx})')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / f'attention_heatmap_head_{head_idx}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Attention heatmap saved to {save_path}")
        
        return fig
    
    def plot_embedding_space(
        self,
        ts_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: List[str] = None,
        method: str = 'tsne',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 2D visualization of embedding space using dimensionality reduction.
        
        Args:
            ts_embeddings: Time series embeddings [n_samples, embed_dim]
            text_embeddings: Text embeddings [n_samples, embed_dim]
            labels: Sample labels for coloring
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Combine embeddings
        ts_emb = ts_embeddings.cpu().numpy()
        text_emb = text_embeddings.cpu().numpy()
        
        all_embeddings = np.concatenate([ts_emb, text_emb], axis=0)
        modality_labels = ['Time Series'] * len(ts_emb) + ['Text'] * len(text_emb)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                logger.warning("UMAP not available, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit and transform
        embeddings_2d = reducer.fit_transform(all_embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot embeddings
        ts_emb_2d = embeddings_2d[:len(ts_emb)]
        text_emb_2d = embeddings_2d[len(ts_emb):]
        
        scatter1 = ax.scatter(ts_emb_2d[:, 0], ts_emb_2d[:, 1], 
                            c='blue', alpha=0.6, s=50, label='Time Series')
        scatter2 = ax.scatter(text_emb_2d[:, 0], text_emb_2d[:, 1], 
                            c='red', alpha=0.6, s=50, label='Text')
        
        # Draw connections between paired samples
        if len(ts_emb_2d) == len(text_emb_2d):
            for i in range(len(ts_emb_2d)):
                ax.plot([ts_emb_2d[i, 0], text_emb_2d[i, 0]], 
                       [ts_emb_2d[i, 1], text_emb_2d[i, 1]], 
                       'k-', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Multimodal Embedding Space Visualization ({method.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / f'embedding_space_{method}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Embedding space plot saved to {save_path}")
        
        return fig
    
    def plot_time_series_reconstruction(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        sample_idx: int = 0,
        feature_names: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series reconstruction comparison.
        
        Args:
            original: Original time series [batch, seq_len, n_features]
            reconstructed: Reconstructed time series [batch, seq_len, n_features]
            sample_idx: Which sample to plot
            feature_names: Names of features
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        orig = original[sample_idx].cpu().numpy()
        recon = reconstructed[sample_idx].cpu().numpy()
        
        n_features = orig.shape[1]
        
        # Create subplots
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
        if n_features == 1:
            axes = [axes]
        
        time_steps = range(len(orig))
        
        for i in range(n_features):
            feature_name = feature_names[i] if feature_names else f'Feature {i+1}'
            
            axes[i].plot(time_steps, orig[:, i], 'b-', label='Original', linewidth=2)
            axes[i].plot(time_steps, recon[:, i], 'r--', label='Reconstructed', linewidth=2)
            
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].set_title(f'Time Series Reconstruction - {feature_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display MSE
            mse = np.mean((orig[:, i] - recon[:, i]) ** 2)
            axes[i].text(0.02, 0.98, f'MSE: {mse:.4f}', 
                        transform=axes[i].transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / f'ts_reconstruction_sample_{sample_idx}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time series reconstruction plot saved to {save_path}")
        
        return fig
    
    def plot_loss_landscape(
        self,
        loss_history: List[float],
        lr_history: List[float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot loss landscape during training.
        
        Args:
            loss_history: Training loss history
            lr_history: Learning rate history
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        epochs = range(len(loss_history))
        
        # Loss curve
        ax1.plot(epochs, loss_history, 'b-', linewidth=2)
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, lr_history, 'g-', linewidth=2)
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Loss gradient (rate of change)
        if len(loss_history) > 1:
            loss_gradient = np.gradient(loss_history)
            ax3.plot(epochs, loss_gradient, 'r-', linewidth=2)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Loss Gradient')
            ax3.set_xlabel('Epoch')
            ax3.set_title('Loss Rate of Change')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / 'loss_landscape.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss landscape plot saved to {save_path}")
        
        return fig
    
    def plot_model_performance_dashboard(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive model performance dashboard.
        
        Args:
            metrics_history: Dictionary of metric histories
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Create a 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_configs = [
            ('loss', 'Training Loss', 'Epoch', 'Loss'),
            ('val_loss', 'Validation Loss', 'Epoch', 'Loss'),
            ('accuracy', 'Accuracy', 'Epoch', 'Accuracy'),
            ('perplexity', 'Perplexity', 'Epoch', 'Perplexity'),
            ('cosine_similarity', 'Cross-Modal Similarity', 'Epoch', 'Cosine Similarity'),
            ('learning_rate', 'Learning Rate', 'Epoch', 'Learning Rate')
        ]
        
        for i, (metric_key, title, xlabel, ylabel) in enumerate(plot_configs):
            if i >= len(axes):
                break
                
            if metric_key in metrics_history:
                values = metrics_history[metric_key]
                epochs = range(len(values))
                
                axes[i].plot(epochs, values, linewidth=2)
                axes[i].set_title(title)
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                if len(values) > 1:
                    z = np.polyfit(epochs, values, 1)
                    p = np.poly1d(z)
                    axes[i].plot(epochs, p(epochs), "--", alpha=0.5, color='red')
                
                # Add min/max annotations
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    min_epoch = values.index(min_val)
                    max_epoch = values.index(max_val)
                    
                    axes[i].annotate(f'Min: {min_val:.4f}', 
                                   xy=(min_epoch, min_val), 
                                   xytext=(10, 10), 
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            else:
                axes[i].text(0.5, 0.5, f'{metric_key}\nNot Available', 
                           transform=axes[i].transAxes, 
                           ha='center', va='center',
                           fontsize=12)
                axes[i].set_title(title)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.save_dir / 'performance_dashboard.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance dashboard saved to {save_path}")
        
        return fig
    
    def create_training_summary_report(
        self,
        training_stats: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive training summary report.
        
        Args:
            training_stats: Dictionary containing training statistics
            save_path: Path to save the report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "# Multimodal LLM Training Summary Report",
            "=" * 50,
            "",
            f"**Training Completed**: {training_stats.get('completion_time', 'Unknown')}",
            f"**Total Training Time**: {training_stats.get('total_time', 'Unknown')}",
            f"**Epochs Completed**: {training_stats.get('epochs_completed', 'Unknown')}",
            f"**Best Validation Loss**: {training_stats.get('best_val_loss', 'Unknown')}",
            "",
            "## Model Configuration",
            "-" * 20,
        ]
        
        # Add model config
        model_config = training_stats.get('model_config', {})
        for key, value in model_config.items():
            report_lines.append(f"- **{key}**: {value}")
        
        report_lines.extend([
            "",
            "## Training Configuration",
            "-" * 25,
        ])
        
        # Add training config
        training_config = training_stats.get('training_config', {})
        for key, value in training_config.items():
            report_lines.append(f"- **{key}**: {value}")
        
        report_lines.extend([
            "",
            "## Final Metrics",
            "-" * 15,
        ])
        
        # Add final metrics
        final_metrics = training_stats.get('final_metrics', {})
        for metric_name, metric_value in final_metrics.items():
            if isinstance(metric_value, float):
                report_lines.append(f"- **{metric_name}**: {metric_value:.6f}")
            else:
                report_lines.append(f"- **{metric_name}**: {metric_value}")
        
        # Join all lines
        report_content = "\n".join(report_lines)
        
        # Save report
        if save_path is None:
            save_path = self.save_dir / 'training_summary_report.md'
        
        with open(save_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Training summary report saved to {save_path}")
        
        return report_content


# Standalone plotting functions
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names)
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage and testing
if __name__ == "__main__":
    # Test visualization utilities
    visualizer = TrainingVisualizer('/tmp/test_plots')
    
    # Mock data
    train_losses = [2.5, 2.1, 1.8, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95]
    val_losses = [2.3, 2.0, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
    
    train_metrics = {
        'accuracy': [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8],
        'perplexity': [12.0, 8.0, 6.0, 5.0, 4.0, 3.7, 3.3, 3.0, 2.7, 2.5]
    }
    
    val_metrics = {
        'accuracy': [0.25, 0.35, 0.45, 0.55, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75],
        'perplexity': [15.0, 10.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.2, 3.0]
    }
    
    # Test training curves
    fig1 = visualizer.plot_training_curves(
        train_losses, val_losses, train_metrics, val_metrics
    )
    
    # Test attention heatmap
    attention_weights = torch.randn(1, 8, 32, 16)  # [batch, heads, seq_out, seq_in]
    fig2 = visualizer.plot_attention_heatmap(attention_weights)
    
    # Test embedding space
    ts_embeddings = torch.randn(50, 512)
    text_embeddings = torch.randn(50, 512)
    fig3 = visualizer.plot_embedding_space(ts_embeddings, text_embeddings, method='pca')
    
    # Test performance dashboard
    metrics_history = {
        'loss': train_losses,
        'val_loss': val_losses,
        'accuracy': train_metrics['accuracy'],
        'perplexity': train_metrics['perplexity'],
        'cosine_similarity': [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.62, 0.65, 0.67],
        'learning_rate': [5e-5, 4.5e-5, 4e-5, 3.5e-5, 3e-5, 2.5e-5, 2e-5, 1.5e-5, 1e-5, 5e-6]
    }
    
    fig4 = visualizer.plot_model_performance_dashboard(metrics_history)
    
    # Close all figures
    plt.close('all')
    
    print("Visualization utilities implementation completed successfully!")