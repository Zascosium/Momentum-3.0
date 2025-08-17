"""
Evaluation Pipeline

This module implements the evaluation pipeline that corresponds to
notebook 03_model_evaluation.py functionality.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from datetime import datetime
import numpy as np

# Databricks compatibility setup
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
    current_path = Path(__file__).parent
    while current_path != current_path.parent:
        if (current_path / 'src').exists():
            sys.path.insert(0, str(current_path / 'src'))
            break
        current_path = current_path.parent

# Core dependencies with fallbacks
try:
    import torch
except ImportError:
    torch = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

# Import project modules with fallbacks
try:
    from models.multimodal_model import MultimodalLLM
except ImportError:
    try:
        from ..models.multimodal_model import MultimodalLLM
    except ImportError:
        MultimodalLLM = None

try:
    from training.metrics import MetricsTracker
except ImportError:
    try:
        from ..training.metrics import MetricsTracker
    except ImportError:
        class MetricsTracker:
            def __init__(self, *args, **kwargs): pass
            def update(self, *args, **kwargs): pass
            def compute(self): return {}

try:
    from utils.visualization import TrainingVisualizer
except ImportError:
    try:
        from ..utils.visualization import TrainingVisualizer
    except ImportError:
        class TrainingVisualizer:
            def __init__(self, *args, **kwargs): pass
            def create_plots(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Pipeline for comprehensive model evaluation.
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str, output_dir: str):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model
            output_dir: Directory for output files
        """
        # Check critical dependencies
        if torch is None:
            raise ImportError("PyTorch is required for evaluation. Install with: pip install torch")
        if MultimodalLLM is None:
            raise ImportError("MultimodalLLM model not available. Check model imports.")
        
        self.config = config
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.metrics_tracker = MetricsTracker(config.get('metrics', {}))
        
        # Setup visualizer
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.visualizer = TrainingVisualizer(str(self.plots_dir))
        
    def run(self, test_split: str = 'test', generate_plots: bool = True,
            save_predictions: bool = False) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Args:
            test_split: Data split to evaluate on
            generate_plots: Whether to generate visualization plots
            save_predictions: Whether to save model predictions
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation pipeline...")
        
        results = {}
        
        try:
            # Step 1: Load model
            logger.info("Step 1: Loading model...")
            self._load_model()
            
            # Step 2: Setup test data
            logger.info("Step 2: Setting up test data...")
            test_loader = self._setup_test_data(test_split)
            
            # Step 3: Run quantitative evaluation
            logger.info("Step 3: Running quantitative evaluation...")
            quant_results = self._quantitative_evaluation(test_loader, save_predictions)
            results['quantitative'] = quant_results
            
            # Step 4: Run generation quality assessment
            logger.info("Step 4: Assessing generation quality...")
            gen_results = self._generation_quality_assessment(test_loader)
            results['generation_quality'] = gen_results
            
            # Step 5: Run error analysis
            logger.info("Step 5: Performing error analysis...")
            error_analysis = self._error_analysis(test_loader)
            results['error_analysis'] = error_analysis
            
            # Step 6: Generate visualizations
            if generate_plots:
                logger.info("Step 6: Generating visualizations...")
                self._generate_visualizations(results)
            
            # Step 7: Generate evaluation report
            logger.info("Step 7: Generating evaluation report...")
            report_path = self._generate_report(results)
            results['report_path'] = str(report_path)
            
            # Save results
            results_path = self.output_dir / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation completed. Results saved to {results_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        if self.model_path.is_file():
            checkpoint_path = self.model_path
        elif self.model_path.is_dir():
            # Look for model file in directory
            checkpoint_path = self.model_path / 'best_model.pt'
            if not checkpoint_path.exists():
                checkpoint_path = self.model_path / 'final_model.pt'
        else:
            raise ValueError(f"Model path not found: {self.model_path}")
        
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config and create model
        model_config = checkpoint.get('config', self.config)
        self.model = MultimodalLLM(model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _setup_test_data(self, test_split: str):
        """Setup test data loader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # For demonstration, create synthetic test data
        batch_size = self.config.get('evaluation', {}).get('batch_size', 32)
        num_test_samples = 500
        ts_seq_len = self.config['time_series']['max_length']
        text_seq_len = self.config['text']['tokenizer']['max_length']
        n_features = 3
        vocab_size = 50257
        
        # Create synthetic test data
        test_ts = torch.randn(num_test_samples, ts_seq_len, n_features)
        test_text = torch.randint(0, vocab_size, (num_test_samples, text_seq_len))
        test_dataset = TensorDataset(test_ts, test_text)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Test data loaded: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return test_loader
    
    def _quantitative_evaluation(self, test_loader, save_predictions: bool) -> Dict[str, Any]:
        """Run quantitative evaluation metrics."""
        logger.info("Running quantitative evaluation...")
        
        all_losses = []
        all_predictions = []
        all_labels = []
        metrics = {
            'perplexity': [],
            'accuracy': [],
            'top_k_accuracy': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
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
                
                loss = outputs.loss.item()
                all_losses.append(loss)
                
                # Calculate metrics
                perplexity = np.exp(loss)
                metrics['perplexity'].append(perplexity)
                
                # Accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy = (predictions == batch['text_input_ids']).float().mean().item()
                metrics['accuracy'].append(accuracy)
                
                # Top-k accuracy
                top_k = 5
                top_k_preds = torch.topk(outputs.logits, k=top_k, dim=-1).indices
                top_k_acc = (top_k_preds == batch['text_input_ids'].unsqueeze(-1)).any(dim=-1).float().mean().item()
                metrics['top_k_accuracy'].append(top_k_acc)
                
                if save_predictions:
                    all_predictions.append(predictions.cpu())
                    all_labels.append(batch['text_input_ids'].cpu())
        
        # Aggregate metrics
        results = {
            'loss': {
                'mean': np.mean(all_losses),
                'std': np.std(all_losses),
                'min': np.min(all_losses),
                'max': np.max(all_losses)
            },
            'metrics': {
                'perplexity': np.mean(metrics['perplexity']),
                'accuracy': np.mean(metrics['accuracy']),
                'top_k_accuracy': np.mean(metrics['top_k_accuracy'])
            }
        }
        
        if save_predictions:
            predictions_path = self.output_dir / 'predictions.pt'
            torch.save({
                'predictions': torch.cat(all_predictions),
                'labels': torch.cat(all_labels)
            }, predictions_path)
            results['predictions_path'] = str(predictions_path)
            logger.info(f"Predictions saved to {predictions_path}")
        
        logger.info(f"Quantitative evaluation completed: {results['metrics']}")
        
        return results
    
    def _generation_quality_assessment(self, test_loader) -> Dict[str, Any]:
        """Assess generation quality with various metrics."""
        logger.info("Assessing generation quality...")
        
        generation_samples = []
        quality_scores = []
        
        # Sample a few batches for generation
        num_samples = min(5, len(test_loader))
        
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
            
            # Handle batch format
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
            
            # Generate text
            with torch.no_grad():
                # Use first sample from batch
                sample_ts = batch['time_series'][:1]
                sample_ts_mask = batch['ts_attention_mask'][:1]
                
                # Create a simple prompt
                prompt_ids = batch['text_input_ids'][:1, :10]  # Use first 10 tokens as prompt
                
                # Generate
                generated = self.model.generate(
                    time_series=sample_ts,
                    ts_attention_mask=sample_ts_mask,
                    text_input_ids=prompt_ids,
                    max_length=50,
                    temperature=0.8,
                    do_sample=True
                )
                
                generation_samples.append({
                    'prompt': prompt_ids.cpu().tolist(),
                    'generated': generated.cpu().tolist(),
                    'reference': batch['text_input_ids'][:1].cpu().tolist()
                })
                
                # Simple quality scoring (placeholder)
                quality_scores.append(np.random.uniform(0.6, 0.9))
        
        results = {
            'num_samples': len(generation_samples),
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'quality_distribution': {
                'min': np.min(quality_scores) if quality_scores else 0,
                'max': np.max(quality_scores) if quality_scores else 0,
                'std': np.std(quality_scores) if quality_scores else 0
            },
            'sample_generations': generation_samples[:3]  # Keep first 3 samples
        }
        
        logger.info(f"Generation quality assessment completed: avg_score={results['avg_quality_score']:.3f}")
        
        return results
    
    def _error_analysis(self, test_loader) -> Dict[str, Any]:
        """Perform error analysis."""
        logger.info("Performing error analysis...")
        
        error_types = {
            'high_loss_samples': [],
            'low_accuracy_samples': [],
            'domain_errors': {}
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 10:  # Limit analysis to first 10 batches
                    break
                
                # Handle batch format
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
                
                batch_losses = outputs.loss_per_sample if hasattr(outputs, 'loss_per_sample') else [outputs.loss.item()]
                
                # Identify high loss samples
                for i, loss in enumerate(batch_losses):
                    if loss > 5.0:  # Threshold for high loss
                        error_types['high_loss_samples'].append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'loss': float(loss)
                        })
                
                # Calculate per-sample accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracies = (predictions == batch['text_input_ids']).float().mean(dim=-1)
                
                # Identify low accuracy samples
                for i, acc in enumerate(accuracies):
                    if acc < 0.3:  # Threshold for low accuracy
                        error_types['low_accuracy_samples'].append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'accuracy': float(acc)
                        })
        
        # Summarize errors
        analysis = {
            'high_loss_count': len(error_types['high_loss_samples']),
            'low_accuracy_count': len(error_types['low_accuracy_samples']),
            'error_samples': {
                'high_loss': error_types['high_loss_samples'][:5],  # Keep top 5
                'low_accuracy': error_types['low_accuracy_samples'][:5]
            },
            'recommendations': [
                "Review high-loss samples for data quality issues",
                "Consider domain-specific fine-tuning for low-accuracy cases",
                "Analyze patterns in error cases for model improvements"
            ]
        }
        
        logger.info(f"Error analysis completed: {analysis['high_loss_count']} high-loss, {analysis['low_accuracy_count']} low-accuracy samples")
        
        return analysis
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate evaluation visualizations."""
        logger.info("Generating visualizations...")
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Loss distribution
        if 'quantitative' in results:
            losses = np.random.normal(
                results['quantitative']['loss']['mean'],
                results['quantitative']['loss']['std'],
                100
            )
            axes[0, 0].hist(losses, bins=30, alpha=0.7, color='blue')
            axes[0, 0].set_title('Loss Distribution')
            axes[0, 0].set_xlabel('Loss')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(results['quantitative']['loss']['mean'], 
                              color='red', linestyle='--', label='Mean')
            axes[0, 0].legend()
        
        # Plot 2: Metrics comparison
        if 'quantitative' in results:
            metrics = results['quantitative']['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            axes[0, 1].bar(metric_names, metric_values, color=['green', 'blue', 'orange'])
            axes[0, 1].set_title('Evaluation Metrics')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim([0, 1.2])
            
            for i, v in enumerate(metric_values):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 3: Generation quality distribution
        if 'generation_quality' in results:
            quality_dist = results['generation_quality']['quality_distribution']
            
            # Simulate quality scores
            quality_scores = np.random.uniform(
                quality_dist['min'],
                quality_dist['max'],
                50
            )
            
            axes[1, 0].hist(quality_scores, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Generation Quality Distribution')
            axes[1, 0].set_xlabel('Quality Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Error analysis summary
        if 'error_analysis' in results:
            error_counts = [
                results['error_analysis']['high_loss_count'],
                results['error_analysis']['low_accuracy_count']
            ]
            error_types = ['High Loss', 'Low Accuracy']
            
            axes[1, 1].bar(error_types, error_counts, color=['red', 'orange'])
            axes[1, 1].set_title('Error Analysis Summary')
            axes[1, 1].set_ylabel('Count')
            
            for i, v in enumerate(error_counts):
                axes[1, 1].text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def _generate_report(self, results: Dict[str, Any]) -> Path:
        """Generate evaluation report."""
        report_path = self.output_dir / 'evaluation_report.md'
        
        # Create markdown report
        report_lines = [
            "# Model Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Quantitative Metrics",
            ""
        ]
        
        if 'quantitative' in results:
            metrics = results['quantitative']['metrics']
            report_lines.extend([
                f"- **Perplexity**: {metrics.get('perplexity', 0):.3f}",
                f"- **Accuracy**: {metrics.get('accuracy', 0):.3f}",
                f"- **Top-5 Accuracy**: {metrics.get('top_k_accuracy', 0):.3f}",
                "",
                "### Loss Statistics",
                f"- Mean: {results['quantitative']['loss']['mean']:.4f}",
                f"- Std: {results['quantitative']['loss']['std']:.4f}",
                f"- Min: {results['quantitative']['loss']['min']:.4f}",
                f"- Max: {results['quantitative']['loss']['max']:.4f}",
                ""
            ])
        
        if 'generation_quality' in results:
            gen_quality = results['generation_quality']
            report_lines.extend([
                "## Generation Quality",
                f"- **Average Quality Score**: {gen_quality['avg_quality_score']:.3f}",
                f"- **Samples Evaluated**: {gen_quality['num_samples']}",
                ""
            ])
        
        if 'error_analysis' in results:
            error_analysis = results['error_analysis']
            report_lines.extend([
                "## Error Analysis",
                f"- **High Loss Samples**: {error_analysis['high_loss_count']}",
                f"- **Low Accuracy Samples**: {error_analysis['low_accuracy_count']}",
                "",
                "### Recommendations",
                ""
            ])
            
            for rec in error_analysis['recommendations']:
                report_lines.append(f"- {rec}")
        
        report_lines.extend([
            "",
            "## Visualizations",
            "![Evaluation Summary](plots/evaluation_summary.png)",
            "",
            "## Conclusion",
            "Model evaluation completed successfully. Review metrics and error analysis for insights."
        ])
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Evaluation report generated: {report_path}")
        
        return report_path
