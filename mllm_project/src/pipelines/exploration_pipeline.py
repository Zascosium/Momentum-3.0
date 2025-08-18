"""
Data Exploration Pipeline

This module implements the data exploration pipeline that corresponds to
notebook 01_data_exploration.py functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Databricks compatibility
if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
    # Try to find the project root in Databricks
    current_path = Path(__file__).parent
    while current_path != current_path.parent:
        if (current_path / 'src').exists():
            sys.path.insert(0, str(current_path / 'src'))
            break
        current_path = current_path.parent

# Import with fallbacks for different execution contexts
try:
    from data.preprocessing import TimeSeriesPreprocessor, TextPreprocessor
except ImportError:
    try:
        # Try absolute import from src
        import sys
        from pathlib import Path
        src_path = Path(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from data.preprocessing import TimeSeriesPreprocessor, TextPreprocessor
    except ImportError:
        # Final fallback - try direct import
        try:
            import preprocessing
            TimeSeriesPreprocessor = preprocessing.TimeSeriesPreprocessor
            TextPreprocessor = preprocessing.TextPreprocessor
        except ImportError as e:
            logger.error(f"Failed to import preprocessing classes: {e}")
            # Create dummy classes for exploration
            class TimeSeriesPreprocessor:
                def __init__(self, config=None):
                    pass
            class TextPreprocessor:
                def __init__(self, config=None):
                    pass
# Import utilities with fallbacks
try:
    from utils.visualization import TrainingVisualizer
    from utils.config_loader import load_config_for_training
except ImportError:
    try:
        # Try with sys.path already updated above
        from utils.visualization import TrainingVisualizer
        from utils.config_loader import load_config_for_training
    except ImportError as e:
        logger.warning(f"Could not import utilities: {e}")
        # Create dummy classes
        class TrainingVisualizer:
            def __init__(self, output_dir):
                self.output_dir = output_dir
        def load_config_for_training():
            return {}

logger = logging.getLogger(__name__)


class DataExplorationPipeline:
    """
    Pipeline for comprehensive data exploration and analysis.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str, cache_dir: Optional[str] = None):
        """
        Initialize the exploration pipeline.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for output files
            cache_dir: Optional cache directory
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.plots_dir = self.output_dir / 'plots'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.visualizer = TrainingVisualizer(str(self.plots_dir))
        
    def run(self, sample_size: int = 1000, generate_report: bool = True) -> Dict[str, Any]:
        """
        Run the complete exploration pipeline.
        
        Args:
            sample_size: Number of samples to analyze
            generate_report: Whether to generate HTML report
            
        Returns:
            Dictionary containing exploration results
        """
        logger.info("Starting data exploration pipeline...")
        
        # Step 1: Load and analyze dataset structure
        logger.info("Step 1: Analyzing dataset structure...")
        dataset_stats = self._analyze_dataset_structure(sample_size)
        self.results['dataset_structure'] = dataset_stats
        
        # Step 2: Analyze time series characteristics
        logger.info("Step 2: Analyzing time series characteristics...")
        ts_analysis = self._analyze_time_series(dataset_stats.get('samples', []))
        self.results['time_series_analysis'] = ts_analysis
        
        # Step 3: Analyze text data
        logger.info("Step 3: Analyzing text data...")
        text_analysis = self._analyze_text_data(dataset_stats.get('samples', []))
        self.results['text_analysis'] = text_analysis
        
        # Step 4: Assess data quality
        logger.info("Step 4: Assessing data quality...")
        quality_assessment = self._assess_data_quality(dataset_stats.get('samples', []))
        self.results['quality_assessment'] = quality_assessment
        
        # Step 5: Analyze multimodal alignment
        logger.info("Step 5: Analyzing multimodal alignment...")
        alignment_analysis = self._analyze_multimodal_alignment(dataset_stats.get('samples', []))
        self.results['alignment_analysis'] = alignment_analysis
        
        # Step 6: Generate visualizations
        logger.info("Step 6: Generating visualizations...")
        self._generate_visualizations()
        
        # Step 7: Generate report
        if generate_report:
            logger.info("Step 7: Generating exploration report...")
            report_path = self._generate_report()
            self.results['report_path'] = str(report_path)
        
        # Save results
        results_path = self.output_dir / 'exploration_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Exploration completed. Results saved to {results_path}")
        
        return self.results
    
    def _analyze_dataset_structure(self, sample_size: int) -> Dict[str, Any]:
        """Analyze dataset structure and load samples."""
        try:
            # Load real data from configured data source
            data_config = self.config.get('data', {})
            data_path = data_config.get('data_dir', './data')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data directory not found: {data_path}")
            
            # Load actual dataset
            # This should be implemented based on your specific data format
            raise NotImplementedError("Real data loading not implemented. Please implement data loading for your specific dataset format.")
            
        except Exception as e:
            logger.error(f"Dataset structure analysis failed: {e}")
            return {'error': str(e), 'samples': []}
    
    def _analyze_time_series(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze time series characteristics."""
        if not samples:
            return {}
        
        ts_lengths = [s['ts_length'] for s in samples if 'ts_length' in s]
        n_features = [s.get('n_features', 1) for s in samples]
        
        analysis = {
            'length_distribution': {
                'mean': np.mean(ts_lengths),
                'median': np.median(ts_lengths),
                'std': np.std(ts_lengths),
                'min': np.min(ts_lengths),
                'max': np.max(ts_lengths)
            },
            'feature_distribution': {
                'mean': np.mean(n_features),
                'median': np.median(n_features),
                'unique_counts': list(set(n_features))
            },
            'temporal_patterns': self._analyze_temporal_patterns(samples[:10])
        }
        
        return analysis
    
    def _analyze_temporal_patterns(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in time series."""
        patterns = {
            'trend_detected': 0,
            'seasonal_detected': 0,
            'stationary': 0
        }
        
        for sample in samples:
            if 'time_series' in sample:
                ts = sample['time_series']
                if isinstance(ts, np.ndarray) and len(ts) > 0:
                    # Simple pattern detection (placeholder)
                    if np.mean(ts[-10:]) > np.mean(ts[:10]):
                        patterns['trend_detected'] += 1
                    # Add more sophisticated pattern detection as needed
        
        return patterns
    
    def _analyze_text_data(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze text data characteristics."""
        if not samples:
            return {}
        
        texts = [s['text'] for s in samples if 'text' in s]
        
        if not texts:
            return {}
        
        text_lengths = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]
        
        analysis = {
            'num_samples': len(texts),
            'char_length_stats': {
                'mean': np.mean(text_lengths),
                'median': np.median(text_lengths),
                'std': np.std(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths)
            },
            'word_count_stats': {
                'mean': np.mean(word_counts),
                'median': np.median(word_counts),
                'std': np.std(word_counts),
                'min': np.min(word_counts),
                'max': np.max(word_counts)
            },
            'sample_texts': texts[:5]
        }
        
        return analysis
    
    def _assess_data_quality(self, samples: List[Dict]) -> Dict[str, Any]:
        """Assess data quality."""
        if not samples:
            return {}
        
        quality_metrics = {
            'total_samples': len(samples),
            'valid_samples': 0,
            'missing_time_series': 0,
            'missing_text': 0,
            'length_outliers': 0
        }
        
        min_ts_length = self.config.get('quality_control', {}).get('min_time_series_length', 50)
        max_ts_length = self.config.get('quality_control', {}).get('max_time_series_length', 1000)
        
        for sample in samples:
            is_valid = True
            
            # Check time series
            if 'time_series' not in sample or sample.get('ts_length', 0) == 0:
                quality_metrics['missing_time_series'] += 1
                is_valid = False
            elif sample['ts_length'] < min_ts_length or sample['ts_length'] > max_ts_length:
                quality_metrics['length_outliers'] += 1
                is_valid = False
            
            # Check text
            if 'text' not in sample or not sample['text']:
                quality_metrics['missing_text'] += 1
                is_valid = False
            
            if is_valid:
                quality_metrics['valid_samples'] += 1
        
        quality_metrics['quality_score'] = (
            quality_metrics['valid_samples'] / quality_metrics['total_samples'] * 100
            if quality_metrics['total_samples'] > 0 else 0
        )
        
        return quality_metrics
    
    def _analyze_multimodal_alignment(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze alignment between time series and text modalities."""
        if not samples:
            return {}
        
        alignment_stats = {
            'samples_analyzed': len(samples),
            'domain_alignment': {},
            'length_correlation': None
        }
        
        # Analyze domain-specific alignment
        for domain in self.config.get('domains', {}).get('included', []):
            domain_samples = [s for s in samples if s.get('domain') == domain]
            if domain_samples:
                alignment_stats['domain_alignment'][domain] = {
                    'count': len(domain_samples),
                    'avg_ts_length': np.mean([s.get('ts_length', 0) for s in domain_samples]),
                    'avg_text_length': np.mean([len(s.get('text', '')) for s in domain_samples])
                }
        
        # Calculate correlation between TS length and text length
        ts_lengths = [s.get('ts_length', 0) for s in samples]
        text_lengths = [len(s.get('text', '')) for s in samples]
        
        if len(ts_lengths) > 1 and len(text_lengths) > 1:
            correlation = np.corrcoef(ts_lengths[:len(text_lengths)], text_lengths[:len(ts_lengths)])[0, 1]
            alignment_stats['length_correlation'] = float(correlation)
        
        return alignment_stats
    
    def _generate_visualizations(self):
        """Generate exploration visualizations."""
        logger.info("Generating visualizations...")
        
        # Create sample plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Domain distribution
        if 'dataset_structure' in self.results:
            domain_dist = self.results['dataset_structure'].get('domain_distribution', {})
            if domain_dist:
                axes[0, 0].bar(domain_dist.keys(), domain_dist.values())
                axes[0, 0].set_title('Sample Distribution by Domain')
                axes[0, 0].set_xlabel('Domain')
                axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Time series length distribution
        if 'time_series_analysis' in self.results:
            samples = self.results.get('dataset_structure', {}).get('samples', [])
            if samples:
                ts_lengths = [s['ts_length'] for s in samples if 'ts_length' in s]
                axes[0, 1].hist(ts_lengths, bins=30, alpha=0.7)
                axes[0, 1].set_title('Time Series Length Distribution')
                axes[0, 1].set_xlabel('Length')
                axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Text length distribution
        if 'text_analysis' in self.results:
            samples = self.results.get('dataset_structure', {}).get('samples', [])
            if samples:
                text_lengths = [len(s.get('text', '')) for s in samples]
                axes[1, 0].hist(text_lengths, bins=30, alpha=0.7, color='orange')
                axes[1, 0].set_title('Text Length Distribution')
                axes[1, 0].set_xlabel('Character Count')
                axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Quality metrics
        if 'quality_assessment' in self.results:
            quality = self.results['quality_assessment']
            metrics = ['valid_samples', 'missing_time_series', 'missing_text', 'length_outliers']
            values = [quality.get(m, 0) for m in metrics]
            axes[1, 1].bar(metrics, values, color=['green', 'red', 'red', 'orange'])
            axes[1, 1].set_title('Data Quality Metrics')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'exploration_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def _generate_report(self) -> Path:
        """Generate HTML exploration report."""
        report_path = self.output_dir / 'exploration_report.html'
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Exploration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .bad {{ color: red; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Data Exploration Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Dataset Overview</h2>
            <div class="metric">
                <strong>Total Samples:</strong> {self.results.get('dataset_structure', {}).get('total_samples', 0)}
            </div>
            <div class="metric">
                <strong>Domains:</strong> {', '.join(self.results.get('dataset_structure', {}).get('domains', []))}
            </div>
            
            <h2>Data Quality</h2>
            <div class="metric">
                <strong>Quality Score:</strong> 
                <span class="{'good' if self.results.get('quality_assessment', {}).get('quality_score', 0) > 80 else 'warning'}">
                    {self.results.get('quality_assessment', {}).get('quality_score', 0):.1f}%
                </span>
            </div>
            
            <h2>Visualizations</h2>
            <img src="plots/exploration_summary.png" alt="Exploration Summary">
            
            <h2>Recommendations</h2>
            <ul>
                <li>Review data quality metrics and address any issues</li>
                <li>Consider data augmentation for underrepresented domains</li>
                <li>Validate preprocessing pipeline with larger sample</li>
            </ul>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_path}")
        return report_path
