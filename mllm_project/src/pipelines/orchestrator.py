"""
Pipeline Orchestrator

This module provides orchestration for running the complete end-to-end pipeline.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

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

# Import pipeline modules with fallbacks
try:
    from .exploration_pipeline import DataExplorationPipeline
except ImportError:
    try:
        from exploration_pipeline import DataExplorationPipeline
    except ImportError:
        DataExplorationPipeline = None

try:
    from .training_pipeline import TrainingPipeline
except ImportError:
    try:
        from training_pipeline import TrainingPipeline
    except ImportError:
        TrainingPipeline = None

try:
    from .evaluation_pipeline import EvaluationPipeline
except ImportError:
    try:
        from evaluation_pipeline import EvaluationPipeline
    except ImportError:
        EvaluationPipeline = None

try:
    from .inference_pipeline import InferencePipeline
except ImportError:
    try:
        from inference_pipeline import InferencePipeline
    except ImportError:
        InferencePipeline = None

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the execution of multiple pipeline stages.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str, verbose: bool = False):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Pipeline configuration dictionary
            output_dir: Directory for pipeline outputs
            verbose: Enable verbose logging
        """
        # Check if any pipelines are available
        if not any([DataExplorationPipeline, TrainingPipeline, EvaluationPipeline, InferencePipeline]):
            raise ImportError("No pipelines available. Check pipeline imports and dependencies.")
        
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.stages = ['explore', 'train', 'evaluate', 'demo']
        self.results = {}
        self.state_file = self.output_dir / 'pipeline_state.json'
        
    def run(self, stages: Optional[List[str]] = None,
           resume: bool = False) -> Dict[str, Any]:
        """
        Run the pipeline with specified stages.
        
        Args:
            stages: List of stages to run (default: all stages)
            resume: Resume from previous run
            
        Returns:
            Dictionary containing results from all stages
        """
        logger.info("Starting pipeline orchestration...")
        
        # Determine stages to run
        if stages is None:
            stages = self.stages
        
        # Load previous state if resuming
        if resume and self.state_file.exists():
            with open(self.state_file, 'r') as f:
                previous_state = json.load(f)
                self.results = previous_state.get('results', {})
                completed_stages = previous_state.get('completed_stages', [])
                stages = [s for s in stages if s not in completed_stages]
                logger.info(f"Resuming pipeline. Completed stages: {completed_stages}")
        
        # Track pipeline execution
        pipeline_start_time = time.time()
        completed_stages = list(self.results.keys())
        
        try:
            # Execute each stage
            for stage in stages:
                logger.info(f"\n{'='*60}")
                logger.info(f"Executing stage: {stage.upper()}")
                logger.info(f"{'='*60}")
                
                stage_start_time = time.time()
                
                if stage == 'explore':
                    stage_results = self._run_exploration()
                elif stage == 'train':
                    stage_results = self._run_training()
                elif stage == 'evaluate':
                    stage_results = self._run_evaluation()
                elif stage == 'demo':
                    stage_results = self._run_demo()
                else:
                    logger.warning(f"Unknown stage: {stage}")
                    continue
                
                stage_time = time.time() - stage_start_time
                
                # Store results
                self.results[stage] = {
                    'status': 'success',
                    'execution_time': stage_time,
                    'timestamp': datetime.now().isoformat(),
                    'results': stage_results,
                    'summary': self._get_stage_summary(stage, stage_results)
                }
                
                completed_stages.append(stage)
                
                # Save state after each stage
                self._save_state(completed_stages)
                
                logger.info(f"Stage '{stage}' completed in {stage_time/60:.2f} minutes")
                
        except Exception as e:
            logger.error(f"Pipeline failed at stage '{stage}': {e}")
            self.results[stage] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self._save_state(completed_stages)
            raise
        
        # Calculate total execution time
        total_time = time.time() - pipeline_start_time
        
        # Create pipeline summary
        pipeline_summary = {
            'total_execution_time': total_time,
            'stages_executed': stages,
            'stages_completed': completed_stages,
            'completion_timestamp': datetime.now().isoformat(),
            'stage_results': self.results
        }
        
        # Save final results
        results_path = self.output_dir / 'pipeline_results.json'
        with open(results_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        # Generate pipeline report
        self._generate_pipeline_report(pipeline_summary)
        
        logger.info(f"\nPipeline completed in {total_time/3600:.2f} hours")
        logger.info(f"Results saved to {results_path}")
        
        return pipeline_summary
    
    def _run_exploration(self) -> Dict[str, Any]:
        """Run data exploration stage."""
        logger.info("Running data exploration...")
        
        config = self.config.get('explore', {})
        
        # Get configuration
        data_dir = config.get('data_dir', self.config.get('data', {}).get('data_dir'))
        output_dir = self.output_dir / 'exploration'
        sample_size = config.get('sample_size', 1000)
        generate_report = config.get('generate_report', True)
        
        # Create exploration pipeline
        pipeline = DataExplorationPipeline(
            config=self.config,
            output_dir=str(output_dir),
            cache_dir=str(self.output_dir / 'cache')
        )
        
        # Run exploration
        results = pipeline.run(
            sample_size=sample_size,
            generate_report=generate_report
        )
        
        return results
    
    def _run_training(self) -> Dict[str, Any]:
        """Run model training stage."""
        logger.info("Running model training...")
        
        config = self.config.get('train', {})
        
        # Get configuration
        checkpoint_dir = self.output_dir / 'checkpoints'
        experiment_name = config.get('experiment_name', 'pipeline_training')
        epochs = config.get('epochs', self.config.get('training', {}).get('epochs', 10))
        
        # Update training config
        training_config = self.config.copy()
        if 'training' in training_config:
            training_config['training']['epochs'] = epochs
        
        # Create training pipeline
        pipeline = TrainingPipeline(
            config=training_config,
            checkpoint_dir=str(checkpoint_dir),
            experiment_name=experiment_name,
            use_wandb=config.get('use_wandb', False)
        )
        
        # Run training
        results = pipeline.run(
            resume_from=config.get('resume_from'),
            verbose=self.verbose
        )
        
        return results
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """Run model evaluation stage."""
        logger.info("Running model evaluation...")
        
        config = self.config.get('evaluate', {})
        
        # Get model path from training results or config
        if 'train' in self.results:
            model_path = self.output_dir / 'checkpoints' / 'best_model.pt'
        else:
            model_path = config.get('model_path')
            if not model_path:
                raise ValueError("No model path available for evaluation")
        
        # Get configuration
        output_dir = self.output_dir / 'evaluation'
        test_split = config.get('test_split', 'test')
        generate_plots = config.get('generate_plots', True)
        save_predictions = config.get('save_predictions', False)
        
        # Create evaluation pipeline
        pipeline = EvaluationPipeline(
            config=self.config,
            model_path=str(model_path),
            output_dir=str(output_dir)
        )
        
        # Run evaluation
        results = pipeline.run(
            test_split=test_split,
            generate_plots=generate_plots,
            save_predictions=save_predictions
        )
        
        return results
    
    def _run_demo(self) -> Dict[str, Any]:
        """Run inference demonstration stage."""
        logger.info("Running inference demo...")
        
        config = self.config.get('demo', {})
        
        # Get model path
        if 'train' in self.results:
            model_path = self.output_dir / 'checkpoints' / 'best_model.pt'
        else:
            model_path = config.get('model_path')
            if not model_path:
                raise ValueError("No model path available for demo")
        
        # Get configuration
        demo_dir = self.output_dir / 'demo'
        num_examples = config.get('num_examples', 10)
        temperature = config.get('temperature', 0.8)
        streaming = config.get('streaming', False)
        
        # Create inference pipeline
        pipeline = InferencePipeline(
            config=self.config,
            model_path=str(model_path),
            demo_dir=str(demo_dir)
        )
        
        # Run demo
        results = pipeline.run_demo(
            num_examples=num_examples,
            streaming=streaming,
            temperature=temperature
        )
        
        return results
    
    def _get_stage_summary(self, stage: str, results: Dict[str, Any]) -> str:
        """Get summary for a stage."""
        if stage == 'explore':
            quality_score = results.get('quality_assessment', {}).get('quality_score', 0)
            return f"Data quality score: {quality_score:.1f}%"
        
        elif stage == 'train':
            epochs = results.get('epochs_completed', 0)
            final_loss = results.get('final_val_loss', 'N/A')
            return f"Trained for {epochs} epochs, final val loss: {final_loss}"
        
        elif stage == 'evaluate':
            metrics = results.get('quantitative', {}).get('metrics', {})
            accuracy = metrics.get('accuracy', 0)
            return f"Test accuracy: {accuracy:.3f}"
        
        elif stage == 'demo':
            num_examples = len(results.get('examples', []))
            avg_time = results.get('performance_stats', {}).get('avg_generation_time', 0)
            return f"Generated {num_examples} examples, avg time: {avg_time:.3f}s"
        
        return "Stage completed"
    
    def _save_state(self, completed_stages: List[str]):
        """Save pipeline state for resuming."""
        state = {
            'completed_stages': completed_stages,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _generate_pipeline_report(self, summary: Dict[str, Any]):
        """Generate comprehensive pipeline report."""
        report_path = self.output_dir / 'pipeline_report.md'
        
        # Create markdown report
        report_lines = [
            "# Pipeline Execution Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- **Total Execution Time**: {summary['total_execution_time']/3600:.2f} hours",
            f"- **Stages Executed**: {', '.join(summary['stages_executed'])}",
            f"- **Stages Completed**: {', '.join(summary['stages_completed'])}",
            "",
            "## Stage Results",
            ""
        ]
        
        for stage in summary['stages_executed']:
            if stage in summary['stage_results']:
                stage_data = summary['stage_results'][stage]
                status_icon = "✅" if stage_data['status'] == 'success' else "❌"
                
                report_lines.extend([
                    f"### {status_icon} {stage.upper()}",
                    f"- **Status**: {stage_data['status']}",
                    f"- **Execution Time**: {stage_data.get('execution_time', 0)/60:.2f} minutes",
                    f"- **Summary**: {stage_data.get('summary', 'N/A')}",
                    ""
                ])
                
                # Add stage-specific details
                if stage == 'explore' and 'results' in stage_data:
                    results = stage_data['results']
                    if 'quality_assessment' in results:
                        qa = results['quality_assessment']
                        report_lines.extend([
                            "#### Data Quality",
                            f"- Total samples: {qa.get('total_samples', 0)}",
                            f"- Valid samples: {qa.get('valid_samples', 0)}",
                            f"- Quality score: {qa.get('quality_score', 0):.1f}%",
                            ""
                        ])
                
                elif stage == 'train' and 'results' in stage_data:
                    results = stage_data['results']
                    report_lines.extend([
                        "#### Training Metrics",
                        f"- Epochs completed: {results.get('epochs_completed', 0)}",
                        f"- Best validation loss: {results.get('best_val_loss', 'N/A')}",
                        f"- Final train loss: {results.get('final_train_loss', 'N/A')}",
                        f"- Final val loss: {results.get('final_val_loss', 'N/A')}",
                        ""
                    ])
                
                elif stage == 'evaluate' and 'results' in stage_data:
                    results = stage_data['results']
                    if 'quantitative' in results:
                        metrics = results['quantitative'].get('metrics', {})
                        report_lines.extend([
                            "#### Evaluation Metrics",
                            f"- Perplexity: {metrics.get('perplexity', 0):.3f}",
                            f"- Accuracy: {metrics.get('accuracy', 0):.3f}",
                            f"- Top-5 Accuracy: {metrics.get('top_k_accuracy', 0):.3f}",
                            ""
                        ])
                
                elif stage == 'demo' and 'results' in stage_data:
                    results = stage_data['results']
                    if 'performance_stats' in results:
                        stats = results['performance_stats']
                        report_lines.extend([
                            "#### Demo Performance",
                            f"- Examples generated: {len(results.get('examples', []))}",
                            f"- Avg generation time: {stats.get('avg_generation_time', 0):.3f}s",
                            f"- Throughput: {stats.get('throughput', 0):.2f} samples/sec",
                            ""
                        ])
        
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        # Add recommendations based on results
        recommendations = []
        
        # Check data quality
        if 'explore' in summary['stage_results']:
            quality_score = summary['stage_results']['explore'].get('results', {}).get('quality_assessment', {}).get('quality_score', 0)
            if quality_score < 80:
                recommendations.append("- Consider data cleaning to improve quality score")
        
        # Check training performance
        if 'train' in summary['stage_results']:
            train_results = summary['stage_results']['train'].get('results', {})
            if train_results.get('epochs_completed', 0) < train_results.get('total_epochs', 10):
                recommendations.append("- Training stopped early - review early stopping criteria")
        
        # Check evaluation metrics
        if 'evaluate' in summary['stage_results']:
            metrics = summary['stage_results']['evaluate'].get('results', {}).get('quantitative', {}).get('metrics', {})
            if metrics.get('accuracy', 0) < 0.5:
                recommendations.append("- Low accuracy detected - consider hyperparameter tuning")
        
        if not recommendations:
            recommendations.append("- Pipeline executed successfully with good performance")
        
        report_lines.extend(recommendations)
        
        report_lines.extend([
            "",
            "## Next Steps",
            "1. Review detailed results in stage-specific output directories",
            "2. Analyze visualizations and plots generated",
            "3. Consider recommendations for improvements",
            "4. Deploy model if performance meets requirements",
            "",
            "---",
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Pipeline report generated: {report_path}")
