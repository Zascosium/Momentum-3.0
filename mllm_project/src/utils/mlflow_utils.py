"""
MLflow integration utilities for experiment tracking and model management.
Provides comprehensive logging, model registration, and experiment management for Databricks.
"""

import os
import json
import pickle
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class MLflowExperimentManager:
    """
    Comprehensive MLflow experiment management for multimodal training.
    Handles experiment creation, run management, and artifact logging.
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize MLflow experiment manager.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Artifact storage location
            tags: Default tags for the experiment
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.default_tags = tags or {}
        
        # Check if MLflow is available
        if mlflow is None:
            raise ImportError("MLflow not available - install mlflow package")
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize client
        if MlflowClient is not None:
            self.client = MlflowClient()
        else:
            self.client = None
            logger.warning("MlflowClient not available - some features disabled")
        
        # Create or get experiment
        self.experiment_id = self._setup_experiment()
        
        logger.info(f"MLflow experiment manager initialized for: {experiment_name}")
    
    def _setup_experiment(self) -> str:
        """Create or get MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=self.artifact_location,
                    tags=self.default_tags
                )
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> mlflow.ActiveRun:
        """
        Start an MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags for the run
            nested: Whether this is a nested run
            
        Returns:
            Active MLflow run
        """
        # Combine default tags with run-specific tags
        all_tags = {**self.default_tags}
        if tags:
            all_tags.update(tags)
        
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=all_tags,
            nested=nested
        )
        
        logger.info(f"Started MLflow run: {run.info.run_name} (ID: {run.info.run_id})")
        return run
    
    def log_config(self, config: Dict[str, Any], prefix: str = ""):
        """
        Log configuration parameters to MLflow.
        
        Args:
            config: Configuration dictionary
            prefix: Prefix for parameter names
        """
        def flatten_dict(d, parent_key='', sep='_'):
            """Flatten nested dictionary."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    # Convert non-string values to strings
                    if not isinstance(v, (str, int, float, bool)) and v is not None:
                        v = str(v)
                    items.append((new_key, v))
            return dict(items)
        
        # Flatten configuration for MLflow
        flat_config = flatten_dict(config)
        
        # Add prefix if specified
        if prefix:
            flat_config = {f"{prefix}_{k}": v for k, v in flat_config.items()}
        
        # Log parameters
        for key, value in flat_config.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log parameter {key}: {e}")
        
        # Also save as artifact
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f, indent=2, default=str)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "config")
            os.unlink(temp_path)
            
        except Exception as e:
            logger.warning(f"Failed to log config artifact: {e}")
    
    def log_metrics_batch(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[int] = None
    ):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Step number
            timestamp: Timestamp
        """
        for metric_name, metric_value in metrics.items():
            try:
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                    mlflow.log_metric(metric_name, metric_value, step=step, timestamp=timestamp)
                else:
                    logger.warning(f"Skipping invalid metric {metric_name}: {metric_value}")
            except Exception as e:
                logger.warning(f"Failed to log metric {metric_name}: {e}")
    
    def log_model_summary(self, model: torch.nn.Module):
        """
        Log model summary and statistics.
        
        Args:
            model: PyTorch model
        """
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Log parameter counts
            mlflow.log_metric("total_parameters", total_params)
            mlflow.log_metric("trainable_parameters", trainable_params)
            mlflow.log_metric("frozen_parameters", total_params - trainable_params)
            
            # Calculate memory usage (rough estimate)
            param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            mlflow.log_metric("estimated_memory_mb", param_memory_mb)
            
            # Create model summary
            summary_lines = [
                f"Model Summary",
                f"=" * 50,
                f"Total Parameters: {total_params:,}",
                f"Trainable Parameters: {trainable_params:,}",
                f"Frozen Parameters: {total_params - trainable_params:,}",
                f"Estimated Memory: {param_memory_mb:.2f} MB",
                f"",
                f"Model Architecture:",
                f"-" * 20,
                str(model)
            ]
            
            summary_text = "\n".join(summary_lines)
            
            # Save as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(summary_text)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "model_info")
            os.unlink(temp_path)
            
            logger.info(f"Model summary logged: {total_params:,} total parameters")
            
        except Exception as e:
            logger.error(f"Failed to log model summary: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset information and statistics.
        
        Args:
            dataset_info: Dataset information dictionary
        """
        try:
            # Log dataset metrics
            for key, value in dataset_info.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"dataset_{key}", value)
                else:
                    mlflow.log_param(f"dataset_{key}", str(value))
            
            # Save detailed dataset info as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dataset_info, f, indent=2, default=str)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "dataset_info")
            os.unlink(temp_path)
            
            logger.info("Dataset information logged")
            
        except Exception as e:
            logger.error(f"Failed to log dataset info: {e}")
    
    def log_training_plots(self, plots_dir: str):
        """
        Log training plots as artifacts.
        
        Args:
            plots_dir: Directory containing plot files
        """
        try:
            plots_path = Path(plots_dir)
            if plots_path.exists():
                # Log all plot files
                for plot_file in plots_path.glob("*.png"):
                    mlflow.log_artifact(str(plot_file), "plots")
                
                for plot_file in plots_path.glob("*.pdf"):
                    mlflow.log_artifact(str(plot_file), "plots")
                
                logger.info(f"Training plots logged from {plots_dir}")
            else:
                logger.warning(f"Plots directory not found: {plots_dir}")
                
        except Exception as e:
            logger.error(f"Failed to log training plots: {e}")
    
    def log_model_checkpoint(
        self,
        model: torch.nn.Module,
        model_name: str = "model",
        save_state_dict: bool = True,
        save_full_model: bool = False
    ):
        """
        Log model checkpoint to MLflow.
        
        Args:
            model: PyTorch model
            model_name: Name for the model artifact
            save_state_dict: Whether to save state dict
            save_full_model: Whether to save the complete model
        """
        try:
            if save_full_model:
                # Log full PyTorch model
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=f"{model_name}_full",
                    conda_env=None,  # Use current environment
                    serialization_format="pickle"
                )
            
            if save_state_dict:
                # Save state dict as artifact
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                    torch.save(model.state_dict(), f.name)
                    mlflow.log_artifact(f.name, f"{model_name}_state_dict")
                    os.unlink(f.name)
            
            logger.info(f"Model checkpoint logged: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model checkpoint: {e}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            description: Model description
            tags: Model tags
            
        Returns:
            Model version
        """
        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags,
                description=description
            )
            
            logger.info(f"Model registered: {model_name} v{result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple MLflow runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            DataFrame with run comparison
        """
        try:
            runs_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'run_name': run.info.run_name,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                }
                
                # Add metrics
                for key, value in run.data.metrics.items():
                    run_data[f"metric_{key}"] = value
                
                # Add parameters
                for key, value in run.data.params.items():
                    run_data[f"param_{key}"] = value
                
                runs_data.append(run_data)
            
            comparison_df = pd.DataFrame(runs_data)
            
            # Save comparison as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                comparison_df.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, "run_comparison")
                os.unlink(f.name)
            
            logger.info(f"Run comparison completed for {len(run_ids)} runs")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            raise
    
    def get_best_run(
        self,
        metric_name: str,
        ascending: bool = True,
        filter_string: str = ""
    ) -> mlflow.entities.Run:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Metric to optimize
            ascending: Whether lower values are better
            filter_string: MLflow filter string
            
        Returns:
            Best MLflow run
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )
            
            if runs.empty:
                raise ValueError("No runs found matching criteria")
            
            run_id = runs.iloc[0]['run_id']
            best_run = self.client.get_run(run_id)
            
            logger.info(f"Best run found: {run_id} ({metric_name}={runs.iloc[0][f'metrics.{metric_name}']})")
            return best_run
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            raise
    
    def create_experiment_report(self, output_path: str = None) -> str:
        """
        Create a comprehensive experiment report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        try:
            # Get all runs in experiment
            runs_df = mlflow.search_runs(experiment_ids=[self.experiment_id])
            
            if runs_df.empty:
                report = "No runs found in experiment."
            else:
                # Generate report
                report_lines = [
                    f"# MLflow Experiment Report: {self.experiment_name}",
                    "=" * 60,
                    "",
                    f"**Experiment ID**: {self.experiment_id}",
                    f"**Total Runs**: {len(runs_df)}",
                    f"**Generated**: {pd.Timestamp.now()}",
                    "",
                    "## Run Summary",
                    "-" * 15,
                    "",
                ]
                
                # Summary statistics
                status_counts = runs_df['status'].value_counts()
                for status, count in status_counts.items():
                    report_lines.append(f"- **{status}**: {count} runs")
                
                report_lines.extend([
                    "",
                    "## Best Runs by Metric",
                    "-" * 25,
                    ""
                ])
                
                # Find best runs for common metrics
                metric_columns = [col for col in runs_df.columns if col.startswith('metrics.')]
                
                for metric_col in metric_columns[:5]:  # Top 5 metrics
                    metric_name = metric_col.replace('metrics.', '')
                    
                    # Try to find best run (assume lower is better for loss-like metrics)
                    ascending = 'loss' in metric_name.lower() or 'error' in metric_name.lower()
                    
                    if not runs_df[metric_col].isna().all():
                        best_run = runs_df.loc[runs_df[metric_col].idxmin() if ascending else runs_df[metric_col].idxmax()]
                        best_value = best_run[metric_col]
                        
                        report_lines.append(f"- **{metric_name}**: {best_value:.6f} (Run: {best_run['run_name']})")
                
                # Add detailed run table
                report_lines.extend([
                    "",
                    "## Detailed Run Information",
                    "-" * 30,
                    "",
                    runs_df.to_string(index=False)
                ])
            
            report = "\n".join(report_lines)
            
            # Save report
            if output_path is None:
                output_path = f"experiment_report_{self.experiment_id}.md"
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            # Log as artifact if in active run
            try:
                mlflow.log_artifact(output_path, "reports")
            except:
                pass  # Not in active run
            
            logger.info(f"Experiment report saved to {output_path}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to create experiment report: {e}")
            raise


class MLflowModelManager:
    """
    Manages MLflow model lifecycle including versioning and deployment.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow model manager.
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
    
    def create_registered_model(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Create a new registered model.
        
        Args:
            name: Model name
            description: Model description
            tags: Model tags
        """
        try:
            self.client.create_registered_model(
                name=name,
                description=description,
                tags=tags
            )
            logger.info(f"Created registered model: {name}")
        except Exception as e:
            logger.error(f"Failed to create registered model: {e}")
            raise
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str,
        description: Optional[str] = None
    ):
        """
        Promote model to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            description: Transition description
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                description=description
            )
            logger.info(f"Promoted model {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def load_model_from_registry(
        self,
        model_name: str,
        stage: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Load model from MLflow registry.
        
        Args:
            model_name: Name of the registered model
            stage: Model stage ('Staging', 'Production', etc.)
            version: Specific model version
            
        Returns:
            Loaded model
        """
        try:
            if stage:
                model_uri = f"models:/{model_name}/{stage}"
            elif version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from registry: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            raise


# Convenience functions
def log_training_session(
    experiment_name: str,
    config: Dict[str, Any],
    model: torch.nn.Module,
    metrics_history: Dict[str, List[float]],
    plots_dir: Optional[str] = None,
    run_name: Optional[str] = None
) -> str:
    """
    Log a complete training session to MLflow.
    
    Args:
        experiment_name: MLflow experiment name
        config: Training configuration
        model: Trained model
        metrics_history: Training metrics history
        plots_dir: Directory containing training plots
        run_name: Name for the MLflow run
        
    Returns:
        MLflow run ID
    """
    manager = MLflowExperimentManager(experiment_name)
    
    with manager.start_run(run_name=run_name) as run:
        # Log configuration
        manager.log_config(config)
        
        # Log model summary
        manager.log_model_summary(model)
        
        # Log final metrics
        final_metrics = {}
        for metric_name, values in metrics_history.items():
            if values:
                final_metrics[f"final_{metric_name}"] = values[-1]
                final_metrics[f"best_{metric_name}"] = min(values) if 'loss' in metric_name else max(values)
        
        manager.log_metrics_batch(final_metrics)
        
        # Log model checkpoint
        manager.log_model_checkpoint(model, "final_model")
        
        # Log training plots if available
        if plots_dir:
            manager.log_training_plots(plots_dir)
        
        logger.info(f"Training session logged to MLflow: {run.info.run_id}")
        return run.info.run_id


# Example usage and testing
if __name__ == "__main__":
    # Test MLflow integration
    experiment_manager = MLflowExperimentManager(
        experiment_name="test_multimodal_llm",
        tags={"project": "multimodal_llm", "version": "1.0"}
    )
    
    # Mock configuration
    config = {
        "model": {
            "name": "multimodal_llm",
            "embedding_dim": 512
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 5e-5,
            "epochs": 10
        }
    }
    
    # Start a test run
    with experiment_manager.start_run(run_name="test_run") as run:
        # Log configuration
        experiment_manager.log_config(config)
        
        # Log mock metrics
        for epoch in range(5):
            metrics = {
                "train_loss": 2.0 - epoch * 0.3,
                "val_loss": 1.8 - epoch * 0.2,
                "accuracy": 0.5 + epoch * 0.1
            }
            experiment_manager.log_metrics_batch(metrics, step=epoch)
        
        # Create mock model
        model = torch.nn.Linear(100, 10)
        experiment_manager.log_model_summary(model)
        
        print(f"Test run completed: {run.info.run_id}")
    
    # Create experiment report
    report = experiment_manager.create_experiment_report()
    print("MLflow integration test completed successfully!")