"""
Simplified Databricks Pipeline Runner
Complete multimodal LLM pipeline optimized for Databricks execution
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabricksPipeline:
    """
    Complete multimodal LLM pipeline for Databricks environment.
    Handles setup, training, evaluation, and inference in one streamlined interface.
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """Initialize the Databricks pipeline."""
        self.project_root = project_root or self._detect_project_root()
        self.setup_paths()
        self.config = self.load_config()
        
        # Setup directories
        self.checkpoint_dir = "/dbfs/mllm_checkpoints"
        self.output_dir = "/dbfs/mllm_outputs"
        self.data_dir = f"{self.project_root}/data/time_mmd"
        
        self._ensure_directories()
        
    def _detect_project_root(self) -> str:
        """Auto-detect project root in Databricks."""
        possible_roots = [
            "/Workspace/Repos/mllm_project",
            "/Workspace/Users/user@company.com/mllm_project",
            "/databricks/driver/mllm_project"
        ]
        
        for root in possible_roots:
            if os.path.exists(root):
                return root
                
        # Fallback: current working directory
        cwd = os.getcwd()
        if "mllm_project" in cwd:
            return cwd
        
        raise RuntimeError("Could not detect project root. Please specify project_root parameter.")
    
    def setup_paths(self):
        """Setup Python paths for imports."""
        src_path = f"{self.project_root}/src"
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
            
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Python paths updated")
    
    def load_config(self) -> Dict[str, Any]:
        """Load and merge all configuration files."""
        import yaml
        
        config_files = [
            "data_config.yaml",
            "model_config.yaml", 
            "pipeline_config.yaml",
            "training_config.yaml"
        ]
        
        merged_config = {}
        config_dir = f"{self.project_root}/config"
        
        for config_file in config_files:
            config_path = f"{config_dir}/{config_file}"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        merged_config.update(file_config)
        
        # Override for Databricks environment
        merged_config.update({
            'environment': 'databricks',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'checkpoint_dir': self.checkpoint_dir,
            'output_dir': self.output_dir,
            'data_dir': self.data_dir
        })
        
        return merged_config
    
    def _ensure_directories(self):
        """Create necessary directories."""
        dirs_to_create = [
            self.checkpoint_dir,
            self.output_dir,
            f"{self.output_dir}/logs",
            f"{self.output_dir}/plots",
            f"{self.output_dir}/results"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate Databricks environment and dependencies."""
        validation_results = {
            'environment': 'databricks',
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        try:
            # Check GPU
            validation_results['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                validation_results['gpu_count'] = torch.cuda.device_count()
                validation_results['gpu_name'] = torch.cuda.get_device_name(0)
                validation_results['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Check data availability
            validation_results['data_available'] = os.path.exists(self.data_dir)
            if validation_results['data_available']:
                validation_results['domains'] = [
                    d for d in os.listdir(f"{self.data_dir}/numerical") 
                    if os.path.isdir(f"{self.data_dir}/numerical/{d}")
                ]
            
            # Check imports
            required_modules = [
                'torch', 'transformers', 'numpy', 'pandas', 
                'yaml', 'mlflow', 'matplotlib', 'tqdm'
            ]
            
            missing_modules = []
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            validation_results['missing_modules'] = missing_modules
            validation_results['all_dependencies_available'] = len(missing_modules) == 0
            
            logger.info("✅ Environment validation completed successfully")
            return validation_results
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['error'] = str(e)
            logger.error(f"❌ Environment validation failed: {e}")
            return validation_results
    
    def run_data_exploration(self, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run data exploration pipeline."""
        logger.info("🔍 Starting data exploration...")
        
        try:
            from pipelines.exploration_pipeline import DataExplorationPipeline
            
            pipeline = DataExplorationPipeline(self.config)
            results = pipeline.run(domains=domains)
            
            # Save results
            output_path = f"{self.output_dir}/exploration_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Data exploration completed. Results saved to: {output_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Data exploration failed: {e}")
            raise e
    
    def run_training(self, 
                    domains: List[str] = ["Agriculture", "Climate", "Economy"],
                    epochs: int = 5,
                    batch_size: int = 4) -> Dict[str, Any]:
        """Run training pipeline."""
        logger.info(f"🏋️ Starting training pipeline...")
        logger.info(f"Domains: {domains}")
        logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}")
        
        try:
            from pipelines.training_pipeline_databricks import DatabricksTrainingPipeline
            
            # Update config for this run
            training_config = self.config.copy()
            training_config.update({
                'domains': domains,
                'epochs': epochs,
                'batch_size': batch_size,
                'save_dir': self.checkpoint_dir
            })
            
            pipeline = DatabricksTrainingPipeline(training_config)
            model, metrics = pipeline.run()
            
            # Save training summary
            training_summary = {
                'timestamp': datetime.now().isoformat(),
                'domains': domains,
                'epochs': epochs,
                'batch_size': batch_size,
                'final_metrics': metrics,
                'model_path': f"{self.checkpoint_dir}/best_model.pt"
            }
            
            summary_path = f"{self.output_dir}/training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2, default=str)
            
            logger.info(f"✅ Training completed. Model saved to: {self.checkpoint_dir}")
            return training_summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise e
    
    def run_evaluation(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Run evaluation pipeline."""
        logger.info("📊 Starting evaluation...")
        
        try:
            from pipelines.evaluation_pipeline import EvaluationPipeline
            
            model_path = model_path or f"{self.checkpoint_dir}/best_model.pt"
            
            pipeline = EvaluationPipeline(self.config)
            results = pipeline.run(model_path=model_path)
            
            # Save results
            output_path = f"{self.output_dir}/evaluation_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Evaluation completed. Results saved to: {output_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise e
    
    def run_inference_demo(self, 
                          model_path: Optional[str] = None,
                          num_samples: int = 5) -> Dict[str, Any]:
        """Run inference demonstration."""
        logger.info(f"🎯 Starting inference demo with {num_samples} samples...")
        
        try:
            from pipelines.demo_pipeline_databricks import DatabricksDemoPipeline
            
            model_path = model_path or f"{self.checkpoint_dir}/best_model.pt"
            
            pipeline = DatabricksDemoPipeline(self.config)
            pipeline.load_model(model_path)
            
            results = pipeline.run_demo(num_samples=num_samples)
            
            # Save results
            output_path = f"{self.output_dir}/demo_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Inference demo completed. Results saved to: {output_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Inference demo failed: {e}")
            raise e
    
    def run_full_pipeline(self,
                         domains: List[str] = ["Agriculture", "Climate", "Economy"],
                         epochs: int = 5,
                         batch_size: int = 4,
                         skip_exploration: bool = False) -> Dict[str, Any]:
        """Run the complete pipeline from data exploration to inference."""
        logger.info("🚀 Starting FULL MULTIMODAL LLM PIPELINE")
        logger.info("=" * 60)
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'configuration': {
                'domains': domains,
                'epochs': epochs,
                'batch_size': batch_size,
                'skip_exploration': skip_exploration
            }
        }
        
        try:
            # Step 1: Environment validation
            logger.info("Step 1/5: Environment Validation")
            validation_results = self.validate_environment()
            pipeline_results['validation'] = validation_results
            
            if validation_results['status'] != 'success':
                raise RuntimeError("Environment validation failed")
            
            # Step 2: Data exploration (optional)
            if not skip_exploration:
                logger.info("Step 2/5: Data Exploration")
                exploration_results = self.run_data_exploration(domains)
                pipeline_results['exploration'] = exploration_results
            else:
                logger.info("Step 2/5: Data Exploration (SKIPPED)")
            
            # Step 3: Model training
            logger.info("Step 3/5: Model Training")
            training_results = self.run_training(domains, epochs, batch_size)
            pipeline_results['training'] = training_results
            
            # Step 4: Model evaluation
            logger.info("Step 4/5: Model Evaluation") 
            evaluation_results = self.run_evaluation()
            pipeline_results['evaluation'] = evaluation_results
            
            # Step 5: Inference demo
            logger.info("Step 5/5: Inference Demo")
            demo_results = self.run_inference_demo()
            pipeline_results['demo'] = demo_results
            
            # Pipeline completion
            pipeline_results['end_time'] = datetime.now().isoformat()
            pipeline_results['status'] = 'success'
            pipeline_results['model_location'] = f"{self.checkpoint_dir}/best_model.pt"
            
            # Save complete results
            results_path = f"{self.output_dir}/pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            logger.info(f"🤖 Model saved to: {pipeline_results['model_location']}")
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            # Save failed results
            results_path = f"{self.output_dir}/pipeline_results_failed.json"
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.error(f"❌ PIPELINE FAILED: {e}")
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status and available models."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'project_root': self.project_root,
            'checkpoint_dir': self.checkpoint_dir,
            'output_dir': self.output_dir
        }
        
        # Check for saved models
        model_files = []
        if os.path.exists(self.checkpoint_dir):
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('.pt'):
                    file_path = os.path.join(self.checkpoint_dir, file)
                    model_files.append({
                        'name': file,
                        'path': file_path,
                        'size_mb': os.path.getsize(file_path) / (1024*1024),
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    })
        
        status['available_models'] = model_files
        
        # Check for results
        results_files = []
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(self.output_dir, file)
                    results_files.append({
                        'name': file,
                        'path': file_path,
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    })
        
        status['available_results'] = results_files
        
        return status


# Convenience functions for notebook use
def create_pipeline(project_root: Optional[str] = None) -> DatabricksPipeline:
    """Create a new pipeline instance."""
    return DatabricksPipeline(project_root)

def quick_train(domains: List[str] = ["Agriculture", "Climate"],
               epochs: int = 3,
               batch_size: int = 2,
               project_root: Optional[str] = None) -> Dict[str, Any]:
    """Quick training with minimal configuration."""
    pipeline = DatabricksPipeline(project_root)
    return pipeline.run_full_pipeline(
        domains=domains,
        epochs=epochs, 
        batch_size=batch_size,
        skip_exploration=True
    )

# Example usage
if __name__ == "__main__":
    # This would typically be run from a Databricks notebook
    pipeline = DatabricksPipeline()
    
    # Run validation
    validation = pipeline.validate_environment()
    print("Validation Results:", json.dumps(validation, indent=2))
    
    # Check status
    status = pipeline.get_status()
    print("Pipeline Status:", json.dumps(status, indent=2))