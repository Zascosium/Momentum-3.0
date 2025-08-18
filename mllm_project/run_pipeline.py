#!/usr/bin/env python3
"""
Terminal-based Multimodal LLM Pipeline Runner
Supports both local execution and Databricks cluster execution via CLI
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log', mode='a')
        ]
    )

def detect_environment() -> str:
    """Detect if running on Databricks or locally."""
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        return 'databricks'
    elif os.path.exists('/databricks'):
        return 'databricks'
    else:
        return 'local'

def setup_paths(project_root: Optional[str] = None) -> str:
    """Setup Python paths for imports."""
    if project_root is None:
        # Auto-detect project root
        current_dir = Path(__file__).parent.absolute()
        if (current_dir / 'src').exists():
            project_root = str(current_dir)
        else:
            # Try parent directories
            for parent in current_dir.parents:
                if (parent / 'src').exists() and (parent / 'config').exists():
                    project_root = str(parent)
                    break
            else:
                project_root = str(current_dir)
    
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root

def run_validation(project_root: str) -> bool:
    """Run environment validation."""
    print("🔍 ENVIRONMENT VALIDATION")
    print("=" * 50)
    
    try:
        from databricks_pipeline import DatabricksPipeline
        pipeline = DatabricksPipeline(project_root)
        validation = pipeline.validate_environment()
        
        print(f"Environment: {validation.get('environment', 'unknown')}")
        print(f"CUDA Available: {validation.get('cuda_available', False)}")
        if validation.get('cuda_available'):
            print(f"GPU: {validation.get('gpu_name', 'Unknown')}")
            print(f"Memory: {validation.get('gpu_memory_gb', 0):.1f} GB")
        
        print(f"Data Available: {validation.get('data_available', False)}")
        print(f"Domains: {validation.get('domains', [])}")
        
        if validation['status'] == 'success' and validation.get('all_dependencies_available', False):
            print("✅ Validation PASSED")
            return True
        else:
            print("❌ Validation FAILED")
            if validation.get('missing_modules'):
                print(f"Missing modules: {validation['missing_modules']}")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def run_exploration(project_root: str, domains: Optional[List[str]] = None) -> bool:
    """Run data exploration."""
    print("\n📊 DATA EXPLORATION")
    print("=" * 50)
    
    try:
        from databricks_pipeline import DatabricksPipeline
        pipeline = DatabricksPipeline(project_root)
        results = pipeline.run_data_exploration(domains)
        
        print("✅ Exploration completed")
        print(f"Results saved to: {pipeline.output_dir}/exploration_results.json")
        return True
        
    except Exception as e:
        print(f"❌ Exploration failed: {e}")
        return False

def run_training(project_root: str, 
                domains: List[str], 
                epochs: int, 
                batch_size: int,
                learning_rate: float = 5e-5,
                save_every: int = 1) -> bool:
    """Run model training."""
    print(f"\n🏋️ MODEL TRAINING")
    print("=" * 50)
    print(f"Domains: {domains}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    
    try:
        from databricks_pipeline import DatabricksPipeline
        pipeline = DatabricksPipeline(project_root)
        
        # Update config for this run
        pipeline.config.update({
            'learning_rate': learning_rate,
            'save_every_n_epochs': save_every
        })
        
        results = pipeline.run_training(domains, epochs, batch_size)
        
        print("✅ Training completed")
        print(f"Model saved to: {results['model_path']}")
        
        # Display final metrics
        if 'final_metrics' in results:
            metrics = results['final_metrics']
            print("\n📊 Final Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.error(f"Training error: {e}", exc_info=True)
        return False

def run_evaluation(project_root: str, model_path: Optional[str] = None) -> bool:
    """Run model evaluation."""
    print(f"\n📈 MODEL EVALUATION")
    print("=" * 50)
    
    try:
        from databricks_pipeline import DatabricksPipeline
        pipeline = DatabricksPipeline(project_root)
        results = pipeline.run_evaluation(model_path)
        
        print("✅ Evaluation completed")
        
        # Display key metrics
        if 'metrics' in results:
            print("\n📊 Performance Metrics:")
            for key, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        print(f"Results saved to: {pipeline.output_dir}/evaluation_results.json")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logging.error(f"Evaluation error: {e}", exc_info=True)
        return False

def run_inference(project_root: str, 
                 model_path: Optional[str] = None, 
                 num_samples: int = 5) -> bool:
    """Run inference demo."""
    print(f"\n🎯 INFERENCE DEMO")
    print("=" * 50)
    
    try:
        from databricks_pipeline import DatabricksPipeline
        pipeline = DatabricksPipeline(project_root)
        results = pipeline.run_inference_demo(model_path, num_samples)
        
        print("✅ Inference completed")
        
        # Display sample predictions
        if 'predictions' in results:
            print(f"\n🔮 Generated {len(results['predictions'])} predictions")
            for i, pred in enumerate(results['predictions'][:3]):
                print(f"\nSample {i+1}:")
                print(f"  Domain: {pred.get('domain', 'Unknown')}")
                text = pred.get('generated_text', 'No text')[:150]
                print(f"  Text: {text}...")
        
        print(f"Results saved to: {pipeline.output_dir}/demo_results.json")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        logging.error(f"Inference error: {e}", exc_info=True)
        return False

def run_full_pipeline(project_root: str,
                     domains: List[str],
                     epochs: int,
                     batch_size: int,
                     learning_rate: float = 5e-5,
                     skip_exploration: bool = False,
                     skip_evaluation: bool = False) -> bool:
    """Run the complete pipeline."""
    print("🚀 FULL MULTIMODAL LLM PIPELINE")
    print("=" * 60)
    
    success = True
    
    # Step 1: Validation
    if not run_validation(project_root):
        return False
    
    # Step 2: Exploration (optional)
    if not skip_exploration:
        if not run_exploration(project_root, domains):
            print("⚠️ Exploration failed, continuing...")
    
    # Step 3: Training
    if not run_training(project_root, domains, epochs, batch_size, learning_rate):
        return False
    
    # Step 4: Evaluation (optional)
    if not skip_evaluation:
        if not run_evaluation(project_root):
            print("⚠️ Evaluation failed, continuing...")
    
    # Step 5: Inference Demo
    if not run_inference(project_root):
        print("⚠️ Inference demo failed, but training completed")
        success = False
    
    if success:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("\n⚠️ PIPELINE COMPLETED WITH WARNINGS")
    
    return success

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal LLM Pipeline - Terminal Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default settings
  python run_pipeline.py full --domains Agriculture Climate --epochs 3

  # Training only
  python run_pipeline.py train --domains Agriculture --epochs 5 --batch-size 4

  # Just validation
  python run_pipeline.py validate

  # Evaluation with custom model
  python run_pipeline.py evaluate --model-path /path/to/model.pt

  # Quick training for testing
  python run_pipeline.py full --domains Agriculture --epochs 1 --batch-size 1 --skip-exploration --skip-evaluation
        """
    )
    
    parser.add_argument('command', choices=['full', 'validate', 'explore', 'train', 'evaluate', 'infer'],
                       help='Pipeline command to run')
    
    # Common arguments
    parser.add_argument('--project-root', type=str, help='Project root directory')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    # Training arguments
    parser.add_argument('--domains', nargs='+', default=['Agriculture', 'Climate'],
                       help='Domains to use for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    
    # Pipeline control
    parser.add_argument('--skip-exploration', action='store_true',
                       help='Skip data exploration step')
    parser.add_argument('--skip-evaluation', action='store_true', 
                       help='Skip evaluation step')
    
    # Model arguments
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation/inference')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples for inference demo')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Detect environment
    env = detect_environment()
    logger.info(f"Running in {env} environment")
    
    # Setup paths
    project_root = setup_paths(args.project_root)
    logger.info(f"Project root: {project_root}")
    
    # Run command
    success = False
    
    try:
        if args.command == 'validate':
            success = run_validation(project_root)
            
        elif args.command == 'explore':
            success = run_exploration(project_root, args.domains)
            
        elif args.command == 'train':
            success = run_training(
                project_root, args.domains, args.epochs, 
                args.batch_size, args.learning_rate, args.save_every
            )
            
        elif args.command == 'evaluate':
            success = run_evaluation(project_root, args.model_path)
            
        elif args.command == 'infer':
            success = run_inference(project_root, args.model_path, args.num_samples)
            
        elif args.command == 'full':
            success = run_full_pipeline(
                project_root, args.domains, args.epochs, args.batch_size,
                args.learning_rate, args.skip_exploration, args.skip_evaluation
            )
    
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Exit with appropriate code
    if success:
        print(f"\n✅ Command '{args.command}' completed successfully")
        sys.exit(0)
    else:
        print(f"\n❌ Command '{args.command}' failed")
        sys.exit(1)

if __name__ == "__main__":
    main()