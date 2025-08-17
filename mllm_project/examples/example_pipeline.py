#!/usr/bin/env python3
"""
Example Python script for running the pipeline programmatically.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipelines import (
    DataExplorationPipeline,
    TrainingPipeline,
    EvaluationPipeline,
    InferencePipeline,
    PipelineOrchestrator
)
from src.utils.config_loader import load_config_for_training
from src.utils.cli_utils import setup_logging, print_banner, print_success


def run_exploration_example():
    """Example: Run data exploration."""
    print("\n=== Data Exploration Example ===\n")
    
    # Load configuration
    config = load_config_for_training('./config')
    
    # Create exploration pipeline
    pipeline = DataExplorationPipeline(
        config=config,
        output_dir='./results/exploration',
        cache_dir='./cache'
    )
    
    # Run exploration
    results = pipeline.run(
        sample_size=500,
        generate_report=True
    )
    
    print(f"Data quality score: {results['quality_assessment']['quality_score']:.1f}%")
    print_success("Exploration completed!")
    
    return results


def run_training_example():
    """Example: Run model training."""
    print("\n=== Model Training Example ===\n")
    
    # Load configuration
    config = load_config_for_training('./config')
    
    # Override some settings
    config['training']['epochs'] = 5  # Quick training for demo
    config['training']['batch_size'] = 16
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        config=config,
        checkpoint_dir='./checkpoints',
        experiment_name='demo_training'
    )
    
    # Run training
    summary = pipeline.run(verbose=True)
    
    print(f"Training completed in {summary['total_training_time']/60:.1f} minutes")
    print(f"Final validation loss: {summary['final_val_loss']:.4f}")
    print_success("Training completed!")
    
    return summary


def run_evaluation_example():
    """Example: Run model evaluation."""
    print("\n=== Model Evaluation Example ===\n")
    
    # Load configuration
    config = load_config_for_training('./config')
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(
        config=config,
        model_path='./checkpoints/best_model.pt',
        output_dir='./results/evaluation'
    )
    
    # Run evaluation
    results = pipeline.run(
        test_split='test',
        generate_plots=True,
        save_predictions=False
    )
    
    metrics = results['quantitative']['metrics']
    print(f"Test perplexity: {metrics['perplexity']:.2f}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print_success("Evaluation completed!")
    
    return results


def run_inference_example():
    """Example: Run inference demo."""
    print("\n=== Inference Demo Example ===\n")
    
    # Load configuration
    config = load_config_for_training('./config')
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        config=config,
        model_path='./checkpoints/best_model.pt',
        demo_dir='./demo'
    )
    
    # Run demo
    results = pipeline.run_demo(
        num_examples=5,
        streaming=False,
        temperature=0.8
    )
    
    # Show some examples
    for i, example in enumerate(results['examples'][:3], 1):
        print(f"\nExample {i}:")
        print(f"  Prompt: {example['prompt']}")
        print(f"  Generated: {example['generated'][:100]}...")
        print(f"  Time: {example['generation_time']:.3f}s")
    
    print_success("Inference demo completed!")
    
    return results


def run_full_pipeline_example():
    """Example: Run complete pipeline."""
    print("\n=== Full Pipeline Example ===\n")
    
    # Load pipeline configuration
    import yaml
    with open('./config/pipeline_config.yaml', 'r') as f:
        pipeline_config = yaml.safe_load(f)
    
    # Load model configuration
    config = load_config_for_training('./config')
    pipeline_config.update(config)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=pipeline_config,
        output_dir='./pipeline_results',
        verbose=True
    )
    
    # Run selected stages
    results = orchestrator.run(
        stages=['explore', 'train', 'evaluate', 'demo'],
        resume=False
    )
    
    print("\n=== Pipeline Summary ===")
    for stage, stage_results in results['stage_results'].items():
        status = "✅" if stage_results['status'] == 'success' else "❌"
        print(f"{status} {stage}: {stage_results.get('summary', 'Completed')}")
    
    print_success("Full pipeline completed!")
    
    return results


def run_custom_pipeline():
    """Example: Custom pipeline with specific configurations."""
    print("\n=== Custom Pipeline Example ===\n")
    
    from src.pipelines.orchestrator import PipelineOrchestrator
    
    # Custom configuration
    custom_config = {
        'explore': {
            'sample_size': 200,
            'generate_report': False
        },
        'train': {
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 0.0001
        },
        'evaluate': {
            'generate_plots': False,
            'save_predictions': True
        },
        'demo': {
            'num_examples': 3,
            'temperature': 0.5
        }
    }
    
    # Load base configuration
    config = load_config_for_training('./config')
    
    # Merge custom config
    for key, value in custom_config.items():
        if key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        output_dir='./custom_pipeline_results',
        verbose=False
    )
    
    # Run only specific stages
    results = orchestrator.run(
        stages=['explore', 'train'],
        resume=False
    )
    
    print_success("Custom pipeline completed!")
    
    return results


def main():
    """Main function to run examples."""
    
    # Setup logging
    setup_logging()
    
    # Print banner
    print_banner()
    
    print("\n Select an example to run:")
    print("1. Data Exploration")
    print("2. Model Training")
    print("3. Model Evaluation")
    print("4. Inference Demo")
    print("5. Full Pipeline")
    print("6. Custom Pipeline")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (0-6): ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            elif choice == '1':
                run_exploration_example()
            elif choice == '2':
                run_training_example()
            elif choice == '3':
                run_evaluation_example()
            elif choice == '4':
                run_inference_example()
            elif choice == '5':
                run_full_pipeline_example()
            elif choice == '6':
                run_custom_pipeline()
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
