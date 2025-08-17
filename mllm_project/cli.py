#!/usr/bin/env python3
"""
Main CLI Entry Point for Multimodal LLM Pipeline
================================================

This script provides a unified command-line interface for running the complete
multimodal LLM pipeline, from data exploration to model deployment.

Usage:
    python cli.py [COMMAND] [OPTIONS]

Commands:
    explore     - Run data exploration and analysis
    train       - Train the multimodal model
    evaluate    - Evaluate trained model performance
    demo        - Run interactive inference demo
    pipeline    - Run complete end-to-end pipeline
    serve       - Start model serving API

Examples:
    python cli.py explore --data-dir /path/to/data --output-dir ./results
    python cli.py train --config config/training.yaml --epochs 10
    python cli.py pipeline --config config/pipeline.yaml
    python cli.py serve --model-path ./models/best --port 8080
"""

import click
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import json
import yaml

# Add project source to path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Databricks compatibility: also try common Databricks paths
if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
    # Common Databricks paths
    databricks_paths = [
        '/databricks/driver',
        '/databricks/driver/src',
        '/Workspace/Repos',
        '/Workspace'
    ]
    for db_path in databricks_paths:
        if os.path.exists(db_path):
            sys.path.insert(0, db_path)

# Enhanced import function for Databricks compatibility
def import_pipeline_class(pipeline_name, module_name):
    """Import pipeline class with multiple fallback strategies"""
    try:
        # Method 1: Standard import
        module = __import__(f"src.pipelines.{module_name}", fromlist=[pipeline_name])
        return getattr(module, pipeline_name)
    except (ImportError, ValueError) as e:
        logging.warning(f"Standard import failed for {pipeline_name}: {e}")
        
        # Method 2: Direct file import (Databricks fallback)
        try:
            import importlib.util
            pipeline_file = src_path / "pipelines" / f"{module_name}.py"
            if pipeline_file.exists():
                spec = importlib.util.spec_from_file_location(module_name, pipeline_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    pipeline_class = getattr(module, pipeline_name)
                    logging.info(f"Successfully imported {pipeline_name} using direct file import")
                    return pipeline_class
            logging.warning(f"Pipeline file not found: {pipeline_file}")
        except Exception as e2:
            logging.warning(f"Direct file import failed for {pipeline_name}: {e2}")
        
        # Method 3: Try without src prefix (last resort)
        try:
            module = __import__(f"pipelines.{module_name}", fromlist=[pipeline_name])
            pipeline_class = getattr(module, pipeline_name)
            logging.info(f"Successfully imported {pipeline_name} without src prefix")
            return pipeline_class
        except Exception as e3:
            logging.warning(f"All import methods failed for {pipeline_name}: {e3}")
        
        return None

# Import pipeline modules with enhanced error handling
DataExplorationPipeline = import_pipeline_class("DataExplorationPipeline", "exploration_pipeline")
TrainingPipeline = import_pipeline_class("TrainingPipeline", "training_pipeline") 
EvaluationPipeline = import_pipeline_class("EvaluationPipeline", "evaluation_pipeline")
InferencePipeline = import_pipeline_class("InferencePipeline", "inference_pipeline")
PipelineOrchestrator = import_pipeline_class("PipelineOrchestrator", "orchestrator")
ModelServingAPI = import_pipeline_class("ModelServingAPI", "serving")

try:
    from src.utils.cli_utils import setup_logging, print_banner, validate_paths
except ImportError as e:
    # Provide fallback implementations
    def setup_logging(level=logging.INFO, log_file=None):
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    def print_banner():
        print("🚀 Multimodal LLM Pipeline CLI")
    def validate_paths(*args, **kwargs):
        return True

try:
    from src.utils.config_loader import load_config_for_training
except ImportError as e:
    def load_config_for_training(config_dir):
        import yaml
        from pathlib import Path
        config_path = Path(config_dir) / 'model_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}


# Configure logging
logger = logging.getLogger(__name__)


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=str, help='Path to log file')
@click.pass_context
def cli(ctx, verbose, log_file):
    """
    Multimodal LLM Pipeline CLI
    
    A comprehensive command-line interface for training and deploying
    multimodal language models that combine time series and text data.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level, log_file=log_file)
    
    # Print banner
    print_banner()
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['log_file'] = log_file


@cli.command()
@click.option('--data-dir', '-d', type=click.Path(exists=True), 
              help='Path to data directory')
@click.option('--config-dir', '-c', type=click.Path(exists=True),
              default='config', help='Path to configuration directory')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./results/exploration', help='Output directory for results')
@click.option('--cache-dir', type=click.Path(),
              default='./cache', help='Cache directory')
@click.option('--domains', multiple=True, 
              help='Specific domains to analyze')
@click.option('--sample-size', type=int, default=1000,
              help='Number of samples to analyze')
@click.option('--generate-report', is_flag=True,
              help='Generate HTML report')
@click.pass_context
def explore(ctx, data_dir, config_dir, output_dir, cache_dir, 
           domains, sample_size, generate_report):
    """
    Run data exploration and analysis pipeline.
    
    This command performs comprehensive data exploration including:
    - Dataset structure analysis
    - Time series characteristics
    - Text data analysis
    - Data quality assessment
    - Multimodal alignment exploration
    """
    logger.info("Starting data exploration pipeline...")
    
    if DataExplorationPipeline is None:
        click.echo(click.style("❌ DataExplorationPipeline not available. Missing dependencies.", fg='red'))
        raise click.Abort()
    
    try:
        # Load configuration
        config = load_config_for_training(config_dir)
        
        # Update config with CLI options
        if data_dir:
            config.setdefault('data', {})['data_dir'] = data_dir
        if domains:
            config.setdefault('domains', {})['included'] = list(domains)
        
        # Initialize exploration pipeline
        pipeline = DataExplorationPipeline(
            config=config,
            output_dir=output_dir,
            cache_dir=cache_dir
        )
        
        # Run exploration
        results = pipeline.run(
            sample_size=sample_size,
            generate_report=generate_report
        )
        
        # Save results
        results_path = Path(output_dir) / 'exploration_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        click.echo(click.style(f"✅ Exploration completed successfully!", fg='green'))
        click.echo(f"📊 Results saved to: {results_path}")
        
        if generate_report:
            report_path = Path(output_dir) / 'exploration_report.html'
            click.echo(f"📄 Report available at: {report_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Exploration failed: {str(e)}")
        click.echo(click.style(f"❌ Exploration failed: {str(e)}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--config-dir', '-c', type=click.Path(exists=True),
              default='config', help='Path to configuration directory')
@click.option('--data-dir', '-d', type=click.Path(exists=True),
              help='Path to data directory')
@click.option('--checkpoint-dir', type=click.Path(),
              default='./checkpoints', help='Directory for model checkpoints')
@click.option('--experiment-name', '-e', type=str,
              help='MLflow experiment name')
@click.option('--epochs', type=int, help='Number of training epochs')
@click.option('--batch-size', type=int, help='Training batch size')
@click.option('--learning-rate', type=float, help='Learning rate')
@click.option('--distributed', is_flag=True, help='Enable distributed training')
@click.option('--mixed-precision', is_flag=True, help='Enable mixed precision training')
@click.option('--resume-from', type=click.Path(exists=True),
              help='Resume training from checkpoint')
@click.option('--use-wandb', is_flag=True, help='Enable Weights & Biases logging')
@click.pass_context
def train(ctx, config_dir, data_dir, checkpoint_dir, experiment_name,
         epochs, batch_size, learning_rate, distributed, mixed_precision,
         resume_from, use_wandb):
    """
    Train the multimodal LLM model.
    
    This command handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization
    - Training with MLflow tracking
    - Checkpoint saving
    - Performance monitoring
    """
    logger.info("Starting training pipeline...")
    
    if TrainingPipeline is None:
        click.echo(click.style("❌ TrainingPipeline not available. Missing dependencies (likely mlflow).", fg='red'))
        click.echo("Install missing dependencies: pip install mlflow")
        raise click.Abort()
    
    try:
        # Load configuration
        config = load_config_for_training(config_dir)
        
        # Update config with CLI options
        if data_dir:
            config.setdefault('data', {})['data_dir'] = data_dir
        if epochs:
            config.setdefault('training', {})['epochs'] = epochs
        if batch_size:
            config.setdefault('training', {})['batch_size'] = batch_size
        if learning_rate:
            config.setdefault('optimizer', {})['learning_rate'] = learning_rate
        if distributed:
            config.setdefault('training', {})['distributed'] = True
        if mixed_precision:
            config.setdefault('mixed_precision', {})['enabled'] = True
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(
            config=config,
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name or "multimodal_llm_training",
            use_wandb=use_wandb
        )
        
        # Run training
        training_summary = pipeline.run(
            resume_from=resume_from,
            verbose=ctx.obj['verbose']
        )
        
        # Save training summary
        summary_path = Path(checkpoint_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        click.echo(click.style(f"✅ Training completed successfully!", fg='green'))
        click.echo(f"📊 Summary saved to: {summary_path}")
        click.echo(f"🏆 Best model saved to: {checkpoint_dir}/best_model.pt")
        
        # Display key metrics
        if 'final_metrics' in training_summary:
            click.echo("\n📈 Final Metrics:")
            for metric, value in training_summary['final_metrics'].items():
                click.echo(f"  • {metric}: {value:.4f}")
        
        return training_summary
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        click.echo(click.style(f"❌ Training failed: {str(e)}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--model-path', '-m', type=click.Path(exists=True),
              required=True, help='Path to trained model')
@click.option('--config-dir', '-c', type=click.Path(exists=True),
              default='config', help='Path to configuration directory')
@click.option('--data-dir', '-d', type=click.Path(exists=True),
              help='Path to test data directory')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./results/evaluation', help='Output directory for results')
@click.option('--test-split', type=str, default='test',
              help='Data split to evaluate on')
@click.option('--batch-size', type=int, help='Evaluation batch size')
@click.option('--generate-plots', is_flag=True, 
              help='Generate visualization plots')
@click.option('--save-predictions', is_flag=True,
              help='Save model predictions')
@click.pass_context
def evaluate(ctx, model_path, config_dir, data_dir, output_dir,
            test_split, batch_size, generate_plots, save_predictions):
    """
    Evaluate trained model performance.
    
    This command performs comprehensive model evaluation including:
    - Quantitative metrics computation
    - Generation quality assessment
    - Error analysis
    - Performance visualization
    """
    logger.info("Starting evaluation pipeline...")
    
    if EvaluationPipeline is None:
        click.echo(click.style("❌ EvaluationPipeline not available. Missing dependencies.", fg='red'))
        raise click.Abort()
    
    try:
        # Load configuration
        config = load_config_for_training(config_dir)
        
        # Update config with CLI options
        if data_dir:
            config.setdefault('data', {})['data_dir'] = data_dir
        if batch_size:
            config.setdefault('evaluation', {})['batch_size'] = batch_size
        
        # Initialize evaluation pipeline
        pipeline = EvaluationPipeline(
            config=config,
            model_path=model_path,
            output_dir=output_dir
        )
        
        # Run evaluation
        eval_results = pipeline.run(
            test_split=test_split,
            generate_plots=generate_plots,
            save_predictions=save_predictions
        )
        
        # Save results
        results_path = Path(output_dir) / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        click.echo(click.style(f"✅ Evaluation completed successfully!", fg='green'))
        click.echo(f"📊 Results saved to: {results_path}")
        
        # Display key metrics
        if 'metrics' in eval_results:
            click.echo("\n📈 Evaluation Metrics:")
            for metric, value in eval_results['metrics'].items():
                click.echo(f"  • {metric}: {value:.4f}")
        
        return eval_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        click.echo(click.style(f"❌ Evaluation failed: {str(e)}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--model-path', '-m', type=click.Path(exists=True),
              required=True, help='Path to trained model')
@click.option('--config-dir', '-c', type=click.Path(exists=True),
              default='config', help='Path to configuration directory')
@click.option('--demo-dir', type=click.Path(),
              default='./demo', help='Directory for demo outputs')
@click.option('--interactive', is_flag=True,
              help='Run in interactive mode')
@click.option('--batch-demo', is_flag=True,
              help='Run batch processing demo')
@click.option('--streaming', is_flag=True,
              help='Enable streaming generation')
@click.option('--num-examples', type=int, default=10,
              help='Number of examples to generate')
@click.option('--temperature', type=float, default=0.8,
              help='Generation temperature')
@click.pass_context
def demo(ctx, model_path, config_dir, demo_dir, interactive,
        batch_demo, streaming, num_examples, temperature):
    """
    Run interactive inference demo.
    
    This command provides interactive demonstrations including:
    - Text generation from time series
    - Batch processing examples
    - Streaming generation
    - Performance benchmarking
    """
    logger.info("Starting inference demo...")
    
    if InferencePipeline is None:
        click.echo(click.style("❌ InferencePipeline not available. Missing dependencies.", fg='red'))
        raise click.Abort()
    
    try:
        # Load configuration
        config = load_config_for_training(config_dir)
        
        # Initialize inference pipeline
        pipeline = InferencePipeline(
            config=config,
            model_path=model_path,
            demo_dir=demo_dir
        )
        
        if interactive:
            # Run interactive mode
            click.echo(click.style("🎮 Starting interactive demo...", fg='cyan'))
            pipeline.run_interactive(
                streaming=streaming,
                temperature=temperature
            )
        
        elif batch_demo:
            # Run batch demo
            click.echo(click.style("📦 Running batch processing demo...", fg='cyan'))
            results = pipeline.run_batch_demo(
                num_examples=num_examples,
                temperature=temperature
            )
            
            # Display results
            click.echo(f"\n✅ Generated {len(results)} examples")
            for i, result in enumerate(results[:3], 1):
                click.echo(f"\n{i}. {result['prompt'][:50]}...")
                click.echo(f"   → {result['generated'][:100]}...")
        
        else:
            # Run standard demo
            click.echo(click.style("🎯 Running standard demo...", fg='cyan'))
            demo_results = pipeline.run_demo(
                num_examples=num_examples,
                streaming=streaming,
                temperature=temperature
            )
            
            # Save demo results
            results_path = Path(demo_dir) / 'demo_results.json'
            with open(results_path, 'w') as f:
                json.dump(demo_results, f, indent=2, default=str)
            
            click.echo(f"📊 Demo results saved to: {results_path}")
        
        click.echo(click.style(f"✅ Demo completed successfully!", fg='green'))
        
    except KeyboardInterrupt:
        click.echo("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        click.echo(click.style(f"❌ Demo failed: {str(e)}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              required=True, help='Pipeline configuration file')
@click.option('--stages', multiple=True, 
              type=click.Choice(['explore', 'train', 'evaluate', 'demo']),
              help='Specific stages to run')
@click.option('--skip-stages', multiple=True,
              type=click.Choice(['explore', 'train', 'evaluate', 'demo']),
              help='Stages to skip')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./pipeline_results', help='Output directory')
@click.option('--resume', is_flag=True,
              help='Resume from previous run')
@click.option('--dry-run', is_flag=True,
              help='Show pipeline plan without executing')
@click.pass_context
def pipeline(ctx, config, stages, skip_stages, output_dir, resume, dry_run):
    """
    Run complete end-to-end pipeline.
    
    This command orchestrates the entire pipeline:
    1. Data exploration and validation
    2. Model training
    3. Evaluation
    4. Inference demonstration
    """
    logger.info("Starting end-to-end pipeline...")
    
    if PipelineOrchestrator is None:
        click.echo(click.style("❌ PipelineOrchestrator not available. Missing dependencies (likely mlflow).", fg='red'))
        click.echo("Install missing dependencies: pip install mlflow")
        raise click.Abort()
    
    try:
        # Load pipeline configuration
        with open(config, 'r') as f:
            pipeline_config = yaml.safe_load(f)
        
        # Determine stages to run
        all_stages = ['explore', 'train', 'evaluate', 'demo']
        if stages:
            stages_to_run = list(stages)
        else:
            stages_to_run = [s for s in all_stages if s not in skip_stages]
        
        click.echo(click.style(f"🚀 Pipeline Stages: {' → '.join(stages_to_run)}", fg='cyan'))
        
        if dry_run:
            click.echo("\n📋 Pipeline Plan (Dry Run):")
            for stage in stages_to_run:
                click.echo(f"  • {stage}: {pipeline_config.get(stage, {}).get('description', 'No description')}")
            click.echo("\n(Use --dry-run=false to execute)")
            return
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(
            config=pipeline_config,
            output_dir=output_dir,
            verbose=ctx.obj['verbose']
        )
        
        # Run pipeline
        results = orchestrator.run(
            stages=stages_to_run,
            resume=resume
        )
        
        # Save pipeline results
        results_path = Path(output_dir) / 'pipeline_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        click.echo(click.style(f"\n✅ Pipeline completed successfully!", fg='green'))
        click.echo(f"📊 Results saved to: {results_path}")
        
        # Display summary
        click.echo("\n📈 Pipeline Summary:")
        for stage, stage_results in results.items():
            if isinstance(stage_results, dict) and 'status' in stage_results:
                status = "✅" if stage_results['status'] == 'success' else "❌"
                click.echo(f"  {status} {stage}: {stage_results.get('summary', 'Completed')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        click.echo(click.style(f"❌ Pipeline failed: {str(e)}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--model-path', '-m', type=click.Path(exists=True),
              required=True, help='Path to trained model')
@click.option('--config-dir', '-c', type=click.Path(exists=True),
              default='config', help='Path to configuration directory')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', '-p', type=int, default=8080, help='Port to listen on')
@click.option('--workers', type=int, default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload on code changes')
@click.option('--enable-cors', is_flag=True, help='Enable CORS support')
@click.option('--api-key', type=str, help='API key for authentication')
@click.pass_context
def serve(ctx, model_path, config_dir, host, port, workers,
         reload, enable_cors, api_key):
    """
    Start model serving API.
    
    This command starts a REST API server for model inference:
    - RESTful endpoints for prediction
    - Batch processing support
    - Health monitoring
    - OpenAPI documentation
    """
    logger.info(f"Starting model serving API on {host}:{port}...")
    
    if ModelServingAPI is None:
        click.echo(click.style("❌ ModelServingAPI not available. Missing dependencies.", fg='red'))
        click.echo("Install missing dependencies: pip install fastapi uvicorn")
        raise click.Abort()
    
    try:
        # Load configuration
        config = load_config_for_training(config_dir)
        
        # Initialize serving API
        api = ModelServingAPI(
            config=config,
            model_path=model_path,
            enable_cors=enable_cors,
            api_key=api_key
        )
        
        click.echo(click.style(f"🚀 Starting API server...", fg='cyan'))
        click.echo(f"📍 Server: http://{host}:{port}")
        click.echo(f"📚 Documentation: http://{host}:{port}/docs")
        click.echo(f"💻 Health check: http://{host}:{port}/health")
        
        if api_key:
            click.echo(f"🔐 API Key authentication enabled")
        
        click.echo("\nPress Ctrl+C to stop the server\n")
        
        # Start server
        api.run(
            host=host,
            port=port,
            workers=workers,
            reload=reload
        )
        
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {str(e)}")
        click.echo(click.style(f"❌ Server failed: {str(e)}", fg='red'))
        raise click.Abort()


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    version_info = {
        'mllm_pipeline': '1.0.0',
        'python': sys.version.split()[0],
        'platform': sys.platform
    }
    
    click.echo("Multimodal LLM Pipeline")
    click.echo("=" * 30)
    for key, value in version_info.items():
        click.echo(f"{key}: {value}")


if __name__ == '__main__':
    cli()
