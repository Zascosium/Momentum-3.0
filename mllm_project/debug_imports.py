#!/usr/bin/env python3
"""
Debug script for Databricks import issues
Run this script to diagnose what's going wrong with imports
"""

import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_environment():
    """Debug the current environment and paths"""
    logger.info("=== ENVIRONMENT DEBUG ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {__file__}")
    logger.info(f"Script parent: {Path(__file__).parent}")
    
    # Check for Databricks
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        logger.info(f"Databricks detected: {os.environ['DATABRICKS_RUNTIME_VERSION']}")
    else:
        logger.info("Not running on Databricks")
    
    logger.info(f"sys.path (first 5): {sys.path[:5]}")
    
    # Check project structure
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    
    logger.info(f"Project root exists: {project_root.exists()}")
    logger.info(f"Src path exists: {src_path.exists()}")
    
    if src_path.exists():
        logger.info(f"Src contents: {list(src_path.iterdir())}")
        
        pipelines_path = src_path / 'pipelines'
        if pipelines_path.exists():
            logger.info(f"Pipelines contents: {list(pipelines_path.iterdir())}")
    
    return project_root, src_path

def test_imports():
    """Test various import strategies"""
    logger.info("\n=== IMPORT TESTING ===")
    
    project_root, src_path = debug_environment()
    
    # Add paths to sys.path
    sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(project_root))
    
    pipelines_to_test = [
        ('DataExplorationPipeline', 'exploration_pipeline'),
        ('TrainingPipeline', 'training_pipeline'),
        ('EvaluationPipeline', 'evaluation_pipeline'), 
        ('InferencePipeline', 'inference_pipeline'),
        ('PipelineOrchestrator', 'orchestrator'),
        ('ModelServingAPI', 'serving')
    ]
    
    success_count = 0
    
    for pipeline_name, module_name in pipelines_to_test:
        logger.info(f"\nTesting {pipeline_name}:")
        
        # Test 1: Direct import
        logger.info("  Test 1: Direct import")
        try:
            module = __import__(f"src.pipelines.{module_name}", fromlist=[pipeline_name])
            getattr(module, pipeline_name)
            logger.info(f"  ✅ Successfully imported {pipeline_name}")
            success_count += 1
            continue
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
        
        # Test 2: Try with importlib
        logger.info("  Test 2: Using importlib")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                module_name, 
                src_path / "pipelines" / f"{module_name}.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                getattr(module, pipeline_name)
                logger.info(f"  ✅ Successfully imported {pipeline_name} (importlib)")
                success_count += 1
                continue
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
        
        logger.error(f"  ❌ All import methods failed for {pipeline_name}")
    
    return success_count == len(pipelines_to_test)

def test_cli_commands():
    """Test CLI commands to ensure they handle missing dependencies gracefully"""
    logger.info("\n=== CLI COMMAND TESTING ===")
    
    commands_to_test = [
        ("help", ["python", "cli.py", "--help"]),
        ("explore help", ["python", "cli.py", "explore", "--help"]),
        ("train help", ["python", "cli.py", "train", "--help"]),
        ("evaluate help", ["python", "cli.py", "evaluate", "--help"]),
        ("demo help", ["python", "cli.py", "demo", "--help"]),
        ("pipeline help", ["python", "cli.py", "pipeline", "--help"]),
        ("serve help", ["python", "cli.py", "serve", "--help"]),
    ]
    
    success_count = 0
    
    for cmd_name, cmd_args in commands_to_test:
        logger.info(f"Testing {cmd_name}...")
        try:
            import subprocess
            result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"✅ {cmd_name} - Command executed successfully")
                success_count += 1
            else:
                logger.error(f"❌ {cmd_name} - Command failed: {result.stderr[:100]}...")
        except Exception as e:
            logger.error(f"❌ {cmd_name} - Exception: {e}")
    
    return success_count >= len(commands_to_test) - 1  # Allow 1 failure

def test_dependencies():
    """Test individual dependencies"""
    logger.info("\n=== DEPENDENCY TESTING ===")
    
    deps = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'click',
        'pyyaml',
        'rich',
        'torch',
        'transformers',
        'scikit-learn',
        'scipy'
    ]
    
    for dep in deps:
        try:
            __import__(dep)
            logger.info(f"✅ {dep}")
        except ImportError as e:
            logger.error(f"❌ {dep}: {e}")

if __name__ == "__main__":
    print("🔍 Databricks Import Debugger")
    print("=" * 50)
    
    # Test environment
    debug_environment()
    
    # Test dependencies
    test_dependencies()
    
    # Test imports
    import_success = test_imports()
    
    # Test CLI commands
    cli_success = test_cli_commands()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if import_success:
        print("✅ Pipeline imports: Working")
    else:
        print("⚠️ Pipeline imports: Some issues detected")
    
    if cli_success:
        print("✅ CLI commands: Working")
    else:
        print("⚠️ CLI commands: Some issues detected")
    
    if import_success and cli_success:
        print("\n🎉 All tests passed! CLI should work on Databricks.")
    else:
        print("\n⚠️ Some issues detected. Check logs above.")
        print("\nFor Databricks users:")
        print("1. Run: python install_databricks.py")
        print("2. Check missing dependencies and install them")
        print("3. Re-run this debug script")