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
    
    # Test 1: Direct exploration pipeline import
    logger.info("Test 1: Direct import")
    try:
        from src.pipelines.exploration_pipeline import DataExplorationPipeline
        logger.info("✅ Successfully imported DataExplorationPipeline")
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
    
    # Test 2: Try without src prefix
    logger.info("Test 2: Import without src prefix")
    try:
        from pipelines.exploration_pipeline import DataExplorationPipeline
        logger.info("✅ Successfully imported DataExplorationPipeline (no src prefix)")
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
    
    # Test 3: Try with importlib
    logger.info("Test 3: Using importlib")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exploration_pipeline", 
            src_path / "pipelines" / "exploration_pipeline.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            DataExplorationPipeline = module.DataExplorationPipeline
            logger.info("✅ Successfully imported DataExplorationPipeline (importlib)")
            return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
    
    return False

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
    success = test_imports()
    
    if success:
        print("\n✅ Imports working! CLI should work.")
    else:
        print("\n❌ Import issues detected. Check the logs above.")
        print("\nTry running this in your Databricks environment:")
        print("python debug_imports.py")