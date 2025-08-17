#!/usr/bin/env python3
"""
Databricks Installation Script
==============================

This script installs the required dependencies for the MLLM pipeline on Databricks.
Run this in a Databricks notebook cell or terminal.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False
    return True

def install_databricks_dependencies():
    """Install core dependencies for Databricks"""
    
    print("🚀 Installing MLLM Pipeline Dependencies for Databricks")
    print("=" * 60)
    
    # Basic dependencies that are usually needed
    basic_deps = [
        "click>=8.1.0",
        "pyyaml>=6.0",
        "rich>=13.3.0",
        "numpy>=1.21.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    # ML dependencies
    ml_deps = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0"
    ]
    
    # Optional but recommended
    optional_deps = [
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "mlflow>=2.4.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0"
    ]
    
    # Install basic dependencies first
    print("\n📦 Installing basic dependencies...")
    for dep in basic_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️ Warning: Failed to install {dep}")
    
    # Install ML dependencies
    print("\n🧠 Installing ML dependencies...")
    for dep in ml_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️ Warning: Failed to install {dep}")
    
    # Install optional dependencies
    print("\n🔧 Installing optional dependencies...")
    for dep in optional_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️ Warning: Failed to install optional dependency {dep}")
    
    print("\n✅ Installation completed!")
    print("\n🔍 Testing imports...")
    
    # Test critical imports
    test_imports()

def test_imports():
    """Test that critical imports work"""
    
    imports_to_test = [
        ("click", "Click CLI framework"),
        ("yaml", "PyYAML configuration"),
        ("rich", "Rich console output"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("sklearn", "Scikit-learn"),
        ("scipy", "SciPy")
    ]
    
    failed_imports = []
    
    for module, description in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Failed imports: {', '.join(failed_imports)}")
        print("You may need to install these manually or check Databricks library configuration.")
    else:
        print("\n🎉 All critical imports successful!")

def setup_databricks_environment():
    """Set up Databricks-specific environment variables"""
    
    print("\n🏗️ Setting up Databricks environment...")
    
    # Check if we're in Databricks
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        print(f"✅ Databricks detected (Runtime: {os.environ['DATABRICKS_RUNTIME_VERSION']})")
        
        # Set environment variables for better compatibility
        os.environ['MLLM_DATABRICKS'] = 'true'
        
        # Add common Databricks paths to Python path
        import sys
        databricks_paths = [
            '/databricks/driver',
            '/databricks/driver/src',
            '/Workspace/Repos',
            '/Workspace'
        ]
        
        for path in databricks_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.append(path)
                print(f"📁 Added to Python path: {path}")
    else:
        print("ℹ️ Not running on Databricks")

if __name__ == "__main__":
    print("🎯 MLLM Pipeline - Databricks Setup")
    print("=" * 40)
    
    # Setup environment
    setup_databricks_environment()
    
    # Install dependencies
    install_databricks_dependencies()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Upload your project files to Databricks")
    print("2. Run: python debug_imports.py")
    print("3. Run: python cli.py explore --help")
    print("4. Start with: python cli.py explore --data-dir /path/to/your/data")