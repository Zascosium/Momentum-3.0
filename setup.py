#!/usr/bin/env python3
"""
Setup script for Momentum-3.0 MLLM project.

This script allows the project to be installed as a Python package,
making imports easier and enabling development mode installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "mlflow>=2.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "jupyter>=1.0.0",
    ]

# Development dependencies
dev_requirements = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.10.0",
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.3.0",
    "pre-commit>=3.3.0",
]

# Optional dependencies for different use cases
extras_require = {
    "dev": dev_requirements,
    "test": [
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.10.0",
        "pytest-xdist>=3.3.0",
    ],
    "docs": [
        "sphinx>=6.2.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=2.0.0",
    ],
    "api": [
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.0",
        "gunicorn>=20.1.0",
    ],
    "cloud": [
        "boto3>=1.26.0",
        "azure-storage-blob>=12.16.0",
        "google-cloud-storage>=2.9.0",
    ],
    "databricks": [
        "databricks-cli>=0.17.0",
    ],
    "all": dev_requirements + [
        "sphinx>=6.2.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=2.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.0",
        "gunicorn>=20.1.0",
        "boto3>=1.26.0",
        "databricks-cli>=0.17.0",
    ]
}

setup(
    name="momentum-mllm",
    version="1.0.0",
    description="Production-ready Multimodal LLM combining time series and text data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MLLM Team",
    author_email="your-email@example.com",
    url="https://github.com/your-username/Momentum-3.0",
    project_urls={
        "Bug Reports": "https://github.com/your-username/Momentum-3.0/issues",
        "Source": "https://github.com/your-username/Momentum-3.0",
        "Documentation": "https://github.com/your-username/Momentum-3.0#readme",
    },
    
    # Package discovery
    packages=find_packages(where="mllm_project"),
    package_dir={"": "mllm_project"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": [
            "config/*.yaml",
            "config/*.yml",
            "config/*.json",
            "notebooks/*.py",
            "*.md",
            "*.txt",
            "*.yaml",
            "*.yml",
        ],
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "mllm-train=src.training.trainer:main",
            "mllm-evaluate=src.utils.inference_utils:main",
            "mllm-preprocess=src.data.preprocessing:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords=[
        "machine learning",
        "deep learning",
        "multimodal",
        "time series",
        "natural language processing",
        "transformers",
        "pytorch",
        "databricks",
        "mlflow",
        "production",
    ],
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Testing
    test_suite="tests",
    tests_require=extras_require["test"],
)
