#!/bin/bash

# Momentum-3.0 MLLM Project Setup Script
# This script creates a conda environment and installs all necessary dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="momentum-mllm"
PYTHON_VERSION="3.10"
ENV_NAME="momentum-mllm-env"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check conda installation
check_conda() {
    if command_exists conda; then
        print_success "Conda is installed"
        conda --version
        return 0
    else
        print_error "Conda is not installed or not in PATH"
        print_status "Please install Miniconda or Anaconda first:"
        print_status "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        print_status "  - Anaconda: https://www.anaconda.com/products/distribution"
        exit 1
    fi
}

# Function to create conda environment
create_conda_env() {
    print_status "Creating conda environment: $ENV_NAME"
    
    # Check if environment already exists
    if conda env list | grep -q "$ENV_NAME"; then
        print_warning "Environment '$ENV_NAME' already exists"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            conda env remove -n "$ENV_NAME" -y
        else
            print_status "Using existing environment"
            return 0
        fi
    fi
    
    # Create new environment
    print_status "Creating new conda environment with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    print_success "Conda environment '$ENV_NAME' created successfully"
}

# Function to install PyTorch with appropriate CUDA support
install_pytorch() {
    print_status "Installing PyTorch with CUDA support..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Check if CUDA is available
    if command_exists nvidia-smi; then
        print_status "NVIDIA GPU detected, installing PyTorch with CUDA support"
        # Install PyTorch with CUDA 11.8 (compatible with most setups)
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        print_warning "No NVIDIA GPU detected, installing CPU-only PyTorch"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    print_success "PyTorch installed successfully"
}

# Function to install other conda packages
install_conda_packages() {
    print_status "Installing conda packages..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Install packages available through conda-forge
    conda install -c conda-forge -y \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        jupyter \
        jupyterlab \
        notebook \
        ipywidgets \
        pyyaml \
        h5py \
        psutil \
        joblib \
        click \
        tqdm \
        requests \
        urllib3 \
        certifi \
        setuptools \
        wheel \
        pip
    
    print_success "Conda packages installed successfully"
}

# Function to install pip packages
install_pip_packages() {
    print_status "Installing pip packages..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install packages from requirements.txt
    if [ -f "requirements.txt" ]; then
        print_status "Installing packages from requirements.txt..."
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing core packages manually..."
        
        # Core ML packages
        pip install \
            transformers>=4.30.0 \
            datasets>=2.12.0 \
            tokenizers>=0.13.0 \
            huggingface_hub>=0.15.0 \
            accelerate>=0.20.0
        
        # MLflow and experiment tracking
        pip install \
            mlflow>=2.4.0 \
            mlflow-skinny>=2.4.0
        
        # Data processing
        pip install \
            pyarrow>=12.0.0 \
            fastparquet>=0.8.0
        
        # Visualization
        pip install \
            plotly>=5.14.0
        
        # Metrics and evaluation
        pip install \
            rouge-score>=0.1.2 \
            sacrebleu>=2.3.0 \
            bert-score>=0.3.13
        
        # Configuration
        pip install \
            omegaconf>=2.3.0 \
            hydra-core>=1.3.0 \
            python-dotenv>=1.0.0
        
        # Utilities
        pip install \
            rich>=13.3.0 \
            loguru>=0.7.0 \
            memory-profiler>=0.60.0
        
        # Testing
        pip install \
            pytest>=7.3.0 \
            pytest-cov>=4.1.0 \
            pytest-mock>=3.10.0 \
            pytest-xdist>=3.3.0
        
        # Development tools
        pip install \
            black>=23.3.0 \
            isort>=5.12.0 \
            flake8>=6.0.0 \
            mypy>=1.3.0 \
            pre-commit>=3.3.0
        
        # Optional: API and deployment
        pip install \
            fastapi>=0.95.0 \
            uvicorn>=0.22.0 \
            pydantic>=1.10.0 \
            gunicorn>=20.1.0
    fi
    
    print_success "Pip packages installed successfully"
}

# Function to install project in development mode
install_project() {
    print_status "Installing project in development mode..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Install project in editable mode
    if [ -f "setup.py" ]; then
        pip install -e .
    else
        print_warning "setup.py not found, skipping project installation"
        print_status "You can manually install the project later with: pip install -e ."
    fi
    
    print_success "Project installation completed"
}

# Function to setup pre-commit hooks
setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Test critical imports
    python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'PyTorch import failed: {e}')

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers import failed: {e}')

try:
    import numpy as np
    print(f'NumPy version: {np.__version__}')
except ImportError as e:
    print(f'NumPy import failed: {e}')

try:
    import pandas as pd
    print(f'Pandas version: {pd.__version__}')
except ImportError as e:
    print(f'Pandas import failed: {e}')

try:
    import mlflow
    print(f'MLflow version: {mlflow.__version__}')
except ImportError as e:
    print(f'MLflow import failed: {e}')

print('\\nInstallation verification completed!')
"
    
    print_success "Installation verification completed"
}

# Function to create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_env.sh << EOF
#!/bin/bash
# Momentum-3.0 MLLM Environment Activation Script

# Activate conda environment
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Set environment variables
export PYTHONPATH="\${PYTHONPATH}:\$(pwd)/mllm_project/src"
export CUDA_VISIBLE_DEVICES=0  # Modify as needed
export TOKENIZERS_PARALLELISM=false  # Avoid warnings

# Print environment info
echo "🚀 Momentum-3.0 MLLM Environment Activated!"
echo "📁 Project directory: \$(pwd)"
echo "🐍 Python: \$(which python)"
echo "📦 Environment: $ENV_NAME"
echo "🔥 CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "💡 Quick commands:"
echo "  - Run training: python mllm_project/notebooks/02_model_training.py"
echo "  - Run tests: pytest tests/"
echo "  - Start Jupyter: jupyter lab"
echo ""
EOF
    
    chmod +x activate_env.sh
    
    print_success "Activation script created: activate_env.sh"
}

# Function to display final instructions
display_instructions() {
    print_success "🎉 Setup completed successfully!"
    echo ""
    print_status "📋 Next steps:"
    echo "  1. Activate the environment:"
    echo "     source activate_env.sh"
    echo "     # OR manually: conda activate $ENV_NAME"
    echo ""
    echo "  2. Set up your data:"
    echo "     # Place your Time-MMD dataset in: mllm_project/data/raw/"
    echo ""
    echo "  3. Configure the project:"
    echo "     # Edit configuration files in: mllm_project/config/"
    echo ""
    echo "  4. Start training:"
    echo "     # Run: python mllm_project/notebooks/02_model_training.py"
    echo ""
    echo "  5. Run tests:"
    echo "     pytest tests/"
    echo ""
    echo "  6. Start Jupyter Lab:"
    echo "     jupyter lab"
    echo ""
    print_status "📚 Documentation:"
    echo "  - README.md: Complete project documentation"
    echo "  - mllm_project/config/: Configuration files"
    echo "  - mllm_project/notebooks/: Training and evaluation notebooks"
    echo ""
    print_success "Happy training! 🚀"
}

# Main execution
main() {
    echo "🚀 Momentum-3.0 MLLM Project Setup"
    echo "=================================="
    echo ""
    
    # Check prerequisites
    check_conda
    
    # Create environment and install packages
    create_conda_env
    install_pytorch
    install_conda_packages
    install_pip_packages
    install_project
    setup_pre_commit
    
    # Verify and finalize
    verify_installation
    create_activation_script
    display_instructions
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Momentum-3.0 MLLM Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-name NAME        Name for conda environment (default: $ENV_NAME)"
            echo "  --python-version VER   Python version (default: $PYTHON_VERSION)"
            echo "  --cpu-only            Install CPU-only PyTorch"
            echo "  --help, -h            Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
