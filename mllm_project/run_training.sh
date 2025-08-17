#!/bin/bash

# Production Multimodal LLM Training Pipeline
# Single command execution script

set -e  # Exit on any error

echo "🚀 Starting Multimodal LLM Production Training Pipeline"
echo "=============================================================="

# Check if we're in the right directory
if [ ! -f "run_production_training.py" ]; then
    echo "❌ Error: run_production_training.py not found"
    echo "Please run this script from the mllm_project directory"
    exit 1
fi

# Check Python installation
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $python_version"

# Check if we're in Databricks
if [ -n "$DATABRICKS_RUNTIME_VERSION" ]; then
    echo "🟦 Databricks Runtime detected: $DATABRICKS_RUNTIME_VERSION"
    DATABRICKS_MODE=true
else
    echo "💻 Local environment detected"
    DATABRICKS_MODE=false
fi

# Install required packages if needed
echo "📦 Checking dependencies..."
python -c "import torch, transformers, numpy, pandas, sklearn, scipy" 2>/dev/null || {
    echo "Installing required packages..."
    pip install torch transformers numpy pandas scikit-learn scipy PyYAML
}

# Set default parameters
EPOCHS=3
BATCH_SIZE=""
EXPERIMENT_NAME=""
SKIP_MOCK_DATA=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --no-mock-data)
            SKIP_MOCK_DATA="--no-mock-data"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N           Number of training epochs (default: 3)"
            echo "  --batch-size N       Training batch size (auto-detected if not specified)"
            echo "  --experiment-name S  MLflow experiment name"
            echo "  --no-mock-data       Skip mock data generation"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Quick training with defaults"
            echo "  $0 --epochs 10 --batch-size 8        # Custom training parameters"
            echo "  $0 --experiment-name 'My_Experiment' # Custom experiment name"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python run_production_training.py --epochs $EPOCHS"

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    CMD="$CMD --experiment-name '$EXPERIMENT_NAME'"
fi

if [ -n "$SKIP_MOCK_DATA" ]; then
    CMD="$CMD $SKIP_MOCK_DATA"
fi

echo "🎯 Training Configuration:"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: ${BATCH_SIZE:-auto-detected}"
echo "   Experiment: ${EXPERIMENT_NAME:-auto-generated}"
echo "   Skip Mock Data: ${SKIP_MOCK_DATA:-false}"
echo ""

# Confirmation prompt (skip in Databricks)
if [ "$DATABRICKS_MODE" = false ]; then
    read -p "🤔 Start training with these settings? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Training cancelled"
        exit 0
    fi
fi

echo "🚀 Executing: $CMD"
echo "=============================================================="

# Execute the training pipeline
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "🎉 TRAINING COMPLETED SUCCESSFULLY!"
    echo "=============================================================="
    echo ""
    echo "📋 Next steps:"
    echo "  1. Check the generated models in: production_data/models/"
    echo "  2. Review training logs in: production_training.log"
    echo "  3. Test inference with the trained model"
    echo "  4. Deploy to production if results are satisfactory"
    echo ""
else
    echo ""
    echo "=============================================================="
    echo "❌ TRAINING FAILED!"
    echo "=============================================================="
    echo ""
    echo "🔍 Check the logs for error details:"
    echo "  - production_training.log"
    echo "  - Console output above"
    echo ""
    exit 1
fi