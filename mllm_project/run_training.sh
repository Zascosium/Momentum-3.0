#!/bin/bash
# Simple training execution script for Multimodal LLM

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
DOMAINS="Agriculture"
EPOCHS=3
BATCH_SIZE=2
LEARNING_RATE=5e-5
METHOD="pipeline"

# Print header
echo -e "${BLUE}🚀 Multimodal LLM Training Script${NC}"
echo "=================================="

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -d, --domains DOMAINS       Domains to train on (default: Agriculture)
    -e, --epochs EPOCHS         Number of epochs (default: 3)
    -b, --batch-size SIZE       Batch size (default: 2)
    -r, --learning-rate RATE    Learning rate (default: 5e-5)
    -m, --method METHOD         Execution method: pipeline|standalone (default: pipeline)
    -q, --quick                 Quick training (1 epoch, batch size 1)
    -f, --full                  Full training (5 epochs, all domains)
    -h, --help                  Show this help message

EXAMPLES:
    $0                          # Default training (Agriculture, 3 epochs)
    $0 -q                       # Quick test training
    $0 -f                       # Full training on all domains
    $0 -d "Agriculture Climate" -e 5 -b 4  # Custom training
    $0 -m standalone -d Climate -e 3       # Standalone training

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--domains)
            DOMAINS="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -r|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -q|--quick)
            DOMAINS="Agriculture"
            EPOCHS=1
            BATCH_SIZE=1
            echo -e "${YELLOW}⚡ Quick training mode enabled${NC}"
            shift
            ;;
        -f|--full)
            DOMAINS="Agriculture Climate Economy"
            EPOCHS=5
            BATCH_SIZE=4
            echo -e "${BLUE}🏋️ Full training mode enabled${NC}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate method
if [[ ! "$METHOD" =~ ^(pipeline|standalone)$ ]]; then
    echo -e "${RED}❌ Invalid method: $METHOD${NC}"
    echo "Valid methods: pipeline, standalone"
    exit 1
fi

# Print configuration
echo -e "${GREEN}📋 Training Configuration:${NC}"
echo "  Domains: $DOMAINS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE" 
echo "  Learning Rate: $LEARNING_RATE"
echo "  Method: $METHOD"
echo

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found. Please install Python.${NC}"
    exit 1
fi

# Check if project files exist
if [[ ! -f "run_pipeline.py" ]]; then
    echo -e "${RED}❌ Project files not found. Are you in the correct directory?${NC}"
    exit 1
fi

# Create directories
mkdir -p logs checkpoints outputs

echo -e "${BLUE}🔍 Validating environment...${NC}"

# Validate environment first
if [[ "$METHOD" == "pipeline" ]]; then
    python run_pipeline.py validate
elif [[ "$METHOD" == "standalone" ]]; then
    # Quick validation for standalone
    python -c "import torch; print('✅ PyTorch available'); print(f'CUDA: {torch.cuda.is_available()}')"
fi

if [[ $? -ne 0 ]]; then
    echo -e "${RED}❌ Environment validation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Environment validated${NC}"

# Execute training based on method
echo -e "${BLUE}🚀 Starting training...${NC}"

case $METHOD in
    "pipeline")
        echo -e "${BLUE}Using full pipeline runner${NC}"
        python run_pipeline.py full \
            --domains $DOMAINS \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --learning-rate $LEARNING_RATE
        ;;
    "standalone")
        echo -e "${BLUE}Using standalone training script${NC}"
        python train_standalone.py \
            --domains $DOMAINS \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --learning-rate $LEARNING_RATE
        ;;
esac

TRAINING_EXIT_CODE=$?

# Check results
if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo
    echo -e "${GREEN}🎉 Training completed successfully!${NC}"
    echo
    echo -e "${GREEN}📁 Generated Files:${NC}"
    
    # List checkpoints
    if [[ -d "checkpoints" ]] && [[ "$(ls -A checkpoints)" ]]; then
        echo "  Models:"
        ls -la checkpoints/*.pt 2>/dev/null | awk '{print "    " $9 " (" $5 " bytes)"}' || echo "    (no .pt files found)"
    fi
    
    # List outputs
    if [[ -d "outputs" ]] && [[ "$(ls -A outputs)" ]]; then
        echo "  Results:"
        ls -la outputs/*.json 2>/dev/null | awk '{print "    " $9}' || echo "    (no .json files found)"
    fi
    
    # List logs
    if [[ -d "logs" ]] && [[ "$(ls -A logs)" ]]; then
        echo "  Logs:"
        ls -la logs/*.log 2>/dev/null | awk '{print "    " $9}' | head -3 || echo "    (no .log files found)"
    fi
    
    echo
    echo -e "${BLUE}🎯 Next Steps:${NC}"
    echo "  1. Check model: ls -la checkpoints/"
    echo "  2. View results: cat outputs/training_summary.json"
    echo "  3. Check logs: tail logs/pipeline.log"
    
    if [[ "$METHOD" == "pipeline" ]]; then
        echo "  4. Run inference: python run_pipeline.py infer"
        echo "  5. Evaluate model: python run_pipeline.py evaluate"
    fi
    
else
    echo
    echo -e "${RED}❌ Training failed with exit code: $TRAINING_EXIT_CODE${NC}"
    echo
    echo -e "${YELLOW}🔍 Troubleshooting:${NC}"
    echo "  1. Check logs: tail logs/pipeline.log"
    echo "  2. Verify GPU: nvidia-smi"
    echo "  3. Check disk space: df -h"
    echo "  4. Reduce batch size: $0 --batch-size 1"
    echo "  5. Try quick training: $0 --quick"
    
    exit $TRAINING_EXIT_CODE
fi