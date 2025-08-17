#!/bin/bash

# Example script for running inference demonstration

echo "Running Inference Demo"
echo "======================"

# Set paths
MODEL_PATH="${MODEL_PATH:-./checkpoints/best_model.pt}"
CONFIG_DIR="${CONFIG_DIR:-./config}"
DEMO_DIR="${DEMO_DIR:-./demo}"

# Demo parameters
NUM_EXAMPLES="${NUM_EXAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-0.8}"

# Run interactive demo
echo "Choose demo mode:"
echo "1. Standard demo"
echo "2. Batch processing demo"
echo "3. Interactive mode"
echo "4. Performance benchmark"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Running standard demo..."
        python cli.py demo \
            --model-path "$MODEL_PATH" \
            --config-dir "$CONFIG_DIR" \
            --demo-dir "$DEMO_DIR" \
            --num-examples $NUM_EXAMPLES \
            --temperature $TEMPERATURE \
            --verbose
        ;;
    2)
        echo "Running batch processing demo..."
        python cli.py demo \
            --model-path "$MODEL_PATH" \
            --config-dir "$CONFIG_DIR" \
            --demo-dir "$DEMO_DIR" \
            --batch-demo \
            --num-examples $NUM_EXAMPLES \
            --verbose
        ;;
    3)
        echo "Starting interactive mode..."
        python cli.py demo \
            --model-path "$MODEL_PATH" \
            --config-dir "$CONFIG_DIR" \
            --demo-dir "$DEMO_DIR" \
            --interactive \
            --streaming \
            --verbose
        ;;
    4)
        echo "Running performance benchmark..."
        python cli.py demo \
            --model-path "$MODEL_PATH" \
            --config-dir "$CONFIG_DIR" \
            --demo-dir "$DEMO_DIR" \
            --num-examples 50 \
            --verbose
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "Demo complete! Results in $DEMO_DIR"
