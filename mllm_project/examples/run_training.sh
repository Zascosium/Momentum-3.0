#!/bin/bash

# Example script for running model training

echo "Running Model Training Pipeline"
echo "==============================="

# Set paths
CONFIG_DIR="${CONFIG_DIR:-./config}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-multimodal_llm_training}"

# Training parameters
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

# Run training
python cli.py train \
    --config-dir "$CONFIG_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --mixed-precision \
    --verbose

echo "Training complete! Model saved to $CHECKPOINT_DIR"
