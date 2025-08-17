#!/bin/bash

# Example script for running data exploration

echo "Running Data Exploration Pipeline"
echo "================================="

# Set paths
DATA_DIR="${DATA_DIR:-./data/time_mmd}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/exploration}"
CONFIG_DIR="${CONFIG_DIR:-./config}"

# Run exploration with various options
python cli.py explore \
    --data-dir "$DATA_DIR" \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --sample-size 1000 \
    --generate-report \
    --verbose

echo "Exploration complete! Check results in $OUTPUT_DIR"
