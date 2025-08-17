#!/bin/bash

# Example script for running the complete end-to-end pipeline

echo "Running Complete Pipeline"
echo "========================="

# Set configuration
CONFIG_FILE="${CONFIG_FILE:-./config/pipeline_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./pipeline_results}"

# Run full pipeline
python cli.py pipeline \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --verbose

echo "Pipeline complete! Results in $OUTPUT_DIR"

# Generate summary
echo ""
echo "Pipeline Summary:"
echo "-----------------"
if [ -f "$OUTPUT_DIR/pipeline_report.md" ]; then
    head -n 20 "$OUTPUT_DIR/pipeline_report.md"
fi
