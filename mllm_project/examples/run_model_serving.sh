#!/bin/bash

# Example script for starting model serving API

echo "Starting Model Serving API"
echo "=========================="

# Set paths
MODEL_PATH="${MODEL_PATH:-./checkpoints/best_model.pt}"
CONFIG_DIR="${CONFIG_DIR:-./config}"

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
WORKERS="${WORKERS:-1}"

# Optional API key for authentication
API_KEY="${API_KEY:-}"

# Build command
CMD="python cli.py serve \
    --model-path $MODEL_PATH \
    --config-dir $CONFIG_DIR \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --enable-cors"

# Add API key if provided
if [ -n "$API_KEY" ]; then
    CMD="$CMD --api-key $API_KEY"
fi

# Add reload flag for development
if [ "$ENV" = "development" ]; then
    CMD="$CMD --reload"
fi

echo "Starting server on http://$HOST:$PORT"
echo "API documentation: http://$HOST:$PORT/docs"
echo "Health check: http://$HOST:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run server
$CMD
