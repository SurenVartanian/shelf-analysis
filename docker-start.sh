#!/bin/bash

# Docker startup script for Shelf Analysis API

set -e

echo "🚀 Starting Shelf Analysis API..."

# Check if required environment variables are set
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  Warning: No API keys configured. Some features may not work."
    echo "   Set OPENAI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS environment variables."
fi

# Create necessary directories
mkdir -p /app/data
mkdir -p /app/logs
mkdir -p /app/models/yolo

# Check if custom YOLO model exists
if [ ! -f "/app/models/yolo/shelf_analysis_custom.pt" ]; then
    echo "⚠️  Warning: Custom YOLO model not found at /app/models/yolo/shelf_analysis_custom.pt"
    echo "   The system will use the standard YOLO model instead."
fi

# Set Python path
export PYTHONPATH=/app/src

# Wait for any dependencies (if using docker-compose with other services)
echo "⏳ Waiting for dependencies..."
sleep 2

# Start the application
echo "🎯 Starting FastAPI server..."
exec uvicorn src.shelf_analyzer.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info 