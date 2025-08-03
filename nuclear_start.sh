#!/bin/bash

# NUCLEAR OPTION: Pain-free Python setup

echo "🚀 NUCLEAR SETUP: Making Python ML just work..."

# Remove any existing environment
rm -rf venv

# Create environment
python3.12 -m venv venv
source venv/bin/activate

echo "📦 Installing core packages in correct order..."

# Install build tools
pip install --upgrade pip setuptools wheel

# Install NumPy 1.x FIRST
pip install "numpy>=1.24.0,<2.0.0"

# Install compatible versions that actually work together
pip install opencv-python==4.8.1.78  # Older, more stable
pip install ultralytics==8.0.196     # Known working version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install web framework
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 python-multipart

# Install LLM APIs
pip install openai==1.3.7 anthropic==0.7.8

# Install utilities
pip install python-dotenv requests aiofiles pandas pydantic

echo "✅ Environment setup complete!"
echo "🧪 Testing imports..."

python -c "
import cv2
import numpy as np
from ultralytics import YOLO
import fastapi
print('✅ All imports working!')
print(f'NumPy version: {np.__version__}')
print(f'OpenCV version: {cv2.__version__}')
"

# Check if .env exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "🔑 Created .env file - add your API keys!"
fi

echo "🌟 Starting server..."
python -m src.shelf_analyzer.main
