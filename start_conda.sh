#!/bin/bash

# Alternative startup script using conda (if needed)

echo "🚀 Starting Shelf Analysis API with conda..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Installing miniconda..."
    echo "Please install miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
if ! conda env list | grep -q "shelf-analysis"; then
    echo "📦 Creating conda environment..."
    conda create -n shelf-analysis python=3.11 -y
fi

echo "🔧 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate shelf-analysis

echo "📥 Installing dependencies..."
conda install -c conda-forge fastapi uvicorn opencv pandas numpy pillow requests -y
pip install ultralytics openai anthropic python-dotenv aiofiles

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file with your API keys!"
fi

echo "🌟 Starting FastAPI server..."
python -m src.shelf_analyzer.main
