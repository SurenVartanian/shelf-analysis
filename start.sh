#!/bin/bash

# Shelf Analysis API Startup Script

echo "🚀 Starting Shelf Analysis API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install build tools first (Python 3.12 fix)
echo "🔨 Installing build tools for Python 3.12..."
pip install --upgrade pip
pip install setuptools>=70.0.0 wheel>=0.42.0 pip-tools

# Install dependencies with better dependency resolution
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file with your API keys!"
    echo "📝 Required: OPENAI_API_KEY"
fi

# Run the server
echo "🌟 Starting FastAPI server..."
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""

uvicorn src.shelf_analyzer.main:app --reload --host 0.0.0.0 --port 8000
