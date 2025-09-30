#!/bin/bash

# SafeLend startup script

set -e

echo "Starting SafeLend application..."

# Check if data directory exists
if [ ! -d "data/raw" ]; then
    echo "Creating data directories..."
    mkdir -p data/raw data/interim data/processed
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Check if reports directory exists
if [ ! -d "reports" ]; then
    echo "Creating reports directory..."
    mkdir -p reports
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if models exist
if [ ! -f "models/random_forest_model.pkl" ]; then
    echo "Warning: No trained models found. Please train models first."
    echo "Run: python src/models/train.py"
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
python -m uvicorn src.service.server:app --host 0.0.0.0 --port 8000 --reload
