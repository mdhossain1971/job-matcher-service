#!/bin/bash
# Run the Job Matcher ML Service locally

set -e

echo "=============================================="
echo "Job Matcher ML Service - Local Setup"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Run this script from the job-matcher-service directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install requirements
echo "Installing requirements (first time takes ~2-3 minutes)..."
pip install -r requirements.txt

echo ""
echo "=============================================="
echo "Starting server..."
echo "=============================================="
echo ""
echo "API Docs: http://localhost:5000/docs"
echo "Health:   http://localhost:5000/health"
echo ""

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
