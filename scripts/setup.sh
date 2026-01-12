#!/bin/bash
# Setup script for Diffusion Boltzmann Sampler
#
# Usage: ./scripts/setup.sh

set -e

echo "======================================"
echo "Diffusion Boltzmann Sampler Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check Node.js version
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required but not found."
    exit 1
fi

NODE_VERSION=$(node -v)
echo "Found Node.js $NODE_VERSION"

# Create virtual environment
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate and install dependencies
echo ""
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create .env files if they don't exist
echo ""
echo "Setting up environment files..."
if [ ! -f ".env" ]; then
    echo "DEBUG=false" > .env
    echo "Created .env"
fi

if [ ! -f "frontend/.env" ]; then
    cp frontend/.env.example frontend/.env
    echo "Created frontend/.env"
fi

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To start development:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Start backend: make backend"
echo "  3. Start frontend: make frontend (in new terminal)"
echo ""
echo "Or use: make dev"
