#!/bin/bash

# Heart Sound Classification - Setup Script (Linux/Mac)
# This script sets up both backend and frontend

echo "======================================"
echo "Heart Sound Classification - Setup"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
echo ""

# Check Node.js
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found!${NC}"
    echo "Please install Node.js 14+ from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
echo -e "${GREEN}✅ Node.js $NODE_VERSION found${NC}"
echo ""

# Setup Backend
echo "======================================"
echo "Setting up Backend..."
echo "======================================"
cd backend

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Backend setup complete!${NC}"
else
    echo -e "${RED}❌ Backend setup failed!${NC}"
    exit 1
fi

# Deactivate virtual environment
deactivate

cd ..
echo ""

# Setup Frontend
echo "======================================"
echo "Setting up Frontend..."
echo "======================================"
cd frontend

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Frontend setup complete!${NC}"
else
    echo -e "${RED}❌ Frontend setup failed!${NC}"
    exit 1
fi

cd ..
echo ""

# Check model file
echo "======================================"
echo "Checking Model File..."
echo "======================================"

MODEL_PATH="../results/models/1dcnn_method_best.h5"
if [ -f "$MODEL_PATH" ]; then
    echo -e "${GREEN}✅ Model file found!${NC}"
else
    echo -e "${YELLOW}⚠️  Model file not found at: $MODEL_PATH${NC}"
    echo "Please ensure the model file exists or train a new model."
fi

echo ""
echo "======================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "======================================"
echo ""
echo "To start the application:"
echo "  ./start.sh"
echo ""
echo "Or manually:"
echo "  Terminal 1: cd backend && source venv/bin/activate && python app.py"
echo "  Terminal 2: cd frontend && npm start"
echo ""
