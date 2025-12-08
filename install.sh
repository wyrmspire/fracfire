#!/bin/bash
# FracFire Installation Script

set -e

echo "======================================"
echo "FracFire Installation"
echo "======================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi
echo "✅ Python 3 found"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi
echo "✅ Node.js found"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed"
    exit 1
fi
echo "✅ npm found"

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install -r api/requirements.txt

echo ""
echo "Installing Node.js dependencies..."
npm install

echo ""
echo "Setting up environment files..."
if [ ! -f "frontend/.env.local" ]; then
    cp frontend/.env.example frontend/.env.local
    echo "✅ Created frontend/.env.local (please configure)"
else
    echo "⚠️  frontend/.env.local already exists"
fi

if [ ! -f "api/.env" ]; then
    cp api/.env.example api/.env
    echo "✅ Created api/.env"
else
    echo "⚠️  api/.env already exists"
fi

echo ""
echo "Creating data directories..."
mkdir -p data/generated data/uploads models/trained

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "To start the application:"
echo "  npm run dev"
echo ""
echo "Or run separately:"
echo "  Terminal 1: npm run dev:backend"
echo "  Terminal 2: npm run dev:frontend"
echo ""
echo "The app will be available at:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
