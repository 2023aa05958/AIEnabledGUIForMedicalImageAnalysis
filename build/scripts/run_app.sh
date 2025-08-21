#!/bin/bash
# AI-Enabled GUI for Medical Image Analysis - Quick Start Script
# This script ensures proper Python environment and starts the secure application

echo "==============================================================="
echo "   AI-Enabled GUI for Medical Image Analysis"
echo "   Starting Secure Application..."
echo "==============================================================="

# Change to the root application directory (go up from build/scripts)
cd "$(dirname "$0")/../.."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.9+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Using system Python."
fi

# Install requirements if needed
if [ -f "config/requirements.txt" ]; then
    echo "📦 Checking dependencies..."
    pip install -r config/requirements.txt > /dev/null 2>&1
fi

# Start the secure application
echo "🚀 Starting secure medical image analysis application..."
python3 src/apps/app_secure.py

echo ""
echo "Application stopped."
