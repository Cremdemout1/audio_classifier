#!/bin/bash

# Remove the old environment if it exists
if [ -d new_env ]; then
    echo "Removing old virtual environment..."
    rm -rf new_env
else
    echo "No existing virtual environment found."
fi

# Create a new virtual environment
echo "Creating a new virtual environment..."
python3 -m venv new_env

# Activate the new virtual environment
echo "Activating the virtual environment..."
source new_env/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate the virtual environment."
    exit 1
fi
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
fi
    echo "Installing dependencies..."
    pip install pandas torch torchcodec torchaudio matplotlib numpy pathlib sounddevice > /dev/null

echo "Environment setup complete!"

echo run this command: source new_env/bin/activate