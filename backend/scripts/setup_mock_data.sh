#!/bin/bash

echo "Setting up mock data for YC demo..."

# Make sure the current directory is the script directory
cd "$(dirname "$0")"

# Create necessary directories
if [[ -d "/app" ]]; then
    # In Docker, use absolute paths
    mkdir -p /app/data/screenshots
    SCREENSHOTS_DIR="/app/data/screenshots"
else
    # In local environment
    mkdir -p ../../data/screenshots
    SCREENSHOTS_DIR="../../data/screenshots"
fi

echo "Screenshots will be saved to: $SCREENSHOTS_DIR"

# Activate virtual environment if it exists and we're not in Docker
if [[ ! -d "/app" ]] && [[ -f "../../venv/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source ../../venv/bin/activate
fi

# Run the data generation script
echo "Running mock data generation script..."
python3 generate_mock_data.py

# Check if data generation was successful by checking if alerts exist
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to generate mock data!"
    exit 1
fi

# Run the screenshot generation script
echo "Running mock screenshot generation script..."
python3 generate_mock_screenshots.py

if [[ $? -ne 0 ]]; then
    echo "Warning: Failed to generate mock screenshots!"
fi

echo "Mock data setup complete!"
echo "You can now view the populated dashboard in your browser." 