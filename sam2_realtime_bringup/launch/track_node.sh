#!/bin/bash

# Exit on error
set -e

# Activate the virtual environment
VENV_PATH=~/venvs/sam2_realtime_venv/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi


# Run the tracking node via launch file
echo "🚀 Launching Track Node..."
ros2 launch sam2_realtime_bringup track_node.launch.py