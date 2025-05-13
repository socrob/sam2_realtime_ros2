#!/bin/bash

# Exit on error
set -e

# Activate the virtual environment
VENV_PATH=~/venvs/sam2_realtime_venv/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH"
else
    echo "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# === RUN NODE ===
echo "🚀 Launching YOLO Prompt node..."
ros2 run sam2_realtime yolo_prompt_node