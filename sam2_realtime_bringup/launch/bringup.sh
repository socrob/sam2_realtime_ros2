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

# Check that the SAM2_ASSETS_DIR env var is set
if [ -z "$SAM2_ASSETS_DIR" ]; then
    echo "❌ SAM2_ASSETS_DIR is not set. Please export it first."
    exit 1
fi

# Change directory to the root of the SAM2 assets for Hydra compatibility
echo "📂 Changing directory to SAM2 assets: $SAM2_ASSETS_DIR"
cd "$SAM2_ASSETS_DIR"

# === RUN NODE ===
echo "🚀 Launching SAM2 node..."
ros2 run sam2_realtime sam2_realtime_node
# ros2 launch sam2_realtime_bringup sam2_realtime_node.launch.py
