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
ros2 run sam2_realtime yolo_prompt_node \
  --ros-args \
  -p image_topic:=/k4a/rgb/image_raw \
  -p yolo_model:=yolov8n.pt \
  -p confidence_threshold:=0.9 \
  -p min_box_area:=2000 \
  -p max_aspect_ratio:=3.0

# ros2 run sam2_realtime yolo_prompt_node \
#   --ros-args \
#   -p image_topic:=/camera/camera/color/image_raw \
#   -p yolo_model:=yolov8n.pt \
#   -p confidence_threshold:=0.9 \
#   -p min_box_area:=2000 \
#   -p max_aspect_ratio:=3.0
