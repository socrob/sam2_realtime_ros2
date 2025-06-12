#!/bin/bash

# Exit on error
set -e

# Default to RealSense
CAMERA_TYPE="realsense"

# Parse optional camera argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA_TYPE="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Activate the virtual environment
VENV_PATH=~/venvs/sam2_realtime_venv/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Choose image topic based on selected camera
if [ "$CAMERA_TYPE" == "azure" ]; then
    IMAGE_TOPIC="/k4a/rgb/image_raw"
elif [ "$CAMERA_TYPE" == "realsense" ]; then
    IMAGE_TOPIC="/camera/camera/color/image_raw"
else
    echo "❌ Unknown camera type: $CAMERA_TYPE (expected: azure or realsense)"
    exit 1
fi

# === RUN NODE ===
echo "🚀 Launching YOLO Prompt node with $CAMERA_TYPE topic..."
ros2 run sam2_realtime yolo_prompt_node \
  --ros-args \
  -p image_topic:="${IMAGE_TOPIC}" \
  -p yolo_model:=yolov8n.pt \
  -p confidence_threshold:=0.9 \
  -p min_box_area:=2000 \
  -p max_aspect_ratio:=3.0


# USAGE EXAMPLE
# ./yolo_prompt_node.sh --camera azure
# ./yolo_prompt_node.sh --camera realsense

