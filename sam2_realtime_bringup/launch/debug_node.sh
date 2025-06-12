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

# Set image topic based on selected camera
if [ "$CAMERA_TYPE" == "azure" ]; then
    IMAGE_TOPIC="/k4a/rgb/image_raw"
elif [ "$CAMERA_TYPE" == "realsense" ]; then
    IMAGE_TOPIC="/camera/camera/color/image_raw"
else
    echo "❌ Unknown camera type: $CAMERA_TYPE (expected: azure or realsense)"
    exit 1
fi

# Run the debug node with parameters
echo "🛠️  Launching sam2_debug_node for $CAMERA_TYPE..."
ros2 run sam2_realtime debug_node --ros-args \
  -p image_topic:="$IMAGE_TOPIC" \
  -p tracker_topic:="/tracked_object"

# USAGE EXAMPLES:
# ./debug_node.sh --camera azure
# ./debug_node.sh --camera realsense
