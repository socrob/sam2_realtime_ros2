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

# Set depth and camera info topics based on selected camera
if [ "$CAMERA_TYPE" == "azure" ]; then
    DEPTH_TOPIC="/k4a/depth_to_rgb/image_raw"
    CAM_INFO="/k4a/rgb/camera_info"
    TARGET_FRAME="camera_base"
    DEPTH_DIVISOR="1"
elif [ "$CAMERA_TYPE" == "realsense" ]; then
    DEPTH_TOPIC="/camera/camera/depth/image_rect_raw"
    CAM_INFO="/camera/camera/color/camera_info"
    TARGET_FRAME="camera_color_optical_frame"
    DEPTH_DIVISOR="1000"
else
    echo "❌ Unknown camera type: $CAMERA_TYPE (expected: azure or realsense)"
    exit 1
fi

# Run the tracking node via launch file
echo "🚀 Launching Track Node for $CAMERA_TYPE..."
ros2 launch sam2_realtime_bringup track_node.launch.py \
  depth_topic:="$DEPTH_TOPIC" \
  cam_info:="$CAM_INFO" \
  target_frame:="$TARGET_FRAME" \
  depth_image_units_divisor:="$DEPTH_DIVISOR"



# USAGE EXAMPLE
# ./track_node.sh --camera azure
# ./track_node.sh --camera realsense
