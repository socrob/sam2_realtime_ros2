#!/bin/bash


# Exit on error
set -e


# Default to RealSense
CAMERA_TYPE="realsense"


# Parse optional camera argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA_TYPE="$2"; shift 2 ;;
        *)
            echo "❌ Unknown parameter passed: $1"; exit 1 ;;
    esac
done


# Activate the virtual environment
VENV_PATH=~/venvs/sam2_realtime_venv/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment from $VENV_PATH"
    # shellcheck source=/dev/null
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found at $VENV_PATH"; exit 1
fi


# Set depth and camera info topics based on selected camera
if [ "$CAMERA_TYPE" == "azure" ]; then
    DEPTH_TOPIC="/k4a/depth_to_rgb/image_raw"
    CAM_INFO="/k4a/rgb/camera_info"
    TARGET_FRAME="rgb_camera_link"
    DEPTH_DIVISOR="1" # Azure K4A typically publishes meters already
elif [ "$CAMERA_TYPE" == "realsense" ]; then
    # Standard RealSense topics
    DEPTH_TOPIC="/camera/camera/depth/image_rect_raw"
    CAM_INFO="/camera/camera/color/camera_info"
    TARGET_FRAME="camera_color_optical_frame"
    DEPTH_DIVISOR="1000" # RealSense depth is in millimeters
elif [ "$CAMERA_TYPE" == "sim_head" ]; then
    DEPTH_TOPIC="/head_front_camera/depth/image_raw"
    CAM_INFO="/head_front_camera/rgb/camera_info"
    # TARGET_FRAME="head_front_camera_link"
    TARGET_FRAME="base_footprint"
    DEPTH_DIVISOR="1"
elif [ "$CAMERA_TYPE" == "sim_wrist" ]; then
    DEPTH_TOPIC="/realsense_d435/aligned_depth_to_color/image_raw"
    CAM_INFO="/realsense_d435/aligned_depth_to_color/camera_info"
    # TARGET_FRAME="realsense_d435_link"
    TARGET_FRAME="base_footprint"
    DEPTH_DIVISOR="1000"
else
    echo "❌ Unknown camera type: $CAMERA_TYPE (expected: azure or realsense)"; exit 1
fi


SAM2_MASK_TOPIC="/sam2/mask"


# Run the node via the launch file
echo "🚀 Launching mask2pcl for $CAMERA_TYPE..."
ros2 launch sam2_realtime_bringup mask2pcl.launch.py \
    namespace:="mask2pcl" \
    depth_topic:="${DEPTH_TOPIC}" \
    cam_info:="${CAM_INFO}" \
    sam2_mask_topic:="${SAM2_MASK_TOPIC}" \
    target_frame:="${TARGET_FRAME}" \
    depth_image_units_divisor:="${DEPTH_DIVISOR}" \
    enable:=true \
    min_mask_area:=200 \
    cloud_stride:=2


# USAGE EXAMPLES
# ./mask2pcl.sh --camera azure
# ./mask2pcl.sh --camera realsense
