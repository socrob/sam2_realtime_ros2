#!/bin/bash
set -e

# Default to RealSense
CAMERA_TYPE="realsense"

# Parse CLI args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Activate venv
VENV_PATH=~/venvs/sam2_realtime_venv/bin/activate
if [ -f "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check assets path
if [ -z "$SAM2_ASSETS_DIR" ]; then
    echo "❌ SAM2_ASSETS_DIR is not set. Please export it first."
    exit 1
fi

echo "📂 Changing directory to SAM2 assets: $SAM2_ASSETS_DIR"
cd "$SAM2_ASSETS_DIR"

# Set image topics based on camera type
if [ "$CAMERA_TYPE" == "azure" ]; then
    IMAGE_TOPIC="/k4a/rgb/image_raw"
elif [ "$CAMERA_TYPE" == "realsense" ]; then
    IMAGE_TOPIC="/camera/camera/color/image_raw"
elif [ "$CAMERA_TYPE" == "sim_head" ]; then
    IMAGE_TOPIC="/head_front_camera/rgb/image_raw"
elif [ "$CAMERA_TYPE" == "sim_wrist" ]; then
    IMAGE_TOPIC="/realsense_d435/color/image_raw"
else
    echo "❌ Unknown camera type: $CAMERA_TYPE (use 'realsense' or 'azure')"
    exit 1
fi

# === RUN NODE ===
echo "🚀 Launching SAM2 node using $CAMERA_TYPE topics..."
ros2 launch sam2_realtime_bringup sam2_realtime_node.launch.py \
    image_topic:=${IMAGE_TOPIC} \
    image_reliability:=2 \
    model_cfg:=configs/sam2.1/sam2.1_hiera_s.yaml \
    checkpoint:=checkpoints/sam2.1_hiera_small.pt \
    live_visualization:=True



# USAGE EXAMPLE
# ./sam2_realtime_node.sh --camera azure
# ./sam2_realtime_node.sh --camera realsense
