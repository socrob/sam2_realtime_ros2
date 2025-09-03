#!/bin/bash

# Exit on error
set -e

# Defaults
CAMERA_TYPE="realsense"          # azure | realsense
DETECT_CLASS="cup"            # e.g., "book", "person, cup" or "all"
YOLO_MODEL="yolov8n.pt"          # can be yolov8s.pt, custom .pt, etc.
CONF=0.4
MIN_AREA=800
MAX_AR=3.0
IMGSZ=640
IMAGE_TOPIC=""

# Parse args
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --camera) CAMERA_TYPE="$2"; shift 2 ;;
    --class|--classes|--detect) DETECT_CLASS="$2"; shift 2 ;;
    --model) YOLO_MODEL="$2"; shift 2 ;;
    --conf) CONF="$2"; shift 2 ;;
    --min-area) MIN_AREA="$2"; shift 2 ;;
    --max-ar) MAX_AR="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --image-topic) IMAGE_TOPIC="$2"; shift 2 ;;
    *) echo "❌ Unknown parameter: $1"; exit 1 ;;
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

# YOLO assets dir check (ultralytics loads from a path you provide)
if [ -z "$YOLO_ASSETS_DIR" ]; then
  echo "❌ YOLO_ASSETS_DIR environment variable not set (should point to directory containing $YOLO_MODEL)"; exit 1
fi

# Choose image topic if not provided
if [ -z "$IMAGE_TOPIC" ]; then
  if [ "$CAMERA_TYPE" == "azure" ]; then
    IMAGE_TOPIC="/k4a/rgb/image_raw"
  elif [ "$CAMERA_TYPE" == "realsense" ]; then
    IMAGE_TOPIC="/camera/color/image_raw"
  else
    echo "❌ Unknown camera type: $CAMERA_TYPE (expected: azure or realsense)"; exit 1
  fi
fi

# Run node (assumes entry point is `yolo_prompt_node` in package `sam2_realtime`)
# If you installed the classname-enabled script as a separate entry point, change the executable accordingly.
echo "🚀 Launching YOLO Prompt node"
echo "   camera=$CAMERA_TYPE | topic=$IMAGE_TOPIC | classes=\"$DETECT_CLASS\" | model=$YOLO_MODEL | conf=$CONF | imgsz=$IMGSZ"
ros2 run sam2_realtime yolo_prompt_node \
  --ros-args \
  -p image_topic:="${IMAGE_TOPIC}" \
  -p yolo_model:="${YOLO_MODEL}" \
  -p detect_class:="${DETECT_CLASS}" \
  -p confidence_threshold:=${CONF} \
  -p min_box_area:=${MIN_AREA} \
  -p max_aspect_ratio:=${MAX_AR} \
  -p imgsz:=${IMGSZ}

# USAGE EXAMPLES
# ./yolo_prompt_node.sh --camera azure --class "book"
# ./yolo_prompt_node.sh --camera realsense --class "person, cup" --conf 0.35 --imgsz 960
# ./yolo_prompt_node.sh --class all --model yolov8s.pt --min-area 600
