#!/bin/bash

set -e

# Default values
CAMERA_TYPE="realsense"
LAUNCH_SAM2="true"
LAUNCH_MASK2PCL="true" 
LAUNCH_TRACK_NODE="true"
LAUNCH_YOLO_PROMPT="true"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --camera) CAMERA_TYPE="$2"; shift 2 ;;
        --sam2) LAUNCH_SAM2="$2"; shift 2 ;;
        --mask2pcl) LAUNCH_MASK2PCL="$2"; shift 2 ;;
        --track) LAUNCH_TRACK_NODE="$2"; shift 2 ;;
        --yolo-prompt) LAUNCH_YOLO_PROMPT="$2"; shift 2 ;;
        --all) 
            LAUNCH_SAM2="true"
            LAUNCH_MASK2PCL="true" 
            LAUNCH_TRACK_NODE="true"
            LAUNCH_YOLO_PROMPT="false"
            shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --camera TYPE          Camera type: azure, realsense, sim_head, sim_wrist"
            echo "  --sam2 true/false      Launch SAM2 node (default: true)"
            echo "  --mask2pcl true/false  Launch mask2pcl node (default: false)"
            echo "  --track true/false     Launch track node (default: false)"
            echo "  --yolo-prompt true/false Launch YOLO prompt node (default: false)"
            echo "  --all                  Launch all nodes except yolo-prompt"
            echo ""
            echo "Examples:"
            echo "  $0                     # Just SAM2 with realsense"
            echo "  $0 --camera azure      # SAM2 with azure kinect"
            echo "  $0 --all               # All nodes except yolo-prompt"
            echo "  $0 --mask2pcl true     # SAM2 + mask2pcl"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set camera-specific topics
case $CAMERA_TYPE in
    azure)
        IMAGE_TOPIC="/k4a/rgb/image_raw"
        DEPTH_TOPIC="/k4a/depth_to_rgb/image_raw" 
        CAM_INFO="/k4a/rgb/camera_info"
        TARGET_FRAME="rgb_camera_link"
        CAMERA_FRAME="rgb_camera_link"
        DEPTH_DIVISOR="1"
        ;;
    realsense)
        IMAGE_TOPIC="/camera/camera/color/image_raw"
        DEPTH_TOPIC="/camera/camera/depth/image_rect_raw"
        CAM_INFO="/camera/camera/color/camera_info"
        TARGET_FRAME="camera_link"
        CAMERA_FRAME="camera_color_optical_frame"
        DEPTH_DIVISOR="1000"
        ;;
    sim_head)
        IMAGE_TOPIC="/head_front_camera/rgb/image_raw"
        DEPTH_TOPIC="/head_front_camera/depth/image_raw"
        CAM_INFO="/head_front_camera/rgb/camera_info"
        TARGET_FRAME="base_footprint"
        CAMERA_FRAME="head_front_camera_link" 
        DEPTH_DIVISOR="1"
        ;;
    sim_wrist)
        IMAGE_TOPIC="/realsense_d435/color/image_raw"
        DEPTH_TOPIC="/realsense_d435/aligned_depth_to_color/image_raw"
        CAM_INFO="/realsense_d435/aligned_depth_to_color/camera_info"
        TARGET_FRAME="base_footprint"
        CAMERA_FRAME="realsense_d435_link"
        DEPTH_DIVISOR="1000"
        ;;
    *)
        echo "Unknown camera type: $CAMERA_TYPE"
        exit 1
        ;;
esac

# Environment setup
SAM2_VENV_PATH=~/venvs/sam2_realtime_venv/bin/activate
if [ -f "$SAM2_VENV_PATH" ]; then
    echo "Activating SAM2 virtual environment"
    source "$SAM2_VENV_PATH"
else
    echo "Warning: SAM2 virtual environment not found"
fi

if [ "$LAUNCH_SAM2" == "true" ] && [ -n "$SAM2_ASSETS_DIR" ]; then
    echo "Changing directory to SAM2 assets: $SAM2_ASSETS_DIR"
    cd "$SAM2_ASSETS_DIR"
fi

# Launch info
echo "==============================================="
echo "SAM2 REALTIME LAUNCH"
echo "==============================================="
echo "Camera: $CAMERA_TYPE"
echo "Nodes:"
echo "  SAM2:          $LAUNCH_SAM2"
echo "  Mask2PCL:      $LAUNCH_MASK2PCL"
echo "  Track:         $LAUNCH_TRACK_NODE"
echo "  YOLO Prompt:   $LAUNCH_YOLO_PROMPT"
echo "==============================================="

# Launch
ros2 launch sam2_realtime_bringup master.launch.py \
    camera_type:=$CAMERA_TYPE \
    launch_sam2:=$LAUNCH_SAM2 \
    launch_mask2pcl:=$LAUNCH_MASK2PCL \
    launch_track_node:=$LAUNCH_TRACK_NODE \
    launch_yolo_prompt:=$LAUNCH_YOLO_PROMPT \
    image_topic:=$IMAGE_TOPIC \
    depth_topic:=$DEPTH_TOPIC \
    cam_info:=$CAM_INFO \
    target_frame:=$TARGET_FRAME \
    camera_frame:=$CAMERA_FRAME \
    depth_divisor:=$DEPTH_DIVISOR