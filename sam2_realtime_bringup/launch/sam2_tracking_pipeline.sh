#!/bin/bash

set -e

# ==============================================================================
# SAM2 Tracking Pipeline Launcher
#
# Launches:
#   - sam2_realtime_node
#   - mask2pcl
#   - track_node
#
# Example:
#   ./launch_sam2_tracking_pipeline.sh --camera azure --all
# ==============================================================================

# ------------------------------------------------------------------------------
# Default node launch options
# ------------------------------------------------------------------------------
CAMERA_TYPE="azure"

LAUNCH_SAM2="true"
LAUNCH_MASK2PCL="false"
LAUNCH_TRACK_NODE="true"

# ------------------------------------------------------------------------------
# Default namespaces
# ------------------------------------------------------------------------------
SAM2_NAMESPACE="sam2"
MASK2PCL_NAMESPACE="mask2pcl"
TRACK_NAMESPACE="track_node"

# ------------------------------------------------------------------------------
# Default SAM2 parameters
# ------------------------------------------------------------------------------
IMAGE_RELIABILITY="2"  # 1 = RELIABLE, 2 = BEST_EFFORT
MODEL_CFG="configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT="checkpoints/sam2.1_hiera_small.pt"
LIVE_VISUALIZATION="True"

# ------------------------------------------------------------------------------
# Default shared parameters
# ------------------------------------------------------------------------------
# Empty by default so it can be selected from the camera profile.
# The user can override it with --target-frame.
TARGET_FRAME=""
TARGET_FRAME_USER_SET="false"

# ------------------------------------------------------------------------------
# Default Mask2PCL parameters
# ------------------------------------------------------------------------------
MASK2PCL_ENABLE="true"
MASK2PCL_MIN_MASK_AREA="100"
CLOUD_STRIDE="4"
MASK2PCL_MAXIMUM_DETECTION_THRESHOLD="0.3"

# ------------------------------------------------------------------------------
# Default Tracker parameters
# ------------------------------------------------------------------------------
TRACK_ENABLE="true"
TRACK_MIN_MASK_AREA="1000"
PREDICT_RATE="10"
PRINT_MEASUREMENT_MARKER="true"
DEPTH_FILTER_PERCENTAGE="0.3"
TRACK_MAXIMUM_DETECTION_THRESHOLD="0.3"
MAX_DEPTH_JUMP="0.3"
RELOCK_WINDOW="1"

# ------------------------------------------------------------------------------
# Environment defaults
# ------------------------------------------------------------------------------
SAM2_VENV_PATH="$HOME/venvs/sam2_realtime_venv/bin/activate"

# Optional.
# If this variable is defined, the script will cd into it before launching SAM2.
# This is useful if Hydra/checkpoint paths are relative to a specific directory.
SAM2_ASSETS_DIR="${SAM2_ASSETS_DIR:-}"

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Camera options:"
    echo "  --camera TYPE"
    echo "      Camera type: azure, realsense, sim_head, sim_wrist"
    echo "      Default: $CAMERA_TYPE"
    echo ""
    echo "Node selection:"
    echo "  --sam2 true/false"
    echo "      Launch SAM2 node"
    echo "      Default: $LAUNCH_SAM2"
    echo ""
    echo "  --mask2pcl true/false"
    echo "      Launch mask2pcl node"
    echo "      Default: $LAUNCH_MASK2PCL"
    echo ""
    echo "  --track true/false"
    echo "      Launch track node"
    echo "      Default: $LAUNCH_TRACK_NODE"
    echo ""
    echo "  --all"
    echo "      Launch SAM2, mask2pcl, and tracker"
    echo ""
    echo "  --sam2-only"
    echo "      Launch only SAM2"
    echo ""
    echo "  --tracking"
    echo "      Launch SAM2 and tracker, without mask2pcl"
    echo ""
    echo "  --pcl"
    echo "      Launch SAM2 and mask2pcl, without tracker"
    echo ""
    echo "Namespaces:"
    echo "  --sam2-namespace NAME"
    echo "      Default: $SAM2_NAMESPACE"
    echo ""
    echo "  --mask2pcl-namespace NAME"
    echo "      Default: $MASK2PCL_NAMESPACE"
    echo ""
    echo "  --track-namespace NAME"
    echo "      Default: $TRACK_NAMESPACE"
    echo ""
    echo "Shared parameters:"
    echo "  --target-frame FRAME"
    echo "      Common target frame for mask2pcl and tracker."
    echo "      If omitted, a camera-specific default is used."
    echo ""
    echo "SAM2 parameters:"
    echo "  --model-cfg PATH"
    echo "      Default: $MODEL_CFG"
    echo ""
    echo "  --checkpoint PATH"
    echo "      Default: $CHECKPOINT"
    echo ""
    echo "  --live-viz true/false"
    echo "      Default: $LIVE_VISUALIZATION"
    echo ""
    echo "Mask2PCL parameters:"
    echo "  --mask2pcl-enable true/false"
    echo "      Default: $MASK2PCL_ENABLE"
    echo ""
    echo "  --mask2pcl-min-area VALUE"
    echo "      Default: $MASK2PCL_MIN_MASK_AREA"
    echo ""
    echo "  --cloud-stride VALUE"
    echo "      Default: $CLOUD_STRIDE"
    echo ""
    echo "Tracker parameters:"
    echo "  --track-enable true/false"
    echo "      Default: $TRACK_ENABLE"
    echo ""
    echo "  --track-min-area VALUE"
    echo "      Default: $TRACK_MIN_MASK_AREA"
    echo ""
    echo "  --predict-rate VALUE"
    echo "      Default: $PREDICT_RATE"
    echo ""
    echo "  --print-measurement-marker true/false"
    echo "      Default: $PRINT_MEASUREMENT_MARKER"
    echo ""
    echo "  --depth-filter-percentage VALUE"
    echo "      Default: $DEPTH_FILTER_PERCENTAGE"
    echo ""
    echo "  --max-depth-jump VALUE"
    echo "      Default: $MAX_DEPTH_JUMP"
    echo ""
    echo "  --relock-window VALUE"
    echo "      Default: $RELOCK_WINDOW"
    echo ""
    echo "Environment:"
    echo "  --venv PATH"
    echo "      Path to SAM2 virtual environment activate script"
    echo "      Default: $SAM2_VENV_PATH"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "      Launch default Azure setup: SAM2 + tracker"
    echo ""
    echo "  $0 --camera realsense --all"
    echo "      Launch all nodes using RealSense topics"
    echo ""
    echo "  $0 --camera azure --mask2pcl true --track false"
    echo "      Launch SAM2 + mask2pcl only"
    echo ""
    echo "  $0 --sam2-only --live-viz true"
    echo "      Launch only SAM2 with visualization"
    echo ""
    echo "  $0 --camera realsense --tracking"
    echo "      Launch SAM2 + tracker for RealSense using the camera-specific target frame"
    echo ""
    echo "  $0 --camera azure --tracking --target-frame base_footprint"
    echo "      Launch SAM2 + tracker for Azure while manually overriding the target frame"
    echo ""
}

# ------------------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------------------
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --camera)
            CAMERA_TYPE="$2"
            shift 2
            ;;

        --sam2)
            LAUNCH_SAM2="$2"
            shift 2
            ;;

        --mask2pcl)
            LAUNCH_MASK2PCL="$2"
            shift 2
            ;;

        --track)
            LAUNCH_TRACK_NODE="$2"
            shift 2
            ;;

        --all)
            LAUNCH_SAM2="true"
            LAUNCH_MASK2PCL="true"
            LAUNCH_TRACK_NODE="true"
            shift
            ;;

        --sam2-only)
            LAUNCH_SAM2="true"
            LAUNCH_MASK2PCL="false"
            LAUNCH_TRACK_NODE="false"
            shift
            ;;

        --tracking)
            LAUNCH_SAM2="true"
            LAUNCH_MASK2PCL="false"
            LAUNCH_TRACK_NODE="true"
            shift
            ;;

        --pcl)
            LAUNCH_SAM2="true"
            LAUNCH_MASK2PCL="true"
            LAUNCH_TRACK_NODE="false"
            shift
            ;;

        --sam2-namespace)
            SAM2_NAMESPACE="$2"
            shift 2
            ;;

        --mask2pcl-namespace)
            MASK2PCL_NAMESPACE="$2"
            shift 2
            ;;

        --track-namespace)
            TRACK_NAMESPACE="$2"
            shift 2
            ;;

        --target-frame)
            TARGET_FRAME="$2"
            TARGET_FRAME_USER_SET="true"
            shift 2
            ;;

        --model-cfg)
            MODEL_CFG="$2"
            shift 2
            ;;

        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;

        --live-viz)
            LIVE_VISUALIZATION="$2"
            shift 2
            ;;

        --mask2pcl-enable)
            MASK2PCL_ENABLE="$2"
            shift 2
            ;;

        --mask2pcl-min-area)
            MASK2PCL_MIN_MASK_AREA="$2"
            shift 2
            ;;

        --cloud-stride)
            CLOUD_STRIDE="$2"
            shift 2
            ;;

        --track-enable)
            TRACK_ENABLE="$2"
            shift 2
            ;;

        --track-min-area)
            TRACK_MIN_MASK_AREA="$2"
            shift 2
            ;;

        --predict-rate)
            PREDICT_RATE="$2"
            shift 2
            ;;

        --print-measurement-marker)
            PRINT_MEASUREMENT_MARKER="$2"
            shift 2
            ;;

        --depth-filter-percentage)
            DEPTH_FILTER_PERCENTAGE="$2"
            shift 2
            ;;

        --max-depth-jump)
            MAX_DEPTH_JUMP="$2"
            shift 2
            ;;

        --relock-window)
            RELOCK_WINDOW="$2"
            shift 2
            ;;

        --venv)
            SAM2_VENV_PATH="$2"
            shift 2
            ;;

        --help|-h)
            show_help
            exit 0
            ;;

        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# Set camera-specific topics, target frame, and depth units
# ------------------------------------------------------------------------------
case "$CAMERA_TYPE" in
    azure)
        IMAGE_TOPIC="/k4a/rgb/image_raw"
        DEPTH_TOPIC="/k4a/depth_to_rgb/image_raw"
        CAM_INFO="/k4a/rgb/camera_info"
        DEFAULT_TARGET_FRAME="rgb_camera_link"
        DEPTH_IMAGE_UNITS_DIVISOR="1"
        ;;

    realsense)
        IMAGE_TOPIC="/camera/camera/color/image_raw"
        DEPTH_TOPIC="/camera/camera/depth/image_rect_raw"
        CAM_INFO="/camera/camera/color/camera_info"
        DEFAULT_TARGET_FRAME="camera_color_frame"
        DEPTH_IMAGE_UNITS_DIVISOR="1000"
        ;;

    *)
        echo "Unknown camera type: $CAMERA_TYPE"
        echo "Valid options: azure, realsense, sim_head, sim_wrist"
        exit 1
        ;;
esac

if [ "$TARGET_FRAME_USER_SET" != "true" ]; then
    TARGET_FRAME="$DEFAULT_TARGET_FRAME"
fi

# ------------------------------------------------------------------------------
# Derived topics
# ------------------------------------------------------------------------------
SAM2_MASK_TOPIC="/${SAM2_NAMESPACE}/mask"

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
if [ -f "$SAM2_VENV_PATH" ]; then
    echo "Activating SAM2 virtual environment:"
    echo "  $SAM2_VENV_PATH"
    source "$SAM2_VENV_PATH"
else
    echo "Warning: SAM2 virtual environment not found:"
    echo "  $SAM2_VENV_PATH"
fi

if [ "$LAUNCH_SAM2" == "true" ] && [ -n "$SAM2_ASSETS_DIR" ]; then
    if [ -d "$SAM2_ASSETS_DIR" ]; then
        echo "Changing directory to SAM2 assets:"
        echo "  $SAM2_ASSETS_DIR"
        cd "$SAM2_ASSETS_DIR"
    else
        echo "Warning: SAM2_ASSETS_DIR is set but does not exist:"
        echo "  $SAM2_ASSETS_DIR"
    fi
fi

# ------------------------------------------------------------------------------
# Launch info
# ------------------------------------------------------------------------------
echo "============================================================"
echo "SAM2 TRACKING PIPELINE LAUNCH"
echo "============================================================"
echo "Camera:"
echo "  Type:                         $CAMERA_TYPE"
echo "  RGB image topic:              $IMAGE_TOPIC"
echo "  Depth topic:                  $DEPTH_TOPIC"
echo "  Camera info:                  $CAM_INFO"
echo "  Depth units divisor:          $DEPTH_IMAGE_UNITS_DIVISOR"
echo ""
echo "Nodes:"
echo "  SAM2:                         $LAUNCH_SAM2"
echo "  Mask2PCL:                     $LAUNCH_MASK2PCL"
echo "  Tracker:                      $LAUNCH_TRACK_NODE"
echo ""
echo "Namespaces:"
echo "  SAM2 namespace:               $SAM2_NAMESPACE"
echo "  Mask2PCL namespace:           $MASK2PCL_NAMESPACE"
echo "  Tracker namespace:            $TRACK_NAMESPACE"
echo ""
echo "Topics:"
echo "  SAM2 mask topic:              $SAM2_MASK_TOPIC"
echo ""
echo "Frames:"
echo "  Target frame:                 $TARGET_FRAME"
echo "  Target frame source:          $([ "$TARGET_FRAME_USER_SET" == "true" ] && echo "manual override" || echo "camera default")"
echo ""
echo "SAM2:"
echo "  Model cfg:                    $MODEL_CFG"
echo "  Checkpoint:                   $CHECKPOINT"
echo "  Live visualization:           $LIVE_VISUALIZATION"
echo ""
echo "Mask2PCL:"
echo "  Enable processing:            $MASK2PCL_ENABLE"
echo "  Min mask area:                $MASK2PCL_MIN_MASK_AREA"
echo "  Cloud stride:                 $CLOUD_STRIDE"
echo "  Max detection threshold:      $MASK2PCL_MAXIMUM_DETECTION_THRESHOLD"
echo ""
echo "Tracker:"
echo "  Enable processing:            $TRACK_ENABLE"
echo "  Min mask area:                $TRACK_MIN_MASK_AREA"
echo "  Predict rate:                 $PREDICT_RATE"
echo "  Print measurement marker:     $PRINT_MEASUREMENT_MARKER"
echo "  Depth filter percentage:      $DEPTH_FILTER_PERCENTAGE"
echo "  Max detection threshold:      $TRACK_MAXIMUM_DETECTION_THRESHOLD"
echo "  Max depth jump:               $MAX_DEPTH_JUMP"
echo "  Relock window:                $RELOCK_WINDOW"
echo "============================================================"

# ------------------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------------------
ros2 launch sam2_realtime_bringup sam2_tracking_pipeline.launch.py \
    launch_sam2:="$LAUNCH_SAM2" \
    launch_mask2pcl:="$LAUNCH_MASK2PCL" \
    launch_track_node:="$LAUNCH_TRACK_NODE" \
    sam2_namespace:="$SAM2_NAMESPACE" \
    mask2pcl_namespace:="$MASK2PCL_NAMESPACE" \
    track_namespace:="$TRACK_NAMESPACE" \
    image_topic:="$IMAGE_TOPIC" \
    depth_topic:="$DEPTH_TOPIC" \
    cam_info:="$CAM_INFO" \
    sam2_mask_topic:="$SAM2_MASK_TOPIC" \
    depth_image_units_divisor:="$DEPTH_IMAGE_UNITS_DIVISOR" \
    target_frame:="$TARGET_FRAME" \
    image_reliability:="$IMAGE_RELIABILITY" \
    model_cfg:="$MODEL_CFG" \
    checkpoint:="$CHECKPOINT" \
    live_visualization:="$LIVE_VISUALIZATION" \
    mask2pcl_enable:="$MASK2PCL_ENABLE" \
    mask2pcl_min_mask_area:="$MASK2PCL_MIN_MASK_AREA" \
    cloud_stride:="$CLOUD_STRIDE" \
    mask2pcl_maximum_detection_threshold:="$MASK2PCL_MAXIMUM_DETECTION_THRESHOLD" \
    track_enable:="$TRACK_ENABLE" \
    track_min_mask_area:="$TRACK_MIN_MASK_AREA" \
    predict_rate:="$PREDICT_RATE" \
    print_measurement_marker:="$PRINT_MEASUREMENT_MARKER" \
    depth_filter_percentage:="$DEPTH_FILTER_PERCENTAGE" \
    track_maximum_detection_threshold:="$TRACK_MAXIMUM_DETECTION_THRESHOLD" \
    max_depth_jump:="$MAX_DEPTH_JUMP" \
    relock_window:="$RELOCK_WINDOW"