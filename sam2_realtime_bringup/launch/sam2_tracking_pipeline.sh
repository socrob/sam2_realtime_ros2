#!/bin/bash
# ==============================================================================
# SAM2 tracking pipeline, started by hand: activates the venv, changes to the
# SAM2 assets folder (relative checkpoint paths resolve there) and runs
# sam2_tracking_pipeline.launch.py with a camera's topics.
#
# SAM2 then waits for a box or mask prompt on /sam2/init_prompt(_mask).
#
# Usage:
#   ./sam2_tracking_pipeline.sh                          # Azure, SAM2 + tracker
#   ./sam2_tracking_pipeline.sh --camera realsense --all
#   ./sam2_tracking_pipeline.sh --target-frame base_footprint -- relock_window:=2
# ==============================================================================

set -e

CAMERA="azure"
LAUNCH_SAM2="true"
LAUNCH_MASK2PCL="false"
LAUNCH_TRACK_NODE="true"
TARGET_FRAME="map"
LIVE_VISUALIZATION="True"
FIX_HEIGHT="true"
FIXED_HEIGHT="1.20"
SAM2_VENV_PATH="$HOME/venvs/sam2_realtime_venv/bin/activate"

# checkpoints/ live in ~/sam2_assets unless SAM2_ASSETS_DIR overrides it
SAM2_ASSETS_DIR="${SAM2_ASSETS_DIR:-$HOME/sam2_assets}"

# Launch arguments after `--`, passed through unchanged
EXTRA_ARGS=()

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS] [-- NAME:=VALUE ...]

  --camera azure|realsense   Camera topics (default: $CAMERA). The RealSense must
                             run with align_depth.enable:=true.
  --all                      SAM2, Mask2PCL and tracker
  --tracking                 SAM2 and tracker (default)
  --pcl                      SAM2 and Mask2PCL
  --sam2-only                SAM2 only
  --target-frame FRAME       Frame of the cloud and tracked position (default: $TARGET_FRAME)
  --live-viz true|false      Publish img_mask (default: $LIVE_VISUALIZATION)
  --venv PATH                venv activate script (default: $SAM2_VENV_PATH)
  -- NAME:=VALUE ...         Any other launch argument of sam2_tracking_pipeline.launch.py

SAM2_ASSETS_DIR: folder holding checkpoints/ (default: \$HOME/sam2_assets).
EOF
}

# Parsing arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --camera)       CAMERA="$2"; shift 2 ;;
        --all)          LAUNCH_MASK2PCL="true";  LAUNCH_TRACK_NODE="true";  shift ;;
        --tracking)     LAUNCH_MASK2PCL="false"; LAUNCH_TRACK_NODE="true";  shift ;;
        --pcl)          LAUNCH_MASK2PCL="true";  LAUNCH_TRACK_NODE="false"; shift ;;
        --sam2-only)    LAUNCH_MASK2PCL="false"; LAUNCH_TRACK_NODE="false"; shift ;;
        --target-frame) TARGET_FRAME="$2"; shift 2 ;;
        --live-viz)     LIVE_VISUALIZATION="$2"; shift 2 ;;
        --venv)         SAM2_VENV_PATH="$2"; shift 2 ;;
        --help|-h)      show_help; exit 0 ;;
        --)             shift; EXTRA_ARGS=("$@"); break ;;
        *)              echo "Unknown option: $1 (see --help)"; exit 1 ;;
    esac
done

# Camera topics; depth is registered to the color image
case "$CAMERA" in
    azure)
        IMAGE_TOPIC="/k4a/rgb/image_raw"
        DEPTH_TOPIC="/k4a/depth_to_rgb/image_raw"
        CAM_INFO="/k4a/rgb/camera_info"
        DEPTH_DIVISOR="1"
        ;;
    realsense)
        IMAGE_TOPIC="/camera/camera/color/image_raw"
        DEPTH_TOPIC="/camera/camera/aligned_depth_to_color/image_raw"
        CAM_INFO="/camera/camera/color/camera_info"
        DEPTH_DIVISOR="1000"
        ;;
    *)
        echo "Unknown camera: $CAMERA (azure or realsense)"
        exit 1
        ;;
esac

# Environment and assets
if [ -f "$SAM2_VENV_PATH" ]; then
    source "$SAM2_VENV_PATH"
else
    echo "Warning: venv not found: $SAM2_VENV_PATH"
fi

if [ ! -f "$SAM2_ASSETS_DIR/checkpoints/sam2.1_hiera_small.pt" ]; then
    echo "SAM2 checkpoint not found in $SAM2_ASSETS_DIR/checkpoints."
    echo "Run setup_env.sh from the sam2_realtime_ros2 repository, or set SAM2_ASSETS_DIR."
    exit 1
fi
cd "$SAM2_ASSETS_DIR"

echo "SAM2 pipeline on '$CAMERA' ($IMAGE_TOPIC), target frame '$TARGET_FRAME':" \
     "SAM2=$LAUNCH_SAM2 Mask2PCL=$LAUNCH_MASK2PCL tracker=$LAUNCH_TRACK_NODE"

ros2 launch sam2_realtime_bringup sam2_tracking_pipeline.launch.py \
    launch_sam2:="$LAUNCH_SAM2" \
    launch_mask2pcl:="$LAUNCH_MASK2PCL" \
    launch_track_node:="$LAUNCH_TRACK_NODE" \
    image_topic:="$IMAGE_TOPIC" \
    depth_topic:="$DEPTH_TOPIC" \
    cam_info:="$CAM_INFO" \
    depth_image_units_divisor:="$DEPTH_DIVISOR" \
    target_frame:="$TARGET_FRAME" \
    live_visualization:="$LIVE_VISUALIZATION" \
    fix_height:="$FIX_HEIGHT" \
    fixed_height:="$FIXED_HEIGHT" \
    "${EXTRA_ARGS[@]}"
