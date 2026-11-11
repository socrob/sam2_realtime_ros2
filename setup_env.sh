#!/bin/bash
set -e

# Paths below are relative to the repository root, wherever the script is run from
REPO_DIR="$(dirname "$(realpath "$0")")"
cd "$REPO_DIR"

# === CONFIG ===
VENV_PATH=~/venvs/sam2_realtime_venv
SAM2_DIR="$REPO_DIR/sam2_realtime/sam2_realtime/segment-anything-2-real-time"

# Checkpoints go to ~/sam2_assets/checkpoints, where the launch scripts look for
# them. Set SAM2_ASSETS_DIR to use another folder (and keep it set when launching).
ASSETS_DIR="${SAM2_ASSETS_DIR:-$HOME/sam2_assets}"
CHECKPOINTS=(sam2.1_hiera_tiny.pt sam2.1_hiera_small.pt sam2.1_hiera_base_plus.pt sam2.1_hiera_large.pt)

echo "Setting up SAM2 Real-Time ROS 2 environment..."

# === STEP 1: Create venv if not exists ===
if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtual environment at $VENV_PATH ..."
  python3 -m venv "$VENV_PATH"
else
  echo "Virtual environment already exists at $VENV_PATH"
fi

# === STEP 2: Activate venv ===
echo "Activating virtual environment ..."
source "$VENV_PATH/bin/activate"

# === STEP 3: Install main ROS2 Python deps ===
echo "Installing ROS2 Python dependencies ..."
pip install -r requirements.txt

# === STEP 4: Install upstream SAM2 repo in editable mode ===
echo "Installing SAM2 dependencies ..."
cd "$SAM2_DIR"
pip install -e ".[notebooks]"

# === STEP 5: Download checkpoints into the assets folder ===
# The upstream download script saves into the current directory, so it runs
# from the assets folder. It is skipped when every checkpoint is already there,
# because wget would otherwise save duplicate copies (*.pt.1).
mkdir -p "$ASSETS_DIR/checkpoints"
cd "$ASSETS_DIR/checkpoints"

MISSING=0
for ckpt in "${CHECKPOINTS[@]}"; do
  if [ ! -f "$ckpt" ]; then
    MISSING=1
  fi
done

if [ "$MISSING" -eq 1 ]; then
  echo "Downloading SAM2 checkpoints into $ASSETS_DIR/checkpoints ..."
  bash "$SAM2_DIR/checkpoints/download_ckpts.sh"
else
  echo "SAM2 checkpoints already in $ASSETS_DIR/checkpoints"
fi

echo "All done! Virtual environment is active."
echo "The launch scripts in sam2_realtime_bringup activate the venv and find the checkpoints by themselves."
