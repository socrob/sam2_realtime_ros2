#!/bin/bash
set -e

# === CONFIG ===
VENV_PATH=~/venvs/sam2_realtime_venv
SAM2_DIR=sam2_realtime/sam2_realtime/segment-anything-2-real-time
ASSETS_DIR=$(realpath "$SAM2_DIR")

echo "🚀 Setting up SAM2 Real-Time ROS 2 environment..."

# === STEP 1: Create venv if not exists ===
if [ ! -d "$VENV_PATH" ]; then
  echo "🔨 Creating virtual environment at $VENV_PATH ..."
  python3 -m venv "$VENV_PATH"
else
  echo "✅ Virtual environment already exists at $VENV_PATH"
fi

# === STEP 2: Activate venv ===
echo "🔑 Activating virtual environment ..."
source "$VENV_PATH/bin/activate"

# === STEP 3: Install main ROS2 Python deps ===
echo "📦 Installing ROS2 Python dependencies ..."
pip install -r requirements.txt

# === STEP 4: Install upstream SAM2 repo in editable mode ===
echo "⬇️  Installing SAM2 dependencies ..."
cd "$SAM2_DIR"
pip install -e ".[notebooks]"

# === STEP 5: Download checkpoints ===
echo "⬇️  Downloading SAM2 checkpoints ..."
cd checkpoints
chmod +x download_ckpts.sh
./download_ckpts.sh

# === STEP 6: Set SAM2_ASSETS_DIR in bashrc if not already ===
echo "🔑 Setting SAM2_ASSETS_DIR in ~/.bashrc ..."
if grep -q "export SAM2_ASSETS_DIR=" ~/.bashrc; then
  echo "✅ SAM2_ASSETS_DIR already set in ~/.bashrc"
else
  echo "export SAM2_ASSETS_DIR=\"$ASSETS_DIR\"" >> ~/.bashrc
  echo "✅ Added: export SAM2_ASSETS_DIR=\"$ASSETS_DIR\""
fi

echo "✅ All done! Virtual environment is active."
echo "👉 Next time, run: source ~/venvs/sam2_realtime_venv/bin/activate"
echo "👉 And: source ~/.bashrc  (or restart your shell) to load SAM2_ASSETS_DIR"
