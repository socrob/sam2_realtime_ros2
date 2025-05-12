#!/bin/bash
set -e

echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

echo "⬇️  Downloading SAM2 checkpoints..."
cd sam2_realtime/sam2_realtime/segment-anything-2-real-time/checkpoints
chmod +x download_ckpts.sh
./download_ckpts.sh

echo "✅ Environment setup complete."
