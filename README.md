# 🎯 **SAM2 Real-Time Tracker ROS 2**

**`sam2_realtime_ros2`** is a ROS 2 wrapper for [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time). It brings **Segment Anything 2** into real-time perception pipelines for robots using YOLO for prompt generation and an EKF for robust 3D tracking.

This repository expects a **Python virtual environment** to isolate dependencies. The **upstream SAM2 repo** is included as a **Git submodule**.

---

## 📂 **Repo structure**

```
sam2_realtime_ros2/
├── sam2_realtime/
│   ├── sam2_realtime_node.py         # SAM2 segmentation node
│   ├── yolo_prompt_node.py           # YOLO bbox prompt
│   ├── yolo_mask_prompt_node.py      # YOLO mask prompt
│   ├── ekf.py                        # EKF filter
│   ├── track_node.py                # 3D tracker
│   ├── segment-anything-2-real-time/ # Upstream submodule
├── sam2_realtime_bringup/            # Launch files & shell scripts
│   ├── launch/
│   │   ├── sam2_realtime_node.sh
│   │   ├── yolo_prompt_node.sh
│   │   ├── track_node.sh
│   │   └── *.launch.py
├── sam2_realtime_msgs/               # Custom ROS messages
│   ├── PromptBbox.msg
│   ├── TrackedObject.msg
├── docker/                           # Docker config
├── requirements.txt
├── setup_env.sh                      # One-step environment setup script
└── ...
```

---

## ⚙️ **How It Works**

➜ **1️⃣ SAM2 Segmentation**  
Run `sam2_realtime_node.py` to:
- Load the **segment-anything-2-real-time** model
- Await a prompt (bounding box or mask)

➜ **2️⃣ YOLO Prompt**  
Run a YOLO model to detect people or objects:
- Outputs bounding box (`PromptBbox`) or mask prompt
- Requires a trigger via the `/sam2_bbox_prompt/event_in` topic
- Example nodes:
  - `yolo_prompt_node.py` (bbox)
  - `yolo_mask_prompt_node.py` (mask)

➜ **3️⃣ EKF Tracking**  
`track_node.py`:
- Synchronizes:
  - Depth image
  - Camera intrinsics
  - SAM2 mask
- Computes robust 3D position in camera frame
- Transforms point to `target_frame`
- Filters position with an EKF
- Publishes:
  - `/tracked_object`
  - `/measurement_marker` (RViz marker)
  - TF transform
- Requires a trigger via the `/track_node/event_in` topic

---

## 🏗️ **Build & Setup**

1️⃣ **Build the workspace:**

```bash
colcon build
source install/setup.bash
```

2️⃣ **Run the environment setup script:**

```bash
./setup_env.sh
```

This script will:
- Create a virtual environment in `~/venvs/sam2_realtime_venv` (if not existing)
- Install Python dependencies
- Install the upstream **SAM2** repo in editable mode
- Download the checkpoints into `~/sam2_assets/checkpoints` (skipped when they are already there)

✅ No environment variables are needed. The launch scripts in `sam2_realtime_bringup/launch` activate the venv, `cd` into `~/sam2_assets` and load `checkpoints/sam2.1_hiera_small.pt` from there.

📂 **Asset folders** (optional overrides):

| Variable | Default | Holds |
|---|---|---|
| `SAM2_ASSETS_DIR` | `~/sam2_assets` | `checkpoints/sam2.1_*.pt` |
| `YOLO_ASSETS_DIR` | `~/yolo_ros/models` | YOLO weights for `yolo_prompt_node` / `yolo_mask_prompt_node` |

Set a variable only to keep the files elsewhere; `setup_env.sh` and the launch scripts both use it. When starting `ros2 launch` directly, run it from the SAM2 assets folder, since the default `checkpoint` is a relative path.

---

## 🎥 **Run Example**

### Step-by-step:

```bash
# 1. Launch SAM2 node (waits for prompt)
./sam2_realtime_node.sh --camera azure

# 2. Launch YOLO prompt (bounding box or mask)
./yolo_prompt_node.sh --camera azure
# Then trigger prompt:
ros2 topic pub -1 /sam2_bbox_prompt/event_in std_msgs/msg/String "{data: 'e_start'}"

# 3. Launch tracking node
./track_node.sh --camera azure
# Then trigger tracking:
ros2 topic pub -1 /track_node/event_in std_msgs/msg/String "{data: 'e_start'}"
```

Use `--camera realsense` to run with RealSense instead.

---

## 🎯 **Camera Depth Scale**

| Camera        | Depth scale |
|---------------|-------------|
| RealSense     | 1000        |
| Orbbec        | 1           |
| Azure Kinect  | 1           |

✅ These values are handled automatically by the launch scripts.

---

## 🐳 **Docker**

A `docker/` folder provides `Dockerfile` + `docker-compose.yml`. Use this to containerize the entire pipeline. The virtual environment must still be activated **inside** the container.

---

## ✅ **TODO**

- [ ] Extend for multi-object tracking and latest updates
- [ ] Final cleanup of Docker setup
- [ ] Test `LifecycleNode` usage

---

## 🏷️ **Credits**

- Upstream: [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
- YOLOv8 (Ultralytics)
- ROS 2 Humble or newer

Built by **SocRob@Home** 🤖