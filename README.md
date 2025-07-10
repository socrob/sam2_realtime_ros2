# 🎯 **SAM2 Real-Time ROS 2**

**`sam2_realtime_ros2`** is a ROS 2 wrapper for [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time). It brings **Segment Anything 2** into real-time perception pipelines for robots using YOLO for prompt generation and an EKF for robust 3D tracking.

This repository expects a **Python virtual environment** to isolate dependencies. The **upstream SAM2 repo** is included as a **Git submodule**.

---

## 📂 **Repo structure**

```
sam2_realtime_ros2/
├── sam2_realtime/
│   ├── sam2_realtime_node.py   # SAM2 segmentation node
│   ├── bbox_prompt_node.py     # YOLO bbox prompt
│   ├── yolo_mask_prompt_node.py# YOLO mask prompt
│   ├── ekf.py                  # EKF filter
│   ├── track_node.py           # 3D tracker
│   ├── segment-anything-2-real-time/  # Upstream submodule
├── sam2_realtime_bringup/      # Launch files & shell scripts
│   ├── launch/
│   │   ├── sam2_realtime_node.sh
│   │   ├── yolo_prompt_node.sh
│   │   ├── track_node_2.sh
│   │   └── *.launch.py
├── sam2_realtime_msgs/         # Custom ROS messages
│   ├── PromptBbox.msg
│   ├── TrackedObject.msg
├── docker/                     # Docker config
├── requirements.txt
├── setup_env.sh                # One-step environment setup script
└── ...
```

---

## ⚙️ **How It Works**

➜ **1️⃣ YOLO Prompt**  
Run a YOLO model to detect people or objects:
- Outputs bounding box (`PromptBbox`) or mask prompt
- Example nodes: `bbox_prompt_node.py` and `yolo_mask_prompt_node.py`

➜ **2️⃣ SAM2 Wrapper**  
`sam2_realtime_node.py`:
- Loads the **segment-anything-2-real-time** model
- Receives the YOLO prompt → segments mask in real-time

➜ **3️⃣ EKF Tracking**  
`track_node.py`:
- Synchronizes:
  - Depth image
  - Camera intrinsics
  - SAM2 mask
- Computes robust 3D position in camera frame
- Transforms point to `target_frame`
- Filters position with an EKF for robust tracking
- Publishes:
  - `/tracked_object`
  - `/measurement_marker` (RViz marker)
  - TF transform

---

## 🏗️ **Build & Setup**

1️⃣ **Build the workspace:**

```bash
colcon build
source install/setup.bash
```

2️⃣ **Create & activate a virtual environment:**

```bash
python3 -m venv ~/venvs/sam2_realtime_venv
source ~/venvs/sam2_realtime_venv/bin/activate
```

3️⃣ **Run the `setup_env.sh` to install everything:**

```bash
./setup_env.sh
```

This script will:
- Install Python dependencies for ROS 2
- Install the upstream **SAM2** as editable
- Download checkpoints
- Create and export `$SAM2_ASSETS_DIR` automatically by adding it to `~/.bashrc`

✅ **Important:** Next time you open a shell, run `source ~/.bashrc` to make sure `$SAM2_ASSETS_DIR` is available.

The virtual environment **must be activated** each time you use the nodes.

---

## 🎥 **Run Example**

```bash
# 1. Run YOLO prompt node
./yolo_prompt_node.sh --camera realsense

# 2. Run SAM2 segmentation node
./sam2_realtime_node.sh --camera realsense

# 3. Run EKF tracking node
./track_node_2.sh --camera realsense
```

Use `--camera azure` for Azure Kinect.

The provided shell scripts:
- Automatically activate the virtual environment
- Use `$SAM2_ASSETS_DIR`
- Configure camera topics, depth scale, and other parameters

---

## 🎯 **Camera Depth Scale**

| Camera        | Depth scale |
|---------------|--------------|
| RealSense     | 1000         |
| Orbbec        | 1            |
| Azure Kinect  | 1            |

✅ These depth unit divisors are configured automatically in the example launch scripts.

---

## 🐳 **Docker**

A `docker/` folder provides `Dockerfile` + `docker-compose.yml`. Use this to containerize the entire pipeline. The virtual environment is still required **inside** the container — make sure you activate it as part of your entrypoint.

---

## ✅ **TODO**

- [ ] Update README with final details
- [ ] Test `LifecycleNode` usage
- [ ] Verify & update Docker setup
- [ ] Add `event_in` for tracking control
- [ ] Update SAM2 wrapper for multi-object segmentation and latest updates

---

## 🏷️ **Credits**

- Upstream: [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time) (included as a Git submodule)
- YOLOv8 (Ultralytics)
- ROS 2 Humble or newer

Built by **SocRob@Home** 🤖
