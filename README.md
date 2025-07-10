# 🚀 **SAM2 Real-Time ROS 2**

**`sam2_realtime_ros2`** is a ROS 2 wrapper for [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time). It brings **Segment Anything 2** into real-time perception pipelines for robots using YOLO for prompt generation and EKF for robust 3D tracking.

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
│   ├── segment-anything-2-real-time/  # Upstream wrapper
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

3️⃣ **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**✅ Note:** This virtual environment is **required** to run all nodes.

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


## 🎯 **Camera Depth Scale**

| Camera        | Depth scale |
|---------------|--------------|
| RealSense     | 1000         |
| Orbbec        | 1            |
| Azure Kinect  | 1            |


## 🐳 **Docker**

A `docker/` folder provides `Dockerfile` + `docker-compose.yml`. Use this to containerize the entire pipeline. The virtualenv is still required **inside** the container.


## ✅ **TODO**

- [ ] Update README
- [ ] Test `LifecycleNode` usage
- [ ] Verify & update Docker setup
- [ ] Add `event_in` for tracking control
- [ ] Apply changes according to original repo
- [ ] Extend to multi-object segmentation according to original repo


## 🏷️ **Credits**

- Upstream: [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
- YOLOv8 (Ultralytics)
- ROS 2 Humble or newer


Built by **SocRob@Home** 🤖
