# **SAM2 Real-Time Tracker ROS 2**

**`sam2_realtime_ros2`** is a ROS 2 wrapper for [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time). It brings **Segment Anything 2** into real-time perception pipelines for robots: an external detector sends the prompt (a box or a mask), SAM2 segments the object in every frame, and an EKF tracks its 3D position.

This repository expects a **Python virtual environment** to isolate dependencies. The **upstream SAM2 repo** is included as a **Git submodule**.

---

## **Repo structure**

```
sam2_realtime_ros2/
├── sam2_realtime/
│   ├── sam2_realtime_node.py         # SAM2 segmentation node
│   ├── mask2pcl.py                   # Mask → object point cloud
│   ├── track_node.py                 # 3D tracker (EKF)
│   ├── ekf.py                        # EKF filter
│   ├── segment-anything-2-real-time/ # Upstream submodule
├── sam2_realtime_bringup/
│   └── launch/
│       ├── sam2_tracking_pipeline.launch.py  # SAM2, Mask2PCL and tracker (each optional)
│       └── sam2_tracking_pipeline.sh         # Same, with venv, assets folder and camera topics
├── sam2_realtime_msgs/               # Custom ROS messages
│   ├── PromptBbox.msg
│   ├── TrackedObject.msg
├── docker/                           # Docker config
├── requirements.txt
├── setup_env.sh                      # One-step environment setup script
└── ...
```

---

## **How It Works**

**1. SAM2 Segmentation**  
`sam2_realtime_node.py`:
- Loads the **segment-anything-2-real-time** model
- Waits for a prompt from an external detector: a box (`PromptBbox`) on `/sam2/init_prompt`, or a mask image on `/sam2/init_prompt_mask`. The package has no detector of its own.
- Publishes the object's mask on `/sam2/mask` (and the image with the mask drawn on `/sam2/img_mask` with `live_visualization`)

**2. Mask2PCL** (optional)  
`mask2pcl.py` turns the mask and the depth image into the object's point cloud (`/mask2pcl/object_cloud`) in `target_frame`.

**3. EKF Tracking**  
`track_node.py`:
- Synchronizes:
  - Depth image
  - Camera intrinsics
  - SAM2 mask
- Computes robust 3D position in camera frame
- Transforms point to `target_frame`
- Filters position with an EKF
- Publishes:
  - `/track_node/tracked_object`
  - `/track_node/measurement_marker` (RViz marker)
  - TF transform `tracked_object_1`

Mask2PCL and the tracker are lifecycle nodes that process while ACTIVE; pause them with a lifecycle deactivate and resume with activate.

---

## **Build & Setup**

1. **Build the workspace:**

```bash
colcon build
source install/setup.bash
```

2. **Run the environment setup script:**

```bash
./setup_env.sh
```

This script will:
- Create a virtual environment in `~/venvs/sam2_realtime_venv` (if not existing)
- Install Python dependencies
- Install the upstream **SAM2** repo in editable mode
- Download the checkpoints into `~/sam2_assets/checkpoints` (skipped when they are already there)

No environment variables are needed. `sam2_tracking_pipeline.sh` activates the venv, `cd` into `~/sam2_assets` and load `checkpoints/sam2.1_hiera_small.pt` from there.

**Asset folders** (optional overrides):

| Variable | Default | Holds |
|---|---|---|
| `SAM2_ASSETS_DIR` | `~/sam2_assets` | `checkpoints/sam2.1_*.pt` |

Set it only to keep the files elsewhere; `setup_env.sh` and the launch script both use it. When starting `ros2 launch` directly, run it from the SAM2 assets folder, since the default `checkpoint` is a relative path.

---

## **Run Example**

```bash
# From sam2_realtime_bringup/launch:
# SAM2 + tracker on the Azure Kinect (the default); waits for a prompt
./sam2_tracking_pipeline.sh

# All three nodes on a RealSense (driver with align_depth.enable:=true)
./sam2_tracking_pipeline.sh --camera realsense --all

# Any other launch argument after --
./sam2_tracking_pipeline.sh --target-frame base_footprint -- relock_window:=2

# Then send a prompt from a detector, e.g. a box by hand:
ros2 topic pub -1 /sam2/init_prompt sam2_realtime_msgs/msg/PromptBbox "{x_min: 400, y_min: 100, x_max: 700, y_max: 600}"
```

`./sam2_tracking_pipeline.sh --help` lists the options.

---

## **Camera Depth Scale**

| Camera        | Depth scale |
|---------------|-------------|
| RealSense     | 1000        |
| Orbbec        | 1           |
| Azure Kinect  | 1           |

`sam2_tracking_pipeline.sh` sets them per camera (`depth_image_units_divisor`).

---

## **Docker**

A `docker/` folder provides `Dockerfile` + `docker-compose.yml`. Use this to containerize the entire pipeline. The virtual environment must still be activated **inside** the container.

---

## **TODO**

- [ ] Extend for multi-object tracking and latest updates
- [ ] Final cleanup of Docker setup
- [ ] Test `LifecycleNode` usage

---

## **Credits**

- Upstream: [Gy920/segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
- ROS 2 Humble or newer

Built by **SocRob@Home**