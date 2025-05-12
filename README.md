
# ros2_template

A modular ROS 2 project template for image-based processing, using lifecycle-aware nodes and good development practices. This repository is designed to be a starting point for robotics projects involving real-time vision and topic-based message handling.

---

## 📁 Repository Structure

```
ros2_template/
├── ros2_template             # Python ROS 2 node with lifecycle and image callbacks
├── ros2_template_msgs        # Custom message definition
├── ros2_template_bringup     # Launch files for simulation or real robot usage
├── docker/                   # Containerization with Docker and Compose
├── .github/workflows/        # CI setup (optional)
```

---

## 🚀 Getting Started

### 🔧 Build the Workspace

Make sure you have ROS 2 Humble installed. Then clone the repository into your workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/socrob/ros2_template.git
cd ..
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

Source the workspace:
```bash
source install/setup.bash
```

---

## 🐳 Docker Support

The `docker/` folder includes everything to build and run this project in a GPU-enabled ROS 2 environment.

Build and run:
```bash
cd docker
make compose-build
make compose-up
```

---

### ⚠️ Note on ROS 2 Topic Echo Issues in Docker

If you are running the RealSense node on the **host machine** and trying to **echo topics from inside the Docker container**, you may encounter a situation where `ros2 topic list` works (topics are visible), but `ros2 topic echo` shows **no data**.

This issue is caused by **Fast DDS defaulting to shared memory (SHM) transport** when the host and container appear to share IPC and network namespaces (which is the case when using `--net=host` and `--ipc=host`).

#### ✅ Workaround

As discussed in [Fast DDS GitHub issue #5396](https://github.com/eProsima/Fast-DDS/issues/5396#issuecomment-2493358758), you can **force Fast DDS to use UDP** instead of SHM by setting the following environment variable inside the container:

```bash
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

This is already included in the Docker Compose setup via:

```yaml
environment:
  - FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

This change ensures topic communication works reliably between the host and container when using DDS-based transports like Fast DDS.

---

## 🧪 Testing & Linting

This project includes:

- `flake8` and `pep257` for Python style checks
- Unit test stubs in `ros2_template/test/`
- GitHub Actions CI workflow

To run tests manually:

```bash
colcon test
colcon test-result --verbose
```

---

## 📜 Custom Messages

The `ros2_template_msgs` package contains a simple message definition used for communication across nodes.

---

## 🧩 Launching

Use the bringup package to launch the template node:
```bash
ros2 launch ros2_template_bringup template_node.launch.py
```

---

## 💻 Local Development Using Python Virtual Environment

To facilitate code development and dependency management, it is recommended to use a **Python virtual environment**.  
This follows the recommendation in [this ROS 2 issue](https://github.com/ros2/ros2/issues/1094#issuecomment-2067759726).

### 🪄 Setup

1. **Modify `setup.cfg`** in your ROS 2 Python package:
   ```ini
   [build_scripts]
   executable = /usr/bin/env python3
   ```

2. **Build and source the workspace**:
   ```bash
   colcon build
   source install/setup.bash
   ```

3. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv ~/venvs/ros2_template_venv
   source ~/venvs/ros2_template_venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run your ROS 2 nodes or launch files**:
   ```bash
   ros2 run ros2_template template_node
   ros2 launch ros2_template_bringup bringup.py
   ```

---

## 📄 License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

---

## 🙋 Citation

If you use this in your work, please cite it as:

```yaml
cff-version: 1.2.0
title: "ros2_template"
version: "0.1.0"
authors:
  - family-names: "Serra"
    given-names: "Rodrigo"
repository-code: "https://github.com/socrob/ros2_template"
license: "GPL-3.0"
```