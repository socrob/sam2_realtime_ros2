# ros2_template

This package provides a ROS 2 lifecycle node example that subscribes to camera images and publishes a custom message. It is designed as a modular starting point for creating computer vision and perception nodes that use standard ROS 2 patterns.

---

## 🚀 Node: `template_node`

This is a [`rclpy.lifecycle.LifecycleNode`](https://docs.ros.org/en/foxy/How-To-Guides/Using-Lifecycle-Nodes.html) that listens to an image topic and publishes a message containing basic information (e.g., resolution) when enabled.

### ✅ Features

- Lifecycle management (`configure`, `activate`, `deactivate`, etc.)
- Dynamic parameters (`enable`, `image_reliability`)
- ROS 2 QoS support
- Camera image subscription via `cv_bridge`
- Custom message publishing (`ros2_template_msgs/MyCustomMsg`)

---

## 🔧 Parameters

| Name                | Type   | Default   | Description                                   |
|---------------------|--------|-----------|-----------------------------------------------|
| `enable`            | bool   | `True`    | Whether the node is actively processing images |
| `image_reliability` | int    | `1`       | QoS setting for image topic (1=Reliable, 2=Best Effort) |

---

## 📦 Dependencies

- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- `std_srvs`
- [`ros2_template_msgs`](https://github.com/socrob/ros2_template_msgs)

---

## ▶️ Running the Node

This node is launched via the `ros2_template_bringup` package. Example:

```bash
ros2 launch ros2_template_bringup template_node.launch.py
```

You can override launch parameters:

```bash
ros2 launch ros2_template_bringup template_node.launch.py input_image_topic:=/camera/image_raw
```

### Simple Node

```bash
ros2 launch ros2_template_bringup simple_node.launch.py
```


---

## 🧪 Testing

You can run the included linters with:

```bash
colcon test --packages-select ros2_template
colcon test-result --verbose
```

---

## 📁 Directory Structure

```
ros2_template/
├── ros2_template/
│   └── template_node.py
│   └── template_simple_node.py
├── test/
│   ├── test_flake8.py
│   └── test_pep257.py
├── setup.py
├── package.xml
└── README.md
```

---

## 📜 License

This package is licensed under the [GPL-3](https://www.gnu.org/licenses/gpl-3.0.html).

Maintainer: Rodrigo Serra (<rodrigo.serra@tecnico.ulisboa.pt>)
