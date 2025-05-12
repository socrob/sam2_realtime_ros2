# ros2_template_bringup

This package provides launch files for the [`ros2_template`](https://github.com/socrob/ros2_template) lifecycle node, which subscribes to camera images and publishes custom messages based on incoming data.

---

## 🚀 Purpose

The bringup package is responsible for:
- Declaring launch arguments
- Configuring and launching the `template_node`
- Managing ROS 2 namespace and topic remappings
- Supporting lifecycle activation with parameters

---

## 📦 Package Dependencies

- [`ros2_template`](https://github.com/socrob/ros2_template)
- `launch`
- `launch_ros`

---

## 🧪 Launch Example

To run the template node with default parameters:

```bash
ros2 launch ros2_template_bringup template_node.launch.py
```

### 🔧 Optional Arguments

| Argument              | Default                   | Description                                     |
|----------------------|---------------------------|-------------------------------------------------|
| `namespace`           | `template`                | Namespace for the node                          |
| `input_image_topic`   | `/camera/image_raw`       | Image topic to subscribe to                     |
| `enable`              | `True`                    | Start the node enabled                          |
| `image_reliability`   | `1`                       | QoS reliability (1 = Reliable, 2 = Best Effort) |

You can override any of these at launch time:

```bash
ros2 launch ros2_template_bringup template_node.launch.py input_image_topic:=/camera/camera/color/image_raw enable:=False
```

---

## 📁 File Structure

```
ros2_template_bringup/
├── launch/
│   └── template_node.launch.py
├── package.xml
└── README.md
```

---

## 📜 License

This package is licensed under the [GPL-3](https://www.gnu.org/licenses/gpl-3.0.html).
