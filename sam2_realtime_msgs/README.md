# ros2_template_msgs

This package defines custom ROS 2 messages for the [`ros2_template`](https://github.com/socrob/ros2_template) node, enabling the communication of processed image metadata or results in a structured format.

---

## 📦 Message(s)

### `MyCustomMsg.msg`

```text
string data
```

A minimal message used for publishing a string response, e.g., describing image dimensions or processing status.

---

## 📁 Directory Structure

```
ros2_template_msgs/
├── msg/
│   └── MyCustomMsg.msg
├── package.xml
├── CMakeLists.txt
└── README.md
```

---

## 🔧 Build and Usage

After cloning this package into your ROS 2 workspace:

```bash
cd ~/ros2_ws
colcon build --packages-select ros2_template_msgs
source install/setup.bash
```

Then you can use it in other packages by adding:

### `package.xml`

```xml
<depend>ros2_template_msgs</depend>
```

### `CMakeLists.txt`

```cmake
find_package(ros2_template_msgs REQUIRED)
```

---

## 📜 License

This package is licensed under the [GPL-3](https://www.gnu.org/licenses/gpl-3.0.html).

Maintainer: Rodrigo Serra (<rodrigo.serra@tecnico.ulisboa.pt>)
