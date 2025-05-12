#!/bin/bash
set -e

# Source ROS 2 and workspace
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash

# Launch the ros2_template node
exec ros2 launch ros2_template_bringup template_node.launch.py \
    namespace:=template \
    input_image_topic:=/camera/camera/color/image_raw \
    enable:=True \
    image_reliability:=1