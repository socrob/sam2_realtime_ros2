#!/bin/bash

xhost +local:docker  # Allow X11 access

docker run --gpus all -it --rm --net=host --ipc=host --pid=host\
  -e DISPLAY=$DISPLAY \
  -e ROS_DOMAIN_ID=7 \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/udev/rules.d:/etc/udev/rules.d \
  --device=/dev/video0 --device=/dev/video1 \
  --device=/dev/usbmon0 --device=/dev/usbmon1 \
  --privileged \
  ros2_template
