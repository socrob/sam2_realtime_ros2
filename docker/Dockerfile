# === Stage 1: Base setup ===
FROM ros:humble AS deps

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Create workspace
WORKDIR /root/ros2_ws
RUN mkdir -p src/ros2_template

# System + GUI dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    libgl1-mesa-glx \
    libgtk-3-dev \
    libudev-dev \
    usbutils \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install RealSense ROS package
RUN apt-get update && \
    apt-get install -y ros-humble-rviz2 && \
    apt-get install -y ros-humble-realsense2-camera && \
    rm -rf /var/lib/apt/lists/*


# Update rosdep database
RUN rosdep update

# === Stage 2: Build workspace ===
FROM deps AS builder

WORKDIR /root/ros2_ws

# Copy your full source tree into the container
COPY ./ ./src/ros2_template
RUN pip3 install --no-cache-dir -r src/ros2_template/requirements.txt

# Install ROS package dependencies and build
RUN source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Add workspace setup to shell
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc


CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /root/ros2_ws/install/setup.bash && exec bash"]

