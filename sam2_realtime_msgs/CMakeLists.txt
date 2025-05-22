cmake_minimum_required(VERSION 3.8)
project(sam2_realtime_msgs)

# ============================================================
# Dependencies
# ============================================================
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# ============================================================
# Interface Generation
# ============================================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PromptBbox.msg"
  "msg/TrackedObject.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# ============================================================
# Export Dependencies
# ============================================================
ament_export_dependencies(rosidl_default_runtime)

ament_package()
