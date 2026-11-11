#!/usr/bin/env python3
"""
SAM2 tracking pipeline: the SAM2 node, Mask2PCL and the tracker, each optional.

SAM2 waits for a prompt: a box (`sam2_realtime_msgs/PromptBbox`) on
`/<sam2_namespace>/init_prompt`, or a mask image on `init_prompt_mask`, from an
external detector. It then publishes the object's mask on `mask`. Mask2PCL turns
the mask into a point cloud (`object_cloud`); the tracker turns it into a
filtered 3D position (`tracked_object`, and the TF `tracked_object_1`) in
`target_frame`.

`checkpoint` and `model_cfg` are relative to the working directory: launch it
from the folder that holds `checkpoints/` (see sam2_tracking_pipeline.sh).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


# (name, default, description), grouped by the node that uses them
ARGUMENTS = [
    # Node selection
    ("launch_sam2", "true", "Launch the SAM2 node"),
    ("launch_mask2pcl", "true", "Launch the mask to point cloud node"),
    ("launch_track_node", "true", "Launch the tracker node"),

    # Namespaces
    ("sam2_namespace", "sam2", "Namespace of the SAM2 node"),
    ("mask2pcl_namespace", "mask2pcl", "Namespace of the Mask2PCL node"),
    ("track_namespace", "track_node", "Namespace of the tracker node"),

    # Camera and shared topics
    ("image_topic", "/k4a/rgb/image_raw", "Color image topic"),
    ("depth_topic", "/k4a/depth_to_rgb/image_raw", "Depth image topic, registered to the color image"),
    ("cam_info", "/k4a/rgb/camera_info", "Camera info topic of the color image"),
    ("sam2_mask_topic", "/sam2/mask", "SAM2 mask topic, read by Mask2PCL and the tracker"),
    ("depth_image_units_divisor", "1", "Raw depth / divisor = metres (1000 for millimetres)"),
    ("target_frame", "base_footprint", "Frame of the point cloud and the tracked position"),

    # SAM2
    ("image_reliability", "2", "Color image QoS: 1 = reliable, 2 = best effort"),
    ("model_cfg", "configs/sam2.1/sam2.1_hiera_s.yaml", "SAM2 model config"),
    ("checkpoint", "checkpoints/sam2.1_hiera_small.pt", "SAM2 checkpoint, relative to the working directory"),
    ("live_visualization", "False", "Publish the image with the mask drawn (img_mask)"),

    # Mask2PCL
    ("mask2pcl_min_mask_area", "100", "Minimum mask area, in pixels"),
    ("cloud_stride", "4", "Pixel stride when building the point cloud"),
    ("mask2pcl_maximum_detection_threshold", "0.3", "Mask2PCL maximum detection threshold"),

    # Tracker
    ("track_min_mask_area", "1000", "Minimum mask area, in pixels"),
    ("predict_rate", "10", "EKF prediction rate, in Hz"),
    ("print_measurement_marker", "true", "Publish the measurement marker"),
    ("depth_filter_percentage", "0.3", "Depth filtering percentage"),
    ("track_maximum_detection_threshold", "0.3", "Tracker maximum detection threshold"),
    ("max_position_jump", "0.3", "Largest accepted jump between measurements, in metres"),
    ("relock_window", "1", "Relock window, in seconds"),
    ("fix_height", "false", "Report the tracked object at a fixed height"),
    ("fixed_height", "0.0", "Height used when fix_height is true, in metres"),
]


def generate_launch_description():
    def arg(name):
        return LaunchConfiguration(name)

    sam2_node = LifecycleNode(
        package="sam2_realtime",
        executable="sam2_realtime_node",
        name="sam2_realtime_node",
        namespace=arg("sam2_namespace"),
        output="screen",
        parameters=[{
            "image_topic": arg("image_topic"),
            "image_reliability": arg("image_reliability"),
            "model_cfg": arg("model_cfg"),
            "checkpoint": arg("checkpoint"),
            "live_visualization": arg("live_visualization"),
        }],
        condition=IfCondition(arg("launch_sam2")),
    )

    mask2pcl_node = Node(
        package="sam2_realtime",
        executable="mask2pcl",
        name="mask2pcl",
        namespace=arg("mask2pcl_namespace"),
        output="screen",
        parameters=[{
            "depth_topic": arg("depth_topic"),
            "cam_info": arg("cam_info"),
            "sam2_mask_topic": arg("sam2_mask_topic"),
            "target_frame": arg("target_frame"),
            "depth_image_units_divisor": arg("depth_image_units_divisor"),
            "min_mask_area": arg("mask2pcl_min_mask_area"),
            "cloud_stride": arg("cloud_stride"),
            "maximum_detection_threshold": arg("mask2pcl_maximum_detection_threshold"),
        }],
        condition=IfCondition(arg("launch_mask2pcl")),
    )

    track_node = Node(
        package="sam2_realtime",
        executable="track_node",
        name="track_node",
        namespace=arg("track_namespace"),
        output="screen",
        parameters=[{
            "depth_topic": arg("depth_topic"),
            "cam_info": arg("cam_info"),
            "sam2_mask_topic": arg("sam2_mask_topic"),
            "target_frame": arg("target_frame"),
            "depth_image_units_divisor": arg("depth_image_units_divisor"),
            "depth_filter_percentage": arg("depth_filter_percentage"),
            "maximum_detection_threshold": arg("track_maximum_detection_threshold"),
            "min_mask_area": arg("track_min_mask_area"),
            "predict_rate": arg("predict_rate"),
            "print_measurement_marker": arg("print_measurement_marker"),
            "max_position_jump": arg("max_position_jump"),
            "relock_window": arg("relock_window"),
            "fix_height": arg("fix_height"),
            "fixed_height": arg("fixed_height"),
        }],
        condition=IfCondition(arg("launch_track_node")),
    )

    declared = [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in ARGUMENTS
    ]
    return LaunchDescription(declared + [sam2_node, mask2pcl_node, track_node])
