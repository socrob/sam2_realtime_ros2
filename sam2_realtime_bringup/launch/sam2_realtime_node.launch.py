#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    image_topic = LaunchConfiguration("image_topic")
    image_reliability = LaunchConfiguration("image_reliability")
    model_cfg = LaunchConfiguration("model_cfg")
    checkpoint = LaunchConfiguration("checkpoint")
    live_visualization = LaunchConfiguration("live_visualization")

    return LaunchDescription([
        DeclareLaunchArgument(
            "namespace",
            default_value="sam2",
            description="Namespace for the SAM2 node"
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/k4a/rgb/image_raw",
            description="Image topic for RGB input"
        ),
        DeclareLaunchArgument(
            "image_reliability",
            default_value="2",  # 1 = RELIABLE, 2 = BEST_EFFORT
            description="QoS reliability setting"
        ),
        DeclareLaunchArgument(
            "model_cfg",
            default_value="configs/sam2.1/sam2.1_hiera_s.yaml",
            description="Path to SAM2 model config"
        ),
        DeclareLaunchArgument(
            "checkpoint",
            default_value="checkpoints/sam2.1_hiera_small.pt",
            description="Path to SAM2 checkpoint"
        ),
        DeclareLaunchArgument(
            "live_visualization",
            default_value="False",
            description="SAM2 mask visualization"
        ),
        LifecycleNode(
            package="sam2_realtime",
            executable="sam2_realtime_node",
            name="sam2_realtime_node",
            namespace=namespace,
            parameters=[{
                "image_topic": image_topic,
                "image_reliability": image_reliability,
                "model_cfg": model_cfg,
                "checkpoint": checkpoint,
                "live_visualization": live_visualization,
            }],
            output="screen"
        )
    ])
