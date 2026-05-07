#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():

    # -------------------------------------------------------------------------
    # Node selection arguments
    # -------------------------------------------------------------------------
    launch_sam2_arg = DeclareLaunchArgument(
        "launch_sam2",
        default_value="true",
        description="Whether to launch the SAM2 realtime lifecycle node"
    )

    launch_mask2pcl_arg = DeclareLaunchArgument(
        "launch_mask2pcl",
        default_value="true",
        description="Whether to launch the mask to pointcloud node"
    )

    launch_track_node_arg = DeclareLaunchArgument(
        "launch_track_node",
        default_value="true",
        description="Whether to launch the tracker node"
    )

    # -------------------------------------------------------------------------
    # Namespace arguments
    # -------------------------------------------------------------------------
    sam2_namespace_arg = DeclareLaunchArgument(
        "sam2_namespace",
        default_value="sam2",
        description="Namespace for the SAM2 node"
    )

    mask2pcl_namespace_arg = DeclareLaunchArgument(
        "mask2pcl_namespace",
        default_value="mask2pcl",
        description="Namespace for the mask2pcl node"
    )

    track_namespace_arg = DeclareLaunchArgument(
        "track_namespace",
        default_value="track_node",
        description="Namespace for the tracker node"
    )

    # -------------------------------------------------------------------------
    # Shared camera/topic arguments
    # -------------------------------------------------------------------------
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/k4a/rgb/image_raw",
        description="RGB image topic"
    )

    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/k4a/depth_to_rgb/image_raw",
        description="Depth image topic"
    )

    cam_info_arg = DeclareLaunchArgument(
        "cam_info",
        default_value="/k4a/rgb/camera_info",
        description="Camera info topic"
    )

    sam2_mask_topic_arg = DeclareLaunchArgument(
        "sam2_mask_topic",
        default_value="/sam2/mask",
        description="SAM2 output mask topic"
    )

    depth_image_units_divisor_arg = DeclareLaunchArgument(
        "depth_image_units_divisor",
        default_value="1",
        description="Depth image units divisor, e.g. 1000 for mm to meters"
    )

    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="base_footprint",
        description="Common target frame for mask2pcl and tracker"
    )

    # -------------------------------------------------------------------------
    # SAM2 parameters
    # -------------------------------------------------------------------------
    image_reliability_arg = DeclareLaunchArgument(
        "image_reliability",
        default_value="2",
        description="QoS reliability setting: 1 = RELIABLE, 2 = BEST_EFFORT"
    )

    model_cfg_arg = DeclareLaunchArgument(
        "model_cfg",
        default_value="configs/sam2.1/sam2.1_hiera_s.yaml",
        description="Path to SAM2 model config"
    )

    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint",
        default_value="checkpoints/sam2.1_hiera_small.pt",
        description="Path to SAM2 checkpoint"
    )

    live_visualization_arg = DeclareLaunchArgument(
        "live_visualization",
        default_value="False",
        description="Enable SAM2 live mask visualization"
    )

    # -------------------------------------------------------------------------
    # Mask2PCL parameters
    # -------------------------------------------------------------------------
    mask2pcl_enable_arg = DeclareLaunchArgument(
        "mask2pcl_enable",
        default_value="true",
        description="Enable mask2pcl processing"
    )

    mask2pcl_min_mask_area_arg = DeclareLaunchArgument(
        "mask2pcl_min_mask_area",
        default_value="100",
        description="Minimum mask area for mask2pcl"
    )

    cloud_stride_arg = DeclareLaunchArgument(
        "cloud_stride",
        default_value="4",
        description="Stride used when generating the pointcloud"
    )

    mask2pcl_maximum_detection_threshold_arg = DeclareLaunchArgument(
        "mask2pcl_maximum_detection_threshold",
        default_value="0.3",
        description="Maximum detection threshold for mask2pcl"
    )

    # -------------------------------------------------------------------------
    # Tracker parameters
    # -------------------------------------------------------------------------
    track_enable_arg = DeclareLaunchArgument(
        "track_enable",
        default_value="false",
        description="Enable tracker processing"
    )

    track_min_mask_area_arg = DeclareLaunchArgument(
        "track_min_mask_area",
        default_value="1000",
        description="Minimum mask area for tracking"
    )

    predict_rate_arg = DeclareLaunchArgument(
        "predict_rate",
        default_value="10",
        description="Tracker prediction rate"
    )

    print_measurement_marker_arg = DeclareLaunchArgument(
        "print_measurement_marker",
        default_value="true",
        description="Whether to publish/print the measurement marker"
    )

    depth_filter_percentage_arg = DeclareLaunchArgument(
        "depth_filter_percentage",
        default_value="0.3",
        description="Depth filtering percentage"
    )

    track_maximum_detection_threshold_arg = DeclareLaunchArgument(
        "track_maximum_detection_threshold",
        default_value="0.3",
        description="Maximum detection threshold for tracker"
    )

    max_depth_jump_arg = DeclareLaunchArgument(
        "max_depth_jump",
        default_value="0.3",
        description="Maximum accepted depth jump"
    )

    relock_window_arg = DeclareLaunchArgument(
        "relock_window",
        default_value="1",
        description="Tracker relock window"
    )

    # -------------------------------------------------------------------------
    # SAM2 Lifecycle Node
    # -------------------------------------------------------------------------
    sam2_node = LifecycleNode(
        package="sam2_realtime",
        executable="sam2_realtime_node",
        name="sam2_realtime_node",
        namespace=LaunchConfiguration("sam2_namespace"),
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
            "image_reliability": LaunchConfiguration("image_reliability"),
            "model_cfg": LaunchConfiguration("model_cfg"),
            "checkpoint": LaunchConfiguration("checkpoint"),
            "live_visualization": LaunchConfiguration("live_visualization"),
        }],
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_sam2"))
    )

    # -------------------------------------------------------------------------
    # Mask2PCL Node
    # -------------------------------------------------------------------------
    mask2pcl_node = Node(
        package="sam2_realtime",
        executable="mask2pcl",
        name="mask2pcl",
        namespace=LaunchConfiguration("mask2pcl_namespace"),
        output="screen",
        parameters=[{
            "depth_topic": LaunchConfiguration("depth_topic"),
            "cam_info": LaunchConfiguration("cam_info"),
            "sam2_mask_topic": LaunchConfiguration("sam2_mask_topic"),
            "target_frame": LaunchConfiguration("target_frame"),
            "depth_image_units_divisor": LaunchConfiguration("depth_image_units_divisor"),
            "enable": LaunchConfiguration("mask2pcl_enable"),
            "min_mask_area": LaunchConfiguration("mask2pcl_min_mask_area"),
            "cloud_stride": LaunchConfiguration("cloud_stride"),
            "maximum_detection_threshold": LaunchConfiguration(
                "mask2pcl_maximum_detection_threshold"
            ),
        }],
        condition=IfCondition(LaunchConfiguration("launch_mask2pcl"))
    )

    # -------------------------------------------------------------------------
    # Tracker Node
    # -------------------------------------------------------------------------
    track_node = Node(
        package="sam2_realtime",
        executable="track_node",
        name="track_node",
        namespace=LaunchConfiguration("track_namespace"),
        output="screen",
        parameters=[{
            "depth_topic": LaunchConfiguration("depth_topic"),
            "cam_info": LaunchConfiguration("cam_info"),
            "sam2_mask_topic": LaunchConfiguration("sam2_mask_topic"),
            "target_frame": LaunchConfiguration("target_frame"),
            "depth_filter_percentage": LaunchConfiguration("depth_filter_percentage"),
            "depth_image_units_divisor": LaunchConfiguration("depth_image_units_divisor"),
            "maximum_detection_threshold": LaunchConfiguration(
                "track_maximum_detection_threshold"
            ),
            "min_mask_area": LaunchConfiguration("track_min_mask_area"),
            "predict_rate": LaunchConfiguration("predict_rate"),
            "print_measurement_marker": LaunchConfiguration("print_measurement_marker"),
            "max_depth_jump": LaunchConfiguration("max_depth_jump"),
            "relock_window": LaunchConfiguration("relock_window"),
            "enable": LaunchConfiguration("track_enable"),
        }],
        condition=IfCondition(LaunchConfiguration("launch_track_node"))
    )

    return LaunchDescription([
        # Node selection
        launch_sam2_arg,
        launch_mask2pcl_arg,
        launch_track_node_arg,

        # Namespaces
        sam2_namespace_arg,
        mask2pcl_namespace_arg,
        track_namespace_arg,

        # Shared topics / camera params
        image_topic_arg,
        depth_topic_arg,
        cam_info_arg,
        sam2_mask_topic_arg,
        depth_image_units_divisor_arg,
        target_frame_arg,

        # SAM2 params
        image_reliability_arg,
        model_cfg_arg,
        checkpoint_arg,
        live_visualization_arg,

        # Mask2PCL params
        mask2pcl_enable_arg,
        mask2pcl_min_mask_area_arg,
        cloud_stride_arg,
        mask2pcl_maximum_detection_threshold_arg,

        # Tracker params
        track_enable_arg,
        track_min_mask_area_arg,
        predict_rate_arg,
        print_measurement_marker_arg,
        depth_filter_percentage_arg,
        track_maximum_detection_threshold_arg,
        max_depth_jump_arg,
        relock_window_arg,

        # Nodes
        sam2_node,
        mask2pcl_node,
        track_node,
    ])