#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    
    # Camera type parameter
    camera_type_arg = DeclareLaunchArgument(
        'camera_type', 
        default_value='realsense',
        description='Camera type: azure, realsense, sim_head, sim_wrist'
    )
    
    # Node selection arguments
    launch_sam2_arg = DeclareLaunchArgument('launch_sam2', default_value='true')
    launch_mask2pcl_arg = DeclareLaunchArgument('launch_mask2pcl', default_value='false')
    launch_track_node_arg = DeclareLaunchArgument('launch_track_node', default_value='false')
    launch_yolo_prompt_arg = DeclareLaunchArgument('launch_yolo_prompt', default_value='false')
    
    # SAM2 arguments
    sam2_namespace_arg = DeclareLaunchArgument('sam2_namespace', default_value='sam2')
    sam2_live_viz_arg = DeclareLaunchArgument('sam2_live_visualization', default_value='True')
    
    # Camera topic arguments (will be set based on camera_type)
    image_topic_arg = DeclareLaunchArgument('image_topic', default_value='/camera/camera/color/image_raw')
    depth_topic_arg = DeclareLaunchArgument('depth_topic', default_value='/camera/camera/depth/image_rect_raw')
    cam_info_arg = DeclareLaunchArgument('cam_info', default_value='/camera/camera/color/camera_info')
    target_frame_arg = DeclareLaunchArgument('target_frame', default_value='camera_link')
    camera_frame_arg = DeclareLaunchArgument('camera_frame', default_value='camera_color_optical_frame')
    depth_divisor_arg = DeclareLaunchArgument('depth_divisor', default_value='1000')
    
    # Other namespace arguments
    mask2pcl_namespace_arg = DeclareLaunchArgument('mask2pcl_namespace', default_value='mask2pcl')
    track_namespace_arg = DeclareLaunchArgument('track_namespace', default_value='track_node')

    # SAM2 Node
    sam2_node = LifecycleNode(
        package="sam2_realtime",
        executable="sam2_realtime_node",
        name="sam2_realtime_node",
        namespace=LaunchConfiguration('sam2_namespace'),
        parameters=[{
            "image_topic": LaunchConfiguration('image_topic'),
            "image_reliability": 2,
            "model_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "checkpoint": "checkpoints/sam2.1_hiera_small.pt",
            "live_visualization": LaunchConfiguration('sam2_live_visualization'),
        }],
        output="screen",
        condition=IfCondition(LaunchConfiguration('launch_sam2'))
    )

    # Mask2PCL Node
    mask2pcl_node = Node(
        package='sam2_realtime',
        executable='mask2pcl',
        namespace=LaunchConfiguration('mask2pcl_namespace'),
        name='mask2pcl',
        output='screen',
        parameters=[{
            'depth_topic': LaunchConfiguration('depth_topic'),
            'cam_info': LaunchConfiguration('cam_info'),
            'sam2_mask_topic': '/sam2/mask',
            'target_frame': LaunchConfiguration('target_frame'),
            'depth_image_units_divisor': LaunchConfiguration('depth_divisor'),
            'enable': False,
            'min_mask_area': 200,
            'cloud_stride': 2,
            'maximum_detection_threshold': 0.3,
        }],
        condition=IfCondition(LaunchConfiguration('launch_mask2pcl'))
    )

    # Track Node
    track_node = Node(
        package='sam2_realtime',
        executable='track_node',
        namespace=LaunchConfiguration('track_namespace'),
        name='track_node',
        output='screen',
        parameters=[{
            'depth_topic': LaunchConfiguration('depth_topic'),
            'cam_info': LaunchConfiguration('cam_info'),
            'sam2_mask_topic': '/sam2/mask',
            'target_frame': LaunchConfiguration('target_frame'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'depth_image_units_divisor': LaunchConfiguration('depth_divisor'),
            'enable': False,
            'min_mask_area': 1000,
            'predict_rate': 10,
            'print_measurement_marker': True,
            'depth_filter_percentage': 0.3,
            'maximum_detection_threshold': 0.3,
            'max_depth_jump': 0.3,
            'relock_window': 1,
        }],
        condition=IfCondition(LaunchConfiguration('launch_track_node'))
    )

    # YOLO Prompt Node
    yolo_prompt_node = Node(
        package='sam2_realtime',
        executable='yolo_prompt_node',
        name='yolo_prompt_node',
        output='screen',
        parameters=[{
            'image_topic': LaunchConfiguration('image_topic'),
            'yolo_model': 'yolov8n.pt',
            'detect_class': 'cup',
            'confidence_threshold': 0.4,
            'min_box_area': 800,
            'max_aspect_ratio': 3.0,
            'imgsz': 640,
        }],
        condition=IfCondition(LaunchConfiguration('launch_yolo_prompt'))
    )

    return LaunchDescription([
        camera_type_arg,
        launch_sam2_arg,
        launch_mask2pcl_arg, 
        launch_track_node_arg,
        launch_yolo_prompt_arg,
        sam2_namespace_arg,
        sam2_live_viz_arg,
        image_topic_arg,
        depth_topic_arg,
        cam_info_arg,
        target_frame_arg,
        camera_frame_arg,
        depth_divisor_arg,
        mask2pcl_namespace_arg,
        track_namespace_arg,
        sam2_node,
        mask2pcl_node,
        track_node,
        yolo_prompt_node,
    ])