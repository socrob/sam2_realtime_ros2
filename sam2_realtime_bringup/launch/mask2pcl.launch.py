from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('depth_topic', default_value='/k4a/depth_to_rgb/image_raw'),
        DeclareLaunchArgument('cam_info', default_value='/k4a/rgb/camera_info'),
        DeclareLaunchArgument('sam2_mask_topic', default_value='/sam2/mask'),
        DeclareLaunchArgument('target_frame', default_value='camera_base'),
        DeclareLaunchArgument('depth_image_units_divisor', default_value='1'), # e.g., 1000 for mm→m
        # Tracking / filtering params
        DeclareLaunchArgument('tracking_active', default_value='true'),
        DeclareLaunchArgument('min_mask_area', default_value='100'),
        DeclareLaunchArgument('cloud_stride', default_value='4'),
        # Kept for compatibility if used elsewhere in node
        DeclareLaunchArgument('maximum_detection_threshold', default_value='0.3'),


        Node(
            package='sam2_realtime',
            executable='mask2pcl',
            name='mask2pcl',
            output='screen',
            parameters=[{
                'depth_topic': LaunchConfiguration('depth_topic'),
                'cam_info': LaunchConfiguration('cam_info'),
                'sam2_mask_topic': LaunchConfiguration('sam2_mask_topic'),
                'target_frame': LaunchConfiguration('target_frame'),
                'depth_image_units_divisor': LaunchConfiguration('depth_image_units_divisor'),
                'tracking_active': LaunchConfiguration('tracking_active'),
                'min_mask_area': LaunchConfiguration('min_mask_area'),
                'cloud_stride': LaunchConfiguration('cloud_stride'),
                'maximum_detection_threshold': LaunchConfiguration('maximum_detection_threshold'),
            }],
        )
    ])