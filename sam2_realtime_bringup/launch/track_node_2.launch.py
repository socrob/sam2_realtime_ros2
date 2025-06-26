from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('depth_topic', default_value='/k4a/depth_to_rgb/image_raw'),
        DeclareLaunchArgument('cam_info', default_value='/k4a/rgb/camera_info'),
        DeclareLaunchArgument('target_frame', default_value='camera_base'),
        DeclareLaunchArgument('depth_image_units_divisor', default_value='1'),
        DeclareLaunchArgument('sam2_mask_topic', default_value='/sam2/mask'),
        DeclareLaunchArgument('depth_filter_percentage', default_value='0.3'),
        DeclareLaunchArgument('maximum_detection_threshold', default_value='0.3'),

        Node(
            package='sam2_realtime',
            executable='track_node_2',
            name='track_node_2',
            output='screen',
            parameters=[{
                'depth_topic': LaunchConfiguration('depth_topic'),
                'cam_info': LaunchConfiguration('cam_info'),
                'sam2_mask_topic': LaunchConfiguration('sam2_mask_topic'),
                'target_frame': LaunchConfiguration('target_frame'),
                'depth_filter_percentage': LaunchConfiguration('depth_filter_percentage'),
                'depth_image_units_divisor': LaunchConfiguration('depth_image_units_divisor'),
            }],
        )
    ])
