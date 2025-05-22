from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('depth_topic', default_value='/camera/camera/extrinsics/depth_to_color'),
        DeclareLaunchArgument('depth_info', default_value='/camera/camera/depth/camera_info'),
        DeclareLaunchArgument('sam2_mask_topic', default_value='/sam2/mask'),
        DeclareLaunchArgument('target_frame', default_value='base_link'),
        DeclareLaunchArgument('depth_filter_percentage', default_value='0.2'),
        DeclareLaunchArgument('depth_image_units_divisor', default_value='1000'),
        DeclareLaunchArgument('depth_image_reliability', default_value='1'),  # BEST_EFFORT
        DeclareLaunchArgument('depth_info_reliability', default_value='1'),   # BEST_EFFORT

        Node(
            package='sam2_realtime',
            executable='track_node',
            name='track_node',
            output='screen',
            parameters=[{
                'depth_topic': LaunchConfiguration('depth_topic'),
                'depth_info': LaunchConfiguration('depth_info'),
                'sam2_mask_topic': LaunchConfiguration('sam2_mask_topic'),
                'target_frame': LaunchConfiguration('target_frame'),
                'depth_filter_percentage': LaunchConfiguration('depth_filter_percentage'),
                'depth_image_units_divisor': LaunchConfiguration('depth_image_units_divisor'),
                'depth_image_reliability': LaunchConfiguration('depth_image_reliability'),
                'depth_info_reliability': LaunchConfiguration('depth_info_reliability'),
            }],
        )
    ])
