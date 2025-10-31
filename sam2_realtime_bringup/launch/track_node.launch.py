from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='track_node'),
        DeclareLaunchArgument('depth_topic', default_value='/k4a/depth_to_rgb/image_raw'),
        DeclareLaunchArgument('cam_info', default_value='/k4a/rgb/camera_info'),
        DeclareLaunchArgument('target_frame', default_value='rgb_camera_link'),
        DeclareLaunchArgument('depth_image_units_divisor', default_value='1'),
        DeclareLaunchArgument('sam2_mask_topic', default_value='/sam2/mask'),
        DeclareLaunchArgument('depth_filter_percentage', default_value='0.3'),
        DeclareLaunchArgument('maximum_detection_threshold', default_value='0.3'),
        DeclareLaunchArgument('min_mask_area', default_value='1000'),
        DeclareLaunchArgument('predict_rate', default_value='10'),
        DeclareLaunchArgument('print_measurement_marker', default_value='true'),
        DeclareLaunchArgument('max_depth_jump', default_value='0.3'),
        DeclareLaunchArgument('relock_window', default_value='1'),
        DeclareLaunchArgument('enable', default_value='false'),

        Node(
            package='sam2_realtime',
            executable='track_node',
            namespace=LaunchConfiguration('namespace'),
            name='track_node',
            output='screen',
            parameters=[{
                'depth_topic': LaunchConfiguration('depth_topic'),
                'cam_info': LaunchConfiguration('cam_info'),
                'sam2_mask_topic': LaunchConfiguration('sam2_mask_topic'),
                'target_frame': LaunchConfiguration('target_frame'),
                'depth_filter_percentage': LaunchConfiguration('depth_filter_percentage'),
                'depth_image_units_divisor': LaunchConfiguration('depth_image_units_divisor'),
                'maximum_detection_threshold': LaunchConfiguration('maximum_detection_threshold'),
                'min_mask_area': LaunchConfiguration('min_mask_area'),
                'predict_rate': LaunchConfiguration('predict_rate'),
                'print_measurement_marker': LaunchConfiguration('print_measurement_marker'),
                'max_depth_jump': LaunchConfiguration('max_depth_jump'),
                'relock_window': LaunchConfiguration('relock_window'),
                'enable': LaunchConfiguration('enable'),
            }],
        )
    ])
