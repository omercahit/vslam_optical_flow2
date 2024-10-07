from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vslam_optical_flow2',
            executable='vslam_optical_flow2',
            name='vslam_optical_flow2',
            output='screen',
        ),
    ])

