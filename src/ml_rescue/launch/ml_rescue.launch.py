import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription  # type: ignore[attr-defined]
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('ml_rescue'), 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='ml_rescue',
            executable='state_machine',
            name='ml_rescue',
            namespace='',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='ml_rescue',
            executable='ml_rescue_node',
            name='ml_rescue_node',
            namespace='',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='ml_rescue',
            executable='ml_vision_node',
            name='ml_vision_node',
            namespace='',
            output='screen',
            parameters=[config],
        ),
    ])
