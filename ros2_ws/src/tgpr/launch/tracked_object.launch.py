#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    tgpr_params = os.path.join(
        get_package_share_directory('tgpr'),
        'config',
        'tgpr-params.yaml'
    )

    traj_publisher_node = Node(
        package='tgpr',
        executable='trajectory_prediction.py',
        name='trajectory_prediction',
        output='screen',
        parameters=[tgpr_params, {'use_sim_time': use_sim_time}]
    )

    ld = LaunchDescription()
    ld.add_action(traj_publisher_node)

    return ld
