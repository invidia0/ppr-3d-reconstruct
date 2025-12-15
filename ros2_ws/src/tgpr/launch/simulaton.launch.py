#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    set_env_vars_resources = AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'models'))

    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'empty_world.world'
    )

    gz_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world]}.items()
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    # RViz config
    rviz_config_file = os.path.join(
        get_package_share_directory('tgpr'),
        'rviz',
        'simulation.rviz'
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    traj_publisher_params = os.path.join(
        get_package_share_directory('tgpr'),
        'config',
        'traj_publisher-params.yaml'
    )

    traj_publisher_node = Node(
        package='tgpr',
        executable='traj_publisher.py',
        name='traj_publisher',
        output='screen',
        parameters=[traj_publisher_params, {'use_sim_time': use_sim_time}]
    )

    world_dummy_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_world_dummy',
        arguments=[
            '0', '0', '0',        # x y z
            '0', '0', '0',        # roll pitch yaw
            'world',              # parent frame
            'world_dummy'         # child frame (never used)
        ]
    )

    ld = LaunchDescription()
    ld.add_action(SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'))
    ld.add_action(SetEnvironmentVariable('MESA_GL_VERSION_OVERRIDE', '3.3'))
    ld.add_action(set_env_vars_resources)
    ld.add_action(gz_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(world_dummy_tf)
    ld.add_action(rviz2_node)
    ld.add_action(traj_publisher_node)

    return ld
