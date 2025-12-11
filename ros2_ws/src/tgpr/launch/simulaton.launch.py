#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'empty_world.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world], 'on_exit_shutdown': 'true'}.items()
    )

    # gzserver_cmd = Node(
    #     package='ros_gz_sim',
    #     executable='gz_sim',
    #     name='gz_sim',
    #     output='screen',
    #     arguments=['-r', '-s', '-v2', world],
    # )

    # (optional) Gazebo client – kept disabled for headless sim
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 '}.items()
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

    set_env_vars_resources = AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'models'))

    # Transform from odom to world
    tf2_odom_to_world_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_odom_to_world',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'world_dummy']
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
    # VRPN Mocap
    # vrpn_client_node = Node(
    #     package='vrpn_mocap',
    #     executable='vrpn_mocap_node',
    #     name='vrpn_mocap_client',
    #     output='screen',
    #     parameters=[{
    #         'server': '192.168.1.103',  # Replace with your VRPN server IP or hostname
    #         'port': 3883,
    #         'frame_id': 'world',
    #         'update_freq': 100.0,
    #         'refresh_freq': 1.0,
    #         'sensor_data_qos': True,
    #         'multi_sensor': False,
    #         'use_vrpn_timestamps': False,
    #     }]
    # )

    ld = LaunchDescription()
    # ld.add_action(gzserver_cmd)
    # ld.add_action(gzclient_cmd)
    # ld.add_action(spawn_turtlebot_cmd)
    # ld.add_action(robot_state_publisher_cmd)
    # ld.add_action(set_env_vars_resources)
    # ld.add_action(tf2_odom_to_world_node)
    ld.add_action(world_dummy_tf)
    ld.add_action(rviz2_node)
    ld.add_action(traj_publisher_node)

    return ld
