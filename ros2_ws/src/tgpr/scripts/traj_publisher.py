#!/usr/bin/env python3

from turtle import st
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from rclpy.qos import qos_profile_sensor_data, QoSProfile

import numpy as np

from tgpr.pi_controller import PI

import os

os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '[{severity}] ({name}): {message}'
os.environ['RCUTILS_COLORIZED_OUTPUT'] = '1'

class TrajPublish(Node):
    def __init__(self):
        super().__init__('traj_publisher')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('simulation_mode', True),
                ('pose_topic', '/pose'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('measurement_topic', '/pose_measurement'),
                ('figure_8_scale', 3.0),
                ('kp_v', 3.0),
                ('ki_v', 0.0),
                ('max_velocity', 0.5),
                ('kp_omega', 3.0),
                ('ki_omega', 0.0),
                ('max_angular_velocity', 1.0),
                ('pure_pursuit_lookahead', 0.5),
                ('measurement_publish_rate', 10.0)
            ]
        )
        self.simulation_mode = self.get_parameter('simulation_mode').get_parameter_value().bool_value
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.measurement_topic = self.get_parameter('measurement_topic').get_parameter_value().string_value
        self.figure_8_scale = self.get_parameter('figure_8_scale').get_parameter_value().double_value
        self.kp_v = self.get_parameter('kp_v').get_parameter_value().double_value
        self.ki_v = self.get_parameter('ki_v').get_parameter_value().double_value
        self.max_velocity = self.get_parameter('max_velocity').get_parameter_value().double_value
        self.kp_omega = self.get_parameter('kp_omega').get_parameter_value().double_value
        self.ki_omega = self.get_parameter('ki_omega').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.pp_lookahead = self.get_parameter('pure_pursuit_lookahead').get_parameter_value().double_value
        self.measurement_publish_rate = self.get_parameter('measurement_publish_rate').get_parameter_value().double_value

        self._current_pose = PoseStamped()

        self._f8_traj = self.figure_8_trajectory(100, scale=self.figure_8_scale)
        self._f8_marker_array = self._build_markers()
        self._next_point = np.array([self._f8_traj[0][0], self._f8_traj[1][0]])
        self._traj_counter = 0
        self._PI_vel = PI(kp=self.kp_v, ki=self.ki_v)
        self._PI_omega = PI(kp=self.kp_omega, ki=self.ki_omega)

        # Subs
        self._pose_sub = self.create_subscription(
            Odometry if self.simulation_mode else PoseStamped,
            '/odom' if self.simulation_mode else self.pose_topic,
            self.control_callback,
            qos_profile=qos_profile_sensor_data
        )

        # Timers
        _marker_traj_period = 0.1
        self._f8_traj_timer = self.create_timer(_marker_traj_period, self.f8_traj_callback)

        ## Timer for measurements publishing
        _measurement_period = 1.0 / self.measurement_publish_rate
        self._measurement_timer = self.create_timer(_measurement_period, self._meaurement_publisher)

        # Pubs
        self._f8_traj_pub = self.create_publisher(MarkerArray, 'f8_trajectory', 1)
        self._odom_world_pub = self.create_publisher(Odometry, 'odom_world', 1)
        self._cmd_vel_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 1)
        self._measurement_pub = self.create_publisher(PoseStamped, self.measurement_topic, 1)

        # Message holders
        self._cmd_vel_msg = TwistStamped()

        # Last time
        self._last_time = self.get_clock().now()

        self.get_logger().info('✅️ Trajectory Publisher Node has been started.')


    def figure_8_trajectory(self, num_points, scale=1.0):
        t = np.linspace(0, 2 * np.pi, num_points)
        x = scale * np.cos(t)
        y = scale * np.sin(2*t) / 2
        return x, y


    def circle_trajectory(self, num_points, radius=1.0):
        t = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        return x, y
    

    def _meaurement_publisher(self):
        self._measurement_pub.publish(self._current_pose)


    def _build_markers(self) -> MarkerArray:
        marker_array = MarkerArray()
        x, y = self._f8_traj  # trajectory defined in world frame

        for i, (xi, yi) in enumerate(zip(x, y)):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.ns = "f8_traj"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(xi)
            marker.pose.position.y = float(yi)
            marker.pose.position.z = 0.0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            marker.color.a = 1.0
            marker.color.r = 0.62
            marker.color.g = 0.62
            marker.color.b = 0.62

            marker.lifetime = Duration(sec=0, nanosec=0)
            marker_array.markers.append(marker)

        return marker_array


    def f8_traj_callback(self):
        now = self.get_clock().now().to_msg()
        for marker in self._f8_marker_array.markers:
            marker.header.stamp = now
            if marker.id == self._traj_counter:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.62
                marker.color.g = 0.62
                marker.color.b = 0.62
        self._f8_traj_pub.publish(self._f8_marker_array)


    def control_callback(self, msg):
        if self.simulation_mode:
            # In simulation mode, msg is of type Odometry
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            self._odom_world_pub.publish(msg)
        else:
            # Convert PoseStamped to Odometry and publish
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            self._odom_world_pub.publish(Odometry(
                header=msg.header,
                pose=Odometry.Pose(msg.pose),
                twist=Odometry.Twist()
            ))

        if self.simulation_mode:
            # In simulation mode, msg is of type Odometry
            pose = msg.pose.pose
            orientation = pose.orientation
            msg = PoseStamped()
            msg.header = msg.header
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position = pose.position
            msg.pose.orientation = orientation

        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 0.1

        position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        if np.linalg.norm(self._next_point - position[0:2]) < self.pp_lookahead:
            self._traj_counter += 1
            if self._traj_counter >= len(self._f8_traj[0]):
                self._traj_counter = 0
            self._next_point = np.array([
                self._f8_traj[0][self._traj_counter],
                self._f8_traj[1][self._traj_counter]
            ])

        # Now compute control w.r.t current target
        dx = self._next_point[0] - position[0]
        dy = self._next_point[1] - position[1]
        error = (dx**2 + dy**2)**0.5

        target_heading = np.arctan2(dy, dx)
        q = msg.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )

        raw_err = target_heading - yaw
        heading_error = np.arctan2(np.sin(raw_err), np.cos(raw_err))
        heading_error = np.clip(heading_error, -np.pi/2, np.pi/2)

        v_cmd = self._PI_vel.control(error, dt=dt, cap=self.max_velocity)
        omega_cmd = self._PI_omega.control(heading_error, dt=dt, cap=self.max_angular_velocity)

        if abs(heading_error) > np.pi / 3:
            v_cmd *= 0.3
        if abs(heading_error) > np.pi / 2:
            v_cmd = 0.0
        if error < 0.01:
            v_cmd = 0.0
            omega_cmd = 0.0

        self._cmd_vel_msg.header.stamp = now.to_msg()
        self._cmd_vel_msg.twist.linear.x = float(v_cmd)
        self._cmd_vel_msg.twist.linear.y = 0.0
        self._cmd_vel_msg.twist.linear.z = 0.0
        self._cmd_vel_msg.twist.angular.x = 0.0
        self._cmd_vel_msg.twist.angular.y = 0.0
        self._cmd_vel_msg.twist.angular.z = float(omega_cmd)

        self._cmd_vel_pub.publish(self._cmd_vel_msg)

        self._current_pose = msg

        self.get_logger().info(f'🚙 Control sent to actuators')


    def stop_robot(self):
        self.get_logger().info("Stopping robot...")
        stop_msg = TwistStamped()
        stop_msg.header.stamp = self.get_clock().now().to_msg()
        stop_msg.twist.linear.x = 0.0
        stop_msg.twist.linear.y = 0.0
        stop_msg.twist.linear.z = 0.0
        stop_msg.twist.angular.x = 0.0
        stop_msg.twist.angular.y = 0.0
        stop_msg.twist.angular.z = 0.0
        for _ in range(10):
            self._cmd_vel_pub.publish(stop_msg)


if __name__ == '__main__':
    rclpy.init()
    node = TrajPublish()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C received, stopping robot...')
    finally:
        # This will run both on Ctrl+C and on normal shutdown
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()