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

class TrajPublish(Node):
    def __init__(self):
        super().__init__('traj_publisher')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('pose_topic', '/pose'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('figure_8_scale', 3.0),
                ('kp_v', 3.0),
                ('ki_v', 0.0),
                ('max_velocity', 0.5),
                ('kp_omega', 3.0),
                ('ki_omega', 0.0),
                ('max_angular_velocity', 1.0),
                ('pure_pursuit_lookahead', 0.5)
            ]
        )

        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.figure_8_scale = self.get_parameter('figure_8_scale').get_parameter_value().double_value
        self.kp_v = self.get_parameter('kp_v').get_parameter_value().double_value
        self.ki_v = self.get_parameter('ki_v').get_parameter_value().double_value
        self.max_velocity = self.get_parameter('max_velocity').get_parameter_value().double_value
        self.kp_omega = self.get_parameter('kp_omega').get_parameter_value().double_value
        self.ki_omega = self.get_parameter('ki_omega').get_parameter_value().double_value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.pp_lookahead = self.get_parameter('pure_pursuit_lookahead').get_parameter_value().double_value

        # _dataset_history = self.get_parameter('dataset_history').get_parameter_value().integer_value

        # System initialization
        # self._f8_traj = self.figure_8_trajectory(100, scale=self.figure_8_scale)
        self._current_position = np.array([0.0, 0.0])
        self._current_yaw = 0.0

        self._f8_traj = self.figure_8_trajectory(100, scale=self.figure_8_scale)
        self._f8_marker_array, self._robot_marker = self._build_markers()
        self._next_point = np.array([self._f8_traj[0][0], self._f8_traj[1][0]])
        self._traj_counter = 0
        self._PI_vel = PI(kp=self.kp_v, ki=self.ki_v)
        self._PI_omega = PI(kp=self.kp_omega, ki=self.ki_omega)

        # Subs
        self._pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.control_callback,
            qos_profile=qos_profile_sensor_data
        )

        # Timers
        timer_period = 0.1
        self.get_logger().info("Creating f8 timer")
        self._f8_traj_timer = self.create_timer(timer_period, self.f8_traj_callback)
        self.get_logger().info("Timer created")

        # Pubs
        self._f8_traj_pub = self.create_publisher(MarkerArray, 'f8_trajectory', 1)
        self._robot_marker_pub = self.create_publisher(Marker, 'robot_marker', 1)
        self._cmd_vel_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 1)

        # Message holders
        self._cmd_vel_msg = TwistStamped()

        # Last time
        self._last_time = self.get_clock().now()


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

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker.lifetime = Duration(sec=0, nanosec=0)
            marker_array.markers.append(marker)

        robot_marker = Marker()
        robot_marker.header.frame_id = "world"
        robot_marker.ns = "robot"
        robot_marker.id = len(marker_array.markers) + 1
        robot_marker.type = Marker.ARROW
        robot_marker.action = Marker.ADD

        robot_marker.scale.x = 0.4
        robot_marker.scale.y = 0.1
        robot_marker.scale.z = 0.1

        robot_marker.color.a = 1.0
        robot_marker.color.r = 0.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 1.0
        robot_marker.lifetime = Duration(sec=0, nanosec=0)

        return marker_array, robot_marker

    def f8_traj_callback(self):
        now = self.get_clock().now().to_msg()
        for marker in self._f8_marker_array.markers:
            marker.header.stamp = now
            if marker.id == self._traj_counter:
                marker.color.r = 1.0
                marker.color.g = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
        self._f8_traj_pub.publish(self._f8_marker_array)


    def _publish_robot_marker(self, msg: PoseStamped):
        self._robot_marker.header.stamp = self.get_clock().now().to_msg()
        # position & orientation from mocap
        self._robot_marker.pose = msg.pose
        self._robot_marker_pub.publish(self._robot_marker)

    def control_callback(self, msg: PoseStamped):
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

        # Advance along trajectory while inside lookahead
        # while True:
        #     dx_look = self._next_point[0] - position[0]
        #     dy_look = self._next_point[1] - position[1]
        #     dist_look = (dx_look**2 + dy_look**2)**0.5

        #     if dist_look < self.pp_lookahead:
        #         self.get_logger().info("Advancing to next trajectory point")
        #         self._traj_counter += 1
        #         if self._traj_counter >= len(self._f8_traj[0]):
        #             self._traj_counter = 0
        #         self._next_point = np.array([
        #             self._f8_traj[0][self._traj_counter],
        #             self._f8_traj[1][self._traj_counter]
        #         ])
        #     else:
        #         break
        if np.linalg.norm(self._next_point - position[0:2]) < self.pp_lookahead:
            self.get_logger().info("Advancing to next trajectory point")
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

        # # choose a comfortable forward speed
        # v_cmd = min(self.max_velocity, 0.15)

        # # pure pursuit curvature
        # Ld = max(self.pp_lookahead, 0.1)
        # omega_cmd = 2.0 * v_cmd * np.sin(heading_error) / Ld

        # # saturate
        # omega_cmd = np.clip(omega_cmd, -self.max_angular_velocity, self.max_angular_velocity)

        if abs(heading_error) > np.pi / 3:
            v_cmd *= 0.3
        if abs(heading_error) > np.pi / 2:
            v_cmd = 0.0
        if error < 0.01:
            v_cmd = 0.0
            omega_cmd = 0.0

        self.get_logger().info(
            f"Error: {error:.3f}, Heading error: {heading_error:.3f}, "
            f"v_cmd: {v_cmd:.3f}, omega_cmd: {omega_cmd:.3f}"
        )

        self._cmd_vel_msg.header.stamp = now.to_msg()
        self._cmd_vel_msg.twist.linear.x = float(v_cmd)
        self._cmd_vel_msg.twist.linear.y = 0.0
        self._cmd_vel_msg.twist.linear.z = 0.0
        self._cmd_vel_msg.twist.angular.x = 0.0
        self._cmd_vel_msg.twist.angular.y = 0.0
        self._cmd_vel_msg.twist.angular.z = float(omega_cmd)

        self._publish_robot_marker(msg)
        self._cmd_vel_pub.publish(self._cmd_vel_msg)

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