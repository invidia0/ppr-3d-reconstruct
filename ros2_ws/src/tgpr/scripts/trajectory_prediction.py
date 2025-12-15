#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseArray

from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.35"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

import jax
import jax.numpy as jnp

from tgpr.tgpr import TGPR

import math

os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '[{severity}] ({name}): {message}'
os.environ['RCUTILS_COLORIZED_OUTPUT'] = '1'

class TrajPred(Node):
    def __init__(self):
        super().__init__('trajectory_prediction')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('measurement_topic', '/pose_measurements'),
                ('dataset_history', 10),
                ('prediction_horizon', 5)
            ]
        )

        self.measurement_topic = self.get_parameter('measurement_topic').get_parameter_value().string_value
        self._dataset_history = self.get_parameter('dataset_history').get_parameter_value().integer_value
        self.prediction_horizon = self.get_parameter('prediction_horizon').get_parameter_value().integer_value

        self._measurements = jnp.empty((0, 4))  # x, y, yaw, timestamp

        # self._meas_array = MarkerArray()
        self._meas_marker_timer = self.create_timer(0.2, self._meas_markers_pub)
        self._meas_pub = self.create_publisher(PoseArray, 'measurements_array', 1)

        self._prior_marker_timer = self.create_timer(0.2, self._prior_markers_pub)
        self._prior_pub = self.create_publisher(PoseArray, 'prior_array', 1)

        self._posterior_marker_timer = self.create_timer(0.2, self._posterior_markers_pub)
        self._posterior_pub = self.create_publisher(PoseArray, 'posterior_array', 1)

        # Timers
        # (Timer for prediction)
        self._pred_timer = self.create_timer(0.02, self._predict_trajectory_timer_callback)  # 50 Hz


        # Subs
        self._obs_sub = self.create_subscription(
            PoseStamped,
            self.measurement_topic,
            self.measurements_callback,
            1)

        # Jax setup
        self.key = jax.random.PRNGKey(0)
        self.get_logger().info('🔑 JAX random key initialized.')

        self.tgpr = TGPR(dataset_history=self._dataset_history,
                        sigma_v=0.1,
                        sigma_w=0.1,
                        C_single=jnp.eye(3),
                        K0=jnp.eye(3)*0.01,
                        R=jnp.eye(3)*0.01)
        
        self.get_logger().info('✅️ Trajectory Prediction Node has been started.')


    def _meas_markers_pub(self) -> None:
        meas = self.tgpr.measurements.T  # shape (3, N): x,y,yaw

        now = self.get_clock().now().to_msg()
        arr = PoseArray()

        arr.header.stamp = now
        arr.header.frame_id = 'world'

        for i in range(meas.shape[1]):
            pose = PoseStamped()
            pose.pose.position.x = float(meas[0, i])
            pose.pose.position.y = float(meas[1, i])
            pose.pose.position.z = 0.0  # Assuming z=0 for measurements

            # Orientation from yaw
            yaw = float(meas[2, i])
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            arr.poses.append(pose.pose)

        self._meas_pub.publish(arr)
        self.get_logger().info(f'📤 Published measurement markers.')


    def _prior_markers_pub(self) -> None:
        prior = self.tgpr.prior.T  # shape (3, N): x,y,yaw

        now = self.get_clock().now().to_msg()
        arr = PoseArray()

        arr.header.stamp = now
        arr.header.frame_id = 'world'

        for i in range(prior.shape[1]):
            pose = PoseStamped()
            pose.pose.position.x = float(prior[0, i])
            pose.pose.position.y = float(prior[1, i])
            pose.pose.position.z = 0.0  # Assuming z=0 for prior

            # Orientation from yaw
            yaw = float(prior[2, i])
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            arr.poses.append(pose.pose)

        self._prior_pub.publish(arr)
        self.get_logger().info(f'📤 Published prior markers.')


    def _posterior_markers_pub(self) -> None:
        x_traj = self.tgpr.predicted_trajectory.T  # shape (3, N): x,y,yaw
        now = self.get_clock().now().to_msg()
        arr = PoseArray()

        arr.header.stamp = now
        arr.header.frame_id = 'world'

        for i in range(x_traj.shape[1]):
            pose = PoseStamped()
            pose.pose.position.x = float(x_traj[0, i])
            pose.pose.position.y = float(x_traj[1, i])
            pose.pose.position.z = 0.2

            # Orientation from yaw
            yaw = float(x_traj[2, i])
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            arr.poses.append(pose.pose)

        self._posterior_pub.publish(arr)
        self.get_logger().info(f'📤 Published posterior markers.')


    def _predict_trajectory_timer_callback(self) -> None:
        if self._measurements.shape[0] < self._dataset_history:
            self.get_logger().warning(f'⚠️ Not enough measurements: {self._measurements.shape[0]}/{self._dataset_history}')
            return
        
        self.tgpr.measurements = self._measurements[:, :3]  # x, y, yaw

        # Average dt from measurements
        time_stamps = self._measurements[:, 3]
        dt_estimates = jnp.diff(time_stamps)
        avg_dt = jnp.mean(dt_estimates)
        
        self.tgpr.predict_trajectory(dt=avg_dt, pred_horizon=self.prediction_horizon)

        self.get_logger().info(f'🚀 Prediction successful.')

    def measurements_callback(self, msg: PoseStamped):
        now = self.get_clock().now()

        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
        measurement = jnp.array([msg.pose.position.x,
                                msg.pose.position.y,
                                yaw,
                                now.nanoseconds * 1e-9])

        self._measurements = jnp.vstack([self._measurements, measurement])
        if self._measurements.shape[0] > self.tgpr._max_hist:
            self._measurements = self._measurements.at[1:, :].get()


def main(args=None):
    rclpy.init(args=args)

    trajectory_prediction = TrajPred()

    rclpy.spin(trajectory_prediction)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    trajectory_prediction.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()