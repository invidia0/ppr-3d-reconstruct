#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

import jax
import jax.numpy as jnp

from tgpr.tgpr import TGPR


class TrajPred(Node):
    def __init__(self):
        super().__init__('trajectory_prediction')
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0
        self.declare_parameters(
            namespace='',
            parameters=[
                ('measurement_topic', '/pose_measurements'),
                ('dataset_history', 10)
            ]
        )

        self.measurement_topic = self.get_parameter('measurement_topic').get_parameter_value().string_value
        _dataset_history = self.get_parameter('dataset_history').get_parameter_value().integer_value

        # Subs
        self._obs_sub = self.create_subscription(
            PoseStamped,
            self.measurement_topic,
            self.measurements_callback,
            1)
        self._obs_sub

        # Jax setup
        self.key = jax.random.PRNGKey(0)

        self.tgpr = TGPR(dataset_history=_dataset_history,
                        sigma_v=0.1,
                        sigma_w=0.1,
                        C_single=jnp.eye(3),
                        K0=jnp.eye(3)*0.01,
                        R=jnp.eye(3)*0.01)


    def measurements_callback(self, msg: PoseStamped):
        # Extract measurement
        measurement = jnp.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 msg.pose.position.z])

        # Update TGPR with new measurement
        self.tgpr.pushback_measurements(measurement)
        self.get_logger().info(f'New measurement received: {measurement}')

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