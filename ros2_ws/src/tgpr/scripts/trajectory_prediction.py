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

        self.tgpr_model = TGPR(dataset_history=_dataset_history)

        self.tgpr_model.hello()


    def measurements_callback(self, msg: PoseStamped):
        pass
    # def timer_callback(self):
    #     msg = String()
    #     msg.data = 'Hello World: %d' % self.i
    #     self.publisher_.publish(msg)
    #     self.get_logger().info('Publishing: "%s"' % msg.data)
    #     self.i += 1


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