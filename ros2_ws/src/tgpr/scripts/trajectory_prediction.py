#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

import jax
import jax.numpy as jnp


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
                ('measurement_topic', '/pose_measurements', rclpy.Parameter.Type.STRING),
            ]
        )

        self.measurement_topic = self.get_parameter('measurement_topic').get_parameter_value().string_value

        # Subs
        self._obs_sub = self.create_subscription(
            PoseStamped,
            self.measurement_topic,
            self.measurements_callback,
            1)
        self._obs_sub


        # Jax setup
        self.key = jax.random.PRNGKey(0)
        

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

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()