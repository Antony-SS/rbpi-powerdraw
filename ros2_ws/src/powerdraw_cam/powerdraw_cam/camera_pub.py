#!/usr/bin/env python3
"""Publish camera frames from a V4L2 loopback device to a ROS 2 topic."""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):
    """ROS 2 node that captures from a V4L2 device and publishes Image msgs.

    Parameters
    ----------
    device_id : int
        V4L2 device index (default: 50, i.e. /dev/video50).
    fps : float
        Publishing frame rate (default: 15.0).
    """

    def __init__(self):
        super().__init__('camera_publisher')

        self.declare_parameter('device_id', 50)
        self.declare_parameter('fps', 10.0)

        device_id = self.get_parameter('device_id').value
        fps = self.get_parameter('fps').value

        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.bridge_ = CvBridge()
        self.cap_ = cv2.VideoCapture(device_id)

        if not self.cap_.isOpened():
            self.get_logger().error(f'Failed to open /dev/video{device_id}')
            raise RuntimeError(f'Cannot open /dev/video{device_id}')

        self.get_logger().info(
            f'Capturing from /dev/video{device_id} at {fps} fps'
        )
        self.timer_ = self.create_timer(1.0 / fps, self._timer_cb)

    def _timer_cb(self):
        """Grab a frame and publish it."""
        ret, frame = self.cap_.read()
        if not ret:
            self.get_logger().warn('Failed to read frame', throttle_duration_sec=5.0)
            return
        msg = self.bridge_.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

    def destroy_node(self):
        """Release the capture device on shutdown."""
        self.cap_.release()
        super().destroy_node()


def main(args=None):
    """Entry point."""
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
