#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from unitree_go.msg import Go2FrontVideoData
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import av
import cv2
import numpy as np

class Go2CameraBridge(Node):
    def __init__(self):
        super().__init__('go2_cam_bridge')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Go2FrontVideoData,
            '/frontvideostream',
            self.video_callback,
            qos_profile
        )

        self.raw_pub = self.create_publisher(Image, '/go2_camera/image_raw', 10)
        self.compressed_pub = self.create_publisher(CompressedImage, '/go2_camera/image_raw/compressed', 10)

        # PyAV H.264 Codec Context
        self.codec = av.CodecContext.create('h264', 'r')
        self.bridge = CvBridge()
        self.frame_count = 0
        self.get_logger().info("Go2 Stream Bridge running. Waiting for keyframe sync...")

    def video_callback(self, msg: Go2FrontVideoData):
        payload = bytes(msg.video720p) if len(msg.video720p) > 0 else bytes(msg.video360p)
        if not payload:
            return

        try:
            # Parse slice chunks into NAL units
            packets = self.codec.parse(payload)
            for packet in packets:
                try:
                    frames = self.codec.decode(packet)
                    for frame in frames:
                        img_bgr = frame.to_ndarray(format='bgr24')
                        stamp = self.get_clock().now().to_msg()
                        self.frame_count += 1

                        # Publish Raw Image
                        raw_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
                        raw_msg.header.stamp = stamp
                        raw_msg.header.frame_id = "go2_front_camera_optical_frame"
                        self.raw_pub.publish(raw_msg)

                        # Publish Compressed Image
                        comp_msg = CompressedImage()
                        comp_msg.header.stamp = stamp
                        comp_msg.header.frame_id = "go2_front_camera_optical_frame"
                        comp_msg.format = "jpeg"
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
                        comp_msg.data = encimg.tobytes()
                        self.compressed_pub.publish(comp_msg)

                        if self.frame_count % 30 == 0:
                            self.get_logger().info(f"Stream synced: Frame #{self.frame_count} published")

                except av.AVError:
                    # Ignore corrupted/intermediate P-frames while waiting for next keyframe
                    pass

        except av.AVError:
            # Catch packet-level parsing errors during NAL assembly
            pass
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Go2CameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()