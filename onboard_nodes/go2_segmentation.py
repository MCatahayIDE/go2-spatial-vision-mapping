#!/usr/bin/env python3
# 1. Import torch first to reserve static TLS
import torch
from ultralytics import YOLOWorld

import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import cv2
import numpy as np

class Go2OutletDetectorNode(Node):
    def __init__(self):
        super().__init__('go2_outlet_detector_node')
        self.bridge = CvBridge()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.get_logger().info(f"Loading YOLO-World on [{self.device}]...")
        self.model = YOLOWorld('yolov8s-worldv2.pt')
        self.model.to(self.device)

        ## SEMANTIC TARGETS
        # Currently targeting wall power outlets
        self.target_classes = [
            "power outlet", 
            "electrical socket", 
            "wall outlet", 
            "electrical plug",
            "light switch"
        ]
        self.model.set_classes(self.target_classes)
        self.get_logger().info(f"Active Prompt Classes: {self.target_classes}")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            Image,
            '/go2_camera/image_raw',
            self.image_callback,
            qos_profile
        )

        self.overlay_pub = self.create_publisher(Image, '/go2_camera/segmented_overlay', 10)
        self.comp_pub = self.create_publisher(CompressedImage, '/go2_camera/segmented_overlay/compressed', 10)

        self.frame_count = 0
        self.last_saved_time = 0.0
        self.get_logger().info("Outlet Detector initialized. Listening to /go2_camera/image_raw")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stamp = msg.header.stamp

            # Zero-shot inference
            results = self.model.predict(
                source=frame,
                device=self.device,
                imgsz=640,
                conf=0.20,
                iou=0.45,
                verbose=False
            )

            res = results[0]
            annotated_frame = res.plot()

            # Inspect detections and auto-save high-confidence snapshots
            if len(res.boxes) > 0:
                confs = res.boxes.conf.cpu().numpy()
                max_conf = np.max(confs)

                # Save Individual Detection per Snapshot
                # Limit snapshot creation to 1 every couple of seconds, prevent spamming
                current_time = self.get_clock().now().nanoseconds / 1e9
                if max_conf > 0.60 and (current_time - self.last_saved_time) > 5.0:
                    cv2.imwrite("detected_outlet_sample.jpg", annotated_frame)
                    self.last_saved_time = current_time
                    self.get_logger().info(f"[+] High-confidence detection ({max_conf:.2f})! Saved snapshot to 'detected_outlet_sample.jpg'")

            # 1. Publish Annotated Raw Image
            overlay_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            overlay_msg.header.stamp = stamp
            overlay_msg.header.frame_id = "go2_front_camera_optical_frame"
            self.overlay_pub.publish(overlay_msg)

            # 2. Publish Annotated Compressed Image for Web Viewer
            comp_msg = CompressedImage()
            comp_msg.header.stamp = stamp
            comp_msg.header.frame_id = "go2_front_camera_optical_frame"
            comp_msg.format = "jpeg"
            _, encimg = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            comp_msg.data = encimg.tobytes()
            self.comp_pub.publish(comp_msg)

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Go2OutletDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


