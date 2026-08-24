#!/usr/bin/env python3
# 1. Import torch first to reserve static TLS
import torch
from ultralytics import YOLOWorld

import sys
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

import cv2
import numpy as np

class Go2RealSenseSpatialLocalizer(Node):
    def __init__(self, prefix="/camera/camera"):            # Utilize macro for repeated and lengthy topic directory 
        super().__init__('go2_realsense_spatial_localizer')
        self.bridge = CvBridge()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 1. Load YOLO-World Model
        self.get_logger().info(f"Loading YOLO-World on [{self.device}]...")
        self.model = YOLOWorld('yolov8s-worldv2.pt')
        self.model.to(self.device)

        # Semantic Prompts
        self.target_classes = [
            "power outlet", 
            "electrical socket", 
            "wall outlet", 
            "electrical plug"
        ]
        self.model.set_classes(self.target_classes)

        # Intrinsic Fallbacks (Updated dynamically from CameraInfo)
        self.fx = 910.0
        self.fy = 910.0
        self.cx = 640.0
        self.cy = 360.0

        # Odometry State Storage
        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.robot_yaw = 0.0

        # Topic paths
        info_topic = f"{prefix}/color/camera_info"
        rgb_topic = f"{prefix}/color/image_raw"
        depth_topic = f"{prefix}/aligned_depth_to_color/image_raw"

        self.get_logger().info(f"Subscribing to RGB-D: {rgb_topic} & {depth_topic}")

        # 2. Camera Info Subscriber
        self.create_subscription(CameraInfo, info_topic, self.camera_info_callback, 10)

        # 3. Robot Odometry Subscriber (LIO-SAM)
        self.create_subscription(Odometry, '/lio_sam_ros2/mapping/odometry', self.odom_callback, 10)

        # 4. Synchronized RGB + Aligned Depth Subscribers
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.08)
        self.ts.registerCallback(self.rgbd_callback)

        # 5. Output Publishers
        self.outlet_point_pub = self.create_publisher(PointStamped, '/go2_vision/detected_outlet_position', 10)
        self.overlay_comp_pub = self.create_publisher(CompressedImage, '/go2_camera/segmented_overlay/compressed', 10)

        self.last_saved_time = 0.0
        self.get_logger().info("Spatial Outlet Localizer initialized and ready.")

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.cx = msg.k[2]
        self.fy = msg.k[4]
        self.cy = msg.k[5]

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        self.robot_pos = np.array([pos.x, pos.y, pos.z])
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

    def rgbd_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            # 16-bit depth in millimeters
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            results = self.model.predict(
                source=color_frame,
                device=self.device,
                imgsz=640,
                conf=0.20,
                iou=0.45,
                verbose=False
            )

            res = results[0]
            annotated_frame = color_frame.copy()

            if len(res.boxes) > 0:
                for box, cls_id, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    u_c = int((x1 + x2) / 2)
                    v_c = int((y1 + y2) / 2)

                    # Extract central ROI depth
                    box_w = max(1, int((x2 - x1) * 0.25))
                    box_h = max(1, int((y2 - y1) * 0.25))
                    depth_roi = depth_frame[max(0, v_c - box_h):min(depth_frame.shape[0], v_c + box_h),
                                            max(0, u_c - box_w):min(depth_frame.shape[1], u_c + box_w)]

                    valid_depths = depth_roi[depth_roi > 0]

                    if len(valid_depths) > 0:
                        # Metric depth in meters
                        z_c = float(np.median(valid_depths)) / 1000.0

                        # 1. Pinhole Ray Back-Projection (Camera Frame)
                        x_c = ((u_c - self.cx) * z_c) / self.fx
                        y_c = ((v_c - self.cy) * z_c) / self.fy

                        # 2. World Frame Coordinates
                        x_world = self.robot_pos[0] + (z_c * np.cos(self.robot_yaw) - x_c * np.sin(self.robot_yaw))
                        y_world = self.robot_pos[1] + (z_c * np.sin(self.robot_yaw) + x_c * np.cos(self.robot_yaw))
                        z_world = self.robot_pos[2] - y_c

                        # 3. Publish 3D Point Landmark
                        point_msg = PointStamped()
                        point_msg.header.stamp = rgb_msg.header.stamp
                        point_msg.header.frame_id = "map"
                        point_msg.point.x = x_world
                        point_msg.point.y = y_world
                        point_msg.point.z = z_world
                        self.outlet_point_pub.publish(point_msg)

                        # Render Bounding Box and 3D Metric Coordinates on Stream
                        cls_name = self.target_classes[int(cls_id)]
                        label_cam = f"{cls_name} {conf:.2f} | Dist: {z_c:.2f}m"
                        label_coord = f"Cam: [{x_c:+.2f}, {y_c:+.2f}, {z_c:.2f}]m"

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(annotated_frame, (u_c, v_c), 5, (0, 0, 255), -1)
                        cv2.putText(annotated_frame, label_cam, (x1, max(20, y1 - 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label_coord, (x1, max(40, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # Save snapshot on high confidence
                        current_time = self.get_clock().now().nanoseconds / 1e9
                        if conf > 0.60 and (current_time - self.last_saved_time) > 4.0:
                            cv2.imwrite("realsense_spatial_outlet_sample.jpg", annotated_frame)
                            self.last_saved_time = current_time
                            self.get_logger().info(f"[+] Outlet at Dist={z_c:.2f}m | Cam=({x_c:+.2f}, {y_c:+.2f}, {z_c:+.2f})m | World=({x_world:.2f}, {y_world:.2f})")

            # Publish Compressed Stream for Browser Viewer
            comp_msg = CompressedImage()
            comp_msg.header.stamp = rgb_msg.header.stamp
            comp_msg.header.frame_id = "camera_color_optical_frame"
            comp_msg.format = "jpeg"
            _, encimg = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            comp_msg.data = encimg.tobytes()
            self.overlay_comp_pub.publish(comp_msg)

        except Exception as e:
            self.get_logger().error(f"RGB-D processing error: {e}")

def main(args=None):
    rclpy.init(args=args)
    # Allows passing topic prefix via command line (default: /camera/camera)
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/camera/camera"
    node = Go2RealSenseSpatialLocalizer(prefix=prefix)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
