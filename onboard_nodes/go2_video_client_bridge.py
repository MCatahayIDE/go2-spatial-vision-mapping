#!/usr/bin/env python3
import sys
import time
import multiprocessing as mp

import cv2
import numpy as np

# -------------------------------------------------------------
# Process A: Dedicated SDK Video Fetcher (No ROS 2 in this process)
# -------------------------------------------------------------
def video_capture_worker(nic: str, frame_queue: mp.Queue, stop_event: mp.Event):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    try:
        ChannelFactoryInitialize(0, nic)
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()
    except Exception as e:
        print(f"[Worker Error] Failed to initialize ChannelFactory on {nic}: {e}")
        return

    while not stop_event.is_set():
        code, data = client.GetImageSample()
        if code == 0 and data is not None and len(data) > 0:
            raw_bytes = bytes(data)
            # Keep queue depth minimal (drop stale frames to guarantee zero latency)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
            frame_queue.put(raw_bytes)
        time.sleep(0.04)  # ~25 FPS polling loop

# -------------------------------------------------------------
# Process B: Dedicated ROS 2 Publisher Node (No SDK Channel here)
# -------------------------------------------------------------
def run_ros2_publisher(frame_queue: mp.Queue):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CompressedImage
    from cv_bridge import CvBridge

    class VideoPublisherNode(Node):
        def __init__(self):
            super().__init__('go2_video_client_bridge')
            self.bridge = CvBridge()
            self.raw_pub = self.create_publisher(Image, '/go2_camera/image_raw', 10)
            self.comp_pub = self.create_publisher(CompressedImage, '/go2_camera/image_raw/compressed', 10)
            self.frame_count = 0

            # 30 Hz timer to drain incoming frames from the queue
            self.timer = self.create_timer(0.033, self.timer_callback)
            self.get_logger().info("Go2 Camera Bridge Active! Streaming on /go2_camera/image_raw")

        def timer_callback(self):
            if frame_queue.empty():
                return

            try:
                raw_bytes = frame_queue.get_nowait()
                np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    return

                stamp = self.get_clock().now().to_msg()
                self.frame_count += 1

                # 1. Publish standard Image
                raw_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
                raw_msg.header.stamp = stamp
                raw_msg.header.frame_id = "go2_front_camera_optical_frame"
                self.raw_pub.publish(raw_msg)

                # 2. Publish CompressedImage (Zero-copy JPEG passthrough)
                comp_msg = CompressedImage()
                comp_msg.header.stamp = stamp
                comp_msg.header.frame_id = "go2_front_camera_optical_frame"
                comp_msg.format = "jpeg"
                comp_msg.data = raw_bytes
                self.comp_pub.publish(comp_msg)

                if self.frame_count % 60 == 0:
                    self.get_logger().info(f"Published Frame #{self.frame_count} ({img_bgr.shape[1]}x{img_bgr.shape[0]})")

            except Exception as e:
                self.get_logger().error(f"Publish error: {e}")

    rclpy.init()
    node = VideoPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

# -------------------------------------------------------------
# Main Process Entry Point
# -------------------------------------------------------------
def main():
    nic = sys.argv[1] if len(sys.argv) > 1 else "ethrobot"
    print(f"Starting Go2 Camera Bridge on interface: {nic}")

    # Use 'spawn' to guarantee isolated memory spaces and C-handles
    mp.set_start_method('spawn', force=True)

    frame_queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()

    # Launch isolated SDK capture worker
    worker = mp.Process(target=video_capture_worker, args=(nic, frame_queue, stop_event))
    worker.daemon = True
    worker.start()

    try:
        # Run ROS 2 Publisher in the main process
        run_ros2_publisher(frame_queue)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        worker.terminate()
        worker.join()

if __name__ == '__main__':
    main()