#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

latest_jpeg = None
jpeg_lock = threading.Lock()

class ROSImageSubscriber(Node):
    def __init__(self):
        super().__init__('web_viewer_subscriber')
        self.sub = self.create_subscription(
            CompressedImage,
            '/go2_camera/segmented_overlay/compressed',                         # Paramaterize the segmented_overlay topic name for dynamic streaming
            self.image_callback,
            10
        )

    def image_callback(self, msg: CompressedImage):
        global latest_jpeg
        with jpeg_lock:
            latest_jpeg = bytes(msg.data)

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        
        while True:
            with jpeg_lock:
                frame_data = latest_jpeg

            if frame_data is not None:
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(frame_data)))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    break
            time.sleep(0.033)  # ~30 FPS sync

def main():
    rclpy.init()

    # Set input topic primary as segmented overlay, with secondary fallback of raw stream if no overlay
    # topic = sys.argv[1] if len(sys.argv) > 1 else '/go2_camera/segmented_overlay/compressed'
    ros_node = ROSImageSubscriber()
    
    # Run ROS spin loop in background thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    # Start HTTP stream server on port 8080
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, StreamHandler)
    print("[+] Live Stream Server active on http://192.168.123.164:8080")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()