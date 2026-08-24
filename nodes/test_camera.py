#!/usr/bin/env python3
import sys
import time
import cv2
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

# Pass network interface (default to eth0 or sys.argv[1])
nic = sys.argv[1] if len(sys.argv) > 1 else "eth0"
print(f"[+] Initializing ChannelFactory on interface: {nic}")

ChannelFactoryInitialize(0, nic)

client = VideoClient()
client.SetTimeout(3.0)
client.Init()

print("[+] VideoClient connected. Capturing 30 test frames...")
count = 0

for i in range(30):
    code, data = client.GetImageSample()
    if code == 0 and data is not None and len(data) > 0:
        np_arr = np.frombuffer(bytes(data), dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            count += 1
            if count % 10 == 0:
                print(f"[+] Captured Frame #{count}: {img.shape[1]}x{img.shape[0]}")
                cv2.imwrite("go2_test_frame.jpg", img)
    time.sleep(0.05)

print(f"[+] Done! Successfully captured {count}/30 frames. Check 'go2_test_frame.jpg'")