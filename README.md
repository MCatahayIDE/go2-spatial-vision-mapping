# Unitree Go2 Real-Time Pipeline for Semantic Segmenation and Spatial Landmark Mapping 


## Overview
An end-to-end spatial perception, open-vocabulary semantic detection, and 3D landmark mapping pipeline deployed on the **Unitree Go2 quadruped** equipped with an **Intel RealSense D435i** depth camera.

This system detects target environmental fixtures (e.g., electrical outlets, switches, infrastructure markers) in real time, back-projects 2D image coordinates into 3D metric camera space using hardware-aligned depth and camera intrinsics, transforms detections into a globally consistent coordinate frame via onboard LiDAR-inertial odometry (`utlidar`), and performs online spatial clustering to generate a persistent 3D landmark map.

## Core Features & Implementation
- **Open-Vocabulary Semantic Detection:** Leverages **YOLO-World** for zero-shot text-prompted target detection, bypassing fixed dataset label limitations.
- **Hardware-Synchronized RGB-D Back-Projection:** Ingests aligned 1080p RGB and 16-bit depth streams (`ApproximateTimeSynchronizer`), applying central-ROI median filtering to suppress specular depth dropouts and computing metric $(X_c, Y_c, Z_c)$ optical vectors.
- **Dynamic Coordinate Frame Transformation:** Fuses camera-relative 3D vectors with high-frequency ($\sim150\text{ Hz}$) onboard LiDAR-inertial odometry (`/utlidar/robot_odom`), anchoring detections to a persistent global map frame.
- **Online Recursive Spatial Clustering:** Employs Euclidean neighborhood gating ($d \le 0.25\text{ m}$) and online running centroid averaging ($\mathcal{O}(1)$ memory) to merge multi-frame observations and reject transient false positives.
- **Live Visual Telemetry & Persistent Export:** Streams real-time annotated video with dynamic odometry heads-up display (HUD) over HTTP MJPEG and serializes verified fixtures to a structured `lab_outlets_map.json` database.

## Pipeline System Architecture 
```
INTEL REALSENSE D435i
               ┌─────────────────────────────┐
               │  RGB8 Stream (1280x720)     │
               │  Aligned Depth (Z16)        │
               │  Camera Intrinsics Matrix K │
               └──────────────┬──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Node 1: go2_rs_spatial_localizer.py                             │
│  ├─ Time Synchronization (ApproximateTimeSynchronizer <= 80ms)  │
│  ├─ Zero-Shot Detection (YOLO-World on cuda:0)                  │
│  ├─ Central-ROI Median Depth Extraction (Z_c)                   │
│  ├─ Pinhole Ray Back-Projection -> (X_c, Y_c, Z_c)              │
│  └─ Global Map Frame Transformation via /utlidar/robot_odom     │
└─────────────────┬─────────────────────────────┬─────────────────┘
│                             │
Topic: /go2_camera/segmented_overlay/compressed│ Topic: /go2_vision/detected_outlet_position
│                             │
▼                             ▼
┌──────────────────────────────┐  ┌───────────────────────────────┐
│ Node 2: vid_stream_server.py │  │ Node 3: go2_landmark_mgr.py   │
│  └─ HTTP MJPEG Stream        │  │  ├─ Euclidean Gating (<25cm)  │
│     (Port 8080 Browser HUD)  │  │  ├─ Recursive Mean Centroid   │
└──────────────────────────────┘  │  ├─ Confirmation Gate (N>=100)│
                                  │  ├─ RViz 3D Marker Broadcast  │
                                  │  └─ Auto-Dump to JSON Map     │
                                  └───────────────────────────────┘
```

## ROS2 Node and Topic Flow
```
UNITREE GO2 ONBOARD JETSON
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                                        │
 │  [ Onboard Front Camera ]                                                                              │
 │             │                                                                                          │
 │     (VideoHub Service)                                                                                 │
 │             │                                                                                          │
 │  [ Internal Bus: ethrobot ]                                                                            │
 │             │                                                                                          │
 │             ▼                                                                                          │
 │  ┌──────────────────────────────────────────────────┐                                                  │
 │  │ Node 1: go2_video_client_bridge.py               │                                                  │
 │  │ ├─ Worker Process: VideoClient.GetImageSample()  │                                                  │
 │  │ │    └─ Pulls 1080p JPEG @ 20-25 FPS via RPC     │                                                  │
 │  │ ├─ Shared Memory: multiprocessing.Queue(maxsize=2)│                                                  │
 │  │ └─ Main Process: ROS 2 Publisher Node            │                                                  │
 │  └──────────────┬───────────────────────────────────┘                                                  │
 │                 │                                                                                      │
 │                 ├─────────────────────────────────────────┐                                            │
 │                 ▼                                         ▼                                            │
 │   Topic: /go2_camera/image_raw          Topic: /go2_camera/image_raw/compressed                        │
 │   Type: sensor_msgs/msg/Image           Type: sensor_msgs/msg/CompressedImage                          │
 │   Payload: 1920x1080 BGR8               Payload: Direct JPEG Passthrough                               │
 │                 │                                                                                      │
 │                 ▼                                                                                      │
 │  ┌──────────────────────────────────────────────────┐                                                  │
 │  │ Node 2: go2_outlet_segmentation.py               │                                                  │
 │  │ ├─ YOLO-World (yolov8s-worldv2.pt) on cuda:0     │                                                  │
 │  │ ├─ Semantic Prompts: "wall outlet", "socket",... │                                                  │
 │  │ ├─ Inference Rate: ~18-20 FPS                    │                                                  │
 │  │ ├─ Snapshot Hook: Saves detected_outlet_sample   │                                                  │
 │  │ └─ Renderer: Generates annotated BGR8 frame      │                                                  │
 │  └──────────────┬───────────────────────────────────┘                                                  │
 │                 │                                                                                      │
 │                 ├─────────────────────────────────────────┐                                            │
 │                 ▼                                         ▼                                            │
 │   Topic: /go2_camera/segmented_overlay  Topic: /go2_camera/segmented_overlay/compressed                │
 │   Type: sensor_msgs/msg/Image           Type: sensor_msgs/msg/CompressedImage                          │
 │   Payload: Annotated BGR8 Frame         Payload: Annotated JPEG Frame                                  │
 │                                                           │                                            │
 │                                                           ▼                                            │
 │                                         ┌───────────────────────────────────┐                          │
 │                                         │ Node 3: vid_stream_server.py      │                          │
 │                                         │ ├─ ROS 2 Subscriber Thread        │                          │
 │                                         │ └─ HTTP Streamer (Port 8080)      │                          │
 │                                         └─────────────────┬─────────────────┘                          │
 │                                                           │                                            │
 └───────────────────────────────────────────────────────────┼────────────────────────────────────────────┘
                                                             │ Ethernet Subnet (xxx.xxx.xxx)
                                                             ▼
                                                ┌─────────────────────────┐
                                                │ Windows Host Browser    │
                                                │ Port:                   │
                                                │        8080             │
                                                └─────────────────────────┘
```


## Core Topics and Message Types
### Nodes:
- `go2_video_client_bridge`
- `go2_outlet_detector_node`
- `web_viewer_subscriber`

### Topics:
- `/go2_camera/image_raw`
- `/go2_camera/image_raw/compressed`
- `/go2_camera/segmented_overlay`
- `/go2_camera/segmented_overlay/compressed`


## Embedded Systems, Hardware
- **Robot Platform:** Unitree Go2 Quadruped (Perception Unit: NVIDIA Jetson Orin / Xavier NX, Ubuntu 20.04 / 22.04)
- **Perception Sensor:** Intel RealSense D435i Depth Camera (USB 3.2)
- **Middleware:** ROS 2 (Foxy) & Eclipse CycloneDDS (`rmw_cyclonedds_cpp`)
- **Computer Vision & ML:** PyTorch (CUDA-accelerated), Ultralytics YOLO-World, OpenCV, NumPy

## Installation and Configuration
- WIP

## Sample JSON Output Entry
```
{
  "coordinate_frame": "map",
  "total_confirmed_outlets": 2,
  "outlets": [
    {
      "id": "outlet_01",
      "class": "power_outlet",
      "position_meters": {
        "x": -2.417,
        "y": 2.492,
        "z": 0.285
      },
      "total_observations": 177,
      "confirmed": true,
      "first_detected_timestamp": 1787357983.37,
      "last_detected_timestamp": 1787357995.31
    }
  ]
}
```
