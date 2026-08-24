#!/usr/bin/env python3
import json
import os
import time
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class Landmark:

  def __init__(self, landmark_id: str, point: np.ndarray, timestamp: float):
    self.id = landmark_id
    self.centroid = np.array(point, dtype=np.float64)
    self.count = 1
    self.first_seen = timestamp
    self.last_seen = timestamp
    self.confirmed = False

  def update(self, point: np.ndarray, timestamp: float):
    # Running average centroid update
    self.centroid = (self.centroid * self.count + point) / (self.count + 1)
    self.count += 1
    self.last_seen = timestamp

  def to_dict(self):
    return {
        "id": self.id,
        "class": "power_outlet",
        "position_meters": {
            "x": round(float(self.centroid[0]), 3),
            "y": round(float(self.centroid[1]), 3),
            "z": round(float(self.centroid[2]), 3),
        },
        "total_observations": self.count,
        "confirmed": self.confirmed,
        "first_detected_timestamp": round(self.first_seen, 2),
        "last_detected_timestamp": round(self.last_seen, 2),
    }


class OutletDetectionPlotter(Node):

  def __init__(self):
    super().__init__("go2_outlet_landmark_manager")

    # Clustering Parameters
    self.cluster_threshold_m = 0.25  # 25 cm neighborhood radius
    self.min_observations_to_confirm = 100  # 15 detections to verify landmark
    self.output_filepath = "outlet_maps/lab_outlets_map.json"

    self.landmarks = []
    self.next_id_index = 1

    # 1. Subscriber to Real-Time 3D Outlet Detections
    self.sub = self.create_subscription(
        PointStamped,
        "/go2_vision/detected_outlet_position",
        self.outlet_point_callback,
        10,
    )

    # 2. RViz Visualizer Marker Publisher
    self.marker_pub = self.create_publisher(
        MarkerArray, "/go2_vision/outlet_markers", 10
    )

    # 3. Periodic Map Exporter & Marker Broadcast (1 Hz)
    self.timer = self.create_timer(1.0, self.periodic_update)

    self.get_logger().info(
        "Go2 Landmark Manager online. Listening to"
        " /go2_vision/detected_outlet_position"
    )

  def outlet_point_callback(self, msg: PointStamped):
    point = np.array(
        [msg.point.x, msg.point.y, msg.point.z], dtype=np.float64
    )
    current_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

    matched_landmark = None
    min_dist = float("inf")

    # Find nearest existing landmark cluster
    for lm in self.landmarks:
      dist = np.linalg.norm(lm.centroid - point)
      if dist < min_dist:
        min_dist = dist
        matched_landmark = lm

    # Association gate
    if matched_landmark is not None and min_dist < self.cluster_threshold_m:
      matched_landmark.update(point, current_time)
      if (
          not matched_landmark.confirmed
          and matched_landmark.count >= self.min_observations_to_confirm
      ):
        matched_landmark.confirmed = True
        self.get_logger().info(
            f"*[CONFIRMED LANDMARK] {matched_landmark.id} anchored at"
            f" X={matched_landmark.centroid[0]:.2f}m,"
            f" Y={matched_landmark.centroid[1]:.2f}m,"
            f" Z={matched_landmark.centroid[2]:.2f}m"
        )
    else:
      # Register new candidate cluster
      new_id = f"outlet_{self.next_id_index:02d}"
      self.next_id_index += 1
      new_lm = Landmark(new_id, point, current_time)
      self.landmarks.append(new_lm)
      self.get_logger().info(
          f"[*] New candidate landmark spotted: {new_id} at"
          f" ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
      )

  def periodic_update(self):
    self.save_map_to_json()
    self.publish_rviz_markers()

  def save_map_to_json(self):
    confirmed_lms = [lm for lm in self.landmarks if lm.confirmed]
    data = {
        "session_timestamp": time.time(),
        "coordinate_frame": "map",
        "total_confirmed_outlets": len(confirmed_lms),
        "outlets": [lm.to_dict() for lm in confirmed_lms],
        "candidates": [
            lm.to_dict() for lm in self.landmarks if not lm.confirmed
        ],
    }
    with open(self.output_filepath, "w") as f:
      json.dump(data, f, indent=2)

  def publish_rviz_markers(self):
    marker_array = MarkerArray()
    stamp = self.get_clock().now().to_msg()

    for idx, lm in enumerate(self.landmarks):
      # 3D Sphere Marker
      sphere = Marker()
      sphere.header.frame_id = "map"
      sphere.header.stamp = stamp
      sphere.ns = "outlets_geometry"
      sphere.id = idx * 2
      sphere.type = Marker.SPHERE
      sphere.action = Marker.ADD
      sphere.pose.position.x = lm.centroid[0]
      sphere.pose.position.y = lm.centroid[1]
      sphere.pose.position.z = lm.centroid[2]
      sphere.pose.orientation.w = 1.0
      sphere.scale.x = 0.15
      sphere.scale.y = 0.15
      sphere.scale.z = 0.15

      # Confirmed = Bright Green, Candidate = Yellow
      if lm.confirmed:
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (
            0.0,
            1.0,
            0.0,
            0.9,
        )
      else:
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (
            1.0,
            1.0,
            0.0,
            0.5,
        )

      # 3D Text Label Marker
      text = Marker()
      text.header.frame_id = "map"
      text.header.stamp = stamp
      text.ns = "outlets_labels"
      text.id = (idx * 2) + 1
      text.type = Marker.TEXT_VIEW_FACING
      text.action = Marker.ADD
      text.pose.position.x = lm.centroid[0]
      text.pose.position.y = lm.centroid[1]
      text.pose.position.z = lm.centroid[2] + 0.20
      text.pose.orientation.w = 1.0
      text.scale.z = 0.12
      text.text = f"{lm.id} ({lm.count} hits)"
      text.color.r, text.color.g, text.color.b, text.color.a = (
          1.0,
          1.0,
          1.0,
          1.0,
      )

      marker_array.markers.append(sphere)
      marker_array.markers.append(text)

    self.marker_pub.publish(marker_array)


def main(args=None):
  rclpy.init(args=args)
  node = OutletDetectionPlotter()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.save_map_to_json()
    node.get_logger().info(
        f"Saved final room landmark map to {node.output_filepath}"
    )
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()