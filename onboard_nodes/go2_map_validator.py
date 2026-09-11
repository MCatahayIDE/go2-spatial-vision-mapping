# Client-side script for landmark extraction from JSON, transmission to mapping validation server
import json
import requests

## DEGUB: Hardcoded Payload Destination
# Switch URL as needed depending if running onboard, or externally via ethernet connection
SERVER_URL = "http://127.0.0.1:8000/validate/landmarks"  # Adjust host IP


def validate_local_map():
  with open("outlet_maps/lab_outlets_map.json", "r") as f:
    map_data = json.load(f)

  # Payload configuration  v v v 
  payload = {
      "room_id": "lab_5428",
      "spatial_tolerance_m": 0.75,
      "outlets": [
          {
              "id": item["id"],
              "x": item["position_meters"]["x"],
              "y": item["position_meters"]["y"],
              "z": item["position_meters"]["z"],
              "total_observations": item["total_observations"],
              "confirmed": item["confirmed"],
          }
          for item in map_data["outlets"]
      ],
  }

  response = requests.post(SERVER_URL, json=payload)
  print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
  validate_local_map()