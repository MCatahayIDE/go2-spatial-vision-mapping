from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field

app = FastAPI(title="AICPS BIM Mapped Outlet Validation Server")

# ---------------------------------------------------------
# 1. GROUND TRUTH BIM STORE (Room Fixture Registry)
# ---------------------------------------------------------
ROOM_BIM_DB = {
    "lab_5428": {
        "room_name": "Physical-AI Lab Room 5428",
        "reference_origin": "Door Threshold (X=0, Y=0, Z=0)",
        "expected_outlets": [
            {
                "bim_id": "BIM_OUTLET__01",
                "x": 0.157,
                "y": -1.72,
                "z": 0.445,
            },
            {
                "bim_id": "BIM_OUTLET__02",
                "x": -1.32,
                "y": 0.52,
                "z": 0.445,
            },
            {
                "bim_id": "BIM_OUTLET__03",
                "x": -4.318,
                "y": 0.52,
                "z": 0.445,
            },
        ],
    }
}

# ---------------------------------------------------------
# 2. PYDANTIC SCHEMAS
# ---------------------------------------------------------
class MappedOutletInput(BaseModel):
  id: str
  x: float
  y: float
  z: float
  total_observations: int
  confirmed: bool


class SpatialMapValidationPayload(BaseModel):
  room_id: str
  coordinate_frame: str = "map"
  spatial_tolerance_m: float = Field(default=0.75, ge=0.05, le=1.0)
  outlets: List[MappedOutletInput]


class FixtureMatchResult(BaseModel):
  detected_id: str
  matched_bim_id: str
  measured_coords: Dict[str, float]
  bim_coords: Dict[str, float]
  spatial_error_m: float
  passed: bool


class SpatialValidationResponse(BaseModel):
  room_id: str
  verdict: str  # "approved", "warning", "rejected"
  timestamp: str
  total_expected_fixtures: int
  total_detected_fixtures: int
  confirmed_matches: int
  missed_fixtures: List[str]
  spurious_detections: List[str]
  mean_spatial_error_m: float
  match_details: List[FixtureMatchResult]
  message: str


# ---------------------------------------------------------
# 3. SPATIAL VALIDATION ENDPOINT
# ---------------------------------------------------------
@app.post("/validate/landmarks", response_model=SpatialValidationResponse)
def validate_spatial_landmarks(payload: SpatialMapValidationPayload):
  if payload.room_id not in ROOM_BIM_DB:
    raise HTTPException(
        status_code=404,
        detail=(
            f"Room '{payload.room_id}' not found in BIM registry."
            f" Available: {list(ROOM_BIM_DB.keys())}"
        ),
    )

  bim_room = ROOM_BIM_DB[payload.room_id]
  expected_outlets = bim_room["expected_outlets"]

  # Only validate confirmed landmarks with sufficient hits
  active_detections = [o for o in payload.outlets if o.confirmed]

  matched_bim_ids = set()
  match_details = []
  spurious_detections = []
  errors = []

  for det in active_detections:
    det_pt = np.array([det.x, det.y, det.z])
    best_match = None
    min_dist = float("inf")

    for exp in expected_outlets:
      if exp["bim_id"] in matched_bim_ids:
        continue  # Avoid duplicate assignment
      exp_pt = np.array([exp["x"], exp["y"], exp["z"]])
      dist = float(np.linalg.norm(det_pt - exp_pt))

      if dist < min_dist:
        min_dist = dist
        best_match = exp

    if best_match and min_dist <= payload.spatial_tolerance_m:
      matched_bim_ids.add(best_match["bim_id"])
      errors.append(min_dist)
      match_details.append(
          FixtureMatchResult(
              detected_id=det.id,
              matched_bim_id=best_match["bim_id"],
              measured_coords={"x": det.x, "y": det.y, "z": det.z},
              bim_coords={
                  "x": best_match["x"],
                  "y": best_match["y"],
                  "z": best_match["z"],
              },
              spatial_error_m=round(min_dist, 3),
              passed=True,
          )
      )
    else:
      spurious_detections.append(det.id)

  missed_fixtures = [
      exp["bim_id"]
      for exp in expected_outlets
      if exp["bim_id"] not in matched_bim_ids
  ]

  mean_error = round(float(np.mean(errors)), 3) if errors else 0.0

  # Global Verdict Formulation
  if len(missed_fixtures) == 0 and len(spurious_detections) == 0:
    verdict = "approved"
    msg = "All BIM fixtures successfully mapped within spatial tolerance."
  elif len(matched_bim_ids) > 0:
    verdict = "warning"
    msg = (
        f"Partial map verification: {len(matched_bim_ids)}/{len(expected_outlets)}"
        f" verified. Missed: {missed_fixtures}, Spurious:"
        f" {spurious_detections}"
    )
  else:
    verdict = "rejected"
    msg = "No mapped fixtures matched BIM specifications."

  return SpatialValidationResponse(
      room_id=payload.room_id,
      verdict=verdict,
      timestamp=datetime.utcnow().isoformat() + "Z",
      total_expected_fixtures=len(expected_outlets),
      total_detected_fixtures=len(active_detections),
      confirmed_matches=len(matched_bim_ids),
      missed_fixtures=missed_fixtures,
      spurious_detections=spurious_detections,
      mean_spatial_error_m=mean_error,
      match_details=match_details,
      message=msg,
  )