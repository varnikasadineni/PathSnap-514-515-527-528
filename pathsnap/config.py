"""Shared constants, paths, and schema definitions across all 4 team members."""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42
CV_N_SPLITS = 5

# Canonical feature schema (Hand-off H2: Member A -> Member B & C)
FEATURE_COLUMNS = [
    "dist_m",              # Perpendicular distance from GPS point to candidate arc (meters)
    "proj_frac",           # Projection position along arc in [0, 1]
    "arc_len_m",           # Physical length of candidate arc (meters)
    "heading_diff_deg",    # Angle difference [0, 180] between GPS movement & arc direction
    "gps_speed_mps",       # Estimated instantaneous GPS speed (m/s)
    "dt_sec",              # Time delta since previous GPS observation (s)
    "dist_rank",           # 0-based rank of candidate distance for this GPS point
    "dist_gap_to_best_m",  # dist_m - min(dist_m) among this point's candidates
    "cand_count",          # Total number of candidates found for this GPS point
    "from_node_deg",       # Out-degree of candidate arc's start node
    "to_node_deg",         # Out-degree of candidate arc's end node
    "prev_connected",      # 1 if arc (or incoming neighbor) was a candidate at t-1, else 0
]

ID_COLUMNS = ["trip_id", "point_idx", "t", "arc_id"]
LABEL_COLUMN = "label"


def load_yaml(rel_path: str | Path) -> dict:
    path = Path(rel_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
