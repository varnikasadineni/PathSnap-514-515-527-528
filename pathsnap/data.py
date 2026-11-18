"""Hand-off H1 (Member A -> B, C, D): Trip data loader & per-trip metric projection.

IMPORTANT:
- Trips span the entire globe (Lon [-100.4, 137.2], Lat [-34.6, 61.7]).
- Do NOT use a fixed regional CRS like EPSG:27700 (British National Grid), which
  produces severe distortion or NaNs outside Great Britain.
- Instead, we project each trip into a trip-centered Local Tangent Plane (ENU)
  in meters around the trip's GPS centroid (lon0, lat0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371008.8


@dataclass
class TripData:
    trip_id: str
    trip_dir: Path
    track: pd.DataFrame       # columns: point_idx, lon, lat, t, x, y
    nodes: pd.DataFrame       # columns: node_id, lon, lat, x, y
    arcs: pd.DataFrame        # columns: arc_id, from_node, to_node, x1, y1, x2, y2, length_m, bearing_deg
    route: Optional[np.ndarray]  # 1D int array of ground-truth arc_ids, or None
    origin_lon: float
    origin_lat: float


def project_lonlat_to_local_m(
    lon: np.ndarray,
    lat: np.ndarray,
    origin_lon: float,
    origin_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 (lon, lat) degrees to local tangent-plane (x, y) meters."""
    lat0_rad = np.deg2rad(origin_lat)
    dlon_rad = np.deg2rad(lon - origin_lon)
    dlat_rad = np.deg2rad(lat - origin_lat)
    x = EARTH_RADIUS_M * np.cos(lat0_rad) * dlon_rad
    y = EARTH_RADIUS_M * dlat_rad
    return x, y


def load_trip(trip_dir: str | Path, load_route: bool = True) -> TripData:
    """Load a single trip directory (.track, .nodes, .arcs, and optionally .route).

    Parameters
    ----------
    trip_dir : str | Path
        Path to a trip folder (e.g. data/raw/train/00000001).
    load_route : bool, default=True
        Set to False inside predict_trip() so ground-truth .route is NEVER read
        during inference (strict leakage prevention).
    """
    tdir = Path(trip_dir)
    trip_id = tdir.name

    track_path = tdir / f"{trip_id}.track"
    nodes_path = tdir / f"{trip_id}.nodes"
    arcs_path = tdir / f"{trip_id}.arcs"
    route_path = tdir / f"{trip_id}.route"

    # 1. Read raw whitespace-separated headerless files
    track = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        dtype={"lon": np.float64, "lat": np.float64, "t": np.float64},
    )
    track.insert(0, "point_idx", np.arange(len(track), dtype=np.int32))

    nodes = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64},
    )
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int32))

    arcs = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": np.int32, "to_node": np.int32},
    )
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int32))

    # 2. Per-trip metric projection centered on GPS trajectory centroid
    origin_lon = float(track["lon"].mean())
    origin_lat = float(track["lat"].mean())

    track["x"], track["y"] = project_lonlat_to_local_m(
        track["lon"].to_numpy(), track["lat"].to_numpy(), origin_lon, origin_lat
    )
    nodes["x"], nodes["y"] = project_lonlat_to_local_m(
        nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), origin_lon, origin_lat
    )

    # 3. Precompute arc geometry (endpoints in meters, length, bearing)
    node_x = nodes["x"].to_numpy()
    node_y = nodes["y"].to_numpy()
    u = arcs["from_node"].to_numpy()
    v = arcs["to_node"].to_numpy()

    x1, y1 = node_x[u], node_y[u]
    x2, y2 = node_x[v], node_y[v]
    dx = x2 - x1
    dy = y2 - y1

    arcs["x1"] = x1
    arcs["y1"] = y1
    arcs["x2"] = x2
    arcs["y2"] = y2
    arcs["length_m"] = np.hypot(dx, dy)
    # Bearing in degrees [0, 360) clockwise from North (y-axis)
    arcs["bearing_deg"] = (np.rad2deg(np.arctan2(dx, dy)) + 360.0) % 360.0

    # 4. Optionally load reference .route (NEVER loaded during inference)
    route: Optional[np.ndarray] = None
    if load_route and route_path.exists():
        route_df = pd.read_csv(route_path, header=None, names=["arc_id"], dtype=np.int32)
        route = route_df["arc_id"].to_numpy(dtype=np.int32)

    return TripData(
        trip_id=trip_id,
        trip_dir=tdir,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
    )


def list_trip_dirs(partition_dir: str | Path) -> list[Path]:
    """Return sorted list of trip subdirectories inside a partition folder."""
    pdir = Path(partition_dir)
    if not pdir.exists():
        return []
    return sorted([d for d in pdir.iterdir() if d.is_dir()])
