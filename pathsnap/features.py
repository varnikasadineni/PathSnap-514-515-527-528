"""Hand-off H2 (Member A): Leakage-free pairwise feature engineering.

Builds the exact FEATURE_COLUMNS defined in pathsnap/config.py for every
(GPS point, candidate arc) pair. Uses ONLY .track, .nodes, .arcs (never .route).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.config import FEATURE_COLUMNS
from pathsnap.data import TripData


def _angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest unsigned angle difference in degrees within [0, 180]."""
    diff = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def build_features_for_trip(trip: TripData, cand_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich candidate DataFrame with all spatial, kinematic, and topological features."""
    df = cand_df.copy()
    track = trip.track
    arcs = trip.arcs

    # 1. Track kinematics (dt, speed, GPS movement bearing)
    tx = track["x"].to_numpy()
    ty = track["y"].to_numpy()
    tt = track["t"].to_numpy()

    dx = np.diff(tx, prepend=tx[0])
    dy = np.diff(ty, prepend=ty[0])
    dt = np.diff(tt, prepend=tt[0])
    dt_safe = np.where(dt > 0, dt, 1.0)
    step_dist = np.hypot(dx, dy)
    speed = np.where(dt > 0, step_dist / dt_safe, 0.0)
    gps_bearing = (np.rad2deg(np.arctan2(dx, dy)) + 360.0) % 360.0
    # Use forward bearing for point 0 (and stationary points where step_dist < 0.5 m)
    if len(tx) > 1:
        fwd_dx = np.diff(tx, append=tx[-1])
        fwd_dy = np.diff(ty, append=ty[-1])
        fwd_bearing = (np.rad2deg(np.arctan2(fwd_dx, fwd_dy)) + 360.0) % 360.0
        gps_bearing = np.where(step_dist < 0.5, fwd_bearing, gps_bearing)

    # 2. Road-network node out-degrees
    out_deg = np.bincount(arcs["from_node"].to_numpy(), minlength=len(trip.nodes))
    from_nodes = arcs["from_node"].to_numpy()
    to_nodes = arcs["to_node"].to_numpy()
    arc_len = arcs["length_m"].to_numpy()
    arc_bearing = arcs["bearing_deg"].to_numpy()

    pidx = df["point_idx"].to_numpy()
    aids = df["arc_id"].to_numpy()

    df["arc_len_m"] = arc_len[aids]
    df["heading_diff_deg"] = _angular_diff_deg(gps_bearing[pidx], arc_bearing[aids])
    df["gps_speed_mps"] = speed[pidx]
    df["dt_sec"] = dt[pidx]
    df["from_node_deg"] = out_deg[from_nodes[aids]].astype(np.float32)
    df["to_node_deg"] = out_deg[to_nodes[aids]].astype(np.float32)

    # 3. Within-point relative distance features
    grp = df.groupby("point_idx", sort=False)["dist_m"]
    df["dist_rank"] = grp.rank(method="min").astype(np.float32) - 1.0
    df["dist_gap_to_best_m"] = df["dist_m"] - grp.transform("min")
    df["cand_count"] = grp.transform("count").astype(np.float32)

    # 4. Temporal-topological feature: was this arc (or an incoming neighbor) a candidate at t-1?
    prev_connected = np.zeros(len(df), dtype=np.float32)
    prev_arcs: set[int] = set()
    prev_to_nodes: set[int] = set()
    for _, g in df.groupby("point_idx", sort=True):
        idxs = g.index.to_numpy()
        g_aids = g["arc_id"].to_numpy()
        if prev_arcs:
            for row_i, aid in zip(idxs, g_aids):
                if aid in prev_arcs or int(from_nodes[aid]) in prev_to_nodes:
                    prev_connected[row_i] = 1.0
        prev_arcs = set(int(a) for a in g_aids)
        prev_to_nodes = set(int(to_nodes[a]) for a in g_aids)
    df["prev_connected"] = prev_connected

    # Verify canonical feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected feature columns: {missing}")

    return df
