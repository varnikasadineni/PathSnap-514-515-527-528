"""Hand-off H2 (Member A): Spatial-index candidate generation & monotonic route alignment.

Member A implements:
1. Fast spatial candidate lookup (cKDTree / STRtree) with adaptive fallback radius
   so high-noise trips (e.g., 120-140 m drift) don't suffer 100% miss rates.
2. Monotonic alignment of GPS points onto .route (train / student_test / server-eval ONLY).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pathsnap.data import TripData


def point_to_segments_projection(
    px: float,
    py: float,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized perpendicular distance (m) and clamped projection fraction [0, 1]."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(seg_len_sq > 1e-12, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0)
    t_clamped = np.clip(t, 0.0, 1.0)
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t_clamped


def generate_candidates_for_trip(
    trip: TripData,
    base_radius_m: float = 50.0,
    fallback_radii_m: tuple[float, ...] = (100.0, 200.0, 350.0),
    min_candidates: int = 5,
    max_candidates: int = 15,
) -> pd.DataFrame:
    """Generate nearby candidate road arcs for every GPS observation in `trip`.

    Uses a cKDTree on arc midpoints + endpoints (fast even on 307k-arc trips).
    Returns DataFrame with columns:
        ['trip_id', 'point_idx', 't', 'arc_id', 'dist_m', 'proj_frac']
    """
    arcs = trip.arcs
    x1 = arcs["x1"].to_numpy()
    y1 = arcs["y1"].to_numpy()
    x2 = arcs["x2"].to_numpy()
    y2 = arcs["y2"].to_numpy()
    arc_ids_all = arcs["arc_id"].to_numpy()

    # Index start, mid, and end points of every arc so long arcs aren't missed
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)
    index_pts = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([mx, my]),
        np.column_stack([x2, y2]),
    ])
    index_arc_ids = np.concatenate([arc_ids_all, arc_ids_all, arc_ids_all])
    tree = cKDTree(index_pts)

    # Extra search buffer to account for long road segments whose midpoint is slightly further
    buffer_m = 80.0
    radii = (base_radius_m,) + tuple(fallback_radii_m)

    tx = trip.track["x"].to_numpy()
    ty = trip.track["y"].to_numpy()
    tt = trip.track["t"].to_numpy()
    pidx = trip.track["point_idx"].to_numpy()

    rows = []
    for i in range(len(tx)):
        px, py = float(tx[i]), float(ty[i])
        chosen_arcs = np.empty(0, dtype=np.int32)
        chosen_dists = np.empty(0, dtype=np.float64)
        chosen_projs = np.empty(0, dtype=np.float64)

        for r in radii:
            hit_idx = tree.query_ball_point([px, py], r=r + buffer_m)
            if not hit_idx:
                continue
            cand_aids = np.unique(index_arc_ids[hit_idx])
            dists, projs = point_to_segments_projection(
                px, py, x1[cand_aids], y1[cand_aids], x2[cand_aids], y2[cand_aids]
            )
            within = dists <= r
            if np.count_nonzero(within) >= min_candidates:
                cand_aids = cand_aids[within]
                dists = dists[within]
                projs = projs[within]
                order = np.argsort(dists)[:max_candidates]
                chosen_arcs = cand_aids[order]
                chosen_dists = dists[order]
                chosen_projs = projs[order]
                break

        # Final safety net: nearest K arcs regardless of radius
        if len(chosen_arcs) == 0:
            k_query = min(max_candidates * 3, len(index_pts))
            _, nn_idx = tree.query([px, py], k=k_query)
            nn_idx = np.atleast_1d(nn_idx)
            cand_aids = np.unique(index_arc_ids[nn_idx])
            dists, projs = point_to_segments_projection(
                px, py, x1[cand_aids], y1[cand_aids], x2[cand_aids], y2[cand_aids]
            )
            order = np.argsort(dists)[:max_candidates]
            chosen_arcs = cand_aids[order]
            chosen_dists = dists[order]
            chosen_projs = projs[order]

        for a, d, pf in zip(chosen_arcs, chosen_dists, chosen_projs):
            rows.append((trip.trip_id, int(pidx[i]), float(tt[i]), int(a), float(d), float(pf)))

    return pd.DataFrame(
        rows,
        columns=["trip_id", "point_idx", "t", "arc_id", "dist_m", "proj_frac"],
    )


def align_track_to_route_monotonic(trip: TripData, window_ahead: int = 25) -> pd.DataFrame:
    """Assign each GPS point to its reference arc along `trip.route` monotonically.

    NEVER call this inside predict_trip() — only for training labels, student_test
    evaluation, and server-side scoring.

    Returns DataFrame with columns: ['trip_id', 'point_idx', 'true_arc', 'route_pos', 'dist_to_route_m']
    """
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no .route loaded.")

    route = trip.route
    n_route = len(route)
    x1 = trip.arcs["x1"].to_numpy()[route]
    y1 = trip.arcs["y1"].to_numpy()[route]
    x2 = trip.arcs["x2"].to_numpy()[route]
    y2 = trip.arcs["y2"].to_numpy()[route]

    tx = trip.track["x"].to_numpy()
    ty = trip.track["y"].to_numpy()
    pidx = trip.track["point_idx"].to_numpy()

    # Member A can refine this into full monotonic Viterbi/DP alignment if needed;
    # windowed forward-only projection prevents loops & reverse twins from stealing points.
    cur_pos = 0
    rows = []
    for i in range(len(tx)):
        lo = cur_pos
        hi = min(n_route, cur_pos + window_ahead)
        dists, _ = point_to_segments_projection(
            float(tx[i]), float(ty[i]), x1[lo:hi], y1[lo:hi], x2[lo:hi], y2[lo:hi]
        )
        best_rel = int(np.argmin(dists))
        best_dist = float(dists[best_rel])

        # If vehicle jumped ahead across a large GPS gap, search further forward
        if best_dist > 150.0 and hi < n_route:
            dists_full, _ = point_to_segments_projection(
                float(tx[i]), float(ty[i]), x1[lo:], y1[lo:], x2[lo:], y2[lo:]
            )
            alt_rel = int(np.argmin(dists_full))
            if float(dists_full[alt_rel]) + 25.0 < best_dist:
                best_rel = alt_rel
                best_dist = float(dists_full[alt_rel])

        cur_pos = lo + best_rel
        rows.append((trip.trip_id, int(pidx[i]), int(route[cur_pos]), int(cur_pos), best_dist))

    return pd.DataFrame(
        rows,
        columns=["trip_id", "point_idx", "true_arc", "route_pos", "dist_to_route_m"],
    )
