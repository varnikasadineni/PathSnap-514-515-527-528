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


def align_track_to_route_monotonic(trip: TripData, skip_penalty_m: float = 0.5) -> pd.DataFrame:
    """Assign each GPS point to its reference arc along `trip.route` via exact monotonic DP.

    Minimizes sum_t [ dist(track[t], route[m_t]) + skip_penalty_m * (m_t - m_{t-1}) ]
    subject to 0 <= m_0 <= m_1 <= ... <= m_{T-1} < len(route).

    Why exact DP instead of greedy windowing?
    A single noisy GPS outlier cannot pull `m_t` permanently forward on long/looping trips
    (e.g., student_test/00000056 with 3,958 pts & 1,323 route arcs).

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
    n_pts = len(tx)

    # Precompute segment geometry once
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    seg_valid = seg_len_sq > 1e-12

    # Exact O(T * M) monotonic DP using running prefix minimum:
    # best_prev[j] = min_{k <= j} (dp[k] - k * skip_penalty_m)
    # => min_{k <= j} (dp[k] + (j - k) * skip_penalty_m) = best_prev[j] + j * skip_penalty_m
    idx_range = np.arange(n_route, dtype=np.float64)
    pen_vec = idx_range * skip_penalty_m

    # Store backpointers as int16/int32 (T x M, e.g. 4000 x 1323 x 2B ~ 10 MB)
    bp = np.empty((n_pts, n_route), dtype=np.int32)
    dist_mat_best = np.empty(n_pts, dtype=np.float64)

    # Step t = 0
    px0, py0 = float(tx[0]), float(ty[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(seg_valid, ((px0 - x1) * dx + (py0 - y1) * dy) / seg_len_sq, 0.0)
    t0 = np.clip(t0, 0.0, 1.0)
    d0 = np.hypot(px0 - (x1 + t0 * dx), py0 - (y1 + t0 * dy))
    # Mild preference for starting near the beginning of the route
    dp = d0 + 0.1 * pen_vec
    bp[0, :] = -1

    # Keep per-step distances only for backtracking recovery (or recompute at end in O(T))
    for i in range(1, n_pts):
        # Prefix argmin of shifted = dp - pen_vec
        shifted = dp - pen_vec
        # Fast vectorized prefix minimum & argmin via Python loop only on improvements
        # Using numpy.minimum.accumulate for values and tracking indices
        pref_min = np.minimum.accumulate(shifted)
        # Identify positions where a new strict minimum occurs
        is_new = np.empty(n_route, dtype=bool)
        is_new[0] = True
        is_new[1:] = pref_min[1:] < pref_min[:-1]
        pref_arg = np.maximum.accumulate(np.where(is_new, np.arange(n_route, dtype=np.int32), 0))
        bp[i, :] = pref_arg

        px, py = float(tx[i]), float(ty[i])
        with np.errstate(divide="ignore", invalid="ignore"):
            proj_t = np.where(seg_valid, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0)
        proj_t = np.clip(proj_t, 0.0, 1.0)
        dists = np.hypot(px - (x1 + proj_t * dx), py - (y1 + proj_t * dy))

        dp = (pref_min + pen_vec) + dists

    # Backtrack optimal monotonic route position sequence
    route_pos = np.empty(n_pts, dtype=np.int32)
    cur = int(np.argmin(dp))
    for i in range(n_pts - 1, -1, -1):
        route_pos[i] = cur
        if i > 0:
            cur = int(bp[i, cur])

    # Compute exact Euclidean distance from each GPS point to its assigned route arc
    rx1, ry1 = x1[route_pos], y1[route_pos]
    rx2, ry2 = x2[route_pos], y2[route_pos]
    rdx, rdy = rx2 - rx1, ry2 - ry1
    rlen_sq = rdx * rdx + rdy * rdy
    with np.errstate(divide="ignore", invalid="ignore"):
        rt = np.where(rlen_sq > 1e-12, ((tx - rx1) * rdx + (ty - ry1) * rdy) / rlen_sq, 0.0)
    rt = np.clip(rt, 0.0, 1.0)
    dist_mat_best = np.hypot(tx - (rx1 + rt * rdx), ty - (ry1 + rt * rdy))

    return pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "point_idx": pidx.astype(np.int32),
            "true_arc": route[route_pos].astype(np.int32),
            "route_pos": route_pos,
            "dist_to_route_m": dist_mat_best,
        }
    )
