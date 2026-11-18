"""Hand-off H5 (Member C): HMM + Log-Space Viterbi Route Reconstruction.

Member C owns this module:
- Hidden states: candidate arcs per GPS point
- Emission probs: classifier predicted probabilities (or Gaussian distance fallback)
- Transition probs: Newson-Krumm exp(-|d_route - d_gps| / beta) using bounded
  cached Dijkstra on a scipy sparse CSR road graph (fast even on 307k arcs!)
- Gap handling: split Viterbi chain whenever dt > split_gap_sec
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from pathsnap.data import TripData


def build_sparse_road_graph(trip: TripData) -> csr_matrix:
    """Build directed weighted road graph (nodes x nodes, weights in meters)."""
    n_nodes = len(trip.nodes)
    u = trip.arcs["from_node"].to_numpy()
    v = trip.arcs["to_node"].to_numpy()
    w = np.maximum(trip.arcs["length_m"].to_numpy(), 1e-3)
    return csr_matrix((w, (u, v)), shape=(n_nodes, n_nodes))


def viterbi_decode_trip(
    trip: TripData,
    cand_df: pd.DataFrame,
    prob_col: Optional[str] = "prob",
    sigma_m: float = 15.0,
    beta_m: float = 10.0,
    max_route_dist_m: float = 500.0,
    split_gap_sec: float = 30.0,
    disconnected_log_penalty: float = -25.0,
    prob_eps: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Decode the most likely sequence of road arcs for every GPS point in `trip`.

    Returns
    -------
    pred_arcs : 1D int32 array of length len(trip.track)
    diagnostics : dict with counts of gap splits, disconnected transitions, etc.
    """
    n_points = len(trip.track)
    pred_arcs = np.full(n_points, -1, dtype=np.int32)
    graph = build_sparse_road_graph(trip)

    from_nodes = trip.arcs["from_node"].to_numpy()
    to_nodes = trip.arcs["to_node"].to_numpy()
    arc_lens = trip.arcs["length_m"].to_numpy()

    tx = trip.track["x"].to_numpy()
    ty = trip.track["y"].to_numpy()
    tt = trip.track["t"].to_numpy()

    # Group candidates by point_idx
    grouped = {
        int(pid): g for pid, g in cand_df.groupby("point_idx", sort=True)
    }

    # Memoize bounded single-source Dijkstra per source node for speed
    dijkstra_cache: dict[int, dict[int, float]] = {}

    def shortest_node_dist(src_node: int, dst_node: int) -> float:
        if src_node == dst_node:
            return 0.0
        cached = dijkstra_cache.get(src_node)
        if cached is None:
            dists = dijkstra(
                graph,
                directed=True,
                indices=src_node,
                limit=max_route_dist_m,
            )
            finite_idx = np.where(np.isfinite(dists))[0]
            cached = {int(k): float(dists[k]) for k in finite_idx}
            dijkstra_cache[src_node] = cached
        return cached.get(dst_node, math.inf)

    # Split point indices into continuous sub-chains when dt > split_gap_sec
    chains: list[list[int]] = []
    current_chain: list[int] = []
    for i in range(n_points):
        if i not in grouped or len(grouped[i]) == 0:
            continue
        if current_chain and (tt[i] - tt[current_chain[-1]]) > split_gap_sec:
            chains.append(current_chain)
            current_chain = [i]
        else:
            current_chain.append(i)
    if current_chain:
        chains.append(current_chain)

    n_disconnected = 0
    n_transitions = 0
    inv_2sigma2 = 1.0 / (2.0 * sigma_m * sigma_m)

    for chain in chains:
        backpointers: list[np.ndarray] = []
        prev_scores: Optional[np.ndarray] = None
        prev_aids: Optional[np.ndarray] = None
        prev_projs: Optional[np.ndarray] = None
        prev_pidx: int = -1

        for step_idx, pid in enumerate(chain):
            g = grouped[pid]
            aids = g["arc_id"].to_numpy(dtype=np.int32)
            dists = g["dist_m"].to_numpy(dtype=np.float64)
            projs = g["proj_frac"].to_numpy(dtype=np.float64)

            if prob_col is not None and prob_col in g.columns:
                probs = np.clip(g[prob_col].to_numpy(dtype=np.float64), prob_eps, 1.0)
                emit_log = np.log(probs)
            else:
                # Distance-only Gaussian emission (Member C can run before Member B finishes!)
                emit_log = -(dists * dists) * inv_2sigma2

            if step_idx == 0:
                prev_scores = emit_log
                backpointers.append(np.full(len(aids), -1, dtype=np.int32))
            else:
                gps_dist = math.hypot(tx[pid] - tx[prev_pidx], ty[pid] - ty[prev_pidx])
                curr_scores = np.full(len(aids), -1e18, dtype=np.float64)
                bp = np.zeros(len(aids), dtype=np.int32)

                for j, (cur_a, cur_pf) in enumerate(zip(aids, projs)):
                    best_val = -1e18
                    best_k = 0
                    cur_u = int(from_nodes[cur_a])
                    cur_along = float(cur_pf * arc_lens[cur_a])

                    for k, (prv_a, prv_pf) in enumerate(zip(prev_aids, prev_projs)):
                        n_transitions += 1
                        if cur_a == prv_a:
                            # Staying on the same arc (penalize backward motion)
                            delta = cur_along - float(prv_pf * arc_lens[prv_a])
                            route_dist = abs(delta) if delta >= -5.0 else abs(delta) + 50.0
                            trans_log = -abs(route_dist - gps_dist) / beta_m
                        else:
                            rem_prev = float((1.0 - prv_pf) * arc_lens[prv_a])
                            d_nodes = shortest_node_dist(int(to_nodes[prv_a]), cur_u)
                            if math.isfinite(d_nodes):
                                route_dist = rem_prev + d_nodes + cur_along
                                trans_log = -abs(route_dist - gps_dist) / beta_m
                            else:
                                n_disconnected += 1
                                trans_log = disconnected_log_penalty

                        score = prev_scores[k] + trans_log + emit_log[j]
                        if score > best_val:
                            best_val = score
                            best_k = k

                    curr_scores[j] = best_val
                    bp[j] = best_k

                prev_scores = curr_scores
                backpointers.append(bp)

            prev_aids = aids
            prev_projs = projs
            prev_pidx = pid

        # Backtrack optimal state sequence for this chain
        best_state = int(np.argmax(prev_scores))
        for step_idx in range(len(chain) - 1, -1, -1):
            pid = chain[step_idx]
            g = grouped[pid]
            pred_arcs[pid] = int(g["arc_id"].to_numpy()[best_state])
            if step_idx > 0:
                best_state = int(backpointers[step_idx][best_state])

    diagnostics = {
        "n_chains": len(chains),
        "n_gap_splits": max(0, len(chains) - 1),
        "n_transitions_eval": n_transitions,
        "n_disconnected_eval": n_disconnected,
        "unique_dijkstra_sources": len(dijkstra_cache),
    }
    return pred_arcs, diagnostics
