"""Hand-off H5 (Member C — Vibhav): HMM + Log-Space Viterbi Route Reconstruction.

Key design choices:
- Hidden states: candidate road arcs at each GPS observation (preserving arc_id).
- Emission log-probabilities:
    * Classifier mode: emission_weight * log(clip(P_clf, prob_eps, 1.0))
    * Distance-only baseline (Phase 1): -(dist_m^2) / (2 * sigma_m^2)
- Transition log-probabilities (Newson-Krumm style):
    * Compare road-network driving distance d_route vs straight-line GPS distance d_gps:
      log P_trans = -|d_route - d_gps| / beta_m
    * Computed via cached bounded single-source Dijkstra on a scipy sparse CSR graph
      so even 307k-arc trips decode in ~0.1–2 seconds.
    * Disconnected transitions receive a finite log penalty (disconnected_log_penalty)
      and are explicitly counted for diagnostics.
- Edge cases:
    * Large GPS gaps (dt > split_gap_sec): split trajectory into independent Viterbi chains.
    * Missing candidates: skipped during decoding and forward/backward filled from nearest
      decoded neighbor within the same chain (counted in diagnostics).
- Continuous route reconstruction:
    * Between consecutive distinct matched arcs within the same chain, if short
      intermediate arcs were traversed between 1 Hz GPS pings (to_node != from_node),
      shortest-path bridge arcs (within bridge_max_dist_m) are inserted to recover a
      topologically continuous road path.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from pathsnap.data import TripData


def build_sparse_road_graph(trip: TripData) -> tuple[csr_matrix, dict[tuple[int, int], int]]:
    """Build directed weighted road graph (nodes x nodes, meters) and (u, v)->arc_id lookup."""
    n_nodes = len(trip.nodes)
    u = trip.arcs["from_node"].to_numpy()
    v = trip.arcs["to_node"].to_numpy()
    w = np.maximum(trip.arcs["length_m"].to_numpy(), 1e-3)
    aids = trip.arcs["arc_id"].to_numpy()

    # If parallel edges share (u, v), keep the shortest arc_id for bridge reconstruction
    edge_to_arc: dict[tuple[int, int], int] = {}
    edge_best_w: dict[tuple[int, int], float] = {}
    for ui, vi, wi, ai in zip(u, v, w, aids):
        key = (int(ui), int(vi))
        wf = float(wi)
        if key not in edge_best_w or wf < edge_best_w[key]:
            edge_best_w[key] = wf
            edge_to_arc[key] = int(ai)

    graph = csr_matrix((w, (u, v)), shape=(n_nodes, n_nodes))
    return graph, edge_to_arc


def reconstruct_continuous_route(
    pred_arcs: Sequence[int],
    trip: TripData,
    graph: csr_matrix,
    edge_to_arc: dict[tuple[int, int], int],
    split_gap_sec: float = 30.0,
    bridge_max_dist_m: float = 400.0,
) -> tuple[list[int], int]:
    """Compress pointwise predicted arcs and bridge short skipped intermediate arcs.

    Returns (continuous_route_arcs, n_bridges_inserted).
    """
    from_nodes = trip.arcs["from_node"].to_numpy()
    to_nodes = trip.arcs["to_node"].to_numpy()
    tt = trip.track["t"].to_numpy()
    n_arcs = len(from_nodes)

    # Extract compressed (arc_id, last_t, first_t) runs
    runs: list[tuple[int, float, float]] = []
    for idx, a in enumerate(pred_arcs):
        ai = int(a)
        if ai < 0 or ai >= n_arcs:
            continue
        t_cur = float(tt[idx])
        if not runs or runs[-1][0] != ai:
            runs.append((ai, t_cur, t_cur))
        else:
            runs[-1] = (ai, runs[-1][1], t_cur)

    if not runs:
        return [], 0

    route_out: list[int] = [runs[0][0]]
    n_bridges = 0

    # Cache predecessor trees only for nodes that actually need bridging
    pred_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for i in range(1, len(runs)):
        prev_a, _, prev_t_end = runs[i - 1]
        cur_a, cur_t_start, _ = runs[i]

        # Do not bridge across an intentional GPS gap split
        if (cur_t_start - prev_t_end) <= split_gap_sec:
            src_node = int(to_nodes[prev_a])
            dst_node = int(from_nodes[cur_a])
            if src_node != dst_node:
                cached = pred_cache.get(src_node)
                if cached is None:
                    dists, preds = dijkstra(
                        graph,
                        directed=True,
                        indices=src_node,
                        limit=bridge_max_dist_m,
                        return_predecessors=True,
                    )
                    cached = (dists, preds)
                    pred_cache[src_node] = cached
                dists, preds = cached
                if math.isfinite(float(dists[dst_node])):
                    # Backtrack node path from dst_node -> src_node
                    node_path = [dst_node]
                    cur_n = dst_node
                    valid_path = True
                    steps = 0
                    while cur_n != src_node:
                        p = int(preds[cur_n])
                        if p < 0 or steps > 50:
                            valid_path = False
                            break
                        node_path.append(p)
                        cur_n = p
                        steps += 1
                    if valid_path:
                        node_path.reverse()
                        bridge_arcs = []
                        for nu, nv in zip(node_path[:-1], node_path[1:]):
                            ba = edge_to_arc.get((nu, nv))
                            if ba is not None and ba != prev_a and ba != cur_a:
                                bridge_arcs.append(ba)
                        if bridge_arcs:
                            route_out.extend(bridge_arcs)
                            n_bridges += len(bridge_arcs)

        if route_out[-1] != cur_a:
            route_out.append(cur_a)

    return route_out, n_bridges


def viterbi_decode_trip(
    trip: TripData,
    cand_df: pd.DataFrame,
    prob_col: Optional[str] = "prob",
    emission_weight: float = 1.0,
    sigma_m: float = 15.0,
    beta_m: float = 10.0,
    max_route_dist_m: float = 500.0,
    split_gap_sec: float = 30.0,
    disconnected_log_penalty: float = -25.0,
    prob_eps: float = 1e-6,
    bridge_max_dist_m: float = 400.0,
) -> tuple[np.ndarray, list[int], dict]:
    """Decode the most likely sequence of road arcs for every GPS point in `trip`.

    Returns
    -------
    pred_arcs : 1D int32 array of length len(trip.track) (pointwise predicted arc_id)
    reconstructed_route : list[int] continuous sequence of arcs with short bridges filled
    diagnostics : dict with gap splits, disconnected counts, bridge counts, cache stats
    """
    n_points = len(trip.track)
    pred_arcs = np.full(n_points, -1, dtype=np.int32)
    graph, edge_to_arc = build_sparse_road_graph(trip)

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
    n_missing_cand = 0
    for i in range(n_points):
        if i not in grouped or len(grouped[i]) == 0:
            n_missing_cand += 1
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
                emit_log = emission_weight * np.log(probs)
            else:
                emit_log = emission_weight * (-(dists * dists) * inv_2sigma2)

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

    # Fill any point that had 0 candidates from nearest valid neighbor
    if n_missing_cand > 0:
        valid_idx = np.where(pred_arcs >= 0)[0]
        if len(valid_idx) > 0:
            for i in range(n_points):
                if pred_arcs[i] < 0:
                    nearest_pos = int(np.argmin(np.abs(valid_idx - i)))
                    pred_arcs[i] = pred_arcs[valid_idx[nearest_pos]]

    reconstructed_route, n_bridges = reconstruct_continuous_route(
        pred_arcs=pred_arcs,
        trip=trip,
        graph=graph,
        edge_to_arc=edge_to_arc,
        split_gap_sec=split_gap_sec,
        bridge_max_dist_m=bridge_max_dist_m,
    )

    diagnostics = {
        "n_chains": len(chains),
        "n_gap_splits": max(0, len(chains) - 1),
        "n_missing_cand_points": n_missing_cand,
        "n_transitions_eval": n_transitions,
        "n_disconnected_eval": n_disconnected,
        "disconnected_eval_rate": round(n_disconnected / n_transitions, 4) if n_transitions else 0.0,
        "unique_dijkstra_sources": len(dijkstra_cache),
        "n_bridge_arcs_inserted": n_bridges,
    }
    return pred_arcs, reconstructed_route, diagnostics
