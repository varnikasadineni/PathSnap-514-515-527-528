"""Hand-off H5 (Member C — Vibhav -> Member D — Guru): End-to-end inference pipeline.

STRICT RULE: predict_trip(trip_dir) MUST NEVER read the .route file!
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import joblib
import numpy as np

from pathsnap.candidates import generate_candidates_for_trip
from pathsnap.config import FEATURE_COLUMNS, REPO_ROOT, load_yaml
from pathsnap.data import load_trip
from pathsnap.features import build_features_for_trip
from pathsnap.hmm import viterbi_decode_trip
from pathsnap.metrics import compress_arc_sequence


def predict_trip(
    trip_dir: str | Path,
    model=None,
    hmm_cfg: Optional[dict] = None,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """Run complete inference pipeline on a single trip directory (NO .route access).

    Agreed H5 Output Contract (Member C -> Member D):
    {
        "trip_id": str,
        "mode": "classifier+hmm" | "distance_hmm_fallback",
        "runtime_sec": float,
        "points": [{"point_idx": int, "lon": float, "lat": float, "t": float, "arc_id": int}, ...],
        "predicted_arcs": [int, ...],          # length == len(track), one arc_id per GPS point
        "compressed_route": [int, ...],        # continuous reconstructed route (with bridges)
        "raw_compressed_route": [int, ...],    # deduplicated pointwise arcs without bridges
        "predicted_arc_coords": [              # WGS84 line segments for UI map rendering
            {"arc_id": int, "coords": [[lat1, lon1], [lat2, lon2]]}, ...
        ],
        "diagnostics": dict,
    }
    """
    t0 = time.perf_counter()

    # Load trip with load_route=False so ground-truth is NEVER accessed here
    trip = load_trip(trip_dir, load_route=False)
    cfg = hmm_cfg or load_yaml(REPO_ROOT / "configs" / "hmm.yaml")
    cand_cfg = cfg.get("candidate_generation", {})
    emit_cfg = cfg.get("emission", {})
    trans_cfg = cfg.get("transition", {})
    gap_cfg = cfg.get("gap_handling", {})

    # 1. Candidate generation
    cand_df = generate_candidates_for_trip(
        trip,
        base_radius_m=float(cand_cfg.get("base_radius_m", 50.0)),
        fallback_radii_m=tuple(cand_cfg.get("fallback_radii_m", [100.0, 200.0, 350.0])),
        min_candidates=int(cand_cfg.get("min_candidates", 5)),
        max_candidates=int(cand_cfg.get("max_candidates", 15)),
    )

    # 2. Feature extraction & Classifier probabilities (if trained model is supplied)
    prob_col: Optional[str] = None
    mode = "distance_hmm_fallback"
    if model is not None:
        feat_df = build_features_for_trip(trip, cand_df)
        cols = feature_cols or FEATURE_COLUMNS
        X = feat_df[cols].to_numpy(dtype=np.float32)
        feat_df["prob"] = model.predict_proba(X)[:, 1]
        cand_df = feat_df
        prob_col = "prob"
        mode = "classifier+hmm"

    # 3. HMM + Viterbi decoding + continuous route bridging
    pred_arcs, reconstructed_route, diagnostics = viterbi_decode_trip(
        trip,
        cand_df,
        prob_col=prob_col,
        emission_weight=float(emit_cfg.get("emission_weight", 1.0)),
        sigma_m=float(emit_cfg.get("sigma_m", 15.0)),
        beta_m=float(trans_cfg.get("beta_m", 10.0)),
        max_route_dist_m=float(trans_cfg.get("max_route_dist_m", 500.0)),
        split_gap_sec=float(gap_cfg.get("split_gap_sec", 30.0)),
        disconnected_log_penalty=float(trans_cfg.get("disconnected_log_penalty", -25.0)),
        prob_eps=float(emit_cfg.get("prob_eps", 1e-6)),
        bridge_max_dist_m=float(trans_cfg.get("bridge_max_dist_m", 400.0)),
    )

    # 4. Format coordinates for UI rendering
    node_lon = trip.nodes["lon"].to_numpy()
    node_lat = trip.nodes["lat"].to_numpy()
    from_nodes = trip.arcs["from_node"].to_numpy()
    to_nodes = trip.arcs["to_node"].to_numpy()

    raw_compressed = compress_arc_sequence(pred_arcs)
    arc_coords = []
    for aid in reconstructed_route:
        u, v = int(from_nodes[aid]), int(to_nodes[aid])
        arc_coords.append(
            {
                "arc_id": int(aid),
                "coords": [
                    [float(node_lat[u]), float(node_lon[u])],
                    [float(node_lat[v]), float(node_lon[v])],
                ],
            }
        )

    points = [
        {
            "point_idx": int(row.point_idx),
            "lon": float(row.lon),
            "lat": float(row.lat),
            "t": float(row.t),
            "arc_id": int(pred_arcs[int(row.point_idx)]),
        }
        for row in trip.track.itertuples(index=False)
    ]

    return {
        "trip_id": trip.trip_id,
        "mode": mode,
        "runtime_sec": round(time.perf_counter() - t0, 4),
        "points": points,
        "predicted_arcs": [int(a) for a in pred_arcs],
        "compressed_route": reconstructed_route,
        "raw_compressed_route": raw_compressed,
        "predicted_arc_coords": arc_coords,
        "diagnostics": diagnostics,
    }


def load_trained_model_if_exists(model_path: str | Path):
    path = Path(model_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        return joblib.load(path)
    return None
