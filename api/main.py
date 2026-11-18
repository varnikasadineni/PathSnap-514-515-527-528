"""Member D (Laptop 1): FastAPI backend for PathSnap distributed evaluation.

Run on Laptop 1:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment overrides (no code edits needed on demo day):
    PATHSNAP_EVAL_DIR   Folder containing the 10 evaluation trips (default: student_test for dev,
                        switch to evaluator_test on demo day)
    PATHSNAP_MODEL_PATH Path to final_model.joblib
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from pathsnap.candidates import align_track_to_route_monotonic
from pathsnap.config import REPO_ROOT, load_yaml
from pathsnap.data import list_trip_dirs, load_trip
from pathsnap.metrics import route_continuity, sequence_route_metrics
from pathsnap.pipeline import load_trained_model_if_exists, predict_trip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap Map-Matching API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server-side state initialized once at startup
STATE: dict = {
    "model": None,
    "model_cfg": {},
    "hmm_cfg": {},
    "eval_dir": None,
    "trip_map": {},  # trip_id -> Path
}


def _resolve_eval_dir(paths_cfg: dict) -> Path:
    env_dir = os.environ.get("PATHSNAP_EVAL_DIR")
    if env_dir:
        return Path(env_dir)
    eval_path = REPO_ROOT / paths_cfg["data"]["evaluator_test_dir"]
    if eval_path.exists() and list_trip_dirs(eval_path):
        return eval_path
    # Development fallback to student_test if evaluator_test folder is not yet dropped in
    return REPO_ROOT / paths_cfg["data"]["student_test_dir"]


@app.on_event("startup")
def startup_event() -> None:
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "paths.yaml")
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model.yaml")
    hmm_cfg = load_yaml(REPO_ROOT / "configs" / "hmm.yaml")

    model_path = os.environ.get("PATHSNAP_MODEL_PATH", paths_cfg["artifacts"]["model_path"])
    model = load_trained_model_if_exists(model_path)

    eval_dir = _resolve_eval_dir(paths_cfg)
    trip_dirs = list_trip_dirs(eval_dir)

    STATE["model"] = model
    STATE["model_cfg"] = model_cfg
    STATE["hmm_cfg"] = hmm_cfg
    STATE["eval_dir"] = eval_dir
    STATE["trip_map"] = {d.name: d for d in trip_dirs}

    logger.info("=" * 64)
    logger.info("PathSnap API STARTED on Laptop 1")
    logger.info("  Eval directory : %s (%d trips)", eval_dir, len(trip_dirs))
    logger.info("  Model loaded   : %s", "YES" if model is not None else "NO (distance-HMM fallback)")
    logger.info("=" * 64)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "eval_dir": str(STATE["eval_dir"]),
        "n_trips": len(STATE["trip_map"]),
        "model_loaded": STATE["model"] is not None,
    }


@app.get("/trips")
def get_trips() -> dict:
    trip_ids = sorted(STATE["trip_map"].keys())
    logger.info("GET /trips -> returning %d trip IDs", len(trip_ids))
    return {"trips": trip_ids}


def _score_server_side(trip_dir: Path, pred_arcs: list[int]) -> dict:
    """Server-side-only evaluation against private .route file.

    Ground-truth .route is NEVER returned in the API response and NEVER passed to predict_trip().
    """
    trip = load_trip(trip_dir, load_route=True)
    if trip.route is None or len(trip.route) == 0:
        return {
            "route_continuity": route_continuity(pred_arcs, trip.arcs),
            "observation_top1_accuracy": None,
            "route_length_f1": None,
            "route_arc_f1": None,
            "point_on_route_rate": None,
            "note": "No server-side .route reference file found.",
        }

    seq_metrics = sequence_route_metrics(pred_arcs, trip.route, trip.arcs)

    # Monotonic alignment of GPS observations onto server-side .route for observation-level accuracy
    aligned = align_track_to_route_monotonic(trip)
    true_arcs = aligned["true_arc"].to_numpy()
    pred_arr = np.asarray(pred_arcs, dtype=np.int32)
    obs_acc = float(np.mean(pred_arr == true_arcs)) if len(true_arcs) == len(pred_arr) else None

    return {
        "observation_top1_accuracy": round(obs_acc, 4) if obs_acc is not None else None,
        "route_continuity": round(seq_metrics["route_continuity"], 4),
        "route_length_f1": round(seq_metrics["route_length_f1"], 4),
        "route_arc_f1": round(seq_metrics["route_arc_f1"], 4),
        "point_on_route_rate": round(seq_metrics["point_on_route_rate"], 4),
    }


@app.post("/predict/{trip_id}")
def predict_endpoint(trip_id: str) -> dict:
    trip_dir: Optional[Path] = STATE["trip_map"].get(trip_id)
    if trip_dir is None:
        logger.warning("POST /predict/%s -> 404 Not Found", trip_id)
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    logger.info("POST /predict/%s -> Inference started...", trip_id)
    result = predict_trip(
        trip_dir=trip_dir,
        model=STATE["model"],
        hmm_cfg=STATE["hmm_cfg"],
        feature_cols=STATE["model_cfg"].get("feature_columns"),
    )

    # Server-side-only scoring (ground-truth never leaves Laptop 1)
    metrics = _score_server_side(trip_dir, result["predicted_arcs"])

    logger.info(
        "POST /predict/%s -> DONE | mode=%s | time=%.3fs | obs_acc=%s | len_F1=%s | continuity=%s",
        trip_id,
        result["mode"],
        result["runtime_sec"],
        metrics.get("observation_top1_accuracy"),
        metrics.get("route_length_f1"),
        metrics.get("route_continuity"),
    )

    # Return ONLY predicted route + scalar evaluation metrics (never .route)
    return {
        "trip_id": result["trip_id"],
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mode": result["mode"],
        "runtime_sec": result["runtime_sec"],
        "n_points": len(result["points"]),
        "n_predicted_arcs": len(result["compressed_route"]),
        "metrics": metrics,
        "gps_points": [[p["lat"], p["lon"]] for p in result["points"]],
        "predicted_arc_coords": result["predicted_arc_coords"],
        "diagnostics": result["diagnostics"],
    }
