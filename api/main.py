"""Member D (Laptop 1): FastAPI backend for PathSnap distributed evaluation.

Run on Laptop 1:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment overrides (no code changes needed on demo day):
    PATHSNAP_EVAL_DIR    Folder with the 10 evaluation trips (defaults to
                         data/raw/evaluator_test if present & non-empty,
                         otherwise falls back to data/raw/student_test for dev).
    PATHSNAP_MODEL_PATH  Path to trained final_model.joblib.

Strict integrity guarantees:
    - predict_trip() NEVER reads .route files.
    - Reference .route files are loaded ONLY inside _score_server_side() on Laptop 1.
    - Ground-truth arc IDs / coordinates are NEVER returned in any JSON response.
    - Every request logs clearly to stdout/CMD on Laptop 1 for evaluator inspection.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from pathsnap.candidates import align_track_to_route_dp
from pathsnap.config import REPO_ROOT, load_yaml
from pathsnap.data import list_trip_dirs, load_trip
from pathsnap.metrics import route_continuity, sequence_route_metrics
from pathsnap.pipeline import load_trained_model_if_exists, predict_trip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pathsnap.api")

# Server-side state initialized once at startup
STATE: dict = {
    "model": None,
    "model_path": "",
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
    st_path = REPO_ROOT / paths_cfg["data"]["student_test_dir"]
    if st_path.exists() and list_trip_dirs(st_path):
        return st_path
    alt = REPO_ROOT / "data" / "raw" / "Student folder" / "student_test"
    return alt


def load_server_state() -> None:
    """Load model, configs, and available trip folders into server memory."""
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "paths.yaml")
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model.yaml")
    hmm_cfg = load_yaml(REPO_ROOT / "configs" / "hmm.yaml")

    model_path = os.environ.get("PATHSNAP_MODEL_PATH", paths_cfg["artifacts"]["model_path"])
    model = load_trained_model_if_exists(model_path)

    eval_dir = _resolve_eval_dir(paths_cfg)
    trip_dirs = list_trip_dirs(eval_dir)

    STATE["model"] = model
    STATE["model_path"] = str(model_path)
    STATE["model_cfg"] = model_cfg
    STATE["hmm_cfg"] = hmm_cfg
    STATE["eval_dir"] = eval_dir
    STATE["trip_map"] = {d.name: d for d in trip_dirs}

    mode_str = "classifier+hmm" if model is not None else "distance_hmm_fallback"
    logger.info("=" * 72)
    logger.info("PATHSNAP API READY (Laptop 1 — Inference & Server-Side Evaluation)")
    logger.info("  Eval folder    : %s (%d trips)", eval_dir, len(trip_dirs))
    logger.info("  Trip IDs       : %s", ", ".join(sorted(STATE["trip_map"].keys())))
    logger.info("  Pipeline mode  : %s (model=%s)", mode_str, model_path if model is not None else "not loaded yet")
    logger.info("  HMM beta / gap : beta_m=%s, split_gap_sec=%s",
                hmm_cfg.get("transition", {}).get("beta_m"),
                hmm_cfg.get("gap_handling", {}).get("split_gap_sec"))
    logger.info("=" * 72)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_server_state()
    yield


app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed map-matching & route reconstruction backend (Laptop 1).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "PathSnap Distributed Map-Matching API",
        "endpoints": ["GET /health", "GET /trips", "POST /predict/{trip_id}"],
        "n_trips": len(STATE["trip_map"]),
        "mode": "classifier+hmm" if STATE["model"] is not None else "distance_hmm_fallback",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "eval_dir": str(STATE["eval_dir"]),
        "n_trips": len(STATE["trip_map"]),
        "model_loaded": STATE["model"] is not None,
        "mode": "classifier+hmm" if STATE["model"] is not None else "distance_hmm_fallback",
    }


@app.get("/trips")
def get_trips(request: Request) -> dict:
    trip_ids = sorted(STATE["trip_map"].keys())
    client_ip = request.client.host if (request is not None and request.client) else "local"
    logger.info("client=%s | GET  /trips -> returned %d trips", client_ip, len(trip_ids))
    return {"trips": trip_ids}


def _score_server_side(
    trip_dir: Path,
    pred_arcs: list[int],
    reconstructed_route: Optional[list[int]] = None,
) -> dict:
    """Server-side-only evaluation against the private .route file.

    Uses:
      - `pred_arcs` (1 per GPS point) for observation_top1_accuracy & point_on_route_rate
      - `reconstructed_route` (continuous bridged arc sequence) for route_continuity,
        route_length_f1, and route_arc_f1
    Ground-truth .route NEVER leaves Laptop 1.
    """
    trip = load_trip(trip_dir, load_route=True)
    seq_for_route = reconstructed_route if reconstructed_route else pred_arcs

    if trip.route is None or len(trip.route) == 0:
        return {
            "observation_top1_accuracy": None,
            "route_continuity": round(route_continuity(seq_for_route, trip.arcs), 4),
            "route_length_f1": None,
            "route_arc_f1": None,
            "point_on_route_rate": None,
            "note": "No server-side .route reference file found.",
        }

    # Sequence metrics on the continuous reconstructed route
    bridged_seq = sequence_route_metrics(seq_for_route, trip.route, trip.arcs)
    # Point-on-route rate on the per-observation predictions
    raw_seq = sequence_route_metrics(pred_arcs, trip.route, trip.arcs)

    # Globally optimal monotonic DP alignment (identical to train.py & eval_hmm.py)
    aligned = align_track_to_route_dp(trip)
    true_arcs = aligned["true_arc"].to_numpy()
    pred_arr = np.asarray(pred_arcs, dtype=np.int32)
    obs_acc = float(np.mean(pred_arr == true_arcs)) if len(true_arcs) == len(pred_arr) else None

    return {
        "observation_top1_accuracy": round(obs_acc, 4) if obs_acc is not None else None,
        "route_continuity": round(bridged_seq["route_continuity"], 4),
        "route_length_f1": round(bridged_seq["route_length_f1"], 4),
        "route_arc_f1": round(bridged_seq["route_arc_f1"], 4),
        "point_on_route_rate": round(raw_seq["point_on_route_rate"], 4),
    }


@app.post("/predict/{trip_id}")
def predict_endpoint(trip_id: str, request: Request) -> dict:
    client_ip = request.client.host if (request is not None and request.client) else "local"
    trip_dir: Optional[Path] = STATE["trip_map"].get(trip_id)
    if trip_dir is None:
        logger.warning("client=%s | POST /predict/%s -> 404 Not Found", client_ip, trip_id)
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    logger.info("client=%s | POST /predict/%s -> Inference STARTED...", client_ip, trip_id)
    try:
        result = predict_trip(
            trip_dir=trip_dir,
            model=STATE["model"],
            hmm_cfg=STATE["hmm_cfg"],
            feature_cols=STATE["model_cfg"].get("feature_columns"),
        )
    except Exception as exc:
        logger.exception("client=%s | POST /predict/%s -> FAILED: %s", client_ip, trip_id, exc)
        raise HTTPException(status_code=500, detail=f"Inference failed for trip '{trip_id}': {exc}") from exc

    # Server-side-only evaluation (ground-truth never leaves Laptop 1)
    metrics = _score_server_side(
        trip_dir=trip_dir,
        pred_arcs=result["predicted_arcs"],
        reconstructed_route=result["compressed_route"],
    )

    logger.info(
        "client=%s | POST /predict/%s -> 200 OK | mode=%s | pts=%d | arcs=%d | time=%.3fs | "
        "Top1=%s | LenF1=%s | Cont=%s",
        client_ip,
        trip_id,
        result["mode"],
        len(result["points"]),
        len(result["compressed_route"]),
        result["runtime_sec"],
        f"{metrics['observation_top1_accuracy']*100:.2f}%" if metrics["observation_top1_accuracy"] is not None else "N/A",
        f"{metrics['route_length_f1']*100:.2f}%" if metrics["route_length_f1"] is not None else "N/A",
        f"{metrics['route_continuity']*100:.2f}%" if metrics["route_continuity"] is not None else "N/A",
    )

    # Strictly return ONLY raw GPS points, predicted route geometry, diagnostics & scalar metrics
    return {
        "trip_id": result["trip_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": result["mode"],
        "runtime_sec": result["runtime_sec"],
        "n_points": len(result["points"]),
        "n_predicted_arcs": len(result["compressed_route"]),
        "metrics": metrics,
        "gps_points": [[p["lat"], p["lon"]] for p in result["points"]],
        "predicted_arc_coords": result["predicted_arc_coords"],
        "diagnostics": result["diagnostics"],
    }
