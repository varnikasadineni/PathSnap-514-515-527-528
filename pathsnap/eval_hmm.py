"""Member C (Vibhav): Phase 1 & Phase 2 HMM/Viterbi evaluation, tuning & Folium maps.

Modes (run from repo root inside .venv):
    1. Phase 1 Distance-HMM baseline on all 10 student_test trips:
       python -m pathsnap.eval_hmm --mode baseline

    2. Phase 2 HMM hyperparameter tuning on Varnika's OOF train probabilities:
       python -m pathsnap.eval_hmm --mode tune --probs results/probs_train_oof.parquet

    3. Phase 2 Classifier-only vs Classifier+HMM comparison on all 10 student_test trips
       (uses results/probs_student_test.parquet if present, or models/final_model.joblib):
       python -m pathsnap.eval_hmm --mode classifier

Strict leakage rules:
    - Tuning uses ONLY training OOF probabilities (80 train trips, NEVER evaluator_test).
    - Comparison evaluates on the 10 student_test trips.
    - Every number reported in Section 5 & 6 is written directly to results/*.csv.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pathsnap.candidates import align_track_to_route_dp as align_track_to_route_monotonic, generate_candidates_for_trip
from pathsnap.config import FEATURE_COLUMNS, RANDOM_SEED, REPO_ROOT, load_yaml
from pathsnap.data import TripData, list_trip_dirs, load_trip
from pathsnap.features import build_features_for_trip
from pathsnap.hmm import viterbi_decode_trip
from pathsnap.maps import build_trip_folium_map
from pathsnap.metrics import (
    observation_top1_accuracy,
    sequence_route_metrics,
)
from pathsnap.pipeline import load_trained_model_if_exists


def _resolve_split_dir(paths_cfg: dict, split: str) -> Path:
    key = {"train": "train_dir", "student_test": "student_test_dir"}[split]
    p = REPO_ROOT / paths_cfg["data"][key]
    if not p.exists():
        alt = REPO_ROOT / "data" / "raw" / "Student folder" / split
        if alt.exists():
            return alt
    return p


def _load_or_build_candidates_with_probs(
    trip: TripData,
    cand_cfg: dict,
    probs_for_trip: Optional[pd.DataFrame] = None,
    model=None,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build candidate DataFrame for `trip` and attach classifier `prob` column if provided."""
    cand_df = generate_candidates_for_trip(
        trip,
        base_radius_m=float(cand_cfg["base_radius_m"]),
        fallback_radii_m=tuple(cand_cfg["fallback_radii_m"]),
        min_candidates=int(cand_cfg["min_candidates"]),
        max_candidates=int(cand_cfg["max_candidates"]),
    )
    if probs_for_trip is not None and len(probs_for_trip) > 0:
        merged = cand_df.merge(
            probs_for_trip[["point_idx", "arc_id", "prob"]],
            on=["point_idx", "arc_id"],
            how="left",
        )
        # Fill any unmatched candidate row safely with a tiny probability
        merged["prob"] = merged["prob"].fillna(1e-6).astype(np.float64)
        return merged

    if model is not None:
        feat_df = build_features_for_trip(trip, cand_df)
        cols = feature_cols or FEATURE_COLUMNS
        X = feat_df[cols].to_numpy(dtype=np.float32)
        feat_df["prob"] = model.predict_proba(X)[:, 1].astype(np.float64)
        return feat_df

    return cand_df


def evaluate_baseline_student_test() -> pd.DataFrame:
    """Phase 1: Benchmark Nearest-Arc (no HMM) vs Distance-Only HMM/Viterbi on 10 student_test trips."""
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "paths.yaml")
    hmm_cfg = load_yaml(REPO_ROOT / "configs" / "hmm.yaml")

    st_dir = _resolve_split_dir(paths_cfg, "student_test")
    trip_dirs = list_trip_dirs(st_dir)
    if not trip_dirs:
        raise FileNotFoundError(f"No trips found in {st_dir}")

    cand_cfg = hmm_cfg["candidate_generation"]
    emit_cfg = hmm_cfg["emission"]
    trans_cfg = hmm_cfg["transition"]
    gap_cfg = hmm_cfg["gap_handling"]

    rows = []
    cached_outputs = {}

    for tdir in trip_dirs:
        trip = load_trip(tdir, load_route=True)
        aligned = align_track_to_route_monotonic(trip)
        true_arcs = aligned["true_arc"].to_numpy()

        t0 = time.perf_counter()
        cand_df = _load_or_build_candidates_with_probs(trip, cand_cfg)
        cand_time = time.perf_counter() - t0

        # 1. Nearest-Arc Baseline (No HMM)
        cand_df["neg_dist"] = -cand_df["dist_m"]
        nn_eval = observation_top1_accuracy(cand_df, "neg_dist", aligned)
        idx_best = cand_df.groupby("point_idx", sort=True)["neg_dist"].idxmax()
        nn_pred = cand_df.loc[idx_best, "arc_id"].to_numpy(dtype=np.int32)
        nn_seq = sequence_route_metrics(nn_pred, trip.route, trip.arcs)

        # 2. Distance-Only HMM + Viterbi
        t1 = time.perf_counter()
        hmm_pred, recon_route, diag = viterbi_decode_trip(
            trip,
            cand_df,
            prob_col=None,
            emission_weight=1.0,
            sigma_m=float(emit_cfg["sigma_m"]),
            beta_m=float(trans_cfg["beta_m"]),
            max_route_dist_m=float(trans_cfg["max_route_dist_m"]),
            split_gap_sec=float(gap_cfg["split_gap_sec"]),
            disconnected_log_penalty=float(trans_cfg["disconnected_log_penalty"]),
            bridge_max_dist_m=float(trans_cfg["bridge_max_dist_m"]),
        )
        hmm_time = time.perf_counter() - t1

        hmm_obs_top1 = float(np.mean(hmm_pred == true_arcs))
        hmm_raw_seq = sequence_route_metrics(hmm_pred, trip.route, trip.arcs)
        hmm_bridged_seq = sequence_route_metrics(recon_route, trip.route, trip.arcs)

        row = {
            "trip_id": trip.trip_id,
            "n_points": len(trip.track),
            "n_arcs_graph": len(trip.arcs),
            "n_ref_route_arcs": len(trip.route),
            "median_gps_to_route_m": round(float(aligned["dist_to_route_m"].median()), 3),
            "candidate_miss_rate": round(nn_eval["candidate_miss_rate"], 4),
            "nn_top1_acc": round(nn_eval["top1_accuracy"], 4),
            "nn_route_continuity": round(nn_seq["route_continuity"], 4),
            "nn_route_len_f1": round(nn_seq["route_length_f1"], 4),
            "nn_route_arc_f1": round(nn_seq["route_arc_f1"], 4),
            "hmm_top1_acc": round(hmm_obs_top1, 4),
            "hmm_raw_continuity": round(hmm_raw_seq["route_continuity"], 4),
            "hmm_bridged_continuity": round(hmm_bridged_seq["route_continuity"], 4),
            "hmm_bridged_len_f1": round(hmm_bridged_seq["route_length_f1"], 4),
            "hmm_bridged_arc_f1": round(hmm_bridged_seq["route_arc_f1"], 4),
            "hmm_point_on_route": round(hmm_raw_seq["point_on_route_rate"], 4),
            "cand_time_s": round(cand_time, 3),
            "hmm_time_s": round(hmm_time, 3),
            "total_time_s": round(cand_time + hmm_time, 3),
            "gap_splits": diag["n_gap_splits"],
            "bridge_arcs": diag["n_bridge_arcs_inserted"],
            "disconnected_eval_rate": diag["disconnected_eval_rate"],
            "unique_dijkstra_sources": diag["unique_dijkstra_sources"],
        }
        rows.append(row)
        cached_outputs[trip.trip_id] = (trip, recon_route)

        print(
            f"[{trip.trip_id}] pts={row['n_points']:4d} | miss={row['candidate_miss_rate']*100:4.1f}% | "
            f"NN Top1={row['nn_top1_acc']*100:5.1f}% -> HMM Top1={row['hmm_top1_acc']*100:5.1f}% | "
            f"Cont: {row['nn_route_continuity']*100:5.1f}% -> {row['hmm_bridged_continuity']*100:5.1f}% | "
            f"LenF1: {row['nn_route_len_f1']*100:5.1f}% -> {row['hmm_bridged_len_f1']*100:5.1f}% | "
            f"time={row['total_time_s']:.2f}s"
        )

    df = pd.DataFrame(rows)
    df_out = _append_summary_row(df)

    csv_path = REPO_ROOT / "results" / "hmm_baseline_student_test.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark table -> {csv_path}")

    _plot_comparison(
        df,
        baseline_label="Nearest-Arc (No HMM)",
        hmm_label="Distance-HMM + Viterbi",
        col_pairs=[
            ("Observation Top-1 Accuracy", "nn_top1_acc", "hmm_top1_acc"),
            ("Route Continuity", "nn_route_continuity", "hmm_bridged_continuity"),
            ("Sequence Route Length F1", "nn_route_len_f1", "hmm_bridged_len_f1"),
        ],
        title="PathSnap Phase 1 — Student-Test (10 Trips): Nearest-Arc vs Distance-HMM + Viterbi",
        out_path=REPO_ROOT / "reports" / "figures" / "hmm_phase1_baseline_comparison.png",
    )

    # Export 3 interactive Folium maps: best, GPS-gap (00000065), and lowest pointwise Top-1 (00000077)
    maps_dir = REPO_ROOT / "reports" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    best_tid = str(df.loc[df["hmm_bridged_len_f1"].idxmax(), "trip_id"])
    hard_tid = str(df.loc[df["hmm_top1_acc"].idxmin(), "trip_id"])
    for tid, tag in [(best_tid, "best"), ("00000065", "gps_gap"), (hard_tid, "challenging")]:
        if tid in cached_outputs:
            trip, recon_route = cached_outputs[tid]
            out_html = maps_dir / f"student_test_{tid}_{tag}_hmm_baseline.html"
            build_trip_folium_map(
                trip,
                predicted_arcs=recon_route,
                show_reference_route=True,
                save_html_path=out_html,
            )
            print(f"Saved Folium map -> {out_html}")

    return df_out


def _generate_subset_oof_probs(
    trip_dirs: list[Path],
    cand_cfg: dict,
    model_cfg: dict,
) -> tuple[list[tuple[TripData, np.ndarray, pd.DataFrame]], dict[str, pd.DataFrame]]:
    """Build genuine GroupKFold(5) OOF probabilities on `trip_dirs` using Member B's tuned params.

    Used automatically when results/probs_train_oof.parquet is not present locally (since large
    parquet files are gitignored). Every trip is scored strictly out-of-fold (zero leakage).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    feature_cols = model_cfg.get("feature_columns") or FEATURE_COLUMNS
    best_params = dict(model_cfg.get("best_params") or {})
    best_params.setdefault("class_weight", "balanced")
    best_params.setdefault("random_state", RANDOM_SEED)

    print(f"Building features & 5-fold GroupKFold OOF probabilities on {len(trip_dirs)} train trips...")
    pre_trips = []
    feat_frames = []
    for tdir in trip_dirs:
        trip = load_trip(tdir, load_route=True)
        aligned = align_track_to_route_monotonic(trip)
        cand_df = generate_candidates_for_trip(
            trip,
            base_radius_m=float(cand_cfg["base_radius_m"]),
            fallback_radii_m=tuple(cand_cfg["fallback_radii_m"]),
            min_candidates=int(cand_cfg["min_candidates"]),
            max_candidates=int(cand_cfg["max_candidates"]),
        )
        feat_df = build_features_for_trip(trip, cand_df)
        true_by_pt = aligned.set_index("point_idx")["true_arc"]
        feat_df["label"] = (
            feat_df["arc_id"].to_numpy() == true_by_pt.reindex(feat_df["point_idx"]).to_numpy()
        ).astype(np.int8)
        pre_trips.append((trip, aligned["true_arc"].to_numpy(), feat_df))
        feat_frames.append(feat_df[["trip_id", "point_idx", "arc_id", "label"] + feature_cols])

    all_df = pd.concat(feat_frames, ignore_index=True)
    X = all_df[feature_cols].to_numpy(dtype=np.float32)
    y = all_df["label"].to_numpy(dtype=np.int32)
    groups = all_df["trip_id"].to_numpy()

    gkf = GroupKFold(n_splits=min(5, len(trip_dirs)))
    oof = np.zeros(len(all_df), dtype=np.float64)
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        clf = HistGradientBoostingClassifier(**best_params)
        clf.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
        print(f"  OOF fold {fold_i}/5 done ({len(va_idx):,} candidate rows)")

    all_df["prob"] = oof
    probs_by_trip = {tid: g[["point_idx", "arc_id", "prob"]].copy() for tid, g in all_df.groupby("trip_id", sort=False)}

    preloaded = []
    for trip, true_arcs, feat_df in pre_trips:
        merged = feat_df.merge(probs_by_trip[trip.trip_id], on=["point_idx", "arc_id"], how="left")
        merged["prob"] = merged["prob"].fillna(1e-6).astype(np.float64)
        preloaded.append((trip, true_arcs, merged))
    return preloaded, probs_by_trip


def tune_hmm_on_train_oof(
    probs_path: Path,
    max_train_trips: int = 20,
    update_config: bool = True,
) -> pd.DataFrame:
    """Phase 2: Grid-search HMM decoder parameters on Member B's train OOF probabilities.

    Tunes emission_weight, beta_m, split_gap_sec, and disconnected_log_penalty
    strictly on training OOF predictions (zero student_test or evaluator_test leakage!).
    """
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "paths.yaml")
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model.yaml")
    hmm_cfg = load_yaml(REPO_ROOT / "configs" / "hmm.yaml")
    cand_cfg = hmm_cfg["candidate_generation"]
    trans_cfg = hmm_cfg["transition"]
    train_dir = _resolve_split_dir(paths_cfg, "train")

    excluded = set(str(x) for x in model_cfg.get("excluded_train_trips", ["00000033", "00000054"]))

    if probs_path.exists():
        probs_df = pd.read_parquet(probs_path)
        probs_df["trip_id"] = probs_df["trip_id"].astype(str)
        avail_tids = set(probs_df["trip_id"].unique().tolist())
        trip_dirs = [d for d in list_trip_dirs(train_dir) if d.name in avail_tids and d.name not in excluded]
    else:
        print(f"Note: {probs_path.name} not on disk (gitignored); computing genuine 5-fold GroupKFold OOF on train subset...")
        trip_dirs = [d for d in list_trip_dirs(train_dir) if d.name not in excluded]
        probs_df = None

    if not trip_dirs:
        raise RuntimeError(f"No matching train trip folders found in {train_dir}")

    # Deterministic subset of training trips for fast tuning (0 = use all)
    if max_train_trips > 0 and len(trip_dirs) > max_train_trips:
        rng = np.random.default_rng(RANDOM_SEED)
        pick_idx = sorted(rng.choice(len(trip_dirs), size=min(max_train_trips, len(trip_dirs)), replace=False))
        trip_dirs = [trip_dirs[i] for i in pick_idx]

    if probs_df is not None:
        print(f"Preloading {len(trip_dirs)} train trips + OOF probabilities for HMM tuning...")
        preloaded = []
        probs_by_trip = {tid: g for tid, g in probs_df.groupby("trip_id", sort=False)}
        for tdir in trip_dirs:
            trip = load_trip(tdir, load_route=True)
            aligned = align_track_to_route_monotonic(trip)
            cand_df = _load_or_build_candidates_with_probs(trip, cand_cfg, probs_for_trip=probs_by_trip[trip.trip_id])
            preloaded.append((trip, aligned["true_arc"].to_numpy(), cand_df))
    else:
        preloaded, _ = _generate_subset_oof_probs(trip_dirs, cand_cfg, model_cfg)

    # Focused grid over the 4 most impactful HMM parameters
    grid_emission_w = [0.5, 1.0, 2.0, 3.5]
    grid_beta = [5.0, 10.0, 20.0, 35.0]
    grid_gap_s = [30.0, 60.0]
    grid_disconn = [-25.0]

    combos = list(itertools.product(grid_emission_w, grid_beta, grid_gap_s, grid_disconn))
    print(f"Running {len(combos)} HMM tuning trials on {len(preloaded)} train trips...")

    trial_rows = []
    for trial_id, (ew, beta, gap_s, disconn) in enumerate(combos, start=1):
        t0 = time.perf_counter()
        top1_list, cont_list, len_f1_list, arc_f1_list = [], [], [], []
        for trip, true_arcs, cand_df in preloaded:
            pred_arcs, recon_route, _ = viterbi_decode_trip(
                trip,
                cand_df,
                prob_col="prob",
                emission_weight=ew,
                beta_m=beta,
                max_route_dist_m=float(trans_cfg["max_route_dist_m"]),
                split_gap_sec=gap_s,
                disconnected_log_penalty=disconn,
                bridge_max_dist_m=float(trans_cfg["bridge_max_dist_m"]),
            )
            top1_list.append(float(np.mean(pred_arcs == true_arcs)))
            seq_m = sequence_route_metrics(recon_route, trip.route, trip.arcs)
            cont_list.append(seq_m["route_continuity"])
            len_f1_list.append(seq_m["route_length_f1"])
            arc_f1_list.append(seq_m["route_arc_f1"])

        elapsed = time.perf_counter() - t0
        mean_top1 = float(np.mean(top1_list))
        mean_cont = float(np.mean(cont_list))
        mean_len_f1 = float(np.mean(len_f1_list))
        mean_arc_f1 = float(np.mean(arc_f1_list))
        # Composite objective balancing pointwise accuracy and sequence route recovery
        composite = 0.5 * mean_top1 + 0.5 * mean_len_f1

        trial_rows.append(
            {
                "trial_id": trial_id,
                "emission_weight": ew,
                "beta_m": beta,
                "split_gap_sec": gap_s,
                "disconnected_log_penalty": disconn,
                "n_train_trips": len(preloaded),
                "mean_oof_top1": round(mean_top1, 4),
                "mean_oof_continuity": round(mean_cont, 4),
                "mean_oof_len_f1": round(mean_len_f1, 4),
                "mean_oof_arc_f1": round(mean_arc_f1, 4),
                "composite_score": round(composite, 4),
                "trial_time_s": round(elapsed, 2),
            }
        )
        print(
            f"  Trial {trial_id:2d}/{len(combos)} | w={ew:3.1f}, beta={beta:4.1f}, gap={gap_s:4.0f}s | "
            f"Top1={mean_top1*100:5.2f}% | LenF1={mean_len_f1*100:5.2f}% | Cont={mean_cont*100:5.2f}% | ({elapsed:.1f}s)"
        )

    tune_df = pd.DataFrame(trial_rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    out_csv = REPO_ROOT / "results" / "hmm_tuning.csv"
    tune_df.to_csv(out_csv, index=False)
    print(f"\nSaved HMM tuning log -> {out_csv}")

    best = tune_df.iloc[0]
    print(
        f"BEST HMM CONFIG: emission_weight={best['emission_weight']}, beta_m={best['beta_m']}, "
        f"split_gap_sec={best['split_gap_sec']}, composite={best['composite_score']:.4f}"
    )

    if update_config:
        hmm_cfg["emission"]["emission_weight"] = float(best["emission_weight"])
        hmm_cfg["transition"]["beta_m"] = float(best["beta_m"])
        hmm_cfg["transition"]["disconnected_log_penalty"] = float(best["disconnected_log_penalty"])
        hmm_cfg["gap_handling"]["split_gap_sec"] = float(best["split_gap_sec"])
        cfg_path = REPO_ROOT / "configs" / "hmm.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(hmm_cfg, f, sort_keys=False)
        print(f"Updated tuned parameters in -> {cfg_path}")

    _plot_tuning_heatmap(tune_df, REPO_ROOT / "reports" / "figures" / "hmm_tuning_heatmap.png")
    return tune_df


def evaluate_classifier_vs_hmm_student_test(
    probs_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Phase 2: Compare Classifier-only vs Classifier+HMM on all 10 student_test trips."""
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "paths.yaml")
    model_cfg = load_yaml(REPO_ROOT / "configs" / "model.yaml")
    hmm_cfg = load_yaml(REPO_ROOT / "configs" / "hmm.yaml")

    probs_by_trip: dict[str, pd.DataFrame] = {}
    model = None
    source_desc = ""

    if probs_path is not None and probs_path.exists():
        pdf = pd.read_parquet(probs_path)
        pdf["trip_id"] = pdf["trip_id"].astype(str)
        probs_by_trip = {tid: g for tid, g in pdf.groupby("trip_id", sort=False)}
        source_desc = f"parquet:{probs_path.name}"
    else:
        mpath = model_path or (REPO_ROOT / paths_cfg["artifacts"]["model_path"])
        model = load_trained_model_if_exists(mpath)
        if model is None:
            raise FileNotFoundError(
                "Neither results/probs_student_test.parquet nor models/final_model.joblib exists yet.\n"
                "Wait for Member B (Varnika) Hand-off H4 before running --mode classifier."
            )
        source_desc = f"model:{Path(mpath).name}"

    st_dir = _resolve_split_dir(paths_cfg, "student_test")
    trip_dirs = list_trip_dirs(st_dir)
    cand_cfg = hmm_cfg["candidate_generation"]
    emit_cfg = hmm_cfg["emission"]
    trans_cfg = hmm_cfg["transition"]
    gap_cfg = hmm_cfg["gap_handling"]
    feature_cols = model_cfg.get("feature_columns") or FEATURE_COLUMNS

    print(f"Running Classifier-only vs Classifier+HMM on {len(trip_dirs)} student_test trips ({source_desc})...")
    rows = []
    cached_outputs = {}

    for tdir in trip_dirs:
        trip = load_trip(tdir, load_route=True)
        aligned = align_track_to_route_monotonic(trip)
        true_arcs = aligned["true_arc"].to_numpy()

        t0 = time.perf_counter()
        cand_df = _load_or_build_candidates_with_probs(
            trip,
            cand_cfg,
            probs_for_trip=probs_by_trip.get(trip.trip_id),
            model=model,
            feature_cols=feature_cols,
        )
        feat_time = time.perf_counter() - t0

        # 1. Classifier-only (pointwise argmax of classifier probability)
        clf_eval = observation_top1_accuracy(cand_df, "prob", aligned)
        idx_best = cand_df.groupby("point_idx", sort=True)["prob"].idxmax()
        clf_pred = cand_df.loc[idx_best, "arc_id"].to_numpy(dtype=np.int32)
        clf_seq = sequence_route_metrics(clf_pred, trip.route, trip.arcs)

        # 2. Classifier + HMM/Viterbi
        t1 = time.perf_counter()
        hmm_pred, recon_route, diag = viterbi_decode_trip(
            trip,
            cand_df,
            prob_col="prob",
            emission_weight=float(emit_cfg.get("emission_weight", 1.0)),
            sigma_m=float(emit_cfg["sigma_m"]),
            beta_m=float(trans_cfg["beta_m"]),
            max_route_dist_m=float(trans_cfg["max_route_dist_m"]),
            split_gap_sec=float(gap_cfg["split_gap_sec"]),
            disconnected_log_penalty=float(trans_cfg["disconnected_log_penalty"]),
            prob_eps=float(emit_cfg.get("prob_eps", 1e-6)),
            bridge_max_dist_m=float(trans_cfg["bridge_max_dist_m"]),
        )
        hmm_time = time.perf_counter() - t1

        hmm_obs_top1 = float(np.mean(hmm_pred == true_arcs))
        hmm_raw_seq = sequence_route_metrics(hmm_pred, trip.route, trip.arcs)
        hmm_bridged_seq = sequence_route_metrics(recon_route, trip.route, trip.arcs)

        row = {
            "trip_id": trip.trip_id,
            "n_points": len(trip.track),
            "candidate_miss_rate": round(clf_eval["candidate_miss_rate"], 4),
            # Classifier-only
            "clf_top1_acc": round(clf_eval["top1_accuracy"], 4),
            "clf_route_continuity": round(clf_seq["route_continuity"], 4),
            "clf_route_len_f1": round(clf_seq["route_length_f1"], 4),
            "clf_route_arc_f1": round(clf_seq["route_arc_f1"], 4),
            "clf_point_on_route": round(clf_seq["point_on_route_rate"], 4),
            # Classifier + HMM
            "hmm_top1_acc": round(hmm_obs_top1, 4),
            "hmm_raw_continuity": round(hmm_raw_seq["route_continuity"], 4),
            "hmm_bridged_continuity": round(hmm_bridged_seq["route_continuity"], 4),
            "hmm_bridged_len_f1": round(hmm_bridged_seq["route_length_f1"], 4),
            "hmm_bridged_arc_f1": round(hmm_bridged_seq["route_arc_f1"], 4),
            "hmm_point_on_route": round(hmm_raw_seq["point_on_route_rate"], 4),
            # Deltas (HMM - Classifier-only)
            "delta_top1_acc": round(hmm_obs_top1 - clf_eval["top1_accuracy"], 4),
            "delta_continuity": round(hmm_bridged_seq["route_continuity"] - clf_seq["route_continuity"], 4),
            "delta_len_f1": round(hmm_bridged_seq["route_length_f1"] - clf_seq["route_length_f1"], 4),
            # Runtime & diagnostics
            "feat_time_s": round(feat_time, 3),
            "hmm_time_s": round(hmm_time, 3),
            "total_time_s": round(feat_time + hmm_time, 3),
            "gap_splits": diag["n_gap_splits"],
            "bridge_arcs": diag["n_bridge_arcs_inserted"],
            "disconnected_eval_rate": diag["disconnected_eval_rate"],
        }
        rows.append(row)
        cached_outputs[trip.trip_id] = (trip, recon_route)

        print(
            f"[{trip.trip_id}] Clf Top1={row['clf_top1_acc']*100:5.1f}% -> HMM={row['hmm_top1_acc']*100:5.1f}% "
            f"(Δ={row['delta_top1_acc']*100:+5.1f}pp) | "
            f"Cont: {row['clf_route_continuity']*100:5.1f}% -> {row['hmm_bridged_continuity']*100:5.1f}% | "
            f"LenF1: {row['clf_route_len_f1']*100:5.1f}% -> {row['hmm_bridged_len_f1']*100:5.1f}% | "
            f"time={row['total_time_s']:.2f}s"
        )

    df = pd.DataFrame(rows)
    df_out = _append_summary_row(df)

    csv_path = REPO_ROOT / "results" / "hmm_vs_classifier_student_test.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved Classifier vs HMM table -> {csv_path}")

    _plot_comparison(
        df,
        baseline_label="Classifier-Only",
        hmm_label="Classifier + HMM/Viterbi",
        col_pairs=[
            ("Observation Top-1 Accuracy", "clf_top1_acc", "hmm_top1_acc"),
            ("Route Continuity", "clf_route_continuity", "hmm_bridged_continuity"),
            ("Sequence Route Length F1", "clf_route_len_f1", "hmm_bridged_len_f1"),
        ],
        title="PathSnap Final — Student-Test (10 Trips): Classifier-Only vs Classifier + HMM/Viterbi",
        out_path=REPO_ROOT / "reports" / "figures" / "hmm_vs_classifier_comparison.png",
    )

    # Export final Folium HTML maps: best, GPS-gap (00000065), and smallest-gain/hardest trip
    maps_dir = REPO_ROOT / "reports" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    best_tid = str(df.loc[df["hmm_bridged_len_f1"].idxmax(), "trip_id"])
    hard_tid = str(df.loc[df["delta_top1_acc"].idxmin(), "trip_id"])
    exported = set()
    for tid, tag in [(best_tid, "best"), ("00000065", "gps_gap"), (hard_tid, "hardest")]:
        if tid in cached_outputs and tid not in exported:
            trip, recon_route = cached_outputs[tid]
            out_html = maps_dir / f"student_test_{tid}_{tag}_classifier_hmm.html"
            build_trip_folium_map(
                trip,
                predicted_arcs=recon_route,
                show_reference_route=True,
                save_html_path=out_html,
            )
            exported.add(tid)
            print(f"Saved Final Folium map -> {out_html}")

    # Print failure/hard-case spotlight for Report Section 5
    print("\n--- Section 5 Spotlight: Trips with smallest HMM Top-1 gain ---")
    print(
        df.sort_values("delta_top1_acc")[
            ["trip_id", "clf_top1_acc", "hmm_top1_acc", "delta_top1_acc", "hmm_bridged_len_f1", "gap_splits"]
        ]
        .head(3)
        .to_string(index=False)
    )
    return df_out


def _append_summary_row(df: pd.DataFrame) -> pd.DataFrame:
    sum_cols = {"n_points", "n_arcs_graph", "n_ref_route_arcs", "gap_splits", "bridge_arcs", "unique_dijkstra_sources"}
    summary = {"trip_id": "MEAN_10_TRIPS"}
    for col in df.columns:
        if col == "trip_id":
            continue
        if col in sum_cols:
            summary[col] = int(df[col].sum())
        else:
            summary[col] = round(float(df[col].mean()), 4)
    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def _plot_comparison(
    df: pd.DataFrame,
    baseline_label: str,
    hmm_label: str,
    col_pairs: list[tuple[str, str, str]],
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(df))
    w = 0.38
    tids = df["trip_id"].tolist()

    for ax, (panel_title, c_base, c_hmm) in zip(axes, col_pairs):
        ax.bar(x - w / 2, df[c_base] * 100, width=w, label=baseline_label, color="#78909c")
        ax.bar(x + w / 2, df[c_hmm] * 100, width=w, label=hmm_label, color="#1e88e5")
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved comparison figure -> {out_path}")


def _plot_tuning_heatmap(tune_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_gap = float(tune_df.iloc[0]["split_gap_sec"])
    sub = tune_df[tune_df["split_gap_sec"] == best_gap]
    piv = sub.pivot(index="emission_weight", columns="beta_m", values="composite_score")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(piv.to_numpy() * 100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{c:.0f} m" for c in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in piv.index])
    ax.set_xlabel("Transition beta_m (meters)")
    ax.set_ylabel("Emission Weight")
    ax.set_title(f"HMM Tuning Composite Score (%) on Train OOF (split_gap={best_gap:.0f}s)")

    vals = piv.to_numpy() * 100
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}%", ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, label="Composite Score (%)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved tuning heatmap -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Member C HMM/Viterbi evaluation & tuning")
    parser.add_argument("--mode", choices=["baseline", "tune", "classifier"], default="baseline")
    parser.add_argument("--probs", type=str, default=None, help="Path to probs_*.parquet from Member B")
    parser.add_argument("--model", type=str, default=None, help="Path to final_model.joblib from Member B")
    parser.add_argument("--tune-trips", type=int, default=20, help="Number of train trips for HMM tuning (0=all)")
    parser.add_argument("--no-update-config", action="store_true", help="Do not overwrite configs/hmm.yaml during tuning")
    args = parser.parse_args()

    if args.mode == "baseline":
        evaluate_baseline_student_test()
    elif args.mode == "tune":
        ppath = Path(args.probs) if args.probs else (REPO_ROOT / "results" / "probs_train_oof.parquet")
        tune_hmm_on_train_oof(ppath, max_train_trips=args.tune_trips, update_config=not args.no_update_config)
    elif args.mode == "classifier":
        ppath = Path(args.probs) if args.probs else (REPO_ROOT / "results" / "probs_student_test.parquet")
        mpath = Path(args.model) if args.model else None
        evaluate_classifier_vs_hmm_student_test(probs_path=ppath, model_path=mpath)


if __name__ == "__main__":
    main()
