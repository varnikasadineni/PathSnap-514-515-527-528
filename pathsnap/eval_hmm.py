"""Member C (Vibhav): Phase 1 & Phase 2 HMM/Viterbi evaluation & Folium map generator.

Usage (from repo root, inside .venv):
    python -m pathsnap.eval_hmm --mode baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.candidates import align_track_to_route_monotonic, generate_candidates_for_trip
from pathsnap.config import REPO_ROOT, load_yaml
from pathsnap.data import list_trip_dirs, load_trip
from pathsnap.hmm import viterbi_decode_trip
from pathsnap.maps import build_trip_folium_map
from pathsnap.metrics import (
    observation_top1_accuracy,
    route_continuity,
    sequence_route_metrics,
)


def evaluate_baseline_student_test() -> pd.DataFrame:
    """Benchmark Nearest-Arc (no HMM) vs Distance-Only HMM/Viterbi on all 10 student_test trips."""
    paths_cfg = load_yaml(REPO_ROOT / "configs" / "paths.yaml")
    hmm_cfg = load_yaml(REPO_ROOT / "configs" / "hmm.yaml")

    st_dir = REPO_ROOT / paths_cfg["data"]["student_test_dir"]
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
        cand_df = generate_candidates_for_trip(
            trip,
            base_radius_m=float(cand_cfg["base_radius_m"]),
            fallback_radii_m=tuple(cand_cfg["fallback_radii_m"]),
            min_candidates=int(cand_cfg["min_candidates"]),
            max_candidates=int(cand_cfg["max_candidates"]),
        )
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
            # Nearest-Arc (No HMM)
            "nn_top1_acc": round(nn_eval["top1_accuracy"], 4),
            "nn_route_continuity": round(nn_seq["route_continuity"], 4),
            "nn_route_len_f1": round(nn_seq["route_length_f1"], 4),
            "nn_route_arc_f1": round(nn_seq["route_arc_f1"], 4),
            # Distance-HMM + Viterbi
            "hmm_top1_acc": round(hmm_obs_top1, 4),
            "hmm_raw_continuity": round(hmm_raw_seq["route_continuity"], 4),
            "hmm_bridged_continuity": round(hmm_bridged_seq["route_continuity"], 4),
            "hmm_bridged_len_f1": round(hmm_bridged_seq["route_length_f1"], 4),
            "hmm_bridged_arc_f1": round(hmm_bridged_seq["route_arc_f1"], 4),
            "hmm_point_on_route": round(hmm_raw_seq["point_on_route_rate"], 4),
            # Runtime & Diagnostics
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
            f"time={row['total_time_s']:.2f}s (gaps={row['gap_splits']}, bridges={row['bridge_arcs']})"
        )

    df = pd.DataFrame(rows)

    # Append macro-average summary row
    summary = {"trip_id": "MEAN_10_TRIPS"}
    for col in df.columns:
        if col == "trip_id":
            continue
        if col in {"n_points", "n_arcs_graph", "n_ref_route_arcs", "gap_splits", "bridge_arcs", "unique_dijkstra_sources"}:
            summary[col] = int(df[col].sum())
        else:
            summary[col] = round(float(df[col].mean()), 4)
    df_out = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "hmm_baseline_student_test.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark table -> {csv_path}")

    # Generate Phase 1 comparison plot
    _plot_phase1_comparison(df, REPO_ROOT / "reports" / "figures" / "hmm_phase1_baseline_comparison.png")

    # Export 3 interactive Folium maps: best, gap trip (00000065 if present), and lowest-scoring trip
    maps_dir = REPO_ROOT / "reports" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    best_tid = str(df.loc[df["hmm_bridged_len_f1"].idxmax(), "trip_id"])
    worst_tid = str(df.loc[df["hmm_bridged_len_f1"].idxmin(), "trip_id"])
    selected_tids = []
    for tid in [best_tid, "00000065", worst_tid]:
        if tid in cached_outputs and tid not in selected_tids:
            selected_tids.append(tid)

    for tid in selected_tids:
        trip, recon_route = cached_outputs[tid]
        tag = "best" if tid == best_tid else ("gps_gap" if tid == "00000065" else "challenging")
        out_html = maps_dir / f"student_test_{tid}_{tag}_hmm_baseline.html"
        build_trip_folium_map(
            trip,
            predicted_arcs=recon_route,
            show_reference_route=True,
            save_html_path=out_html,
        )
        print(f"Saved Folium map -> {out_html}")

    return df_out


def _plot_phase1_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Create a clean side-by-side comparison plot of Nearest-Arc vs Distance-HMM."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(df))
    w = 0.38
    tids = df["trip_id"].tolist()

    panels = [
        ("Observation Top-1 Accuracy", "nn_top1_acc", "hmm_top1_acc"),
        ("Route Continuity", "nn_route_continuity", "hmm_bridged_continuity"),
        ("Sequence Route Length F1", "nn_route_len_f1", "hmm_bridged_len_f1"),
    ]
    for ax, (title, c_nn, c_hmm) in zip(axes, panels):
        ax.bar(x - w / 2, df[c_nn] * 100, width=w, label="Nearest-Arc (No HMM)", color="#78909c")
        ax.bar(x + w / 2, df[c_hmm] * 100, width=w, label="Distance-HMM + Viterbi", color="#1e88e5")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "PathSnap Phase 1 — Student-Test (10 Trips): Nearest-Arc vs Distance-HMM + Viterbi",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved comparison figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Member C HMM/Viterbi evaluation")
    parser.add_argument("--mode", choices=["baseline"], default="baseline")
    args = parser.parse_args()
    if args.mode == "baseline":
        evaluate_baseline_student_test()


if __name__ == "__main__":
    main()
