"""Hand-off H3 (Member B -> C, D): Shared evaluation metrics for PathSnap.

Covers every required metric in Models and Training Guidelines.pdf & Deliverables.pdf:
- Pair-level classification: Accuracy, Precision, Recall, Macro F1, Log Loss, Confusion Matrix
- Observation-level Top-1 Accuracy (missed candidates count as wrong) & Candidate Miss Rate
- Route Continuity (consecutive distinct predicted arcs connected in the road graph)
- Sequence-level Route Performance (arc-set & length-weighted Precision, Recall, F1 vs .route)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def pair_classification_metrics(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict:
    """Compute candidate-pair classification metrics on 0/1 labels."""
    y_true = np.asarray(y_true, dtype=np.int32)
    if y_pred is None:
        if y_prob is None:
            raise ValueError("Provide either y_prob or y_pred.")
        y_pred = (np.asarray(y_prob) >= threshold).astype(np.int32)
    else:
        y_pred = np.asarray(y_pred, dtype=np.int32)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_prob is not None:
        probs_clipped = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-15, 1.0 - 1e-15)
        out["log_loss"] = float(log_loss(y_true, probs_clipped, labels=[0, 1]))
    else:
        out["log_loss"] = None
    return out


def observation_top1_accuracy(
    cand_df: pd.DataFrame,
    score_col: str,
    true_points_df: pd.DataFrame,
) -> dict:
    """Compute observation-level Top-1 accuracy and Candidate Miss Rate.

    Parameters
    ----------
    cand_df : DataFrame with columns ['trip_id', 'point_idx', 'arc_id', score_col]
    score_col : column name containing predicted probability / score
    true_points_df : DataFrame with columns ['trip_id', 'point_idx', 'true_arc']
        One row per GPS observation. `true_arc` is the aligned reference arc_id
        (or -1 if unalignable / excluded). Points whose true_arc is absent from
        their candidate set count as WRONG in Top-1 accuracy.
    """
    valid_pts = true_points_df[true_points_df["true_arc"] >= 0].copy()
    n_total = len(valid_pts)
    if n_total == 0:
        return {"top1_accuracy": 0.0, "candidate_miss_rate": 0.0, "n_points": 0, "n_missed": 0}

    # Check candidate recall (miss rate)
    merged_all = valid_pts.merge(
        cand_df[["trip_id", "point_idx", "arc_id"]],
        on=["trip_id", "point_idx"],
        how="left",
    )
    has_true = (
        merged_all.assign(hit=(merged_all["arc_id"] == merged_all["true_arc"]))
        .groupby(["trip_id", "point_idx"], sort=False)["hit"]
        .any()
    )
    n_missed = int((~has_true).sum())

    # Pick highest-scoring candidate per (trip_id, point_idx)
    idx_best = cand_df.groupby(["trip_id", "point_idx"], sort=False)[score_col].idxmax()
    top1_preds = cand_df.loc[idx_best, ["trip_id", "point_idx", "arc_id"]].rename(
        columns={"arc_id": "pred_arc"}
    )
    eval_df = valid_pts.merge(top1_preds, on=["trip_id", "point_idx"], how="left")
    n_correct = int((eval_df["pred_arc"] == eval_df["true_arc"]).sum())

    return {
        "top1_accuracy": float(n_correct / n_total),
        "candidate_miss_rate": float(n_missed / n_total),
        "n_points": int(n_total),
        "n_missed": n_missed,
    }


def compress_arc_sequence(arc_ids: Sequence[int]) -> list[int]:
    """Collapse consecutive identical arc_ids (ignoring negative/invalid -1 entries)."""
    seq: list[int] = []
    for a in arc_ids:
        ai = int(a)
        if ai < 0:
            continue
        if not seq or seq[-1] != ai:
            seq.append(ai)
    return seq


def route_continuity(pred_arcs: Sequence[int], arcs_df: pd.DataFrame) -> float:
    """Share of consecutive DISTINCT predicted arcs where prev.to_node == next.from_node."""
    seq = compress_arc_sequence(pred_arcs)
    if len(seq) < 2:
        return 1.0
    from_nodes = arcs_df["from_node"].to_numpy()
    to_nodes = arcs_df["to_node"].to_numpy()
    n_arcs = len(from_nodes)

    connected = 0
    total = len(seq) - 1
    for prev_a, next_a in zip(seq[:-1], seq[1:]):
        if 0 <= prev_a < n_arcs and 0 <= next_a < n_arcs:
            if to_nodes[prev_a] == from_nodes[next_a]:
                connected += 1
    return float(connected / total)


def sequence_route_metrics(
    pred_arcs: Sequence[int],
    ref_route: Sequence[int],
    arcs_df: pd.DataFrame,
) -> dict:
    """Evaluate compressed predicted arc sequence against reference .route.

    Computes both arc-count F1 and length-weighted Route F1 (standard map-matching
    sequence metric), plus Route Continuity.
    """
    pred_seq = compress_arc_sequence(pred_arcs)
    ref_seq = compress_arc_sequence(ref_route)

    pred_set = set(pred_seq)
    ref_set = set(ref_seq)
    matched = pred_set & ref_set

    # Arc-count Precision / Recall / F1
    arc_prec = len(matched) / len(pred_set) if pred_set else 0.0
    arc_rec = len(matched) / len(ref_set) if ref_set else 0.0
    arc_f1 = (2 * arc_prec * arc_rec / (arc_prec + arc_rec)) if (arc_prec + arc_rec) > 0 else 0.0

    # Length-weighted Route Precision / Recall / F1
    lengths = arcs_df["length_m"].to_numpy()
    matched_len = float(sum(lengths[a] for a in matched if 0 <= a < len(lengths)))
    pred_len = float(sum(lengths[a] for a in pred_set if 0 <= a < len(lengths)))
    ref_len = float(sum(lengths[a] for a in ref_set if 0 <= a < len(lengths)))

    len_prec = matched_len / pred_len if pred_len > 0 else 0.0
    len_rec = matched_len / ref_len if ref_len > 0 else 0.0
    len_f1 = (2 * len_prec * len_rec / (len_prec + len_rec)) if (len_prec + len_rec) > 0 else 0.0

    # Fraction of GPS points whose predicted arc lies on the reference route
    valid_preds = [int(a) for a in pred_arcs if int(a) >= 0]
    on_route_rate = (
        sum(1 for a in valid_preds if a in ref_set) / len(valid_preds)
        if valid_preds
        else 0.0
    )

    return {
        "route_continuity": route_continuity(pred_arcs, arcs_df),
        "route_arc_precision": float(arc_prec),
        "route_arc_recall": float(arc_rec),
        "route_arc_f1": float(arc_f1),
        "route_length_precision": float(len_prec),
        "route_length_recall": float(len_rec),
        "route_length_f1": float(len_f1),
        "point_on_route_rate": float(on_route_rate),
        "n_pred_unique_arcs": len(pred_set),
        "n_ref_unique_arcs": len(ref_set),
    }
