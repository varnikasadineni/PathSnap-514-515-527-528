"""Hand-off H4 (Member B): Model experimentation (5+ models) & Optuna Bayesian HPO.

Member B owns this module:
- Train & evaluate >= 5 distinct models across families with GroupKFold(5) by trip_id
- Record ALL required metrics into results/model_comparison.csv
- Run Optuna Bayesian HPO on the selected model -> results/optuna_trials.csv
- Export models/final_model.joblib + configs/model.yaml + out-of-fold & student-test probs
"""

from __future__ import annotations

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.config import CV_N_SPLITS, FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_SEED
from pathsnap.metrics import observation_top1_accuracy, pair_classification_metrics


def get_candidate_models(seed: int = RANDOM_SEED) -> dict:
    """Return >= 5 distinct models spanning diverse algorithm families.

    Member B: expand/tune configurations and record every experiment cleanly.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "LogisticRegression": (
            "Linear",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)),
            ]),
        ),
        "GaussianNB": (
            "Probabilistic",
            GaussianNB(),
        ),
        "KNN": (
            "Instance-based",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
            ]),
        ),
        "RandomForest": (
            "Ensemble",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=14,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        "HistGradientBoosting": (
            "Boosting",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_depth=8,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "MLP": (
            "Neural Network",
            Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=40,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]),
        ),
    }
    return models


def run_grouped_cv_skeleton(
    train_df: pd.DataFrame,
    true_train_pts: pd.DataFrame,
    model,
) -> tuple[np.ndarray, dict]:
    """Run GroupKFold(5) by trip_id on 80 training trips, returning OOF probabilities."""
    X = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = train_df[LABEL_COLUMN].to_numpy(dtype=np.int32)
    groups = train_df["trip_id"].to_numpy()

    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    oof_probs = np.zeros(len(train_df), dtype=np.float64)
    fold_f1s = []

    t0 = time.perf_counter()
    for tr_idx, va_idx in gkf.split(X, y, groups):
        from sklearn.base import clone

        est = clone(model)
        est.fit(X[tr_idx], y[tr_idx])
        va_prob = est.predict_proba(X[va_idx])[:, 1]
        oof_probs[va_idx] = va_prob
        fold_f1s.append(pair_classification_metrics(y[va_idx], y_prob=va_prob)["macro_f1"])
    train_time = time.perf_counter() - t0

    oof_pair = pair_classification_metrics(y, y_prob=oof_probs)
    eval_cand = train_df[["trip_id", "point_idx", "arc_id"]].copy()
    eval_cand["prob"] = oof_probs
    oof_top1 = observation_top1_accuracy(eval_cand, "prob", true_train_pts)

    summary = {
        "cv_macro_f1_mean": float(np.mean(fold_f1s)),
        "cv_macro_f1_std": float(np.std(fold_f1s)),
        "oof_top1": oof_top1["top1_accuracy"],
        "oof_log_loss": oof_pair["log_loss"],
        "cv_time_sec": float(train_time),
    }
    return oof_probs, summary
