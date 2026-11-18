"""Hand-off H4 (Member B): Model experimentation (5+ models) & Optuna Bayesian HPO.

Member B owns this module:
- Train & evaluate >= 5 distinct models across families with GroupKFold(5) by trip_id
- Record ALL required metrics into results/model_comparison.csv
- Run Optuna Bayesian HPO on the selected model -> results/optuna_trials.csv
- Export models/final_model.joblib + configs/model.yaml + out-of-fold & student-test probs

Usage (from the repo root):
    python -m pathsnap.train build                        # cache features_*.parquet + points_*.parquet
    python -m pathsnap.train cv                           # all models: CV + student-test + plots
    python -m pathsnap.train cv --models LogisticRegression,HistGradientBoosting
    python -m pathsnap.train tune --n-trials 50           # Optuna on HistGradientBoosting
    python -m pathsnap.train tune --model RandomForest --n-trials 30
Options: --data-root data/raw (folder holding train/ and student_test/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.config import (CV_N_SPLITS, FEATURE_COLUMNS, LABEL_COLUMN, RANDOM_SEED,
                             REPO_ROOT, load_yaml)
from pathsnap.metrics import (observation_top1_accuracy, pair_classification_metrics,
                              route_continuity, sequence_route_metrics)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
PROBS_DIR = RESULTS_DIR / "probs"
MODELS_DIR = REPO_ROOT / "models"
FIG_DIR = REPO_ROOT / "reports" / "figures"

# Points further than this from their own reference route get true_arc = -1
# (unalignable). Trips 00000033 / 00000054 sit 120-140 m off their route.
MAX_ALIGN_DIST_M = 60.0
# Training trips whose GPS does not follow their reference route; excluded from FITTING.
EXCLUDE_TRAIN_TRIPS = ["00000033", "00000054"]
# KNN prediction cost grows with training size; fit it on a fixed random subsample.
KNN_MAX_TRAIN_ROWS = 150_000


# --------------------------------------------------------------------------- models
def get_candidate_models(seed: int = RANDOM_SEED) -> dict:
    """Return >= 5 distinct models spanning diverse algorithm families."""
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
                max_samples=0.5,
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


def _fit(name: str, model, X: np.ndarray, y: np.ndarray):
    if name == "KNN" and len(X) > KNN_MAX_TRAIN_ROWS:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(X), KNN_MAX_TRAIN_ROWS, replace=False)
        X, y = X[idx], y[idx]
    return model.fit(X, y)


# --------------------------------------------------------------------------- dataset
def _build_trip(trip_dir: Path):
    from pathsnap.candidates import generate_candidates_for_trip
    try:  # DP alignment (robust to gaps / route revisits); fall back to the windowed version
        from pathsnap.candidates import align_track_to_route_dp as align
    except ImportError:
        from pathsnap.candidates import align_track_to_route_monotonic as align
    from pathsnap.data import load_trip
    from pathsnap.features import build_features_for_trip

    trip = load_trip(trip_dir, load_route=True)
    cand = generate_candidates_for_trip(trip)
    feats = build_features_for_trip(trip, cand)
    pts = align(trip)
    pts.loc[pts["dist_to_route_m"] > MAX_ALIGN_DIST_M, "true_arc"] = -1
    truth = pts.set_index("point_idx")["true_arc"]
    feats[LABEL_COLUMN] = (feats["arc_id"].to_numpy() == truth.reindex(feats["point_idx"]).to_numpy()).astype(np.int8)
    return feats, pts


def build_split(data_root: Path, split: str, n_jobs: int = -1) -> None:
    """Cache features_<split>.parquet and points_<split>.parquet (H2 format) from Member A's code."""
    from joblib import Parallel, delayed
    from pathsnap.data import list_trip_dirs

    dirs = list_trip_dirs(Path(data_root) / split)
    if not dirs:
        raise SystemExit(f"No trips found in {Path(data_root) / split}")
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_build_trip)(d) for d in dirs)
    feats = pd.concat([o[0] for o in out], ignore_index=True)
    pts = pd.concat([o[1] for o in out], ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(PROCESSED_DIR / f"features_{split}.parquet", index=False)
    pts.to_parquet(PROCESSED_DIR / f"points_{split}.parquet", index=False)
    hit = feats.loc[feats[LABEL_COLUMN] == 1, ["trip_id", "point_idx"]].drop_duplicates()
    valid = pts[pts["true_arc"] >= 0]
    print(f"{split}: {len(dirs)} trips, {len(feats):,} pairs, {len(pts):,} points, "
          f"unalignable {1 - len(valid) / len(pts):.2%}, candidate miss {1 - len(hit) / max(len(valid), 1):.2%}, "
          f"{time.perf_counter() - t0:.0f}s")


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = PROCESSED_DIR / f"features_{split}.parquet"
    p = PROCESSED_DIR / f"points_{split}.parquet"
    if not f.exists():
        raise SystemExit(f"{f} missing - run `python -m pathsnap.train build` first (or get it from Member A).")
    feats, pts = pd.read_parquet(f), pd.read_parquet(p)
    feats["trip_id"] = feats["trip_id"].astype(str)
    pts["trip_id"] = pts["trip_id"].astype(str)
    return feats, pts


def load_train_test():
    tr, tr_pts = load_split("train")
    tr = tr[~tr["trip_id"].isin(EXCLUDE_TRAIN_TRIPS)].reset_index(drop=True)
    tr_pts = tr_pts[~tr_pts["trip_id"].isin(EXCLUDE_TRAIN_TRIPS)].reset_index(drop=True)
    # Unalignable points (> MAX_ALIGN_DIST_M from route) have no trustworthy label: drop them from fitting.
    ok = tr_pts.loc[tr_pts["true_arc"] >= 0, ["trip_id", "point_idx"]]
    tr = tr.merge(ok, on=["trip_id", "point_idx"], how="inner")
    te, te_pts = load_split("student_test")
    return tr, tr_pts, te, te_pts


# --------------------------------------------------------------------------- evaluation
def run_grouped_cv_skeleton(
    train_df: pd.DataFrame,
    true_train_pts: pd.DataFrame,
    model,
    name: str = "",
) -> tuple[np.ndarray, dict]:
    """Run GroupKFold(5) by trip_id on the training trips, returning OOF probabilities.

    summary["folds"] holds per-fold Macro F1, Top-1, log loss and fit time.
    """
    from sklearn.base import clone

    X = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = train_df[LABEL_COLUMN].to_numpy(dtype=np.int32)
    groups = train_df["trip_id"].to_numpy()

    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    oof_probs = np.zeros(len(train_df), dtype=np.float64)
    folds = []

    t0 = time.perf_counter()
    for k, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        est = clone(model)
        t1 = time.perf_counter()
        _fit(name, est, X[tr_idx], y[tr_idx])
        fit_s = time.perf_counter() - t1
        va_prob = est.predict_proba(X[va_idx])[:, 1]
        oof_probs[va_idx] = va_prob
        pm = pair_classification_metrics(y[va_idx], y_prob=va_prob)
        va = train_df.iloc[va_idx][["trip_id", "point_idx", "arc_id"]].assign(prob=va_prob)
        top = observation_top1_accuracy(va, "prob", true_train_pts[true_train_pts["trip_id"].isin(set(groups[va_idx]))])
        folds.append(dict(model=name, fold=k, macro_f1=pm["macro_f1"], log_loss=pm["log_loss"],
                          top1=top["top1_accuracy"], fit_sec=fit_s))
        print(f"   fold {k}: macroF1 {pm['macro_f1']:.4f}  top1 {top['top1_accuracy']:.4f}  ({fit_s:.0f}s)")
    train_time = time.perf_counter() - t0

    oof_pair = pair_classification_metrics(y, y_prob=oof_probs)
    eval_cand = train_df[["trip_id", "point_idx", "arc_id"]].copy()
    eval_cand["prob"] = oof_probs
    oof_top1 = observation_top1_accuracy(eval_cand, "prob", true_train_pts)

    f1s = [f["macro_f1"] for f in folds]
    summary = {
        "cv_macro_f1_mean": float(np.mean(f1s)),
        "cv_macro_f1_std": float(np.std(f1s)),
        "cv_top1_mean": float(np.mean([f["top1"] for f in folds])),
        "oof_top1": oof_top1["top1_accuracy"],
        "oof_log_loss": oof_pair["log_loss"],
        "cv_time_sec": float(train_time),
        "folds": folds,
    }
    return oof_probs, summary


def _top1_frame(cand: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    df = cand[["trip_id", "point_idx", "arc_id"]].assign(prob=prob)
    idx = df.groupby(["trip_id", "point_idx"], sort=False)["prob"].idxmax()
    return df.loc[idx].sort_values(["trip_id", "point_idx"])


def evaluate_student_test(test_df, test_pts, prob, data_root: Path) -> dict:
    """Pair metrics + Top-1 + classifier-only route metrics on the student-test trips."""
    from pathsnap.data import load_trip

    pm = pair_classification_metrics(test_df[LABEL_COLUMN].to_numpy(), y_prob=prob)
    top = observation_top1_accuracy(test_df[["trip_id", "point_idx", "arc_id"]].assign(prob=prob), "prob", test_pts)
    seq = []
    for trip_id, g in _top1_frame(test_df, prob).groupby("trip_id"):
        trip = load_trip(Path(data_root) / "student_test" / trip_id, load_route=True)
        seq.append(sequence_route_metrics(g["arc_id"].to_numpy(), trip.route, trip.arcs))
    seq = pd.DataFrame(seq)
    return {
        "test_top1_acc": top["top1_accuracy"],
        "test_candidate_miss_rate": top["candidate_miss_rate"],
        "test_accuracy": pm["accuracy"],
        "test_precision": pm["precision"],
        "test_recall": pm["recall"],
        "test_macro_f1": pm["macro_f1"],
        "test_log_loss": pm["log_loss"],
        "test_confusion_matrix": json.dumps(pm["confusion_matrix"]),
        "test_route_continuity": float(seq["route_continuity"].mean()),
        "test_route_length_f1": float(seq["route_length_f1"].mean()),
    }


def _save_probs(name: str, train_df, oof, test_df, ptest):
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["trip_id", "point_idx", "arc_id", LABEL_COLUMN]
    train_df[keys].assign(prob=oof).to_parquet(PROBS_DIR / f"{name}_train_oof.parquet", index=False)
    test_df[keys].assign(prob=ptest).to_parquet(PROBS_DIR / f"{name}_student_test.parquet", index=False)


def run_model(name, family, model, tr, tr_pts, te, te_pts, data_root):
    print(f"== {name}")
    oof, cv = run_grouped_cv_skeleton(tr, tr_pts, model, name)
    from sklearn.base import clone
    final = clone(model)
    X = tr[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = tr[LABEL_COLUMN].to_numpy(dtype=np.int32)
    t0 = time.perf_counter(); _fit(name, final, X, y); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); ptest = final.predict_proba(te[FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1]
    infer_s = time.perf_counter() - t0
    test = evaluate_student_test(te, te_pts, ptest, data_root)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / f"{name}.joblib")
    _save_probs(name, tr, oof, te, ptest)
    est = final[-1] if hasattr(final, "steps") else final
    if hasattr(est, "loss_curve_"):
        json.dump({"train_loss": list(map(float, est.loss_curve_)),
                   "val_score": list(map(float, getattr(est, "validation_scores_", None) or []))},
                  open(RESULTS_DIR / f"{name}_history.json", "w"))

    row = {"model": name, "family": family, "config": str(est.get_params()),
           **{k: v for k, v in cv.items() if k != "folds"}, **test,
           "train_time_sec": fit_s, "inference_time_sec": infer_s,
           "inference_ms_per_point": 1000 * infer_s / max(len(te_pts), 1)}
    print(f"   student-test top1 {test['test_top1_acc']:.4f}  macroF1 {test['test_macro_f1']:.4f}  "
          f"logloss {test['test_log_loss']:.4f}  continuity {test['test_route_continuity']:.3f}")
    return row, cv["folds"]


def _upsert(path: Path, new: pd.DataFrame, key: str = "model") -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old[key].isin(new[key].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


# --------------------------------------------------------------------------- plots
def make_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    folds = pd.read_csv(RESULTS_DIR / "cv_folds.csv")
    order = comp.sort_values("test_top1_acc", ascending=False)["model"].tolist()
    c = comp.set_index("model").loc[order]
    short = [m.replace("HistGradientBoosting", "HistGB").replace("LogisticRegression", "LogReg") for m in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([folds[folds.model == m].macro_f1 for m in order], tick_labels=short)
    ax.set_ylabel("Macro F1 per fold"); ax.set_title("Cross-validation Macro F1 (GroupKFold(5) by trip)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cv_macro_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, c.test_top1_acc, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, c.test_macro_f1, w, label="Macro F1")
    ax.set_xticks(x, short); ax.set_ylim(0, 1); ax.legend(loc="lower right")
    ax.set_title("Student-test Top-1 accuracy and Macro F1")
    fig.tight_layout(); fig.savefig(FIG_DIR / "student_test_top1_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(short, c.test_log_loss); ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Student-test log loss")
    fig.tight_layout(); fig.savefig(FIG_DIR / "student_test_log_loss.png", dpi=150); plt.close(fig)

    for m in order:
        cm = np.array(json.loads(c.loc[m, "test_confusion_matrix"]))
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{m}: student-test")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"confusion_{m}.png", dpi=150); plt.close(fig)

    for h in RESULTS_DIR.glob("*_history.json"):
        cv = json.load(open(h)); name = h.stem.replace("_history", "")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cv["train_loss"], label="training loss"); ax.set_xlabel("epoch"); ax.set_ylabel("log loss")
        if cv["val_score"]:
            ax2 = ax.twinx(); ax2.plot(cv["val_score"], color="tab:orange"); ax2.set_ylabel("validation accuracy (orange)")
        ax.set_title(f"{name}: training history")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"{name}_loss_curve.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- Optuna
def search_space(trial, model_name: str) -> dict:
    if model_name == "HistGradientBoosting":
        return dict(
            max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
        )
    if model_name == "RandomForest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            max_depth=trial.suggest_int("max_depth", 6, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            max_samples=trial.suggest_float("max_samples", 0.2, 0.8),
        )
    if model_name == "MLP":
        return dict(
            clf__hidden_layer_sizes=trial.suggest_categorical(
                "hidden_layer_sizes", ["32", "64", "64-32", "128-64", "128-64-32"]),
            clf__alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            clf__learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            clf__batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
        )
    raise SystemExit(f"No search space defined for {model_name}; add one in search_space().")


def _apply_params(model, params: dict):
    from sklearn.base import clone
    m = clone(model)
    p = dict(params)
    if "clf__hidden_layer_sizes" in p:
        p["clf__hidden_layer_sizes"] = tuple(int(v) for v in str(p["clf__hidden_layer_sizes"]).split("-"))
    return m.set_params(**p)


def run_optuna(model_name: str, n_trials: int, n_folds: int, data_root: Path) -> None:
    import optuna

    tr, tr_pts, te, te_pts = load_train_test()
    family, base = get_candidate_models()[model_name]
    X = tr[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = tr[LABEL_COLUMN].to_numpy(dtype=np.int32)
    g = tr["trip_id"].to_numpy()
    splits = list(GroupKFold(n_splits=n_folds).split(X, y, g))

    def objective(trial):
        model = _apply_params(base, search_space(trial, model_name))
        scores = []
        for k, (tr_i, va_i) in enumerate(splits):
            m = _fit(model_name, model, X[tr_i], y[tr_i])
            prob = m.predict_proba(X[va_i])[:, 1]
            va = tr.iloc[va_i][["trip_id", "point_idx", "arc_id"]].assign(prob=prob)
            scores.append(observation_top1_accuracy(va, "prob", tr_pts[tr_pts["trip_id"].isin(set(g[va_i]))])["top1_accuracy"])
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name=f"{model_name}_top1",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    search_s = time.perf_counter() - t0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(RESULTS_DIR / "optuna_trials.csv", index=False)
    best = study.best_params
    best_sk = search_space(optuna.trial.FixedTrial(best), model_name)
    json.dump(dict(model=model_name, best_params=best, best_cv_top1=study.best_value,
                   n_trials=len(study.trials),
                   n_complete=sum(t.state.name == "COMPLETE" for t in study.trials),
                   n_pruned=sum(t.state.name == "PRUNED" for t in study.trials),
                   search_time_sec=search_s, objective=f"mean GroupKFold({n_folds}) Top-1 on training trips",
                   sampler="TPE (seed 42)", pruner="MedianPruner"),
              open(RESULTS_DIR / "optuna_best_params.json", "w"), indent=2)
    print(f"best CV Top-1 {study.best_value:.4f}: {best}  ({search_s / 60:.1f} min)")

    rows = []
    for label, model in [("untuned", base), ("tuned", _apply_params(base, best_sk))]:
        name = f"{model_name}_{label}"
        oof, cv = run_grouped_cv_skeleton(tr, tr_pts, model, model_name)
        from sklearn.base import clone
        final = clone(model)
        t1 = time.perf_counter(); _fit(model_name, final, X, y); fit_s = time.perf_counter() - t1
        t1 = time.perf_counter(); ptest = final.predict_proba(te[FEATURE_COLUMNS].to_numpy(dtype=np.float32))[:, 1]
        infer_s = time.perf_counter() - t1
        test = evaluate_student_test(te, te_pts, ptest, data_root)
        rows.append({"model": name, "cv_macro_f1_mean": cv["cv_macro_f1_mean"], "cv_top1_mean": cv["cv_top1_mean"],
                     "oof_log_loss": cv["oof_log_loss"], **test,
                     "train_time_sec": fit_s, "inference_time_sec": infer_s})
        if label == "tuned":
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(final, MODELS_DIR / "final_model.joblib")
            keys = ["trip_id", "point_idx", "arc_id", LABEL_COLUMN]
            tr[keys].assign(prob=oof).to_parquet(RESULTS_DIR / "probs_train_oof.parquet", index=False)
            te[keys].assign(prob=ptest).to_parquet(RESULTS_DIR / "probs_student_test.parquet", index=False)
    cmp = pd.DataFrame(rows)
    cmp.to_csv(RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
    print(cmp[["model", "cv_macro_f1_mean", "test_top1_acc", "test_macro_f1", "test_log_loss",
               "inference_time_sec"]].round(4).to_string(index=False))

    # configs/model.yaml: keep any keys teammates already rely on, update ours
    import yaml
    cfg_path = REPO_ROOT / "configs" / "model.yaml"
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    cfg.update({
        "model_name": model_name,
        "model_family": family,
        "model_path": "models/final_model.joblib",
        "feature_columns": list(FEATURE_COLUMNS),
        "label_column": LABEL_COLUMN,
        "random_seed": RANDOM_SEED,
        "best_params": {k: (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in best.items()},
        "best_cv_top1": float(study.best_value),
        "excluded_train_trips": EXCLUDE_TRAIN_TRIPS,
        "oof_probs_path": "results/probs_train_oof.parquet",
        "student_test_probs_path": "results/probs_student_test.parquet",
    })
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe(); done = df[df.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done.number, done.value, s=14, label="completed trial")
    ax.plot(done.number, done.value.cummax(), color="tab:orange", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 accuracy"); ax.legend()
    ax.set_title(f"Optuna TPE search: {model_name}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "optuna_history.png", dpi=150); plt.close(fig)
    if len(done) >= 3:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_xlabel("importance (fANOVA)")
        ax.set_title("Hyperparameter importance")
        fig.tight_layout(); fig.savefig(FIG_DIR / "optuna_importance.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "cv", "tune", "plots"])
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--models", default="all", help="comma-separated names for `cv`")
    ap.add_argument("--model", default="HistGradientBoosting", help="model to tune")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--folds", type=int, default=CV_N_SPLITS)
    ap.add_argument("--splits", default="train,student_test", help="for `build`")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel trips for `build`")
    a = ap.parse_args()
    data_root = Path(a.data_root)

    if a.command == "build":
        for s in a.splits.split(","):
            build_split(data_root, s, a.n_jobs)
    elif a.command == "cv":
        tr, tr_pts, te, te_pts = load_train_test()
        print(f"train: {tr.trip_id.nunique()} trips / {len(tr):,} pairs (excluded {EXCLUDE_TRAIN_TRIPS}); "
              f"student-test: {te.trip_id.nunique()} trips / {len(te):,} pairs")
        models = get_candidate_models()
        names = list(models) if a.models == "all" else a.models.split(",")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for n in names:
            family, model = models[n]
            row, folds = run_model(n, family, model, tr, tr_pts, te, te_pts, data_root)
            comp = _upsert(RESULTS_DIR / "model_comparison.csv", pd.DataFrame([row]))
            _upsert(RESULTS_DIR / "cv_folds.csv", pd.DataFrame(folds))
        json.dump({n: dict(family=f, params=str((m[-1] if hasattr(m, "steps") else m).get_params()))
                   for n, (f, m) in models.items()} |
                  {"_features": FEATURE_COLUMNS, "_seed": RANDOM_SEED, "_excluded_train_trips": EXCLUDE_TRAIN_TRIPS,
                   "_max_align_dist_m": MAX_ALIGN_DIST_M, "_knn_max_train_rows": KNN_MAX_TRAIN_ROWS},
                  open(RESULTS_DIR / "model_configs.json", "w"), indent=2)
        make_plots()
        print(comp[["model", "cv_macro_f1_mean", "test_top1_acc", "test_macro_f1", "test_log_loss",
                    "test_route_continuity", "inference_time_sec"]].round(4).to_string(index=False))
    elif a.command == "tune":
        run_optuna(a.model, a.n_trials, a.folds, data_root)
    elif a.command == "plots":
        make_plots()


if __name__ == "__main__":
    main()
