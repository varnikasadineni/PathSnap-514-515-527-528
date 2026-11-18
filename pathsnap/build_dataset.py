"""Member A - build the Hand-off H2 files (candidates + labels + features) for train / student_test.

Run from the repo root:
    python -m pathsnap.build_dataset                     # both splits
    python -m pathsnap.build_dataset --splits train      # one split
    python -m pathsnap.build_dataset --max-trips 3       # quick test on 3 trips per split

Writes
    data/processed/features_<split>.parquet   one row per (GPS point, candidate arc): ids + 12 features + label
    data/processed/points_<split>.parquet     one row per GPS point: true_arc (-1 when missed), ...
    results/candidate_miss_rate.csv           per-trip miss rates
    results/candidate_miss_summary.csv        pooled miss rates (all trips / train / student_test / without anomalies)

Candidate settings are read from configs/hmm.yaml (candidate_generation) so that the training features are
built EXACTLY like the ones predict_trip() builds at demo time. Change them there (and tell Member C), not here.
The .route file is used ONLY to make the labels; no feature column uses it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pathsnap.candidates import align_track_to_route_monotonic, generate_candidates_for_trip
from pathsnap.config import FEATURE_COLUMNS, ID_COLUMNS, LABEL_COLUMN, REPO_ROOT, load_yaml
from pathsnap.data import list_trip_dirs, load_trip
from pathsnap.features import build_features_for_trip

FIXED_RADII_M = (20, 30, 40, 50, 75, 100, 150)
# trips whose reference route sits 120-140 m away from the GPS (see results/data_quality.csv, issue "off_route")
ANOMALOUS_TRIPS_NOTE = "off_route"


def _split_dir(cfg: dict, split: str) -> Path:
    key = {"train": "train_dir", "student_test": "student_test_dir"}[split]
    p = REPO_ROOT / cfg["data"][key]
    if not p.exists():  # also accept data/raw/Student folder/<split>
        alt = REPO_ROOT / "data" / "raw" / "Student folder" / split
        if alt.exists():
            return alt
    return p


def _feature_schema() -> pa.Schema:
    fields = [pa.field("trip_id", pa.string()), pa.field("point_idx", pa.int32()),
              pa.field("t", pa.float64()), pa.field("arc_id", pa.int32())]
    fields += [pa.field(c, pa.float32()) for c in FEATURE_COLUMNS]
    fields += [pa.field(LABEL_COLUMN, pa.int8())]
    return pa.schema(fields)


def _to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    out = {}
    for f in schema:
        out[f.name] = pa.array(df[f.name].to_numpy(), type=f.type)
    return pa.Table.from_pydict(out, schema=schema)


def build_split(split: str, cfg_paths: dict, cand_cfg: dict, out_dir: Path, max_trips: int | None):
    trip_dirs = list_trip_dirs(_split_dir(cfg_paths, split))
    if max_trips:
        trip_dirs = trip_dirs[:max_trips]
    if not trip_dirs:
        raise SystemExit(f"No trips found for split '{split}'. Check data/raw/{split}.")

    schema = _feature_schema()
    feat_writer = pq.ParquetWriter(out_dir / f"features_{split}.parquet", schema, compression="snappy")
    point_frames, miss_rows = [], []
    n = len(trip_dirs)
    try:
        for k, d in enumerate(trip_dirs, 1):
            t0 = time.time()
            trip = load_trip(d)
            cand = generate_candidates_for_trip(
                trip,
                base_radius_m=float(cand_cfg.get("base_radius_m", 50.0)),
                fallback_radii_m=tuple(cand_cfg.get("fallback_radii_m", (100.0, 200.0, 350.0))),
                min_candidates=int(cand_cfg.get("min_candidates", 5)),
                max_candidates=int(cand_cfg.get("max_candidates", 15)),
            )
            feat = build_features_for_trip(trip, cand)               # uses track / nodes / arcs only
            ali = align_track_to_route_monotonic(trip)               # uses .route -> labels only

            true_by_pt = ali.set_index("point_idx")["true_arc"]
            feat[LABEL_COLUMN] = (feat["arc_id"].to_numpy() == true_by_pt.reindex(feat["point_idx"]).to_numpy()).astype(np.int8)
            feat_writer.write_table(_to_table(feat, schema))

            # ---- point table (H2): true_arc = -1 when the true arc is not among the candidates
            hit = feat.groupby("point_idx")[LABEL_COLUMN].max().reindex(ali["point_idx"].to_numpy(), fill_value=0).to_numpy() == 1
            n_cand = feat.groupby("point_idx").size().reindex(ali["point_idx"].to_numpy(), fill_value=0).to_numpy()
            pts = pd.DataFrame({
                "trip_id": trip.trip_id,
                "point_idx": ali["point_idx"].to_numpy(np.int32),
                "t": trip.track["t"].to_numpy(),
                "lon": trip.track["lon"].to_numpy(),
                "lat": trip.track["lat"].to_numpy(),
                "true_arc": np.where(hit, ali["true_arc"].to_numpy(), -1).astype(np.int32),
                "true_arc_ref": ali["true_arc"].to_numpy(np.int32),     # reference arc even when missed
                "dist_to_route_m": ali["dist_to_route_m"].to_numpy(np.float32),
                "n_candidates": n_cand.astype(np.int16),
                "missed": ~hit,
            })
            point_frames.append(pts)

            dist = ali["dist_to_route_m"].to_numpy()
            row = {"partition": split, "trip_id": trip.trip_id, "n_points": len(pts), "n_candidate_rows": len(feat),
                   "cands_per_point": round(len(feat) / len(pts), 2), "n_missed": int((~hit).sum()),
                   "miss_pct_actual": round(100 * float((~hit).mean()), 3)}
            for r in FIXED_RADII_M:
                row[f"miss_pct_r{r}m"] = round(100 * float((dist > r).mean()), 3)
            miss_rows.append(row)
            print(f"[{k:3d}/{n}] {split:12s} {trip.trip_id}  {len(pts):6d} pts  {len(feat)/len(pts):5.1f} cand/pt  "
                  f"miss {row['miss_pct_actual']:6.2f}%  {time.time() - t0:4.1f}s", flush=True)
    finally:
        feat_writer.close()

    points = pd.concat(point_frames, ignore_index=True)
    points.to_parquet(out_dir / f"points_{split}.parquet", index=False)
    return pd.DataFrame(miss_rows)


def summarise(miss: pd.DataFrame, anomalous: list[str]) -> pd.DataFrame:
    cols = ["miss_pct_actual"] + [f"miss_pct_r{r}m" for r in FIXED_RADII_M]
    def pooled(df: pd.DataFrame, scope: str) -> dict:
        w = df["n_points"].to_numpy()
        out = {"scope": scope, "trips": len(df), "points": int(w.sum()), "candidate_rows": int(df["n_candidate_rows"].sum())}
        for c in cols:
            out[c] = round(float((df[c] * w).sum() / w.sum()), 3)
        return out
    rows = [pooled(miss, "all trips")]
    for part, g in miss.groupby("partition"):
        rows.append(pooled(g, part))
    ok = miss[~miss["trip_id"].isin(anomalous)]
    if len(ok) and len(ok) < len(miss):
        rows.append(pooled(ok, f"all trips except off-route trips {', '.join(sorted(anomalous))}"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "student_test"], choices=["train", "student_test"])
    ap.add_argument("--max-trips", type=int, default=None, help="only the first N trips per split (quick test)")
    args = ap.parse_args()

    cfg_paths = load_yaml("configs/paths.yaml")
    cand_cfg = load_yaml("configs/hmm.yaml").get("candidate_generation", {})
    out_dir = REPO_ROOT / cfg_paths["data"]["processed_dir"]
    res_dir = REPO_ROOT / cfg_paths["artifacts"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)
    print("candidate settings (configs/hmm.yaml):", cand_cfg)

    t0 = time.time()
    miss = pd.concat([build_split(s, cfg_paths, cand_cfg, out_dir, args.max_trips) for s in args.splits], ignore_index=True)

    # anomalous = trips flagged "off_route" by the EDA notebook (if results/data_quality.csv exists)
    dq_path = res_dir / "data_quality.csv"
    anomalous = []
    if dq_path.exists():
        dq = pd.read_csv(dq_path, dtype={"trip_id": str})
        anomalous = sorted(set(dq.loc[dq["issue"] == ANOMALOUS_TRIPS_NOTE, "trip_id"]) & set(miss["trip_id"]))
    miss.to_csv(res_dir / "candidate_miss_rate.csv", index=False)
    summ = summarise(miss, anomalous)
    summ.to_csv(res_dir / "candidate_miss_summary.csv", index=False)

    print(f"\nDone in {time.time() - t0:.0f}s.")
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print("\nFiles in", out_dir)
    for p in sorted(out_dir.glob("*.parquet")):
        print(f"  {p.name:34s} {p.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
