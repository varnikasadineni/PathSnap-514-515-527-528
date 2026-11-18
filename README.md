# PathSnap — Distributed Map-Matching & Route Reconstruction

**PES University · UE24CS352A: Machine Learning Hackathon**  
**Team Members:** Vaishnavi (A — Data), Varnika (B — Models), Vibhav (C — HMM/Viterbi), Guru (D — API/UI/Report)

PathSnap is a two-laptop distributed machine learning + HMM/Viterbi system that reconstructs realistic vehicle trajectories by snapping noisy GPS observations onto a road network.

---

## 1. Repository Structure

```text
PathSnap-514-515-527-528/
├── pathsnap/              # Reusable Python package
│   ├── config.py          # Shared constants, seed=42, canonical feature list
│   ├── data.py            # [H1] Trip loader + per-trip local metric projection (ENU meters)
│   ├── candidates.py      # [H2] cKDTree adaptive candidate generation + monotonic route alignment
│   ├── features.py        # [H2] Leakage-free pairwise feature engineering
│   ├── metrics.py         # [H3] Macro F1, Top-1, Log Loss, Route Continuity, Sequence Route F1
│   ├── train.py           # [H4] 5+ models GroupKFold(5) CV + Optuna Bayesian HPO
│   ├── hmm.py             # [H5] Log-space HMM + Viterbi decoder with cached bounded Dijkstra
│   ├── pipeline.py        # [H5] predict_trip(trip_dir) end-to-end inference (never reads .route)
│   └── maps.py            # Interactive Folium map generator (GPS, road network, ref & pred routes)
├── api/
│   └── main.py            # Laptop 1: FastAPI backend + server-side evaluation + CMD request logs
├── ui/
│   └── app.py             # Laptop 2: Streamlit UI calling Laptop 1 over LAN
├── configs/
│   ├── paths.yaml         # Configurable train / student_test / evaluator_test folders
│   ├── model.yaml         # Final model metadata, feature schema, best hyperparameters
│   └── hmm.yaml           # Tuned HMM/Viterbi parameters & gap thresholds
├── notebooks/             # 01_eda.ipynb, 02_models.ipynb, 03_hmm_analysis.ipynb
├── models/                # Saved final_model.joblib (<50 MB)
├── results/               # Experiment CSVs (data_quality, model_comparison, optuna_trials, hmm_tuning)
├── reports/
│   ├── figures/           # All real plots for the PDF report
│   ├── maps/              # Exported interactive Folium HTML maps
│   └── REPORT_TEMPLATE.md # Consolidated report template (Deliverables.pdf Sections 2.2–8)
├── data/                  # Local raw & processed data (GITIGNORED — never committed)
├── requirements.txt
└── README.md
```

---

## 2. Team Branches & Hand-Off Contracts

| Member | Branch | Owns | Key Hand-off |
| :--- | :--- | :--- | :--- |
| **Member A — Vaishnavi (Data)** | `a-data` | `data.py`, `candidates.py`, `features.py`, `01_eda.ipynb` | **H1** `load_trip()`, **H2** `features_*.parquet` + `points_*.parquet` |
| **Member B — Varnika (Models)** | `b-models` | `metrics.py`, `train.py`, `02_models.ipynb`, Optuna HPO | **H3** `metrics.py`, **H4** `final_model.joblib` + `probs_*.parquet` |
| **Member C — Vibhav (HMM/Viz)** | `c-hmm` | `hmm.py`, `pipeline.py`, `maps.py`, `03_hmm_analysis.ipynb` | **H5** `predict_trip(trip_dir)` + Folium HTML maps |
| **Member D — Guru (App/Report)** | `d-app` | `api/main.py`, `ui/app.py`, two-laptop LAN, final PDF | Live 2-laptop deployment + consolidated PDF |

### Agreed `predict_trip(trip_dir)` Output Format (H5)
`predict_trip(trip_dir)` **never reads `.route`** and returns:
```json
{
  "trip_id": "00000000",
  "mode": "classifier+hmm",
  "runtime_sec": 1.24,
  "points": [{"point_idx": 0, "lon": 54.9635, "lat": 56.2515, "t": 1.0, "arc_id": 17630}],
  "predicted_arcs": [17630, 17630, 17633],
  "compressed_route": [17630, 17633],
  "predicted_arc_coords": [{"arc_id": 17630, "coords": [[lat1, lon1], [lat2, lon2]]}],
  "diagnostics": {"n_chains": 1, "n_gap_splits": 0, "n_disconnected_eval": 0}
}
```

### Coordinate Projection Note
Trips span global coordinates (`Lon [-100.4, 137.2]`, `Lat [-34.6, 61.7]`). `pathsnap/data.py` uses a **per-trip Local Tangent Plane (ENU)** centered at each trip's GPS centroid so all distances are accurate in meters worldwide (never use a single fixed regional CRS like `EPSG:27700`).

---

## 3. Installation & Full Reproduction Commands

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw partitions locally (gitignored)
#    data/raw/train/           (80 trips)
#    data/raw/student_test/    (10 trips)
#    data/raw/evaluator_test/  (10 trips - demo day only)
```

### Reproduce Every Result CSV, Figure & Map From Scratch
```powershell
# Stage 1 — Member A (EDA figures + H2 feature/point parquets + candidate miss rates)
python -m pathsnap.eda_figures
python -m pathsnap.build_dataset

# Stage 2 — Member B (6-model GroupKFold(5) CV + Optuna Bayesian HPO -> models/final_model.joblib)
python -m pathsnap.train --build --train-all --tune

# Stage 3 — Member C (Phase 1 baseline, Phase 2 OOF HMM tuning & Classifier+HMM comparison + Folium maps)
python -m pathsnap.eval_hmm --mode baseline
python -m pathsnap.eval_hmm --mode tune --tune-trips 20
python -m pathsnap.eval_hmm --mode classifier

# Stage 4 — Member D (End-to-end 10-trip FastAPI verification & zero-leakage check)
python tests/test_api.py
```

### Verified Final `student_test` Summary (10 Trips / 19,160 GPS Observations)
| Pipeline | Obs Top-1 | Route Continuity | Route Length F1 | Mean Runtime |
| :--- | :---: | :---: | :---: | :---: |
| Nearest-Arc Baseline | `56.93%` | `45.75%` | `63.73%` | — |
| Optuna-Tuned `HistGradientBoosting` Only | `83.45%` | `73.36%` | `84.90%` | `0.14 s` |
| **PathSnap Final (Tuned Classifier + Tuned HMM)** | **`91.74%`** | **`99.67%`** | **`97.07%`** | **`0.79 s`** |

---

## 4. Two-Laptop Distributed Deployment

### Laptop 1 — Model & FastAPI Backend
1. Put both laptops on the same Wi-Fi (or a mobile hotspot if college Wi-Fi blocks peer-to-peer traffic).
2. Find Laptop 1's LAN IP (`ipconfig` on Windows -> IPv4 Address, e.g. `192.168.1.42`).
3. Allow TCP port `8000` through Windows Firewall.
4. Start the API (point `PATHSNAP_EVAL_DIR` to the 10 evaluator trips on demo day):
```powershell
$env:PATHSNAP_EVAL_DIR="data/raw/evaluator_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Keep the CMD window visible so evaluators can inspect live inference logs.

### Laptop 2 — Evaluator-Facing UI
```powershell
$env:PATHSNAP_API_URL="http://192.168.1.42:8000"
streamlit run ui/app.py
```
Select any of the 10 evaluation trips in the browser UI, click **Reconstruct Route**, and inspect both the rendered route/metrics on Laptop 2 and the live API logs on Laptop 1.
