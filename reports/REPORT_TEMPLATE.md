# PathSnap: Distributed Map-Matching & Route Reconstruction
**PES University — UE24CS352A: Machine Learning Hackathon Technical Report**  
**Team:** Guru, Vibhav, Varnika, Vaishnavi  
**GitHub Repository:** https://github.com/varnikasadineni/PathSnap-514-515-527-528

> **Checklist before exporting PDF:** Every number in this report must match a CSV in `results/`, every plot must come from actual experiment outputs in `reports/figures/`, and every team member must be able to explain every section during the viva.

---

## 2.2 Dataset Understanding and Exploratory Data Analysis *(Owner: Member A)*

*Clearly state which partition (`train` = 80 trips, `student_test` = 10 trips; `evaluator_test` untouched) was used for each table/plot.*

### 1. How did we preprocess the data?
- Raw file structure (`.track`, `.nodes`, `.arcs`, `.route`) and row-count relationships.
- **Coordinate projection justification:** Why Euclidean degrees fail and why per-trip Local Tangent Plane (ENU meters) was used (since trips span global coordinates `Lon [-100.4, 137.2]`, `Lat [-34.6, 61.7]`).
- **Missing-value, timestamp, and anomaly analysis** (from `results/data_quality.csv`): missing files, bad rows, sampling intervals, large GPS gaps (e.g., trips `00000078`, `00000090`, `00000065`), off-route drift trips, road-graph connectivity & isolated nodes, rare `.route` discontinuities.
- **Reference route alignment & candidate generation:** How compressed `.route` sequences are monotonically aligned to 1 Hz `.track` observations; adaptive candidate radius study and **Candidate Miss Rate** table.

### 2. Patterns observed & feature engineering rationale
- Explain each engineered feature in `configs/model.yaml` and why it helps disambiguate parallel roads / reverse twin arcs.
- Leakage prevention statement (scalers/imputers fitted on training folds only; `GroupKFold(5)` by `trip_id`).

### Required Visuals
- Missing-value & GPS gap summary table / plot (`reports/figures/...`)
- GPS trajectory & road-network plots (`reports/figures/...`)
- Representative Folium EDA map screenshot (`reports/figures/...`, HTML committed in `reports/maps/`)

---

## 3. Model Experimentation and Comparison *(Owner: Member B)*

### 3.1 Overall Model Comparison Table *(from `results/model_comparison.csv`)*

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1-score | Log Loss | Training Time (s) | Inference Time (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model 1 (Logistic Regression) | Linear | | | | | | | | |
| Model 2 (Gaussian NB) | Probabilistic | | | | | | | | |
| Model 3 (KNN) | Instance-based | | | | | | | | |
| Model 4 (Random Forest) | Ensemble | | | | | | | | |
| Model 5 (LightGBM / HistGB) | Boosting | | | | | | | | |
| Model 6 (MLP) | Neural Network | | | | | | | | |

*State `N/A` and explain why if a metric does not apply. Explain the primary selection metric and why it fits PathSnap + HMM.*

### 3.2 Compulsory Analysis for EVERY Model
*(Duplicate this subsection for ALL 5+ models; tie every answer to actual experimental numbers, not textbook definitions.)*

#### Model X: `<Model Name>`
- **A. Model Understanding**
  1. Why did you consider this algorithm suitable for the PathSnap problem?
  2. What are the major strengths and limitations of this algorithm for this task?
- **B. Training and Configuration**
  5. What were the important hyperparameters and their initial values?
  6. What loss function or optimisation objective was used?
- **C. Experimental Results**
  7. What was the model's performance on the student-testing dataset (Accuracy, Precision, Recall, Macro F1, Top-1, Route Continuity)?
  8. What were the observation-level Top-1 Accuracy, Macro F1-score, and Log Loss on student-testing?
  9. Did you observe overfitting or underfitting? What quantitative evidence (Train/CV vs. Student-Test gap) supports your conclusion?

### 3.3 Model Visualisations *(real plots from `reports/figures/`)*
- Cross-validation Macro F1-score comparison across all models (with fold spread)
- Student-testing Top-1 Accuracy & Macro F1-score comparison
- Log Loss comparison
- Confusion matrices
- Epoch-wise train/validation loss curve (**ONLY** for models with epoch history such as MLP — do not fabricate curves for non-epoch models!)

---

## 4. Bayesian Hyperparameter Optimisation *(Owner: Member B)*

1. **Hyperparameters searched:**
2. **Ranges / distributions:**
3. **Optimisation objective (with `GroupKFold(5)` on 80 train trips):**
4. **Number of trials & computational cost (total wall-clock time, hardware):** *(from `results/optuna_trials.csv`)*
5. **Best hyperparameter configuration:**
6. **Did the tuned model outperform the untuned model?**

| Variant | CV Macro F1 | Student-Test Top-1 | Student-Test Macro F1 | Log Loss | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Untuned Selected Model | | | | | |
| Tuned Selected Model (Optuna) | | | | | |

*(Include Optuna optimization history & parameter importance plots from `reports/figures/`.)*

---

## 5. HMM and Viterbi Route Reconstruction *(Owner: Member C)*

1. **Hidden states:**
2. **Emission probabilities:**
3. **Transition probabilities (Newson–Krumm / bounded shortest-path on road graph):**
4. **Handling missing candidates & large GPS gaps:**
5. **Handling disconnected road segments:**
6. **Did HMM decoding improve the final route? (Quantitative evidence from `results/hmm_tuning.csv`):**

| Pipeline | Obs Top-1 Accuracy | Route Continuity | Route Length F1 | Route Arc F1 |
| :--- | :---: | :---: | :---: | :---: |
| Classifier-only | | | | |
| Classifier + HMM / Viterbi | | | | |

- **Failure / edge-case analysis:** Discuss 2–3 specific student-test trips where HMM helped most and cases where HMM did not help, and why.

---

## 6. Geographical Visualisation Using Folium *(Owner: Member C)*

- Screenshots of interactive Folium maps showing **Raw GPS observations**, **Local Road Network**, **Reference Route**, and **Predicted Route** with distinct colors, legend, and LayerControl.
- Reference paths to committed interactive HTML maps in `reports/maps/` and generation script `pathsnap/maps.py`.

---

## 7. API, UI and Two-Laptop Deployment *(Owner: Member D)*

- Architecture overview (Laptop 1 FastAPI + model/HMM/server-side eval vs. Laptop 2 Streamlit UI over LAN; zero ground-truth exposure to model or UI).
- **Screenshots of working UI** on Laptop 2 showing trip selection, map, and metrics.
- **Screenshots of Laptop 1 CMD terminal logs** showing live `GET /trips` and `POST /predict/{trip_id}` requests and responses.

---

## 8. Academic Integrity & AI-Assistance Disclosure *(Owner: All)*

- Summary of AI tools used (e.g., brainstorming, boilerplate scaffolding, debugging assistance) and confirmation that all experiments, training runs, metrics, plots, and evaluations were executed independently by the team on the provided dataset, with zero leakage of the 10 evaluator trips.
