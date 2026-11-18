"""Member D (Laptop 2): Evaluator-facing Streamlit UI for PathSnap.

Run on Laptop 2 (or Terminal 2 locally during development):
    streamlit run ui/app.py

Connects over HTTP/LAN to Laptop 1's FastAPI backend (configurable in sidebar
or via PATHSNAP_API_URL env var). Displays ONLY the 10 evaluation trip options,
raw GPS observations, predicted route, and server-computed metrics — ground-truth
reference routes never leave Laptop 1.
"""

from __future__ import annotations

import os
import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(
    page_title="PathSnap — Distributed Map-Matching",
    page_icon="🛣️",
    layout="wide",
)

st.title("PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption("Two-Laptop Architecture · Laptop 2 Evaluator UI → Laptop 1 FastAPI + ML Classifier + HMM/Viterbi")

if "history" not in st.session_state:
    st.session_state["history"] = []

default_api = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")

with st.sidebar:
    st.header("Laptop 1 API Connection")
    api_url = st.text_input("FastAPI Base URL", value=default_api).strip().rstrip("/")
    st.caption("Local dev: `http://127.0.0.1:8000`  \nTwo-laptop LAN: `http://<Laptop1-IP>:8000`")

    health_info = None
    try:
        h_resp = requests.get(f"{api_url}/health", timeout=4)
        h_resp.raise_for_status()
        health_info = h_resp.json()
        st.success(f"Connected ({health_info.get('n_trips', 0)} trips ready)")
        mode_badge = health_info.get("mode", "unknown")
        if mode_badge == "classifier+hmm":
            st.info("Pipeline Mode: **Classifier + HMM/Viterbi**")
        else:
            st.warning("Pipeline Mode: **Distance-HMM Fallback** *(waiting for final_model.joblib)*")
    except Exception as exc:
        st.error(f"API Unreachable: {exc}")

    if st.button("Refresh Connection & Trips", width="stretch"):
        st.rerun()

# Fetch the 10 evaluation trip IDs dynamically from Laptop 1
try:
    t_resp = requests.get(f"{api_url}/trips", timeout=5)
    t_resp.raise_for_status()
    trips = t_resp.json().get("trips", [])
except Exception as exc:
    st.error(f"Could not load trip list from `{api_url}/trips`: {exc}")
    st.info("Make sure Laptop 1 is running: `uvicorn api.main:app --host 0.0.0.0 --port 8000`")
    st.stop()

if not trips:
    st.warning("Laptop 1 returned 0 trips. Check `PATHSNAP_EVAL_DIR` on Laptop 1.")
    st.stop()

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    selected_trip = st.selectbox(
        f"Select Evaluation Trip ({len(trips)} available)",
        options=trips,
        index=0,
    )
with col_btn:
    st.write("")
    st.write("")
    run_clicked = st.button("Reconstruct Route", type="primary", width="stretch")

if run_clicked:
    with st.spinner(f"Sending trip `{selected_trip}` to Laptop 1 API (`{api_url}/predict/{selected_trip}`)..."):
        try:
            pred_resp = requests.post(f"{api_url}/predict/{selected_trip}", timeout=180)
            pred_resp.raise_for_status()
            payload = pred_resp.json()
            st.session_state["last_result"] = payload

            m = payload.get("metrics", {})
            rec = {
                "Trip ID": payload["trip_id"],
                "Mode": payload["mode"],
                "GPS Points": payload["n_points"],
                "Predicted Arcs": payload["n_predicted_arcs"],
                "Obs Top-1 (%)": round(m["observation_top1_accuracy"] * 100, 2) if m.get("observation_top1_accuracy") is not None else None,
                "Route Len F1 (%)": round(m["route_length_f1"] * 100, 2) if m.get("route_length_f1") is not None else None,
                "Route Continuity (%)": round(m["route_continuity"] * 100, 2) if m.get("route_continuity") is not None else None,
                "Runtime (s)": payload["runtime_sec"],
            }
            # Replace existing entry for same trip_id if re-run
            st.session_state["history"] = [
                h for h in st.session_state["history"] if h["Trip ID"] != rec["Trip ID"]
            ] + [rec]
        except Exception as exc:
            st.error(f"Inference request failed: {exc}")

data = st.session_state.get("last_result")
if data:
    st.divider()
    st.subheader(f"Trip `{data['trip_id']}` — Reconstructed Route & Server-Side Evaluation")
    metrics = data.get("metrics", {})

    def _fmt_pct(val):
        return f"{val * 100:.2f}%" if val is not None else "N/A"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Obs Top-1 Accuracy", _fmt_pct(metrics.get("observation_top1_accuracy")))
    c2.metric("Route Length F1", _fmt_pct(metrics.get("route_length_f1")))
    c3.metric("Route Arc F1", _fmt_pct(metrics.get("route_arc_f1")))
    c4.metric("Route Continuity", _fmt_pct(metrics.get("route_continuity")))
    c5.metric("Points on Route", _fmt_pct(metrics.get("point_on_route_rate")))
    c6.metric("API Runtime", f"{data['runtime_sec']:.2f} s")

    # Build interactive OpenStreetMap Folium map (Raw GPS + Predicted Route ONLY — no ground truth!)
    gps_pts = data.get("gps_points", [])
    pred_arcs = data.get("predicted_arc_coords", [])

    if gps_pts:
        lats = [p[0] for p in gps_pts]
        lons = [p[1] for p in gps_pts]
        fmap = folium.Map(
            location=[sum(lats) / len(lats), sum(lons) / len(lons)],
            zoom_start=15,
            tiles="OpenStreetMap",
        )

        gps_layer = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
        stride = max(1, len(gps_pts) // 1200)
        for lat, lon in gps_pts[::stride]:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2.2,
                color="#1565c0",
                fill=True,
                fill_opacity=0.65,
            ).add_to(gps_layer)
        gps_layer.add_to(fmap)

        # Draw predicted route ON TOP of raw GPS dots so it pops at every zoom level
        pred_layer = folium.FeatureGroup(name="Predicted Route (PathSnap)", show=True)
        for seg in pred_arcs:
            folium.PolyLine(
                locations=seg["coords"],
                color="#d32f2f",
                weight=4.5,
                opacity=0.95,
            ).add_to(pred_layer)
        pred_layer.add_to(fmap)

        legend_html = """
        <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                    background-color: white; color: #111111; padding: 8px 12px; border-radius: 6px;
                    border: 1px solid #bbb; font-size: 13px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
          <b style="color:#111111;">PathSnap Map</b><br>
          <span style="color:#1565c0;">&#9679;</span> Raw GPS Observations<br>
          <span style="color:#d32f2f;">&#9473;</span> Predicted Route
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        st_folium(fmap, width=None, height=560, returned_objects=[])

    with st.expander("Inference Metadata & HMM Diagnostics"):
        st.json({
            "trip_id": data["trip_id"],
            "mode": data["mode"],
            "timestamp_utc": data["timestamp_utc"],
            "n_points": data["n_points"],
            "n_predicted_arcs": data["n_predicted_arcs"],
            "diagnostics": data.get("diagnostics", {}),
        })

if st.session_state["history"]:
    st.divider()
    st.subheader("Evaluated Trips Summary (Current Session)")
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df, width="stretch", hide_index=True)
    if len(hist_df) > 1:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Trips Tested", len(hist_df))
        if hist_df["Obs Top-1 (%)"].notna().any():
            s2.metric("Mean Obs Top-1", f"{hist_df['Obs Top-1 (%)'].mean():.2f}%")
        if hist_df["Route Len F1 (%)"].notna().any():
            s3.metric("Mean Route Len F1", f"{hist_df['Route Len F1 (%)'].mean():.2f}%")
        s4.metric("Mean Runtime", f"{hist_df['Runtime (s)'].mean():.2f} s")
