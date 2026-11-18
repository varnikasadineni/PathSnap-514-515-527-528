"""Member D (Laptop 2): Evaluator-facing Streamlit UI for PathSnap.

Run on Laptop 2:
    streamlit run ui/app.py

Connects over LAN to Laptop 1's FastAPI server (configurable in sidebar or via PATHSNAP_API_URL).
Displays ONLY the 10 evaluation trip options, raw GPS observations, predicted route, and
server-computed metrics (ground-truth labels never leave Laptop 1).
"""

from __future__ import annotations

import os
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap — Distributed Map-Matching", layout="wide")
st.title("PathSnap: Distributed Map-Matching & Route Reconstruction")

# Configurable Laptop 1 API address
default_api = os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000")
with st.sidebar:
    st.header("Laptop 1 API Connection")
    api_url = st.text_input("FastAPI Base URL", value=default_api).rstrip("/")
    st.caption("Example over LAN: http://192.168.1.42:8000")
    if st.button("Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            r.raise_for_status()
            st.success(f"Connected: {r.json()}")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

# Fetch available evaluation trips from Laptop 1
try:
    resp = requests.get(f"{api_url}/trips", timeout=5)
    resp.raise_for_status()
    trips = resp.json().get("trips", [])
except Exception as exc:
    st.error(f"Failed to fetch trip list from `{api_url}/trips`: {exc}")
    st.stop()

if not trips:
    st.warning("No trips returned by the API.")
    st.stop()

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    selected_trip = st.selectbox("Select Evaluation Trip", options=trips)
with col_btn:
    st.write("")
    st.write("")
    run_clicked = st.button("Reconstruct Route", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner(f"Sending trip `{selected_trip}` to Laptop 1 API..."):
        try:
            pred_resp = requests.post(f"{api_url}/predict/{selected_trip}", timeout=120)
            pred_resp.raise_for_status()
            st.session_state["last_result"] = pred_resp.json()
        except Exception as exc:
            st.error(f"Inference request failed: {exc}")

data = st.session_state.get("last_result")
if data:
    st.subheader(f"Results for Trip `{data['trip_id']}`  ({data['mode']})")
    metrics = data.get("metrics", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    obs_acc = metrics.get("observation_top1_accuracy")
    len_f1 = metrics.get("route_length_f1")
    cont = metrics.get("route_continuity")
    on_route = metrics.get("point_on_route_rate")

    m1.metric("Observation Top-1 Acc", f"{obs_acc * 100:.2f}%" if obs_acc is not None else "N/A")
    m2.metric("Route Length F1", f"{len_f1 * 100:.2f}%" if len_f1 is not None else "N/A")
    m3.metric("Route Continuity", f"{cont * 100:.2f}%" if cont is not None else "N/A")
    m4.metric("Points on Route", f"{on_route * 100:.2f}%" if on_route is not None else "N/A")
    m5.metric("API Runtime", f"{data['runtime_sec']:.2f} s")

    # Build interactive Folium map from API response (GPS points + predicted route ONLY)
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

        pred_layer = folium.FeatureGroup(name="Predicted Route (PathSnap)", show=True)
        for seg in pred_arcs:
            folium.PolyLine(
                locations=seg["coords"],
                color="#d32f2f",
                weight=4,
                opacity=0.9,
            ).add_to(pred_layer)
        pred_layer.add_to(fmap)

        gps_layer = folium.FeatureGroup(name="Raw GPS Observations", show=True)
        stride = max(1, len(gps_pts) // 1200)
        for lat, lon in gps_pts[::stride]:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2.5,
                color="#1565c0",
                fill=True,
                fill_opacity=0.75,
            ).add_to(gps_layer)
        gps_layer.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        st_folium(fmap, width=None, height=580, returned_objects=[])

    with st.expander("Inference Diagnostics"):
        st.json(data.get("diagnostics", {}))
