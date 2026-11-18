"""Hand-off H5 / Section 6 (Member C — Vibhav): Interactive Folium geographical maps.

Mandatory layers per Deliverables.pdf Section 6:
1. Local road-network geometry (grey)
2. Original GPS observations (blue dots)
3. Reference route (thick green line, when available & permitted — never evaluator UI)
4. Predicted route (crisp red line drawn on top for clear comparison)
Plus clear legend, fit_bounds zoom, and Folium LayerControl.

Uses CartoDB Voyager + Esri WorldStreetMap tile URLs (both work reliably even when
opened directly via file:// in Edge/Chrome, where tile.openstreetmap.org returns 403).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import folium
import numpy as np

from pathsnap.data import TripData


def build_trip_folium_map(
    trip: TripData,
    predicted_arcs: Optional[Sequence[int]] = None,
    show_reference_route: bool = True,
    max_road_arcs: int = 2500,
    max_gps_markers: int = 1500,
    save_html_path: Optional[str | Path] = None,
) -> folium.Map:
    """Create an interactive Folium map with separate colored layers, legend, and LayerControl."""
    center_lat = float(trip.track["lat"].mean())
    center_lon = float(trip.track["lon"].mean())

    # Base map works cleanly both over http:// and directly via file:// in Chrome/Edge
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles=None)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Carto Voyager (Default)",
        subdomains="abcd",
        max_zoom=20,
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri Street Map",
        max_zoom=19,
    ).add_to(fmap)

    node_lat = trip.nodes["lat"].to_numpy()
    node_lon = trip.nodes["lon"].to_numpy()
    from_nodes = trip.arcs["from_node"].to_numpy()
    to_nodes = trip.arcs["to_node"].to_numpy()

    # 1. Local road network layer (cropped to GPS bounding box)
    road_layer = folium.FeatureGroup(name="Road Network (Local)", show=True)
    lat_min, lat_max = trip.track["lat"].min() - 0.002, trip.track["lat"].max() + 0.002
    lon_min, lon_max = trip.track["lon"].min() - 0.002, trip.track["lon"].max() + 0.002
    u_lat, u_lon = node_lat[from_nodes], node_lon[from_nodes]
    in_bbox = (u_lat >= lat_min) & (u_lat <= lat_max) & (u_lon >= lon_min) & (u_lon <= lon_max)
    bbox_aids = np.where(in_bbox)[0]
    if len(bbox_aids) > max_road_arcs:
        step = max(1, len(bbox_aids) // max_road_arcs)
        bbox_aids = bbox_aids[::step]

    for aid in bbox_aids:
        u, v = int(from_nodes[aid]), int(to_nodes[aid])
        folium.PolyLine(
            locations=[[float(node_lat[u]), float(node_lon[u])], [float(node_lat[v]), float(node_lon[v])]],
            color="#888888",
            weight=1.5,
            opacity=0.45,
        ).add_to(road_layer)
    road_layer.add_to(fmap)

    # 2. Raw GPS observations layer (underneath reference/predicted routes so lines pop clearly)
    gps_layer = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    stride = max(1, len(trip.track) // max_gps_markers)
    for row in trip.track.iloc[::stride].itertuples(index=False):
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=2.2,
            color="#1565c0",
            fill=True,
            fill_opacity=0.65,
            tooltip=f"idx={row.point_idx}, t={row.t:.1f}s",
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # 3. Reference route layer (thick green halo so both green & red are visible when overlapping)
    if show_reference_route and trip.route is not None and len(trip.route) > 0:
        ref_layer = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        for aid in np.unique(trip.route):
            u, v = int(from_nodes[aid]), int(to_nodes[aid])
            folium.PolyLine(
                locations=[[float(node_lat[u]), float(node_lon[u])], [float(node_lat[v]), float(node_lon[v])]],
                color="#2e7d32",
                weight=6.5,
                opacity=0.7,
            ).add_to(ref_layer)
        ref_layer.add_to(fmap)

    # 4. Predicted route layer (crisp red line on top)
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        pred_layer = folium.FeatureGroup(name="Predicted Route (PathSnap)", show=True)
        seen = set()
        for aid in predicted_arcs:
            ai = int(aid)
            if ai < 0 or ai in seen:
                continue
            seen.add(ai)
            u, v = int(from_nodes[ai]), int(to_nodes[ai])
            folium.PolyLine(
                locations=[[float(node_lat[u]), float(node_lon[u])], [float(node_lat[v]), float(node_lon[v])]],
                color="#d32f2f",
                weight=3.0,
                opacity=0.95,
            ).add_to(pred_layer)
        pred_layer.add_to(fmap)

    # 5. Legend & LayerControl (explicit dark text for readability in all themes)
    legend_html = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background-color: white; color: #111111; padding: 10px 14px; border-radius: 6px;
                border: 1px solid #bbb; font-size: 13px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
      <b style="color:#111111;">PathSnap Legend</b><br>
      <span style="color:#1565c0;">&#9679;</span> Raw GPS Observations<br>
      <span style="color:#888888;">&#9473;</span> Local Road Network<br>
      <span style="color:#2e7d32;">&#9473;</span> Reference Route<br>
      <span style="color:#d32f2f;">&#9473;</span> Predicted Route
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    # Fit map bounds tightly to trajectory
    fmap.fit_bounds(
        [[float(trip.track["lat"].min()), float(trip.track["lon"].min())],
         [float(trip.track["lat"].max()), float(trip.track["lon"].max())]],
        padding=(20, 20),
    )

    if save_html_path is not None:
        out_path = Path(save_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out_path))

    return fmap
