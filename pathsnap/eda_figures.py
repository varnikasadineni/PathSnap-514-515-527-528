"""Member A - static EDA figures for report section 2 (GPS trajectories, road network, candidates).

Run from the repo root:
    python -m pathsnap.eda_figures
    python -m pathsnap.eda_figures --network-trip 00000037

Writes (reports/figures/)
    eda_gps_trajectories.png   six trips: normal, longest, looping, off-route, big time gap, irregular sampling
    eda_road_network.png       one trip: road network + GPS + reference route, a zoom with arc directions,
                               and the candidate arcs of one GPS point
Partitions: train and student_test only (evaluator trips are never touched).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter

from pathsnap.build_dataset import _split_dir
from pathsnap.candidates import align_track_to_route_monotonic, generate_candidates_for_trip
from pathsnap.config import REPO_ROOT, load_yaml
from pathsnap.data import load_trip

TRAJ_TRIPS = [
    ("00000037", "typical trip"),
    ("00000021", "longest trip"),
    ("00000056", "looping route"),
    ("00000054", "reference route ~140 m off"),
    ("00000090", "42.5 h time gap"),
    ("00000004", "irregular sampling"),
]


def find_trip(cfg, trip_id):
    for part in ("train", "student_test"):
        d = _split_dir(cfg, part) / trip_id
        if d.exists():
            return part, d
    raise FileNotFoundError(f"trip {trip_id} not found in train or student_test")


def fig_trajectories(cfg, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    sc = None
    for ax, (tid, label) in zip(axes.ravel(), TRAJ_TRIPS):
        part, d = find_trip(cfg, tid)
        tr = load_trip(d).track
        x, y = (tr.x - tr.x.min()) / 1000, (tr.y - tr.y.min()) / 1000
        sc = ax.scatter(x, y, c=np.linspace(0, 1, len(tr)), s=3, cmap="viridis", vmin=0, vmax=1, zorder=2)
        gaps = np.where(np.diff(tr.t.to_numpy()) > 60)[0] + 1
        if len(gaps):
            ax.scatter(x.iloc[gaps], y.iloc[gaps], marker="x", c="red", s=45, zorder=4)
        ax.scatter(x.iloc[0], y.iloc[0], marker="o", s=90, c="lime", edgecolors="k", zorder=5)
        ax.scatter(x.iloc[-1], y.iloc[-1], marker="s", s=90, c="red", edgecolors="k", zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Trip {tid}: {label}\n{part}, {len(tr):,} GPS points, {len(gaps)} gap(s) > 60 s", fontsize=10)
        ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    handles = [Line2D([], [], marker="o", ls="", mfc="lime", mec="k", ms=9, label="start"),
               Line2D([], [], marker="s", ls="", mfc="red", mec="k", ms=9, label="end"),
               Line2D([], [], marker="x", ls="", c="red", ms=8, label="point after a gap > 60 s")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, label="position in the trip (0 = start, 1 = end)")
    fig.suptitle("Raw GPS trajectories (local metre coordinates; partitions train and student_test)", fontsize=13)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def _segments(ar, mask):
    return np.stack([np.column_stack([ar.x1.to_numpy()[mask], ar.y1.to_numpy()[mask]]),
                     np.column_stack([ar.x2.to_numpy()[mask], ar.y2.to_numpy()[mask]])], axis=1)


def _in_window(ar, xmin, xmax, ymin, ymax):
    a = ((ar.x1 >= xmin) & (ar.x1 <= xmax) & (ar.y1 >= ymin) & (ar.y1 <= ymax)).to_numpy()
    b = ((ar.x2 >= xmin) & (ar.x2 <= xmax) & (ar.y2 >= ymin) & (ar.y2 <= ymax)).to_numpy()
    return a | b


def _arrows(ax, ar, idx, color, lw=0.0035, zorder=2, alpha=1.0):
    ax.quiver(ar.x1.to_numpy()[idx], ar.y1.to_numpy()[idx],
              (ar.x2 - ar.x1).to_numpy()[idx], (ar.y2 - ar.y1).to_numpy()[idx],
              angles="xy", scale_units="xy", scale=1, width=lw, headwidth=5, headlength=6,
              color=color, zorder=zorder, alpha=alpha)


def fig_network(cfg, tid, out, radius_m=50.0, half=130.0):
    part, d = find_trip(cfg, tid)
    trip = load_trip(d)
    tr, ar = trip.track, trip.arcs
    cand = generate_candidates_for_trip(trip)
    ali = align_track_to_route_monotonic(trip)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

    # ---- (a) overview
    pad = 150.0
    xmin, xmax, ymin, ymax = tr.x.min() - pad, tr.x.max() + pad, tr.y.min() - pad, tr.y.max() + pad
    m = _in_window(ar, xmin, xmax, ymin, ymax)
    ax = axes[0]
    ax.add_collection(LineCollection(_segments(ar, m), colors="0.65", linewidths=0.4))
    rmask = np.zeros(len(ar), bool); rmask[trip.route] = True
    ax.add_collection(LineCollection(_segments(ar, rmask), colors="red", linewidths=1.8))
    ax.plot(tr.x, tr.y, ".", ms=2, c="tab:blue")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_title(f"(a) Road network, reference route and GPS\ntrip {tid} ({part}): {int(m.sum()):,} arcs shown", fontsize=10)
    ax.legend(handles=[Line2D([], [], c="0.65", lw=1.5, label="road arcs (.arcs)"),
                       Line2D([], [], c="red", lw=2, label="reference route (.route)"),
                       Line2D([], [], marker=".", ls="", c="tab:blue", label="GPS points (.track)")],
              loc="best", fontsize=8)

    # ---- window around one GPS point in the middle of the trip
    k = len(tr) // 2
    px, py = float(tr.x.iloc[k]), float(tr.y.iloc[k])
    wx0, wx1, wy0, wy1 = px - half, px + half, py - half, py + half
    mw = _in_window(ar, wx0 - 60, wx1 + 60, wy0 - 60, wy1 + 60)
    idx_w = np.where(mw)[0]
    win_pts = (tr.x.between(wx0, wx1) & tr.y.between(wy0, wy1)).to_numpy()

    # ---- (b) zoom with directions
    ax = axes[1]
    _arrows(ax, ar, idx_w, "0.45")
    ridx = np.intersect1d(idx_w, trip.route)
    _arrows(ax, ar, ridx, "red", lw=0.006, zorder=3)
    ax.plot(tr.x[win_pts], tr.y[win_pts], "o", ms=3.5, c="tab:blue", zorder=4)
    ax.set_xlim(wx0, wx1); ax.set_ylim(wy0, wy1); ax.set_aspect("equal")
    ax.set_title("(b) Zoom: every arc is directed; two-way roads have a reverse twin\n(arrows show arc direction)", fontsize=10)
    ax.legend(handles=[Line2D([], [], c="0.45", lw=1.5, label="road arc"),
                       Line2D([], [], c="red", lw=2.5, label="reference route arc"),
                       Line2D([], [], marker="o", ls="", c="tab:blue", label="GPS points")], loc="upper right", fontsize=8)

    # ---- (c) candidates of one GPS point
    ax = axes[2]
    pid = int(tr["point_idx"].iloc[k])
    c = cand[cand.point_idx == pid].sort_values("dist_m")
    true_arc = int(ali.loc[ali.point_idx == pid, "true_arc"].iloc[0])
    _arrows(ax, ar, idx_w, "0.8", lw=0.003)
    cidx = c.arc_id.to_numpy()
    cs = ax.quiver(ar.x1.to_numpy()[cidx], ar.y1.to_numpy()[cidx],
                   (ar.x2 - ar.x1).to_numpy()[cidx], (ar.y2 - ar.y1).to_numpy()[cidx], c.dist_m.to_numpy(),
                   angles="xy", scale_units="xy", scale=1, width=0.006, headwidth=4.5, headlength=5,
                   cmap="plasma_r", zorder=4)
    _arrows(ax, ar, np.array([true_arc]), "limegreen", lw=0.011, zorder=5)
    ax.add_patch(Circle((px, py), radius_m, fill=False, ls="--", ec="k", lw=1.2, zorder=6))
    ax.scatter([px], [py], marker="*", s=260, c="gold", edgecolors="k", zorder=7)
    ax.set_xlim(wx0, wx1); ax.set_ylim(wy0, wy1); ax.set_aspect("equal")
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02); cb.set_label("distance to the GPS point (m)")
    ax.set_title(f"(c) Candidates of GPS point {pid}: {len(c)} arcs found\n(dashed circle = {radius_m:.0f} m search radius)", fontsize=10)
    ax.legend(handles=[Line2D([], [], marker="*", ls="", c="gold", mec="k", ms=14, label="GPS point"),
                       Line2D([], [], c="limegreen", lw=4, label="true arc (label = 1)"),
                       Line2D([], [], c="purple", lw=2, label="candidate arcs (label = 0 or 1)")], loc="upper right", fontsize=8)

    axes[0].set_xlabel("east (m)"); axes[0].set_ylabel("north (m)")
    for a in axes[1:]:   # zoom panels: metres relative to the GPS point in the centre
        a.set_xticks(px + np.arange(-100, 101, 50)); a.set_yticks(py + np.arange(-100, 101, 50))
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - px:.0f}"))
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - py:.0f}"))
        a.set_xlabel("east of the GPS point (m)"); a.set_ylabel("north of the GPS point (m)")
    fig.suptitle(f"Road network and candidate generation, trip {tid} (partition: {part})", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network-trip", default="00000037")
    args = ap.parse_args()
    cfg = load_yaml("configs/paths.yaml")
    figs = REPO_ROOT / cfg["artifacts"]["figures_dir"]
    figs.mkdir(parents=True, exist_ok=True)
    fig_trajectories(cfg, figs / "eda_gps_trajectories.png")
    fig_network(cfg, args.network_trip, figs / "eda_road_network.png")


if __name__ == "__main__":
    main()
