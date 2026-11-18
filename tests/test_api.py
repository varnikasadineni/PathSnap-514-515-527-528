"""Member D: End-to-end verification of PathSnap FastAPI backend across all 10 trips.

Modes:
    python tests/test_api.py          # Fast in-process endpoint verification (works anywhere)
    python tests/test_api.py --live   # Spins up real uvicorn HTTP server & probes with requests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from fastapi import HTTPException
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.main import get_trips, health, load_server_state, predict_endpoint, root

PORT = 8019
BASE_URL = f"http://127.0.0.1:{PORT}"
FORBIDDEN_LEAK_KEYS = {"route", "ref_route", "reference_route", "true_arc", "true_arcs", "ground_truth"}


def _assert_prediction_payload(tid: str, data: dict) -> dict:
    leaked = FORBIDDEN_LEAK_KEYS.intersection(set(data.keys()))
    assert not leaked, f"Ground-truth keys leaked in response: {leaked}"
    assert data["trip_id"] == tid
    assert data["n_points"] > 0
    assert len(data["gps_points"]) == data["n_points"]
    assert len(data["predicted_arc_coords"]) == data["n_predicted_arcs"]
    return data["metrics"]


def run_inprocess() -> None:
    load_server_state()

    r_root = root()
    assert r_root["n_trips"] == 10, r_root
    r_health = health()
    assert r_health["status"] == "ok", r_health
    print("HEALTH:", r_health)

    trips = get_trips()["trips"]
    assert len(trips) == 10, f"Expected 10 evaluation trips, got {len(trips)}"

    # Verify 404 on unknown trip
    try:
        predict_endpoint("does_not_exist")
        raise AssertionError("Expected HTTPException(404) for invalid trip_id")
    except HTTPException as exc:
        assert exc.status_code == 404

    top1s, conts, lenf1s, times = [], [], [], []
    for tid in trips:
        data = predict_endpoint(tid)
        m = _assert_prediction_payload(tid, data)
        top1s.append(m["observation_top1_accuracy"])
        conts.append(m["route_continuity"])
        lenf1s.append(m["route_length_f1"])
        times.append(data["runtime_sec"])

    print("\n=== ALL 10 TRIPS PASSED END-TO-END API VERIFICATION (IN-PROCESS) ===")
    print(f"  Mean Obs Top-1 Accuracy : {sum(top1s)/len(top1s)*100:.2f}%")
    print(f"  Mean Route Continuity   : {sum(conts)/len(conts)*100:.2f}%")
    print(f"  Mean Route Length F1    : {sum(lenf1s)/len(lenf1s)*100:.2f}%")
    print(f"  Mean API Runtime / Trip : {sum(times)/len(times):.2f} s")
    print("  Ground-truth leakage    : ZERO (verified on all 10 JSON responses)")


def run_live_http() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(REPO_ROOT),
    )
    try:
        deadline = time.time() + 25.0
        while time.time() < deadline:
            try:
                if requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.4)
        else:
            raise RuntimeError("Live uvicorn server did not become healthy in time")

        assert requests.get(f"{BASE_URL}/", timeout=5).status_code == 200
        assert requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200
        trips = requests.get(f"{BASE_URL}/trips", timeout=5).json()["trips"]
        assert len(trips) == 10
        assert requests.post(f"{BASE_URL}/predict/does_not_exist", timeout=5).status_code == 404

        top1s, conts, lenf1s, times = [], [], [], []
        for tid in trips:
            resp = requests.post(f"{BASE_URL}/predict/{tid}", timeout=120)
            assert resp.status_code == 200, resp.text
            data = resp.json()
            m = _assert_prediction_payload(tid, data)
            top1s.append(m["observation_top1_accuracy"])
            conts.append(m["route_continuity"])
            lenf1s.append(m["route_length_f1"])
            times.append(data["runtime_sec"])

        print("\n=== ALL 10 TRIPS PASSED LIVE HTTP API VERIFICATION ===")
        print(f"  Mean Obs Top-1 Accuracy : {sum(top1s)/len(top1s)*100:.2f}%")
        print(f"  Mean Route Continuity   : {sum(conts)/len(conts)*100:.2f}%")
        print(f"  Mean Route Length F1    : {sum(lenf1s)/len(lenf1s)*100:.2f}%")
        print(f"  Mean API Runtime / Trip : {sum(times)/len(times):.2f} s")
        print("  Ground-truth leakage    : ZERO (verified on all 10 JSON responses)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap API end-to-end tests")
    parser.add_argument("--live", action="store_true", help="Test over live uvicorn HTTP port")
    args = parser.parse_args()
    if args.live:
        run_live_http()
    else:
        run_inprocess()


if __name__ == "__main__":
    main()
