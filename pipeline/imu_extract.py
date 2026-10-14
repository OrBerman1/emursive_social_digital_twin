"""Stage `imu`: full-rate gyro/accelerometer log from the Insta360 trailer -> CSV (cached, seconds).

Runs in the separate Python 3.9 env (.venv-imu) because telemetry-parser only ships cp39 Windows wheels:
    .venv-imu\\Scripts\\python pipeline\\imu_extract.py walk1 walk2
Columns: t_ms (video time), gx gy gz (deg/s), ax ay az (m/s^2), in the camera's IMU axes.
Stdlib only (no numpy/pandas in that env).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import telemetry_parser

from common import INTERIM, WALKS


def run(walk: str) -> None:
    src = WALKS[walk]
    out_dir = INTERIM / walk
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "imu.csv"
    st = src.stat()
    key = f"{src.name}:{st.st_size}:{int(st.st_mtime)}:v1"
    man = out_dir / "imu.manifest.json"
    if out.exists() and man.exists() and json.loads(man.read_text()).get("key") == key:
        print(f"[imu] {walk}: cached")
        return
    p = telemetry_parser.Parser(str(src))
    imu = p.normalized_imu()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "gx", "gy", "gz", "ax", "ay", "az"])
        for s in imu:
            g, a = s["gyro"], s["accl"] or (0.0, 0.0, 0.0)
            w.writerow([f"{s['timestamp_ms']:.3f}", *(f"{v:.5f}" for v in g), *(f"{v:.5f}" for v in a)])
    man.write_text(json.dumps({"key": key, "camera": f"{p.camera} {p.model}", "samples": len(imu)}))
    print(f"[imu] {walk}: {len(imu)} samples ({p.camera} {p.model}) -> {out}")


if __name__ == "__main__":
    for w in (sys.argv[1:] or list(WALKS)):
        run(w)
