"""Expensive stage (cached): camera-motion-free motion of people from dense optical flow ("optical flow of people").

Why: a person's image motion is dominated by the moving camera. The ground right next to a person's feet is at
the same distance, so its image motion is exactly what a STATIC point at the person's position would do (camera
translation, rotation residuals and horizon tilt included). Feet motion minus local ground flow = the person's
own motion; converted to metres on the ground with the distance at the feet.

Method : 1. DIS optical flow (OpenCV, preset medium) between consecutive stabilised analysis frames (5 fps),
            on a half-resolution band of the equirect frame (pitch -65..+20 deg).
         2. per tracked person (and, for validation, per static ground object) and frame pair f -> f+1:
            ground flow = median flow in a band beside the feet (pitch [feet - D, feet + 0.3 D], yaw box +- 0.6
            box widths), excluding every person box; feet motion = change of the tracked (smoothed) box:
            yaw of the box centre and pitch of the box bottom.
         3. residual (feet - ground) in deg/frame -> ground velocity in the camera-yaw frame:
            lateral = d * dyaw, radial = h / sin^2(depression) * dpitch, with d = h / tan(depression).
Output : data/interim/<walk>/person_flow<_variant>/person_flow.parquet
         rows: kind (person|static), track, frame, ryaw, rpitch (deg/frame), vx, vy (m/s, camera-yaw frame:
         x = right of view yaw 0, y = along view yaw 0), n_ground (pixels used), dist_m

usage:  python pipeline/person_flow.py [walk1 walk2] [--variant stab] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import cv2
import numpy as np
import pandas as pd

from common import INTERIM, WALKS, Stage

VERSION = 1
W_FLOW = 1920                    # flow computed at half resolution
PITCH_HI, PITCH_LO = 20.0, -65.0
MIN_GROUND_PX = 25
STATIC = {"motorcycle", "chair", "potted plant", "dining table", "display table", "stack of crates", "trash bin", "stool",
          "plastic crate", "cardboard box", "plastic chair", "bench"}
FPS = 5.0


def fhash(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run(walk: str, variant: str, force: bool) -> dict:
    sfx = f"_{variant}" if variant else ""
    tdir = INTERIM / walk / f"tracks{sfx}"
    fdir = INTERIM / walk / f"frames{sfx}"
    out_dir = INTERIM / walk / f"person_flow{sfx}"
    fman = json.loads((fdir / "manifest.json").read_text(encoding="utf-8"))
    inputs = {"frames": fman.get("key"), "persons": fhash(tdir / "persons.parquet"),
              "objects": fhash(tdir / "objects.parquet"), "level2": fhash(tdir / "level2.parquet")}
    params = {"w_flow": W_FLOW, "pitch_band": [PITCH_LO, PITCH_HI], "min_ground_px": MIN_GROUND_PX, "dis": "medium"}
    st = Stage("person_flow", VERSION, out_dir, params, inputs)
    if st.done() and not force:
        print(f"[person_flow] {walk}{sfx}: cached")
        return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8")).get("summary", {})
    st.start()
    h = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))["camera_height_used_m"]
    per = pd.read_parquet(tdir / "persons.parquet").assign(kind="person")
    obj = pd.concat([pd.read_parquet(tdir / f"{k}.parquet").assign(src=k) for k in ("objects", "level2")])
    obj = obj[obj.cls.isin(STATIC)].copy()
    obj["track"] = obj.groupby(["src", "track"]).ngroup()
    obj["kind"] = "static"
    cols = ["kind", "track", "frame", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot"]
    ent = pd.concat([per[cols], obj[cols]], ignore_index=True)
    nxt = ent.set_index(["kind", "track", "frame"])
    frames = sorted(int(p.stem) for p in fdir.glob("*.jpg"))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    H_FLOW = W_FLOW // 2
    r0, r1 = int((0.5 - PITCH_HI / 180) * H_FLOW), int((0.5 - PITCH_LO / 180) * H_FLOW)
    X = lambda yaw: (np.asarray(yaw) / 360 + 0.5) * W_FLOW
    Y = lambda pt: (0.5 - np.asarray(pt) / 180) * H_FLOW - r0
    by_frame = {f: g for f, g in ent.groupby("frame")}
    rows, prev = [], None
    for i, f in enumerate(frames[:-1]):
        g = by_frame.get(f)
        if g is None or f + 1 != frames[i + 1]:
            prev = None
            continue
        a = prev if prev is not None else cv2.resize(cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE), (W_FLOW, H_FLOW))
        b = cv2.resize(cv2.imread(str(fdir / f"{f + 1:06d}.jpg"), cv2.IMREAD_GRAYSCALE), (W_FLOW, H_FLOW))
        prev = b
        flow = dis.calc(a[r0:r1], b[r0:r1], None)                         # px/frame, band only
        hb, wb = flow.shape[:2]
        free = np.ones((hb, wb), bool)                                    # ground reference must not be on a person
        for p in g[g.kind == "person"].itertuples():
            xl, xr = int(X(p.yaw_l)) - 3, int(X(p.yaw_r)) + 3
            yt, yb = max(0, int(Y(p.pitch_top))), min(hb, int(Y(p.pitch_bot)) + 3)
            if xr > xl and yb > yt:
                free[yt:yb, max(0, xl):min(wb, xr)] = False
        for e in g.itertuples():
            try:
                e2 = nxt.loc[(e.kind, e.track, f + 1)]
            except KeyError:
                continue
            if e.pitch_bot > -2 or e.pitch_bot < PITCH_LO + 3:
                continue
            bw = ((e.yaw_r - e.yaw_l) % 360) or 1.0
            bh = e.pitch_top - e.pitch_bot
            D = max(1.5, 0.15 * bh)
            ya, yb_ = int(Y(e.pitch_bot + 0.3 * D)), int(Y(e.pitch_bot - D))
            xa, xb = int(X(e.yaw_l - 0.6 * bw)), int(X(e.yaw_l + bw + 0.6 * bw))
            ya, yb_ = max(0, ya), min(hb, yb_)
            xa, xb = max(0, xa), min(wb, xb)
            if yb_ <= ya or xb <= xa:
                continue
            m = free[ya:yb_, xa:xb]
            if e.kind == "static":                                        # do not use the object itself as its ground
                ox0, ox1 = int(X(e.yaw_l)) - xa, int(X(e.yaw_r)) - xa
                oy0 = int(Y(e.pitch_top)) - ya
                m = m.copy(); m[max(0, oy0):, max(0, ox0):max(0, ox1)] = False
            if m.sum() < MIN_GROUND_PX:
                continue
            fl = flow[ya:yb_, xa:xb][m]
            gu, gv = np.median(fl[:, 0]) * 360 / W_FLOW, -np.median(fl[:, 1]) * 180 / H_FLOW   # deg/frame (pitch up +)
            dyaw = ((e2.yaw - e.yaw + 540) % 360) - 180
            dpitch = e2.pitch_bot - e.pitch_bot
            ry, rp = dyaw - gu, dpitch - gv
            phi = np.radians(-e.pitch_bot)
            d = h / np.tan(phi)
            lat_v = d * np.radians(ry) * FPS                               # + = clockwise (to the right)
            rad_v = h / np.sin(phi) ** 2 * np.radians(rp) * FPS            # + = away from the camera
            psi = np.radians(e.yaw)
            vx = rad_v * np.sin(psi) + lat_v * np.cos(psi)
            vy = rad_v * np.cos(psi) - lat_v * np.sin(psi)
            rows.append((e.kind, int(e.track), int(f), float(ry), float(rp), float(vx), float(vy), int(m.sum()), float(d)))
        if i % 100 == 0:
            print(f"[person_flow] {walk}{sfx}: frame {f}/{frames[-1]}  rows {len(rows)}", flush=True)
    df = pd.DataFrame(rows, columns=["kind", "track", "frame", "ryaw", "rpitch", "vx", "vy", "n_ground", "dist_m"])
    df.to_parquet(out_dir / "person_flow.parquet", index=False)
    s = df.assign(spd=np.hypot(df.vx, df.vy)).groupby("kind").spd.median().round(2).to_dict()
    summary = {"rows": len(df), "median_raw_speed_m_s": s, "camera_height_m": h}
    st.finish({"summary": summary})
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.variant, a.force)))


if __name__ == "__main__":
    main()
