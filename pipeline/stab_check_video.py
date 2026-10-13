"""Diagnostic: stabilisation quality at full video rate, on the encoded equirect.mp4 the viewer plays.

Decodes a segment at 960x480, measures visual rotation between consecutive video frames, and reports the
median residual rotation  E_n = Q[n+1+k]^T R_vis Q[n+k]  for index offsets k (finds timing errors).

usage:  python pipeline/stab_check_video.py walk1 [start_s 50] [seconds 10]
"""
from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from common import PROCESSED
from stabilize import ransac_rotation, unit

W, H = 960, 480


def main() -> None:
    walk = sys.argv[1] if len(sys.argv) > 1 else "walk1"
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    st = json.loads((PROCESSED / walk / "stab.json").read_text(encoding="utf-8"))
    q = np.array(st["q_video"]).reshape(-1, 3, 3)
    fps = st["video_fps"]
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", str(PROCESSED / walk / "video" / "equirect.mp4"), "-t", str(dur),
           "-vf", f"scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    orb = cv2.ORB_create(3000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((np.abs(90 - (yy + .5) / H * 180) < 60)).astype(np.uint8) * 255
    feats = [orb.detectAndCompute(f, mask) for f in frames]
    rots, idx = [], []
    for i in range(len(frames) - 1):
        (ka, da), (kb, db) = feats[i], feats[i + 1]
        if da is None or db is None:
            continue
        m = bf.match(da, db)
        if len(m) < 40:
            continue
        pa = np.array([ka[x.queryIdx].pt for x in m]); pb = np.array([kb[x.trainIdx].pt for x in m])
        va = unit(pa[:, 0] / W * 360 - 180, 90 - pa[:, 1] / H * 180)
        vb = unit(pb[:, 0] / W * 360 - 180, 90 - pb[:, 1] / H * 180)
        r, n = ransac_rotation(va, vb, iters=200, thr_deg=0.8)
        if n >= 40:
            rots.append(r); idx.append(int(round(t0 * fps)) + i)
    R = np.array(rots); n0 = np.array(idx)
    raw_deg = np.degrees(Rot.from_matrix(R).magnitude())
    print(f"{walk} {t0:.0f}-{t0 + dur:.0f}s, {len(R)} frame pairs, raw median rotation per video frame: {np.median(raw_deg):.3f} deg")
    for k in range(-8, 9, 2):
        a = np.clip(n0 + k, 0, len(q) - 2)
        E = np.einsum("nji,njk,nkl->nil", q[a + 1], R, q[a])
        print(f"  offset {k:+d} frames: stabilised median {np.median(np.degrees(Rot.from_matrix(E).magnitude())):.3f} deg")
    a = np.clip(n0, 0, len(q) - 2)
    Et = np.einsum("nij,njk,nlk->nil", q[a + 1], R, q[a])   # transposed Q (wrong-direction check)
    print(f"  Q transposed: {np.median(np.degrees(Rot.from_matrix(Et).magnitude())):.3f} deg")


if __name__ == "__main__":
    main()
