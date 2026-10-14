"""Cheap stage: walking direction per analysis frame, from optical flow on the stabilised viewer video.

Walking forward makes the scene stream away from the heading: on a horizon band of the equirect frame the
horizontal flow follows u(yaw) ~ A*sin(yaw - f) + c (c = leftover rotation). f is fitted per 5 fps frame,
then smoothed (circular, sigma 2 s) and held while the camera stands still (low flow amplitude).
Checked by eye on both walks: f looks down the street; f + 180 is the person holding the camera.

Output: data/processed/<walk>/forward.json  {fwd_deg: [...], moving: [0/1 ...]}  (view yaw, stabilised video)
usage:  python pipeline/forward_dir.py [walk1 walk2]
"""
from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from common import PROCESSED, WALKS

W, H, FPS = 960, 480, 5.0
SIGMA_S = 2.0


def raw_directions(src: str) -> tuple[np.ndarray, np.ndarray]:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={FPS:g},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE)
    yaw = np.radians((np.arange(W) + 0.5) / W * 360 - 180)
    X = np.c_[np.sin(yaw), np.cos(yaw), np.ones(W)]
    band = slice(int(H * 0.40), int(H * 0.62))          # pitch ~ +18 .. -22 deg: stalls and ground near the horizon
    f, amp, prev = [np.nan], [0.0], None
    while True:
        buf = p.stdout.read(W * H)
        if len(buf) < W * H:
            break
        g = np.frombuffer(buf, np.uint8).reshape(H, W)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 21, 3, 5, 1.1, 0)
            (a, b, _), *_ = np.linalg.lstsq(X, np.median(fl[band, :, 0], axis=0), rcond=None)
            f.append(np.arctan2(-b, a)); amp.append(float(np.hypot(a, b)))
        prev = g
    return np.array(f[:len(amp)]), np.array(amp)


def run(walk: str) -> dict:
    src = PROCESSED / walk / "video_stab" / "equirect.mp4"
    f, amp = raw_directions(str(src))
    moving = amp > np.percentile(amp, 35)
    # circular smoothing, weighted by how clearly the camera moves; standing frames inherit the neighbours' heading
    wgt = np.where(moving & ~np.isnan(f), amp, 0.0)
    z = np.where(np.isnan(f), 0, np.exp(1j * np.nan_to_num(f))) * wgt
    s = SIGMA_S * FPS
    zs = gaussian_filter1d(z.real, s, mode="nearest") + 1j * gaussian_filter1d(z.imag, s, mode="nearest")
    fwd = np.degrees(np.angle(zs))
    out = {"walk": walk, "fps": FPS, "fwd_deg": np.round(fwd, 1).tolist(), "moving": moving.astype(int).tolist(),
           "median_fwd_deg": round(float(np.degrees(np.angle(np.exp(1j * np.radians(fwd)).mean()))), 1)}
    (PROCESSED / walk / "forward.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return {"walk": walk, "frames": len(fwd), "median_fwd_deg": out["median_fwd_deg"], "moving_share": round(float(moving.mean()), 2)}


if __name__ == "__main__":
    for w in (sys.argv[1:] or list(WALKS)):
        print(json.dumps(run(w)))
