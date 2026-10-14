"""Cheap stage: thumbnail strip for the Analysis walk profile - one forward-looking frame every few seconds.

The frame is cut from the stabilised viewer video around the walking direction (forward_dir.py, median yaw) on a
band around the horizon, and all thumbnails are tiled side by side into one image the viewer slices with CSS.

Output: data/processed/<walk>/thumbs.jpg + thumbs.json {every_s, count, w, h}
usage:  python pipeline/thumbs.py [walk1 walk2] [--every 4]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess

from common import PROCESSED

W, H = 160, 90


def run(walk: str, every: float) -> dict:
    out_dir = PROCESSED / walk
    src = out_dir / "video_stab" / "equirect.mp4"
    fwd = json.loads((out_dir / "forward.json").read_text(encoding="utf-8"))["median_fwd_deg"]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                               capture_output=True, text=True, check=True).stdout)
    n = math.ceil(dur / every)
    # 90 deg wide around the walking direction (a quarter of the equirect width), 56 deg tall around the horizon
    cx = (fwd / 360 + 0.5) % 1
    x0 = f"(iw*{cx:.4f}-iw/8)"
    vf = f"fps=1/{every:g},crop=iw/4:ih*0.31:max(0\\,min(iw-iw/4\\,{x0})):ih*0.345,scale={W}:{H},tile={n}x1"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-frames:v", "1", "-q:v", "4", str(out_dir / "thumbs.jpg")], check=True)
    meta = {"walk": walk, "every_s": every, "count": n, "w": W, "h": H, "fwd_deg": fwd}
    (out_dir / "thumbs.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=["walk1", "walk2"])
    ap.add_argument("--every", type=float, default=4.0)
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.every)))
