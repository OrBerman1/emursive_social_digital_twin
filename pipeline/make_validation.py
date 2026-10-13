"""Cheap stage: pick frames for the hand-count check and store the model's counts next to them.

Writes data/processed/validation/sample.json + downscaled frames, used by viewer/validate.html.
usage:  python pipeline/make_validation.py [--n 15 per walk] [--conf 0.6]
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, WALKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="frames per walk")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--variant", default="stab", choices=["", "stab"], help="match what the viewer shows (stab by default)")
    a = ap.parse_args()
    v = "_stab" if a.variant == "stab" else ""
    out = PROCESSED / "validation"
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for walk in WALKS:
        tr = pd.read_parquet(INTERIM / walk / f"tracks{v}" / "persons.parquet")
        n_frames = json.loads((INTERIM / walk / f"frames{v}" / "manifest.json").read_text(encoding="utf-8"))["n_frames"]
        for f in np.linspace(10, n_frames - 10, a.n).astype(int):
            g = tr[(tr.frame == f) & (tr.conf >= a.conf)]
            img = cv2.imread(str(INTERIM / walk / f"frames{v}" / f"{f:06d}.jpg"))
            name = f"{walk}_{f:06d}.jpg"
            cv2.imwrite(str(out / name), cv2.resize(img, (1920, 960)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            items.append({"walk": walk, "frame": int(f), "t": round(f / 5, 1), "image": name,
                          "model_count": int(len(g)), "model_count_10m": int((g.dist_m <= 10).sum())})
    (out / "sample.json").write_text(json.dumps({"conf": a.conf, "variant": a.variant or "raw", "items": items}, indent=1), encoding="utf-8")
    print(f"{len(items)} frames -> {out}")


if __name__ == "__main__":
    main()
