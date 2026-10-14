"""Stage `det_<name>`: run a detector on perspective tiles of every equirect frame (cached, GPU).

Raw detections are saved at a LOW confidence threshold with their tile box and spherical extent, so
thresholds, class filters, cross-tile de-duplication and tracking can all be redone later without the GPU.

  level 1:  python pipeline/detect.py walk1 --model yolo11x.pt                         (COCO classes)
  level 2:  python pipeline/detect.py walk1 --model yolov8x-worldv2.pt --prompts level2 (open vocabulary)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd

from common import INTERIM, WALKS, Stage
from spherical import TileSampler, box_to_sphere, default_tiles

VERSION = 1

# Level-2 open-vocabulary prompts (see PLAN.md, "Detection strategy").
PROMPT_SETS = {
    "level2": [
        "plastic crate", "stack of crates", "cardboard box", "flattened cardboard", "plastic bag",
        "organic waste", "fruit and vegetable waste", "packaging", "trash bin", "garbage bag",
        "hand truck", "trolley cart", "wheeled shopping trolley", "walker", "mobility scooter",
        "wheelchair", "broom", "e-bike", "delivery box", "pallet", "display table", "plastic chair",
        "stool", "awning", "price sign", "shop sign", "flag", "hanging lamp",
    ],
}


def run(walk: str, model_name: str, prompts: str | None, conf: float, imgsz: int, batch_frames: int,
        limit: int | None, force: bool, frames: str = "frames") -> None:
    frames_dir = INTERIM / walk / frames
    fm = json.loads((frames_dir / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((frames_dir / "index.json").read_text(encoding="utf-8"))
    if limit:
        index = index[:limit]
    tiles = default_tiles()
    tag = model_name.replace(".pt", "") + (f"_{prompts}" if prompts else "") + ("_stab" if frames == "frames_stab" else "")
    out_dir = INTERIM / walk / f"det_{tag}"
    params = {"model": model_name, "prompts": PROMPT_SETS.get(prompts) if prompts else None, "conf": conf,
              "imgsz": imgsz, "tiles": [t.__dict__ for t in tiles], "limit": limit}
    st = Stage(f"det_{tag}", VERSION, out_dir, params, {"frames_key": fm["key"]})
    if st.done() and not force:
        print(f"[detect] {walk} {tag}: cached ({out_dir})")
        return
    st.start()

    from ultralytics import YOLO, YOLOWorld  # imported late so --help is fast
    if prompts:
        model = YOLOWorld(model_name)
        model.set_classes(PROMPT_SETS[prompts])
    else:
        model = YOLO(model_name)
    names = model.names
    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")  # ultralytics won't pick MPS itself

    first = cv2.imread(str(frames_dir / index[0]["file"]))
    sampler = TileSampler(tiles, first.shape[1], first.shape[0])
    rows, t0 = [], time.time()
    for b in range(0, len(index), batch_frames):
        chunk = index[b:b + batch_frames]
        imgs, meta = [], []
        for fr in chunk:
            equi = cv2.imread(str(frames_dir / fr["file"]))
            for tile, img in zip(tiles, sampler.cut(equi)):
                imgs.append(img)
                meta.append((fr, tile))
        results = model.predict(imgs, conf=conf, imgsz=imgsz, verbose=False, half=dev != "cpu", device=dev)
        for (fr, tile), res in zip(meta, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf):
                rows.append({"frame": fr["frame"], "t": fr["t"], "tile": tile.idx, "cls": names[int(c)],
                             "conf": float(p), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                             **box_to_sphere(tile, x1, y1, x2, y2)})
        done = min(b + batch_frames, len(index))
        rate = done / max(time.time() - t0, 1e-6)
        print(f"\r[detect] {walk} {tag}: {done}/{len(index)} frames  {rate:.1f} fps  {len(rows)} boxes", end="", flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "detections.parquet", index=False)
    st.finish({"n_frames": len(index), "n_boxes": len(df),
               "class_counts": df["cls"].value_counts().head(30).to_dict() if len(df) else {}})
    print(f"[detect] {walk} {tag}: {len(df)} boxes -> {out_dir / 'detections.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--prompts", choices=list(PROMPT_SETS), default=None, help="open-vocabulary prompt set")
    ap.add_argument("--conf", type=float, default=0.05, help="keep low: thresholds are applied downstream")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch-frames", type=int, default=2, help="frames per GPU batch (x8 tiles)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (quick test)")
    ap.add_argument("--frames", default="frames", choices=["frames", "frames_stab"], help="frames_stab = offline-stabilised frames")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.walks:
        if w not in WALKS:
            sys.exit(f"unknown walk {w!r}")
        run(w, a.model, a.prompts, a.conf, a.imgsz, a.batch_frames, a.limit, a.force, a.frames)


if __name__ == "__main__":
    main()
