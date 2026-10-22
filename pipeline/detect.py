"""Stage `det_<name>`: run a detector on perspective tiles of every equirect frame (cached, GPU).

Raw detections are saved at a LOW confidence threshold with their tile box and spherical extent, so
thresholds, class filters, cross-tile de-duplication and tracking can all be redone later without the GPU.

  level 1:  python pipeline/detect.py walk1 --model yolo11x.pt                         (COCO classes)
  level 2:  python pipeline/detect.py walk1 --model yolov8x-worldv2.pt --prompts level2 (open vocabulary)
"""
from __future__ import annotations

import argparse
import json
import os
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
    # floor waste (waste.py). The last five are look-alikes that are NOT waste, so the open-vocabulary
    # classifier has somewhere else to put them; waste.py drops them.
    "waste": [
        "litter on the ground", "crumpled paper", "paper cup", "plastic bottle", "drink can", "plastic wrapper",
        "plastic bag on the ground", "flattened cardboard on the ground", "torn cardboard box", "garbage bag",
        "vegetable leaves on the ground", "crushed fruit on the ground", "food waste", "napkin",
        "shoe", "manhole cover", "drain grate", "crate of produce", "puddle",
    ],
}


CHECKPOINT_S = 300         # save the boxes so far every 5 minutes (resume after an interruption)


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
    # checkpoint (CPU-only computers take many hours per walk): every few minutes the boxes so far are saved, and an
    # interrupted run of the same stage (same cache key) continues after the last saved frame instead of from frame 0
    ck_rows, ck_meta = out_dir / "partial.parquet", out_dir / "partial.json"
    rows, start = [], 0
    try:
        ck = json.loads(ck_meta.read_text(encoding="utf-8"))
        if ck.get("key") == st.key and not force:
            start = int(ck["frames_done"])
            rows = pd.read_parquet(ck_rows).to_dict("records") if ck_rows.exists() else []
            print(f"[detect] {walk} {tag}: continuing after frame {start}/{len(index)} (saved by an interrupted run)", flush=True)
    except (OSError, ValueError, KeyError):
        pass
    t0, t_ck = time.time(), time.time()
    for b in range(start, len(index), batch_frames):
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
        if time.time() - t_ck > CHECKPOINT_S and done < len(index):
            pd.DataFrame(rows).to_parquet(ck_rows.with_suffix(".tmp"), index=False)
            os.replace(ck_rows.with_suffix(".tmp"), ck_rows)
            ck_meta.write_text(json.dumps({"key": st.key, "frames_done": done}), encoding="utf-8")
            t_ck = time.time()
        rate = (done - start) / max(time.time() - t0, 1e-6)
        print(f"\r[detect] {walk} {tag}: {done}/{len(index)} frames  {rate:.1f} fps  {len(rows)} boxes", end="", flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "detections.parquet", index=False)
    st.finish({"n_frames": len(index), "n_boxes": len(df),
               "class_counts": df["cls"].value_counts().head(30).to_dict() if len(df) else {}})
    for f in (ck_meta, ck_rows):
        f.unlink(missing_ok=True)
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
