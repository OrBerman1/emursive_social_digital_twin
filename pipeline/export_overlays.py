"""Cheap stage: tracks + objects -> compact per-frame overlay JSON for the viewer.

Also writes a first GIS-ready observation table (GeoJSON) with persistent IDs. Until Boaz's anchor
points arrive, position along the route uses an assumed walking pace and lat/lon is left out.

usage:  python pipeline/export_overlays.py walk1 [--pace 0.9]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, WALKS

FPS_ANALYSIS = 5.0
R = lambda v, n=1: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


def run(walk: str, pace: float, level2: str | None, variant: str = "") -> dict:
    tdir = INTERIM / walk / ("tracks" + variant)
    persons = pd.read_parquet(tdir / "persons.parquet")
    objects = pd.read_parquet(tdir / "objects.parquet")
    l2 = None
    if level2 and (tdir / f"{level2}.parquet").exists():
        l2 = pd.read_parquet(tdir / f"{level2}.parquet")
    attr_p = INTERIM / walk / f"attributes{variant}" / "attributes.parquet"
    attrs = pd.read_parquet(attr_p).set_index("track") if attr_p.exists() else None
    ocr_dir = INTERIM / walk / f"ocr_vlm{variant}"                      # vision-model readings if available ...
    if not (ocr_dir / "signs.parquet").exists():
        ocr_dir = INTERIM / walk / f"ocr{variant}"                      # ... else the Tesseract result
    signs = pd.read_parquet(ocr_dir / "signs.parquet") if (ocr_dir / "signs.parquet").exists() else None
    sign_boxes = pd.read_parquet(ocr_dir / "sign_boxes.parquet") if (ocr_dir / "sign_boxes.parquet").exists() else None
    n_frames = json.loads((INTERIM / walk / ("frames" + variant) / "manifest.json").read_text(encoding="utf-8"))["n_frames"]
    summary = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))

    box = lambda r: [R(r.yaw), R(r.yaw_l), R(r.yaw_r), R(r.pitch_top), R(r.pitch_bot)]
    frames = [{"p": [], "o": [], "l2": [], "s": []} for _ in range(n_frames)]
    for r in persons.itertuples():
        frames[int(r.frame)]["p"].append([int(r.track), R(r.conf, 2), *box(r), R(r.dist_m)])
    for r in objects.itertuples():
        frames[int(r.frame)]["o"].append([r.cls, R(r.conf, 2), *box(r), int(r.track)])
    if l2 is not None:
        for r in l2.itertuples():
            frames[int(r.frame)]["l2"].append([r.cls, R(r.conf, 2), *box(r), R(getattr(r, "dist_m", None)), int(r.track)])

    sign_len = {}
    if sign_boxes is not None:
        smoothed = "interp" in sign_boxes.columns        # tracked + gap-filled + smoothed (ocr_vlm merge)
        for r in sign_boxes.itertuples():
            fs = (int(r.frame),) if smoothed else (int(r.frame), int(r.frame) + 1)   # old Tesseract output: every 2nd frame
            for f in fs:
                if f < n_frames:
                    frames[f]["s"].append([int(r.track), 1, *box(r)])                 # [sign, conf, yaw, yaw_l, yaw_r, pitch_top, pitch_bot]
        sign_len = {int(k): int(v) for k, v in sign_boxes.groupby("track").size().items()}
    tracks = (persons.groupby("track")
              .agg(first=("frame", "min"), last=("frame", "max"), n=("frame", "count"), conf=("conf", "max"),
                   dist=("dist_m", "median"))
              .reset_index())
    obj_len = {int(k): int(v) for k, v in objects.groupby("track").size().items()}
    if l2 is not None:
        obj_len.update({int(k): int(v) for k, v in l2.groupby("track").size().items()})
    out = {
        "walk": walk, "obj_len": obj_len, "sign_len": sign_len, "variant": variant or "raw",
        "video": "video_stab/equirect.mp4" if variant else "video/equirect.mp4", "frames_dir": "frames" + variant, "fps": FPS_ANALYSIS, "n_frames": n_frames, "pace_m_s": pace,
        "camera_height_m": summary["camera_height_used_m"],
        "fields": {"p": ["track", "conf", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot", "dist_m"],
                   "o": ["cls", "conf", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot", "track"],
                   "l2": ["cls", "conf", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot", "dist_m", "track"],
                   "s": ["sign", "conf", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot"]},
        "tracks_fields": ["first", "last", "n", "conf", "dist", "sex", "age"], "signs_fields": ["text", "lang", "conf", "readable", "raw"], "ocr_source": ocr_dir.name,
        "tracks": {int(t.track): [int(t.first), int(t.last), int(t.n), R(t.conf, 2), R(t.dist),
                                   *( [attrs.at[t.track, "sex"], attrs.at[t.track, "age"]] if attrs is not None and t.track in attrs.index else ["uncertain", "uncertain"])]
                   for t in tracks.itertuples()},
        "signs": ({int(r.track): [r.text if r.readable else "", r.lang, R(r.conf), bool(r.readable), r.raw] for r in signs.itertuples()} if signs is not None else {}),
        "frames": frames,
    }
    pdir = PROCESSED / walk
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"overlays{variant}.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    # GIS-ready observations: one feature per persistent entity (geometry pending anchors)
    feats = []
    for t in tracks.itertuples():
        feats.append({"type": "Feature", "geometry": None, "properties": {
            "entity_id": f"P-{int(t.track):04d}", "entity_type": "person", "layer": "social", "class": "person",
            "frame_start": int(t.first), "frame_end": int(t.last), "t_start": R(t.first / FPS_ANALYSIS),
            "t_end": R(t.last / FPS_ANALYSIS), "chainage_m": R(t.first / FPS_ANALYSIS * pace),
            "chainage_basis": f"assumed pace {pace} m/s (replace with anchors)", "confidence": R(t.conf, 2),
            "median_distance_m": R(t.dist), "model": "YOLO11x + angular-IoU tracker", "walk_id": walk}})
    (pdir / f"observations{variant}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    size = (pdir / f"overlays{variant}.json").stat().st_size
    return {"walk": walk, "frames": n_frames, "tracks": len(tracks), "overlay_bytes": size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--pace", type=float, default=0.9, help="assumed walking pace (m/s) until anchors exist")
    ap.add_argument("--level2", default="level2", help="name of the level-2 tracks file, if present")
    ap.add_argument("--variant", default="", choices=["", "stab"])
    a = ap.parse_args()
    for w in a.walks:
        print(run(w, a.pace, a.level2, "_stab" if a.variant == "stab" else ""))


if __name__ == "__main__":
    main()
