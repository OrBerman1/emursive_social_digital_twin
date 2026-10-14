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
    # people on signs/posters (category possible_rabi_sign) and the operator are not counted as people
    dep_p = INTERIM / walk / f"depicted{variant}" / "depicted.parquet"
    dep = pd.read_parquet(dep_p) if dep_p.exists() else None
    depicted = persons.iloc[0:0]
    if dep is not None:
        cat = dict(zip(dep.track, dep.category))
        c = persons.track.map(cat).fillna("person")
        depicted = persons[c == "possible_rabi_sign"]
        persons = persons[c == "person"]
    n_frames = json.loads((INTERIM / walk / ("frames" + variant) / "manifest.json").read_text(encoding="utf-8"))["n_frames"]
    summary = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))

    box = lambda r: [R(r.yaw), R(r.yaw_l), R(r.yaw_r), R(r.pitch_top), R(r.pitch_bot)]
    frames = [{"p": [], "o": [], "l2": [], "s": [], "dp": [], "fd": [], "w": []} for _ in range(n_frames)]
    waste_p = tdir / "waste.parquet"                                    # waste on the floor (waste.py)
    waste = pd.read_parquet(waste_p) if waste_p.exists() else None
    if waste is not None:
        for r in waste.itertuples():
            frames[int(r.frame)]["w"].append([r.type, R(r.p_waste, 2), *box(r), int(r.track)])
    food_p = tdir / "food.parquet"                                      # food on display (food.py, Open Images V7)
    food = pd.read_parquet(food_p) if food_p.exists() else None
    if food is not None:
        for r in food.itertuples():
            frames[int(r.frame)]["fd"].append([r.cls, R(r.conf, 2), *box(r), int(r.track)])
    for r in persons.itertuples():
        frames[int(r.frame)]["p"].append([int(r.track), R(r.conf, 2), *box(r), R(r.dist_m)])
    for r in depicted.itertuples():
        frames[int(r.frame)]["dp"].append([int(r.track), R(r.conf, 2), *box(r)])
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
        per_row = 1 if smoothed else 2                   # Tesseract rows stand for 2 frames each (see above)
        sign_len = {int(k): int(v) * per_row for k, v in sign_boxes.groupby("track").size().items()}
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
                   "s": ["sign", "conf", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot"],
                   "dp": ["track", "conf", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot"],
                   "fd": ["cls", "conf", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot", "track"],
                   "w": ["type", "p_waste", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot", "track"]},
        "waste_len": ({int(k): int(v) for k, v in waste.groupby("track").size().items()} if waste is not None else {}),
        "food_len": ({int(k): int(v) for k, v in food.groupby("track").size().items()} if food is not None else {}),
        "food_group": ({c: g for c, g in zip(food.cls, food.group)} if food is not None else {}),
        "dp_fields": ["first", "last", "n", "p_depicted", "reason"],
        "dp_tracks": ({int(t): [int(g.frame.min()), int(g.frame.max()), int(len(g)),
                                 R(dep.set_index("track").at[t, "p_depicted"], 2), dep.set_index("track").at[t, "reason"]]
                       for t, g in depicted.groupby("track")} if dep is not None else {}),
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
    for t, g in depicted.groupby("track"):
        feats.append({"type": "Feature", "geometry": None, "properties": {
            "entity_id": f"P-{int(t):04d}", "entity_type": "possible_rabi_sign", "layer": "material", "class": "possible_rabi_sign",
            "frame_start": int(g.frame.min()), "frame_end": int(g.frame.max()), "t_start": R(g.frame.min() / FPS_ANALYSIS),
            "t_end": R(g.frame.max() / FPS_ANALYSIS), "confidence": R(g.conf.max(), 2),
            "model": "YOLO11x person + CLIP proposal + vision review (pipeline/depicted.py)", "walk_id": walk}})
    cr_p = pdir / f"crowds{variant}.json"                              # crowds (crowds.py): one feature per crowd, ID C<walk>-<nn>
    if cr_p.exists():
        for c in json.loads(cr_p.read_text(encoding="utf-8")).get("crowds_list", []):
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]}, "properties": {
                "entity_id": c.get("crowd_id", f"C-{c['id']}"), "entity_type": "crowd", "layer": "social", "class": "crowd",
                "frame_start": c["frame_start"], "frame_end": c["frame_end"], "t_start": c["t_start_s"], "t_end": c["t_end_s"],
                "chainage_m": c["chainage_m"], "people_max_at_once": c["people_max_at_once"], "people_median": c["people_median"],
                "people_unique": c["people_unique"], "speed_mean_m_s": c["speed_mean_m_s"], "speed_std_m_s": c["speed_std_m_s"],
                "age_group": c["age_group"], "perceived_gender": c["perceived_gender"], "members": c["members"],
                "model": "crowds.py heuristic (overlapping similar-size person boxes)", "walk_id": walk}})
    (pdir / f"observations{variant}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    size = (pdir / f"overlays{variant}.json").stat().st_size
    return {"walk": walk, "frames": n_frames, "tracks": len(tracks), "possible_rabi_sign": int(depicted.track.nunique()), "overlay_bytes": size}


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
