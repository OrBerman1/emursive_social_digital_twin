"""Stage (floor masks + CLIP cached): waste on the floor -> items with type, map position, density per 10 m of route.

Inputs : data/interim/<walk>/det_yolov8x-worldv2_waste<_stab>/detections.parquet   (detect.py --prompts waste)
         data/interim/<walk>/det_yolov8x-worldv2_level2<_stab>/detections.parquet  (waste-like level-2 classes)
         data/interim/<walk>/frames<_stab>/*.jpg, data/processed/<walk>/georef.json
Method : 1. FLOOR MASK per frame: SegFormer-B2 (ADE20K, nvidia/segformer-b2-finetuned-ade-512-512) on the lower band
            of the equirect frame (pitch +5..-75 deg, 1536 px wide); floor = floor, road, sidewalk, earth, path,
            runway, dirt track. Cached as PNG per frame (also usable later for walkway width / encroachment).
         2. CANDIDATES: open-vocabulary detections of waste (conf >= 0.1), below the horizon, small; kept only when
            the ring around the box is mostly floor (>= 50%): waste lies ON the walkway, goods sit on stalls.
            Look-alike prompts (shoe, manhole cover, drain grate, crate of produce, puddle) are dropped.
         3. TRACKING as one class with the level-2 tracker -> persistent W-#### items.
         4. CLIP ViT-L/14 on up to 3 crops per item: waste types vs not-waste (stock boxes/crates, clean floor,
            feet, drains, puddles, trolleys). Waste when the waste prompts together have p >= P_WASTE.
         5. Position = camera position (georef) + ground distance along the item's bearing; route position =
            camera chainage at the item's middle frame; density = items per 10 m of route.
Output : data/interim/<walk>/tracks<_stab>/waste.parquet, data/processed/<walk>/waste<_stab>.json

usage:  python pipeline/waste.py [walk1 walk2] [--variant stab] [--force]
"""
from __future__ import annotations

from pathlib import Path

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

from attributes import MODEL, crop, text_features
from common import INTERIM, PROCESSED, ROOT, WALKS, Stage
from tracks import dedup_frame, track_per_class

VERSION = 2
FLOOR_VERSION = 2       # floor masks keep their own cache key (independent of the waste logic)
SEG_DIR = ROOT / "models" / "segformer-b2-ade"
FLOOR_IDS = [3, 6, 11, 13, 52, 54, 91]          # ADE20K: floor, road, sidewalk, earth, path, runway, dirt track
BAND_TOP, BAND_BOT, BAND_W = 5.0, -75.0, 1536
WASTE_CLASSES = {"litter on the ground", "crumpled paper", "paper cup", "plastic bottle", "drink can", "plastic wrapper",
                 "plastic bag on the ground", "flattened cardboard on the ground", "torn cardboard box", "garbage bag",
                 "vegetable leaves on the ground", "crushed fruit on the ground", "food waste", "napkin"}
L2_WASTE = {"flattened cardboard", "plastic bag", "organic waste", "fruit and vegetable waste", "packaging", "garbage bag"}
MERGE_M = 1.0
FLOOR_RING_MIN = 0.5
FLOOR_BELOW_MIN = 0.6      # the strip just under the item must be floor (not a table or a stall)
PERSON_OVERLAP_MAX = 0.3   # items overlapping a person box by more than this are carried bags, not waste
P_WASTE = 0.5
N_CROPS = 3
TYPES = {   # type -> (emoji, CLIP prompts)
    "cardboard": ("📦", ["flattened cardboard lying on the ground as trash", "a torn cardboard box thrown on the floor"]),
    "paper & litter": ("🧻", ["crumpled paper and small litter on the ground", "a paper cup and napkins thrown on the floor"]),
    "plastic bag": ("🛍️", ["a plastic bag lying on the ground as trash"]),
    "bottle / can": ("🥤", ["a plastic bottle or drink can lying on the ground"]),
    "organic waste": ("🍂", ["vegetable leaves and crushed fruit lying on the ground", "rotten food waste on the floor"]),
    "garbage bag": ("🗑️", ["a full black garbage bag on the ground"]),
}
NOT_WASTE = {
    "stock": ["boxes and crates of produce for sale on the floor", "sacks and boxes of goods waiting to be sold"],
    "clean floor": ["a clean floor of a market street", "an empty asphalt road surface"],
    "feet": ["a person's feet and shoes", "the legs of a person walking"],
    "drain": ["a manhole cover or a drain grate in the street"],
    "puddle": ["a wet puddle of water on the floor"],
    "trolley": ["a shopping trolley on wheels"],
    "carried bag": ["a plastic bag carried in a person's hand", "a shopping bag held by a shopper"],
    "bin": ["a trash bin with a bin bag", "a garbage can"],
    "table": ["cups and items on a table", "goods on a stall table"],
    "vehicle": ["an e-bike or a scooter", "a parked motorcycle"],
}


def band_rows(H: int):
    return int((0.5 - BAND_TOP / 180) * H), int((0.5 - BAND_BOT / 180) * H)


def floor_masks(walk: str, sfx: str, force: bool) -> Path:
    fdir = INTERIM / walk / f"frames{sfx}"
    out = INTERIM / walk / f"floor{sfx}"
    fman = json.loads((fdir / "manifest.json").read_text(encoding="utf-8"))
    st = Stage("floor", FLOOR_VERSION, out, {"model": "segformer-b2-ade", "band": [BAND_TOP, BAND_BOT], "w": BAND_W, "floor_ids": FLOOR_IDS},
               {"frames": fman.get("key")})
    if st.done() and not force:
        print(f"[waste] {walk}{sfx}: floor masks cached")
        return out
    st.start()
    from transformers import SegformerForSemanticSegmentation
    # GPU when there is one (fp16 on NVIDIA); otherwise Apple MPS or the CPU in fp32 (a computer without an NVIDIA driver)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.float16 if dev == "cuda" else torch.float32
    m = SegformerForSemanticSegmentation.from_pretrained(str(SEG_DIR)).to(dev).eval().to(dt)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    files = sorted(fdir.glob("*.jpg"))
    # an interrupted run of the same stage (same cache key, partial.json) keeps the masks it already wrote
    ck = out / "partial.json"
    try:
        resume = json.loads(ck.read_text(encoding="utf-8")).get("key") == st.key and not force
    except (OSError, ValueError):
        resume = False
    ck.write_text(json.dumps({"key": st.key}), encoding="utf-8")
    for i, p in enumerate(files):
        if resume and (out / f"{p.stem}.png").exists():
            continue
        im = cv2.imread(str(p)); H, W = im.shape[:2]
        r0, r1 = band_rows(H)
        band = cv2.resize(im[r0:r1], (BAND_W, int(BAND_W * (r1 - r0) / W)))
        x = torch.from_numpy(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).to(dev).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            lg = m(pixel_values=((x - mean) / std)[None].to(dt)).logits
            lg = torch.nn.functional.interpolate(lg.float(), size=band.shape[:2], mode="bilinear")
        fl = np.isin(lg.argmax(1)[0].cpu().numpy(), FLOOR_IDS)
        cv2.imwrite(str(out / f"{p.stem}.tmp.png"), (fl * 255).astype(np.uint8))
        os.replace(out / f"{p.stem}.tmp.png", out / f"{p.stem}.png")      # never a half-written mask
        if i % 200 == 0:
            print(f"[waste] {walk}{sfx}: floor {i}/{len(files)}", flush=True)
    st.finish({"frames": len(files)})
    ck.unlink(missing_ok=True)
    return out


def candidates(walk: str, sfx: str, fdir) -> pd.DataFrame:
    per = pd.read_parquet(INTERIM / walk / f"tracks{sfx}" / "persons.parquet")
    pboxes = {f: g[["yaw_l", "yaw_r", "pitch_top", "pitch_bot"]].to_numpy() for f, g in per.groupby("frame")}
    parts = []
    pw = INTERIM / walk / f"det_yolov8x-worldv2_waste{sfx}" / "detections.parquet"
    if pw.exists():
        d = pd.read_parquet(pw); parts.append(d[d.cls.isin(WASTE_CLASSES) & (d.conf >= 0.1)])
    pl = INTERIM / walk / f"det_yolov8x-worldv2_level2{sfx}" / "detections.parquet"
    if pl.exists():
        d = pd.read_parquet(pl); parts.append(d[d.cls.isin(L2_WASTE) & (d.conf >= 0.1)])
    d = pd.concat(parts, ignore_index=True)
    d["ang_h"] = d.pitch_top - d.pitch_bot
    d = d[(d.pitch_top < 0) & (d.pitch_bot > BAND_BOT + 2) & (d.ang_h < 20) & (((d.yaw_r - d.yaw_l) % 360) < 30)].copy()
    d["det_cls"] = d.cls
    # floor ring test per frame
    keep, ring = [], []
    for f, g in d.groupby("frame"):
        mp = fdir / f"{int(f):06d}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
        hb, wb = m.shape
        X = lambda yaw: (np.asarray(yaw) / 360 + 0.5) * wb
        Y = lambda pt: (BAND_TOP - np.asarray(pt)) / (BAND_TOP - BAND_BOT) * hb
        for r in g.itertuples():
            x0, x1, y0, y1 = X(r.yaw_l), X(r.yaw_r), Y(r.pitch_top), Y(r.pitch_bot)
            if x1 < x0:
                continue                                             # wraps around +-180: skip (rare)
            ex, ey = (x1 - x0) * 0.5 + 2, (y1 - y0) * 0.5 + 2
            X0, X1, Y0, Y1 = int(max(0, x0 - ex)), int(min(wb, x1 + ex)), int(max(0, y0 - ey)), int(min(hb, y1 + ey))
            win = m[Y0:Y1, X0:X1]
            inner = np.zeros_like(win)
            inner[int(y0) - Y0:int(y1) - Y0 + 1, int(x0) - X0:int(x1) - X0 + 1] = True
            ringpx = win[~inner]
            fr = float(ringpx.mean()) if ringpx.size else 0.0
            yb0, yb1 = int(y1), int(min(hb, y1 + max(2, (y1 - y0) * 0.25)))
            below = m[yb0:yb1, int(x0):int(x1) + 1]
            fb = float(below.mean()) if below.size else 0.0
            # overlap with any person box (carried bags)
            ov = 0.0
            pb = pboxes.get(f)
            if pb is not None:
                aw = max(1e-6, (r.yaw_r - r.yaw_l) * (r.pitch_top - r.pitch_bot))
                ix = np.clip(np.minimum(pb[:, 1], r.yaw_r) - np.maximum(pb[:, 0], r.yaw_l), 0, None)
                iy = np.clip(np.minimum(pb[:, 2], r.pitch_top) - np.maximum(pb[:, 3], r.pitch_bot), 0, None)
                ov = float((ix * iy).max() / aw)
            if fr >= FLOOR_RING_MIN and fb >= FLOOR_BELOW_MIN and ov <= PERSON_OVERLAP_MAX:
                keep.append(r.Index); ring.append(fr)
    d = d.loc[keep].assign(floor_ring=ring)
    d["cls"] = "waste"
    return pd.concat([dedup_frame(g) for _, g in d.groupby("frame")]) if len(d) else d


def run(walk: str, variant: str, force: bool) -> dict:
    sfx = f"_{variant}" if variant else ""
    tdir = INTERIM / walk / f"tracks{sfx}"
    cam_h = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))["camera_height_used_m"]
    fdir = floor_masks(walk, sfx, force)
    d = candidates(walk, sfx, fdir)
    extra = d[["frame", "yaw", "det_cls", "floor_ring"]]
    w = track_per_class(d.drop(columns=["det_cls", "floor_ring"]), id_offset=0)
    w = w.merge(extra, on=["frame", "yaw"], how="left")
    w = w[w.groupby("track").frame.transform("count") >= 3]
    w["dist_m"] = np.where(w.pitch_bot < -1, cam_h / np.tan(np.radians(-w.pitch_bot.clip(upper=-1))), np.nan)

    # CLIP: waste type vs not waste (cached)
    out_dir = INTERIM / walk / f"waste{sfx}"
    prompts = {**{k: v[1] for k, v in TYPES.items()}, **{f"not:{k}": v for k, v in NOT_WASTE.items()}}
    key = hashlib.sha256(pd.util.hash_pandas_object(w[["track", "frame", "yaw", "pitch_top"]]).values.tobytes()).hexdigest()[:16]
    st = Stage("waste_clip", VERSION, out_dir, {"model": MODEL, "prompts": prompts, "n_crops": N_CROPS}, {"items": key})
    cache = out_dir / "clip.parquet"
    if st.done() and not force and cache.exists():
        lab = pd.read_parquet(cache)
    else:
        st.start()
        import clip
        from PIL import Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(MODEL, device=dev)
        names, tf = text_features(model, clip, prompts, dev)
        real = w[~w.interp] if "interp" in w.columns else w
        picks = real.sort_values("ang_h", ascending=False).groupby("track").head(N_CROPS)
        probs = {}
        for fr, g in picks.groupby("frame"):
            equi = cv2.imread(str(INTERIM / walk / f"frames{sfx}" / f"{int(fr):06d}.jpg"))
            rows = [r._replace(pitch_top=r.pitch_top + 2, pitch_bot=r.pitch_bot - 2) for r in g.itertuples()]
            ims = [preprocess(Image.fromarray(cv2.cvtColor(crop(equi, r, isolate=False), cv2.COLOR_BGR2RGB))) for r in rows]
            with torch.no_grad():
                im = model.encode_image(torch.stack(ims).to(dev)).float()
                im = im / im.norm(dim=-1, keepdim=True)
                p = (100 * im @ tf.T).softmax(-1).cpu().numpy()
            for r, pp in zip(g.itertuples(), p):
                probs.setdefault(int(r.track), []).append(pp)
        rows = []
        for tid, ps in probs.items():
            mp = np.mean(ps, axis=0)
            wi = [i for i, n in enumerate(names) if not n.startswith("not:")]
            ni = [i for i, n in enumerate(names) if n.startswith("not:")]
            best_w, best_n = max(wi, key=lambda i: mp[i]), max(ni, key=lambda i: mp[i])
            rows.append((tid, float(mp[wi].sum()), names[best_w], float(mp[best_w]), names[best_n][4:], float(mp[best_n])))
        lab = pd.DataFrame(rows, columns=["track", "p_waste", "type", "p_type", "not_type", "p_not"])
        lab.to_parquet(cache, index=False)
        st.finish({"items": len(lab)})
    w = w.merge(lab, on="track", how="inner")
    w = w[w.p_waste >= P_WASTE].copy()
    w["cls"] = w["type"]

    # map position, route chainage, density per 10 m
    g = json.loads((PROCESSED / walk / "georef.json").read_text(encoding="utf-8"))
    lat, lon, b0 = np.array(g["lat"]), np.array(g["lon"]), np.array(g["yaw0_bearing_deg"])
    kx, ky = 111320 * np.cos(np.radians(lat.mean())), 110574
    cx, cy = (lon - lon.mean()) * kx, (lat - lat.mean()) * ky
    chain = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
    items = []
    for tid, t in w.groupby("track"):
        fm = int(t.frame.median())
        tt = t[t.dist_m.notna()]
        if len(tt):
            r = tt.iloc[(tt.frame - fm).abs().argmin()]
            b = np.radians(b0[int(r.frame)] + r.yaw_foot)
            px, py = cx[int(r.frame)] + r.dist_m * np.sin(b), cy[int(r.frame)] + r.dist_m * np.cos(b)
            la, lo = lat.mean() + py / ky, lon.mean() + px / kx
        else:
            la, lo = lat[fm], lon[fm]
        items.append({"id": int(tid), "type": t["type"].iloc[0], "emoji": TYPES[t["type"].iloc[0]][0], "p_waste": round(float(t.p_waste.iloc[0]), 2),
                      "frame0": int(t.frame.min()), "frame1": int(t.frame.max()), "lat": round(float(la), 7), "lon": round(float(lo), 7),
                      "chainage_m": round(float(chain[fm]), 1), "det": str(t.det_cls.dropna().mode().iloc[0]) if t.det_cls.notna().any() else None})
    # the tracker sometimes splits one item into several tracks: merge items at the same spot (<= MERGE_M, <= 10 s apart)
    items.sort(key=lambda it: it["frame0"])
    merged = []
    for it in items:
        for mi in merged:
            dx = (it["lon"] - mi["lon"]) * kx; dy = (it["lat"] - mi["lat"]) * ky
            if np.hypot(dx, dy) <= MERGE_M and it["frame0"] - mi["frame1"] <= 50:
                mi["frame1"] = max(mi["frame1"], it["frame1"]); mi.setdefault("tracks", [mi["id"]]).append(it["id"])
                break
        else:
            merged.append(dict(it))
    n_tracks_items = len(items)
    items = merged
    to_item = {t: mi["id"] for mi in items for t in mi.get("tracks", [mi["id"]])}
    w["track"] = w.track.map(to_item).astype(int)                               # merged pieces share one item ID in the viewer
    w = w.drop_duplicates(["track", "frame"])
    # fill short gaps (<= 1 s) inside an item so it does not blink, e.g. between two merged pieces
    num = ["yaw", "yaw_l", "yaw_r", "yaw_foot", "pitch", "pitch_top", "pitch_bot", "ang_h", "dist_m"]
    filled = []
    for tid, t in w.groupby("track"):
        t = t.sort_values("frame").set_index("frame")
        full = t.reindex(range(t.index.min(), t.index.max() + 1))
        gap = full.yaw.isna()
        full[num] = full[num].interpolate(limit=5, limit_area="inside")
        full = full.ffill()
        full["interp"] = full["interp"].astype(bool) | gap
        filled.append(full[full.yaw.notna()].reset_index().rename(columns={"index": "frame"}))
    w = pd.concat(filled, ignore_index=True)
    w.to_parquet(tdir / "waste.parquet", index=False)
    for mi in items:
        mi["frames_seen"] = int((w.track == mi["id"]).sum())
    nb = int(np.ceil(chain[-1] / 10)) or 1
    per10 = np.bincount([min(nb - 1, int(it["chainage_m"] // 10)) for it in items], minlength=nb).tolist() if items else [0] * nb
    res = {"walk": walk, "items": len(items), "item_tracks_before_merge": n_tracks_items, "by_type": pd.Series([it["type"] for it in items]).value_counts().to_dict() if items else {},
           "route_m": round(float(chain[-1]), 1), "per_10m": per10, "candidates_tracked": int(lab.shape[0]),
           "rejected_as": lab[lab.p_waste < P_WASTE].not_type.value_counts().to_dict()}
    (PROCESSED / walk / f"waste{sfx}.json").write_text(json.dumps({**res, "items_list": items}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {k: v for k, v in res.items() if k != "per_10m"} | {"max_per_10m": max(per10) if per10 else 0}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.variant, a.force), ensure_ascii=False))


if __name__ == "__main__":
    main()
