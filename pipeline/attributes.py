"""Stage `attributes`: perceived gender expression (man/woman) + broad age group per person TRACK (cached, GPU).

Method (level 1 attribute, body-based because faces are tiny in 360 footage):
  - per track pick the N largest, confident, non-interpolated boxes
  - cut an undistorted crop around each box from the stabilised frame (perspective re-projection)
  - CLIP ViT-L/14 zero-shot against prompt ensembles; average probabilities over the track
  - label only when the track average is clear; otherwise "uncertain" (also when every crop is too small)
Privacy: crops are not stored; only per-track labels and probabilities. Show results only in aggregate.
Vocabulary follows the team's coding scheme (perceived gender expression; age groups 18-35 / 36-60 / 61+).

usage:  python pipeline/attributes.py walk1 walk2 [--variant stab] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import torch

from common import INTERIM, WALKS, Stage
from spherical import Tile, tile_to_sphere

VERSION = 2   # v2: crops isolate the person (neighbours greyed out)
MODEL = "ViT-L/14"
SEX_PROMPTS = {
    "man": ["a photo of a man", "a full-body photo of a man", "a man walking in a market", "a photo of a male shopper", "a man standing at a market stall"],
    "woman": ["a photo of a woman", "a full-body photo of a woman", "a woman walking in a market", "a photo of a female shopper", "a woman standing at a market stall"],
}
AGE_PROMPTS = {
    "child": ["a photo of a child", "a photo of a kid"],
    "young adult 18-35": ["a photo of a young adult in their twenties", "a photo of a young person"],
    "middle adult 36-60": ["a photo of a middle-aged person", "a photo of a person in their fifties"],
    "older adult 61+": ["a photo of an elderly person", "a photo of an old person with grey hair"],
}
MIN_ANG_H = 6.0   # degrees; ~64 px at 3840 px equirect width. Smaller people are left uncertain.


def crop(equi: np.ndarray, r, size: int = 288, isolate: bool = True) -> np.ndarray:
    """Undistorted square crop centred on a spherical box, with ~20% context. With isolate=True everything
    outside the person's own box is greyed out, so neighbours in the crop don't sway the classifier."""
    pm = (r.pitch_top + r.pitch_bot) / 2
    span = max(r.pitch_top - r.pitch_bot, ((r.yaw_r - r.yaw_l) % 360)) * 1.25
    t = Tile(0, float(r.yaw), float(pm), float(np.clip(span, 8, 110)), size)
    uu, vv = np.meshgrid(np.arange(size) + .5, np.arange(size) + .5)
    yaw, pitch = tile_to_sphere(t, uu, vv)
    h, w = equi.shape[:2]
    mx = (((yaw + 180) / 360 * w - .5) % w).astype(np.float32)
    my = np.clip((90 - pitch) / 180 * h - .5, 0, h - 1).astype(np.float32)
    out = cv2.remap(equi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    if isolate:
        dy = ((yaw - r.yaw + 180) % 360) - 180
        half_w = (((r.yaw_r - r.yaw_l) % 360) / 2) * 1.08
        inside = (np.abs(dy) <= half_w) & (pitch <= r.pitch_top + 1.0) & (pitch >= r.pitch_bot - 1.0)
        out[~inside] = 127
    return out


def text_features(model, clip, prompts: dict, dev) -> tuple[list[str], torch.Tensor]:
    names, feats = [], []
    with torch.no_grad():
        for name, ps in prompts.items():
            f = model.encode_text(clip.tokenize(ps).to(dev)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            m = f.mean(0)
            feats.append(m / m.norm())
            names.append(name)
    return names, torch.stack(feats)


def run(walk: str, variant: str, n_crops: int, sure: float, force: bool) -> dict:
    tpath = INTERIM / walk / f"tracks{variant}" / "persons.parquet"
    out_dir = INTERIM / walk / f"attributes{variant}"
    tkey = hashlib.sha256(tpath.read_bytes()).hexdigest()[:16]
    params = {"model": MODEL, "n_crops": n_crops, "min_ang_h": MIN_ANG_H, "sex_prompts": SEX_PROMPTS, "age_prompts": AGE_PROMPTS}
    st = Stage("attributes", VERSION, out_dir, params, {"tracks": tkey})
    if st.done() and not force:
        print(f"[attributes] {walk}{variant}: cached")
        df = pd.read_parquet(out_dir / "attributes_raw.parquet")
    else:
        st.start()
        import clip
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(MODEL, device=dev)
        sex_names, sex_f = text_features(model, clip, SEX_PROMPTS, dev)
        age_names, age_f = text_features(model, clip, AGE_PROMPTS, dev)
        tr = pd.read_parquet(tpath)
        cand = tr[(~tr.interp) & (tr.conf >= 0.5) & (tr.pitch_bot > -60) & (tr.ang_h >= MIN_ANG_H)]
        picks = cand.sort_values("ang_h", ascending=False).groupby("track").head(n_crops)
        from PIL import Image
        frames_dir = INTERIM / walk / f"frames{variant}"
        rows = []
        for f, g in picks.groupby("frame"):
            equi = cv2.imread(str(frames_dir / f"{int(f):06d}.jpg"))
            ims = [preprocess(Image.fromarray(cv2.cvtColor(crop(equi, r), cv2.COLOR_BGR2RGB))) for r in g.itertuples()]
            with torch.no_grad():
                im = model.encode_image(torch.stack(ims).to(dev)).float()
                im = im / im.norm(dim=-1, keepdim=True)
                ps = (100 * im @ sex_f.T).softmax(-1).cpu().numpy()
                pa = (100 * im @ age_f.T).softmax(-1).cpu().numpy()
            for r, p_s, p_a in zip(g.itertuples(), ps, pa):
                rows.append({"track": int(r.track), "frame": int(f), "ang_h": float(r.ang_h),
                             **{f"p_{n}": float(v) for n, v in zip(sex_names, p_s)},
                             **{f"p_{n}": float(v) for n, v in zip(age_names, p_a)}})
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "attributes_raw.parquet", index=False)        # per-crop probabilities (no images)
        st.finish({"crops": len(df), "tracks_with_crops": int(df.track.nunique()) if len(df) else 0})

    # --- per-track decision (cheap; thresholds can change without the GPU) ---
    all_tracks = pd.read_parquet(tpath, columns=["track"]).track.unique()
    g = df.groupby("track")
    agg = g[[c for c in df.columns if c.startswith("p_")]].mean()
    agg["n_crops"] = g.size()
    agg = agg.reindex(all_tracks)
    agg["sex"] = np.where(agg.p_man >= sure, "man", np.where(agg.p_woman >= sure, "woman", "uncertain"))
    age_cols = [c for c in agg.columns if c.startswith("p_") and c not in ("p_man", "p_woman")]
    best = agg[age_cols].fillna(-1).idxmax(axis=1).str[2:]           # tracks without crops are all-NaN
    agg["age"] = np.where(agg[age_cols].max(axis=1) >= 0.5, best, "uncertain")
    agg.loc[agg.n_crops.isna(), ["sex", "age"]] = "uncertain"
    agg = agg.reset_index().rename(columns={"index": "track"})
    agg.to_parquet(out_dir / "attributes.parquet", index=False)
    return {"walk": walk + variant, "tracks": len(agg), "sex": agg.sex.value_counts().to_dict(), "age": agg.age.value_counts().to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab", choices=["", "stab"])
    ap.add_argument("--n-crops", type=int, default=5)
    ap.add_argument("--sure", type=float, default=0.75, help="track-average probability needed to assign man/woman")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    v = "_stab" if a.variant == "stab" else ""
    for w in a.walks:
        print(json.dumps(run(w, v, a.n_crops, a.sure, a.force), indent=2))


if __name__ == "__main__":
    main()
