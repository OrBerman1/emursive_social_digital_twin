"""Stage (CLIP part cached): people detected on signs, posters and portraits -> category "possible_rabi_sign".

The person detector also fires on pictured people (rabbi portraits, advertisements, posters). They must not be
counted as people. Four cues per person track, combined:
  1. CLIP ViT-L/14 zero-shot on the best crops WITH context (the sign frame matters): "real person in a market"
     vs "printed picture / portrait / poster of a person"  -> p_depicted.
  2. implied body height on flat ground: top of the box above the ground at the foot distance. Real people
     are ~1.0-2.1 m; pictures come out implausible, or their "feet" are above the horizon (hanging).
  3. closeness to signs: share of frames where the person box, enlarged by 30%, touches a detected text region
     (ocr_vlm sign boxes) -> pictures sit on signs next to text.
  4. no own motion (motion.py speed) - only as a tie-breaker (standing real people are also still).
Automatic cues alone were not reliable enough (CLIP >= 0.75 also fires on people behind glass, drivers and the
operator's printed T-shirt; the size rule fires on far real people). So the cues PROPOSE candidates
(p_depicted >= 0.3) and a vision review decides (config/depicted_review.json: depicted / operator / real),
like the sign reading. With --api (new projects, ANTHROPIC_API_KEY set) the candidates are reviewed by Claude through
the API (vlm.py; one context crop per track on numbered sheets; cached in depicted<_variant>/api_review.json).
Unreviewed tracks fall back to p_depicted >= 0.9.
Operator heuristic (crowds.operator_tracks: median box height >= 58 deg, or >= 48 deg near the operator direction)
marks the remaining camera-operator tracks as category "operator"; every downstream stage keeps only "person".
Output : data/interim/<walk>/depicted<_variant>/depicted.parquet
         (track, p_depicted, height_m, feet_up, near_sign, med_spd, is_depicted, reason)

usage:  python pipeline/depicted.py [walk1 walk2] [--variant stab] [--force] [--api]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import cv2
import numpy as np
import pandas as pd
import torch

from attributes import MODEL, crop, text_features
from common import CONFIG, INTERIM, PROCESSED, ROOT, WALKS, Stage

VERSION = 1
CATEGORY = "possible_rabi_sign"
N_CROPS = 5
PROMPTS = {
    "real": ["a photo of a real person standing in a street market", "a photo of a shopper walking in a market",
             "a photo of a vendor at a market stall", "a candid photo of a person on the street"],
    "depicted": ["a printed picture of a person on a shop sign", "a portrait photograph of a rabbi on a poster",
                 "an advertisement poster showing a person", "a framed portrait of a bearded man hanging on a wall",
                 "a painted picture of a face on a sign", "a sticker with a photo of a man"],
}
H_MIN, H_MAX = 0.9, 2.2        # plausible standing body height (m); children ~1.0 m
REVIEW = CONFIG / "depicted_review.json"
P_CANDIDATE, P_FALLBACK = 0.3, 0.9


def fhash(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def clip_scores(walk: str, sfx: str, tr: pd.DataFrame, force: bool) -> pd.DataFrame:
    tpath = INTERIM / walk / f"tracks{sfx}" / "persons.parquet"
    out_dir = INTERIM / walk / f"depicted{sfx}"
    st = Stage("depicted_clip", VERSION, out_dir, {"model": MODEL, "n_crops": N_CROPS, "prompts": PROMPTS}, {"tracks": fhash(tpath)})
    cache = out_dir / "clip.parquet"
    if st.done() and not force and cache.exists():
        print(f"[depicted] {walk}{sfx}: CLIP cached")
        return pd.read_parquet(cache)
    st.start()
    import clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL, device=dev)
    names, tf = text_features(model, clip, PROMPTS, dev)
    from PIL import Image
    cand = tr[(~tr.interp) & (tr.conf >= 0.4)]
    picks = cand.sort_values("ang_h", ascending=False).groupby("track").head(N_CROPS)
    rows = []
    for f, g in picks.groupby("frame"):
        equi = cv2.imread(str(INTERIM / walk / f"frames{sfx}" / f"{int(f):06d}.jpg"))
        ims = [preprocess(Image.fromarray(cv2.cvtColor(crop(equi, r, isolate=False), cv2.COLOR_BGR2RGB))) for r in g.itertuples()]
        with torch.no_grad():
            im = model.encode_image(torch.stack(ims).to(dev)).float()
            im = im / im.norm(dim=-1, keepdim=True)
            p = (100 * im @ tf.T).softmax(-1).cpu().numpy()
        for r, pp in zip(g.itertuples(), p):
            rows.append((int(r.track), int(f), float(pp[names.index("depicted")])))
    df = pd.DataFrame(rows, columns=["track", "frame", "p_depicted"])
    df.to_parquet(cache, index=False)
    st.finish({"crops": len(df)})
    return df


API_PROMPT = """Each numbered cell shows one person detection (green box) from a 360-degree walk through a street market,
with its surroundings. The camera is held upside down by an operator, so the operator's arm, hand or T-shirt can appear
as a large close-up. For EVERY id decide what the green box contains:
- real: a real, physically present person (shopper, vendor, passer-by, driver, a person behind glass)
- depicted: a picture of a person - on a sign, poster, advertisement, portrait, sticker, packaging or screen
- operator: the camera operator (close-up arm, hand, body or T-shirt print)
- unclear: cannot be decided from this crop"""


def api_review(walk: str, sfx: str, tr: pd.DataFrame, cand: list[int], skip: set) -> dict:
    """Claude reviews the candidate tracks (one context crop each, 12 per sheet); cached per tracks file."""
    import vlm
    from collections import namedtuple
    out_dir = INTERIM / walk / f"depicted{sfx}"
    cache_p = out_dir / "api_review.json"
    th = fhash(INTERIM / walk / f"tracks{sfx}" / "persons.parquet")
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    if cache.get("tracks_hash") != th:
        cache = {"tracks_hash": th, "model": vlm.MODEL, "labels": {}}
    todo = [t for t in cand if str(t) not in cache["labels"] and t not in skip]
    R = namedtuple("R", "yaw yaw_l yaw_r pitch_top pitch_bot")
    best = tr[(~tr.interp) & tr.track.isin(todo)].sort_values("ang_h", ascending=False).groupby("track").head(1)
    best = best.sort_values("track")
    item = {"type": "string", "enum": ["real", "depicted", "operator", "unclear"]}
    n = -(-len(best) // 12)
    for k in range(n):
        part = best.iloc[k * 12:(k + 1) * 12]
        ims, ids = [], []
        for r in part.itertuples():
            equi = cv2.imread(str(INTERIM / walk / f"frames{sfx}" / f"{int(r.frame):06d}.jpg"))
            hw, hh = ((r.yaw_r - r.yaw_l) % 360) / 2, (r.pitch_top - r.pitch_bot) / 2
            pm = (r.pitch_top + r.pitch_bot) / 2
            big = R(r.yaw, r.yaw - 2 * hw, r.yaw + 2 * hw, pm + 2 * hh, pm - 2 * hh)     # 2x context around the box
            im = crop(equi, big, size=360, isolate=False)
            span = max(4 * hh, 4 * hw) * 1.25
            bw, bh = 360 * 2 * hw / span, 360 * 2 * hh / span
            cv2.rectangle(im, (int(180 - bw / 2), int(180 - bh / 2)), (int(180 + bw / 2), int(180 + bh / 2)), (0, 220, 0), 2)
            ims.append(im); ids.append(f"T{int(r.track)}")
        try:
            ans = vlm.ask(vlm.sheet(ims, ids), API_PROMPT, vlm.id_map_schema(ids, item))
            cache["labels"].update({i[1:]: lab for i, lab in ans.items()})
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_p.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        except Exception as e:                                         # noqa: BLE001 - unreviewed tracks keep the fallback
            print(f"[depicted api] {walk}: sheet {k + 1} failed: {e}", flush=True)
        print(f"[depicted api] {walk} sheet {k + 1}/{n}", flush=True)
    return {int(t): lab for t, lab in cache["labels"].items() if lab in ("real", "depicted", "operator")}


def run(walk: str, variant: str, force: bool, use_api: bool = False) -> dict:
    sfx = f"_{variant}" if variant else ""
    tdir = INTERIM / walk / f"tracks{sfx}"
    tr = pd.read_parquet(tdir / "persons.parquet")
    h = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))["camera_height_used_m"]
    cl = clip_scores(walk, sfx, tr, force)
    p_dep = cl.groupby("track").p_depicted.median()

    # implied height (top of the box above flat ground at the foot distance) and feet above the horizon
    ok = tr.pitch_bot < -1
    d = np.where(ok, h / np.tan(np.radians(-tr.pitch_bot.clip(upper=-1))), np.nan)
    tr = tr.assign(height=h + d * np.tan(np.radians(tr.pitch_top)), feet_up=tr.pitch_bot > 0)

    # closeness to text regions (signs)
    sb_path = INTERIM / walk / f"ocr_vlm{sfx}" / "sign_boxes.parquet"
    near = pd.Series(0.0, index=tr.track.unique())
    if sb_path.exists():
        sb = pd.read_parquet(sb_path)
        sbf = {f: g[["yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot"]].to_numpy() for f, g in sb.groupby("frame")}
        hit = np.zeros(len(tr), bool)
        for i, r in enumerate(tr.itertuples()):
            s = sbf.get(r.frame)
            if s is None:
                continue
            w = ((r.yaw_r - r.yaw_l) % 360) * 1.3 / 2; hh = (r.pitch_top - r.pitch_bot) * 1.3 / 2; pm = (r.pitch_top + r.pitch_bot) / 2
            dy = np.abs(((s[:, 0] - r.yaw + 540) % 360) - 180)
            sw = ((s[:, 2] - s[:, 1]) % 360) / 2
            hit[i] = bool(np.any((dy <= w + sw) & (s[:, 4] <= pm + hh) & (s[:, 3] >= pm - hh)))
        near = pd.Series(hit, index=tr.index).groupby(tr.track).mean()

    mo_path = PROCESSED / walk / f"motion{sfx}.json"
    spd = {}
    if mo_path.exists():
        mo = json.loads(mo_path.read_text(encoding="utf-8"))
        spd = {int(t): mo["tracks"][str(g)]["med_spd"] for t, g in mo["id2g"].items()}

    review = json.loads(REVIEW.read_text(encoding="utf-8")).get(walk, {}) if REVIEW.exists() else {}
    from crowds import operator_tracks                                   # camera operator (arm / T-shirt) heuristic
    ops = operator_tracks(tr)
    label_of = {int(t): lab for lab in ("depicted", "operator", "real") for t in review.get(lab, [])}
    api_of = {}
    if use_api:
        cand = sorted(int(t) for t, p in p_dep.items() if p >= P_CANDIDATE)
        api_of = api_review(walk, sfx, tr, cand, set(label_of))
    rows = []
    for tid, t in tr.groupby("track"):
        p = float(p_dep.get(tid, np.nan))
        hm = float(np.nanmedian(t.height)) if t.height.notna().any() else np.nan
        fu = float(t.feet_up.mean())
        ns = float(near.get(tid, 0.0))
        sp = spd.get(int(tid), np.nan)
        implausible = (fu > 0.5) or (np.isfinite(hm) and (hm < H_MIN or hm > H_MAX))
        lab = label_of.get(int(tid))
        alab = api_of.get(int(tid))
        if lab:
            category, reason = {"depicted": CATEGORY, "operator": "operator", "real": "person"}[lab], f"vision review: {lab}"
        elif alab:
            category, reason = {"depicted": CATEGORY, "operator": "operator", "real": "person"}[alab], f"Claude API review: {alab}"
        elif int(tid) in ops:
            category, reason = "operator", "operator heuristic (huge box in the operator direction)"
        elif np.isfinite(p) and p >= P_FALLBACK:
            category, reason = CATEGORY, f"CLIP p={p:.2f} (not reviewed)"
        else:
            category, reason = "person", ""
        rows.append((int(tid), round(p, 3) if np.isfinite(p) else None, round(hm, 2) if np.isfinite(hm) else None, round(fu, 2),
                     round(ns, 2), sp if np.isfinite(sp) else None, bool(np.isfinite(p) and p >= P_CANDIDATE), implausible,
                     category, category == CATEGORY, reason))
    df = pd.DataFrame(rows, columns=["track", "p_depicted", "height_m", "feet_up", "near_sign", "med_spd", "candidate", "implausible",
                                     "category", "is_depicted", "reason"])
    out_dir = INTERIM / walk / f"depicted{sfx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "depicted.parquet", index=False)
    return {"walk": walk, "tracks": len(df), "candidates": int(df.candidate.sum()), "categories": df.category.value_counts().to_dict()}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--api", action="store_true", help="review the candidates with Claude (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.variant, a.force, a.api), ensure_ascii=False))


if __name__ == "__main__":
    main()
