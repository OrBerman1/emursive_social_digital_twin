"""Stage (crops + CLIP cached): what the stalls are made of, per GIS store, scored against the team's survey.

Inputs : data/processed/shops.json (store -> walk, frame, view yaw where the walk passes closest; georef.py)
         data/interim/<walk>/frames<_stab>/*.jpg
Method : 1. per store: search +-8 s around the closest approach (georef camera path) for frames where the store is
            2-8 m away and seen from the side (40-140 deg off the walking direction: frontage visible, operator
            behind); the 3 best views (>= 1 s apart) are cropped undistorted, width scaled to the distance.
            Stores without such a view are marked not visible. (v1 used the closest-approach frame only: many
            crops showed the street or the operator, 10-21 m away.)
         2. CLIP ViT-L/14 zero-shot, averaged over the 3 views, on two survey questions:
            facade (Open / Glass / Metal shutter) and structure (steel frame / blocks / cement), plus a set of
            detailed materials (each material vs its absence).
         3. scored against the survey fields Facade_Mat and Skeleton_Mat (Shops_In_hatikva_30_7_26.xlsx).
         4. vision review (Claude, contact sheets of the crops) in config/stall_review.json: facade, structure and
            visible materials per store, or null when the crop does not show the store; merged into the output.
            With --api (new projects, ANTHROPIC_API_KEY set) stores without a review are reviewed by Claude through the
            API (vlm.py; the store's views side by side, 8 stores per sheet) and added to the project's
            config/stall_review.json (source "Claude API"); stores already reviewed are not sent again.
Findings (13 Oct 2026): the survey is near-constant among visible stores (Facade Open 93%, Skeleton steel 97%), so
         accuracy against it cannot separate methods (CLIP 86% / 89% is below the majority baseline 93% / 97%);
         the review disagrees systematically (block-built butcher shops coded "steel", glass fronts coded "Open"),
         and dense 2-3 m wide stores + ~2-3 m route error put several stores' crops on the same frontage.
         Not shown in the viewer until store association (e.g. via shop names read on signs) and the survey
         definitions are settled.
Output : data/interim/<walk>/stalls<_stab>/crops/<OBJECTID>_<k>.jpg, data/processed/stall_materials.json

usage:  python pipeline/stall_materials.py [--variant stab] [--force] [--api]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import namedtuple

import cv2
import numpy as np
import torch

from attributes import MODEL, crop, text_features
from common import CONFIG, INTERIM, PROCESSED, Stage

VERSION = 2
SEARCH_F = 40                   # +-8 s around the closest approach
D_MIN, D_MAX = 2.0, 8.0
SIDE_MIN, SIDE_MAX = 40.0, 140.0
SPAN_UP, SPAN_DOWN = 35.0, -25.0
FACADE = {
    "Open": ["an open market stall with goods displayed at the front and no front wall",
             "an open-front market stall under an awning"],
    "Glass": ["a shop front with large glass windows and a glass door", "a glazed shop window"],
    "Metal": ["a shop front with a metal roller shutter", "a closed metal shutter on a shop"],
}
SKELETON = {
    "steel": ["a market stall built from a light steel frame and metal poles", "a stall under a metal roof structure"],
    "blocks": ["a small shop built of concrete blocks with plastered walls", "a masonry shop building"],
    "cement": ["a raw concrete building", "a cast concrete structure"],
}
DETAIL = {   # material -> (present prompt, absent prompt)
    "canvas / fabric awning": ("a market stall with a canvas fabric awning", "a market stall without any fabric awning"),
    "corrugated metal": ("a stall with corrugated metal sheets", "a stall without corrugated metal sheets"),
    "wood / timber": ("a stall built with wooden boards and timber", "a stall without wood"),
    "plastic sheeting": ("a stall covered with plastic sheeting or a tarp", "a stall without plastic sheeting"),
    "plastic crates as display": ("goods displayed in stacked plastic crates", "goods displayed without plastic crates"),
    "ceramic tiles": ("a shop with ceramic tiled walls", "a shop without tiled walls"),
    "glass": ("a shop with glass windows", "a shop without glass windows"),
    "steel frame": ("a stall with an exposed steel frame", "a stall without a visible steel frame"),
}


API_PROMPT = """Each numbered cell shows 1-3 side views (left to right) of the same market stall or shop front, taken
2-8 m away during a 360-degree walk through a street market (Shuk HaTikvah, Tel Aviv). For EVERY store id answer:
- visible: false if the views do not show a shop front (only street, sky, the camera operator or people blocking it)
- facade: Open (open front, goods displayed, no front wall), Glass (glass windows / glass door) or Metal (closed metal shutter)
- structure: steel (light steel frame / poles / metal roof), blocks (concrete-block or plastered masonry) or cement (cast concrete)
- materials: the materials you can actually see, from the given list only
When visible is false, give your best guess for facade/structure and an empty materials list; they are ignored."""


def api_review(views: dict, review: dict) -> int:
    """Claude reviews stores that have views but no review yet; answers are added to review (in place)."""
    import vlm
    todo = [oid for oid, ps in sorted(views.items()) if ps and str(oid) not in review]
    item = {"type": "object", "additionalProperties": False, "required": ["visible", "facade", "structure", "materials"],
            "properties": {"visible": {"type": "boolean"}, "facade": {"type": "string", "enum": list(FACADE)},
                           "structure": {"type": "string", "enum": list(SKELETON)},
                           "materials": {"type": "array", "items": {"type": "string", "enum": list(DETAIL)}}}}
    n, done = -(-len(todo) // 8), 0
    for k in range(n):
        part = todo[k * 8:(k + 1) * 8]
        ims = [np.hstack([cv2.resize(cv2.imread(str(p)), (448, 448)) for p in views[oid]]) for oid in part]
        ids = [f"S{oid}" for oid in part]
        try:
            ans = vlm.ask(vlm.sheet(ims, ids, cols=2, cell_w=900, cell_h=360), API_PROMPT, vlm.id_map_schema(ids, item))
            for i, a in ans.items():
                review[i[1:]] = {"facade": a["facade"], "structure": a["structure"], "materials": a["materials"],
                                 "source": "Claude API"} if a["visible"] else None
                done += 1
        except Exception as e:                                         # noqa: BLE001 - those stores stay unreviewed
            print(f"[stalls api] sheet {k + 1} failed: {e}", flush=True)
        print(f"[stalls api] sheet {k + 1}/{n}", flush=True)
    return done


def run(variant: str, force: bool, use_api: bool = False) -> dict:
    sfx = f"_{variant}" if variant else ""
    shops = json.loads((PROCESSED / "shops.json").read_text(encoding="utf-8"))["shops"]
    seen = [s for s in shops if s.get("seen")]
    R = namedtuple("R", "yaw yaw_l yaw_r pitch_top pitch_bot")
    geo = {w: json.loads((PROCESSED / w / "georef.json").read_text(encoding="utf-8")) for w in {s["seen"]["walk"] for s in seen}}
    views, picked = {}, {}
    for s in seen:
        w, f0 = s["seen"]["walk"], int(s["seen"]["frame"])
        g = geo[w]
        lat, lon, b0 = np.array(g["lat"]), np.array(g["lon"]), np.array(g["yaw0_bearing_deg"])
        kx, ky = 111320 * np.cos(np.radians(lat.mean())), 110574
        cx, cy = (lon - lon.mean()) * kx, (lat - lat.mean()) * ky
        sx, sy = (s["lon"] - lon.mean()) * kx, (s["lat"] - lat.mean()) * ky
        trav = np.degrees(np.arctan2(np.gradient(cx), np.gradient(cy)))     # walking direction (map bearing)
        cand = []
        for f in range(max(0, f0 - SEARCH_F), min(len(cx), f0 + SEARCH_F + 1)):
            d = float(np.hypot(sx - cx[f], sy - cy[f]))
            brg = float(np.degrees(np.arctan2(sx - cx[f], sy - cy[f])))
            side = abs((brg - trav[f] + 540) % 360 - 180)
            if D_MIN <= d <= D_MAX and SIDE_MIN <= side <= SIDE_MAX:
                cand.append((abs(side - 90) + 5 * abs(d - 4), f, (brg - b0[f] + 540) % 360 - 180, d))
        cand.sort()
        chosen = []
        for c in cand:
            if all(abs(c[1] - o[1]) >= 5 for o in chosen):
                chosen.append(c)
            if len(chosen) == 3:
                break
        out = INTERIM / w / f"stalls{sfx}" / "crops"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for k, (_, f, y, d) in enumerate(chosen):
            fp = INTERIM / w / f"frames{sfx}" / f"{f:06d}.jpg"
            cp = out / f"{s['OBJECTID']}_{k}.jpg"
            if not fp.exists():
                continue
            if not cp.exists() or force:
                half = float(np.degrees(np.arctan2(2.5, d))) + 10
                r = R(y, y - half, y + half, SPAN_UP, SPAN_DOWN)
                cv2.imwrite(str(cp), crop(cv2.imread(str(fp)), r, size=448, isolate=False))
            paths.append(cp)
        views[s["OBJECTID"]] = paths
        picked[s["OBJECTID"]] = [{"frame": int(f), "yaw": round(float(y), 1), "dist_m": round(float(d), 1)} for _, f, y, d in chosen]

    st = Stage("stall_clip", VERSION, INTERIM / f"stalls{sfx}_clip", {"model": MODEL, "facade": FACADE, "skeleton": SKELETON, "detail": DETAIL},
               {"stores": sorted(views), "views": {str(k): v for k, v in picked.items()}})
    cache = INTERIM / f"stalls{sfx}_clip" / "clip.json"
    if st.done() and not force and cache.exists():
        res = json.loads(cache.read_text(encoding="utf-8"))
    else:
        st.start()
        import clip
        from PIL import Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(MODEL, device=dev)
        fn, ff = text_features(model, clip, FACADE, dev)
        sn, sf = text_features(model, clip, SKELETON, dev)
        dn = list(DETAIL)
        with torch.no_grad():
            tok = clip.tokenize([p for m in dn for p in DETAIL[m]]).to(dev)
            df = model.encode_text(tok).float(); df = df / df.norm(dim=-1, keepdim=True)
        res = {}
        for oid, paths in views.items():
            if not paths:
                continue
            ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(dev)
            with torch.no_grad():
                im = model.encode_image(ims).float(); im = im / im.norm(dim=-1, keepdim=True)
                pf = (100 * im @ ff.T).softmax(-1).mean(0).cpu().numpy()
                ps = (100 * im @ sf.T).softmax(-1).mean(0).cpu().numpy()
                lg = (100 * im @ df.T).view(len(paths), len(dn), 2).softmax(-1)[..., 0].mean(0).cpu().numpy()
            res[str(oid)] = {"facade": dict(zip(fn, np.round(pf, 3).tolist())), "skeleton": dict(zip(sn, np.round(ps, 3).tolist())),
                             "detail": dict(zip(dn, np.round(lg, 3).tolist()))}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res), encoding="utf-8")
        st.finish({"stores": len(res)})

    # score against the survey
    rows, cf, cs = [], {}, {}
    for s in seen:
        r = res.get(str(s["OBJECTID"]))
        if not r or not views.get(s["OBJECTID"]):
            continue
        pf = max(r["facade"], key=r["facade"].get); ps = max(r["skeleton"], key=r["skeleton"].get)
        det = [m for m, p in r["detail"].items() if p >= 0.6]
        rows.append({"OBJECTID": s["OBJECTID"], "name": s["name"], "walk": s["seen"]["walk"], "frame": s["seen"]["frame"],
                     "dist_m": s["seen"]["dist_m"], "views": picked.get(s["OBJECTID"], []), "facade_pred": pf, "facade_p": r["facade"][pf], "facade_survey": s.get("Facade_Mat"),
                     "skeleton_pred": ps, "skeleton_p": r["skeleton"][ps], "skeleton_survey": s.get("Skeleton_Mat"),
                     "materials": det, "detail": r["detail"]})
        if s.get("Facade_Mat") in FACADE:
            cf.setdefault(s["Facade_Mat"], {}).setdefault(pf, 0); cf[s["Facade_Mat"]][pf] += 1
        if s.get("Skeleton_Mat") in SKELETON:
            cs.setdefault(s["Skeleton_Mat"], {}).setdefault(ps, 0); cs[s["Skeleton_Mat"]][ps] += 1
    review_p = CONFIG / "stall_review.json"
    rv_doc = json.loads(review_p.read_text(encoding="utf-8")) if review_p.exists() else {"stores": {}}
    review = rv_doc["stores"]
    if use_api and api_review(views, review):
        rv_doc.setdefault("_note", "Vision review of stall frontage crops (stall_materials.py); null = store not visible.")
        review_p.parent.mkdir(parents=True, exist_ok=True)
        review_p.write_text(json.dumps(rv_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        v = review.get(str(r["OBJECTID"]), "missing")
        r["vision"] = None if v in (None, "missing") else v
        r["visible"] = v is not None and v != "missing"
    rv = [r for r in rows if r["vision"]]
    agree = lambda key, sk: round(sum(r["vision"][key] == r[sk] for r in rv) / max(1, len(rv)), 3)
    acc = lambda c: round(sum(v.get(k, 0) for k, v in c.items()) / max(1, sum(sum(v.values()) for v in c.values())), 3)
    summary = {"stores": len(rows), "facade_accuracy": acc(cf), "facade_confusion_survey_to_pred": cf,
               "skeleton_accuracy": acc(cs), "skeleton_confusion_survey_to_pred": cs,
               "majority_baseline": {"facade": "Open", "skeleton": "steel"},
               "vision_review": {"reviewed": len(rows), "visible": len(rv), "facade_agreement_with_survey": agree("facade", "facade_survey"),
                                 "structure_agreement_with_survey": agree("structure", "skeleton_survey"),
                                 "materials_seen": {m: sum(m in r["vision"]["materials"] for r in rv) for m in sorted({m for r in rv for m in r["vision"]["materials"]})}}}
    (PROCESSED / "stall_materials.json").write_text(json.dumps({**summary, "stores_list": rows}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="stab")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--api", action="store_true", help="review unreviewed stores with Claude (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args()
    print(json.dumps(run(a.variant, a.force, a.api), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
