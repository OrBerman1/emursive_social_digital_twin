"""Cheap stage: frontage materials as LOCATIONS along the walks (not per GIS store).

Why: the stall crops (stall_materials.py) show what the camera really saw, but stores are 2-3 m wide and the
route is ~2-3 m accurate, so a crop often shows the neighbour of the store it was made for. Attaching the reading
to the place the camera looked at is honest; attaching it to a store name is not (yet).
Inputs : data/processed/stall_materials.json (best views per crop: walk, frame, view yaw, distance)
         config/stall_review.json (vision review: facade, structure, visible materials; null = nothing visible)
         data/processed/<walk>/georef.json, the crops in data/interim/<walk>/stalls<_stab>/crops/
Method : point = camera position at the view frame + distance along the view bearing; route chainage and side
         (left/right of the walking direction) from the camera path; points of the same walk within MERGE_M with
         the same facade are merged (materials united) - several crops showed the same frontage.
Output : data/processed/frontage_materials.json, data/processed/<walk>/frontage/<id>.jpg (crop for the viewer)

usage:  python pipeline/frontage.py [--variant stab]
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from common import INTERIM, PROCESSED, ROOT

MERGE_M = 3.0


def run(variant: str) -> dict:
    sfx = f"_{variant}" if variant else ""
    sm = json.loads((PROCESSED / "stall_materials.json").read_text(encoding="utf-8"))["stores_list"]
    review = json.loads((ROOT / "config" / "stall_review.json").read_text(encoding="utf-8"))["stores"]
    geo, pts = {}, []
    for s in sm:
        r = review.get(str(s["OBJECTID"]))
        if not r or not s.get("views"):
            continue
        w = s["walk"]
        if w not in geo:
            g = json.loads((PROCESSED / w / "georef.json").read_text(encoding="utf-8"))
            lat, lon, b0 = np.array(g["lat"]), np.array(g["lon"]), np.array(g["yaw0_bearing_deg"])
            kx, ky = 111320 * np.cos(np.radians(lat.mean())), 110574
            cx, cy = (lon - lon.mean()) * kx, (lat - lat.mean()) * ky
            chain = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
            trav = np.degrees(np.arctan2(np.gradient(cx), np.gradient(cy)))
            geo[w] = (lat, lon, b0, kx, ky, cx, cy, chain, trav)
        lat, lon, b0, kx, ky, cx, cy, chain, trav = geo[w]
        v = s["views"][0]
        f, y, d = int(v["frame"]), float(v["yaw"]), float(v["dist_m"])
        brg = np.radians(b0[f] + y)
        px, py = cx[f] + d * np.sin(brg), cy[f] + d * np.cos(brg)
        side = "right" if ((np.degrees(brg) - trav[f] + 540) % 360 - 180) > 0 else "left"
        pts.append({"walk": w, "frame": f, "yaw": round(y, 1), "dist_m": round(d, 1), "x": px, "y": py,
                    "lat": round(float(lat.mean() + py / ky), 7), "lon": round(float(lon.mean() + px / kx), 7),
                    "chainage_m": round(float(chain[f]), 1), "side": side, "facade": r["facade"], "structure": r["structure"],
                    "materials": list(r["materials"]), "crop_src": str(INTERIM / w / f"stalls{sfx}" / "crops" / f"{s['OBJECTID']}_0.jpg"),
                    "crops_from_stores": [s["OBJECTID"]]})
    # merge points that show the same frontage
    pts.sort(key=lambda p: (p["walk"], p["frame"]))
    merged = []
    for p in pts:
        for m in merged:
            if m["walk"] == p["walk"] and m["facade"] == p["facade"] and np.hypot(m["x"] - p["x"], m["y"] - p["y"]) <= MERGE_M:
                m["materials"] = sorted(set(m["materials"]) | set(p["materials"]))
                m["crops_from_stores"].append(p["crops_from_stores"][0])
                break
        else:
            merged.append(dict(p))
    out = []
    for i, m in enumerate(merged, 1):
        dst = PROCESSED / m["walk"] / "frontage"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m["crop_src"], dst / f"{i}.jpg")
        out.append({"id": i, **{k: v for k, v in m.items() if k not in ("x", "y", "crop_src")}, "crop": f"{m['walk']}/frontage/{i}.jpg"})
    summary = {"points": len(out), "by_walk": {w: sum(p["walk"] == w for p in out) for w in sorted({p["walk"] for p in out})},
               "crops_merged": len(pts) - len(out),
               "materials": {k: sum(k in p["materials"] for p in out) for k in sorted({m for p in out for m in p["materials"]})}}
    (PROCESSED / "frontage_materials.json").write_text(json.dumps({**summary, "note": "Location-based: what the camera saw at this "
        "point of the walk (vision review of frontage crops). Not attached to GIS store names (store match uncertain).",
        "points_list": out}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="stab")
    a = ap.parse_args()
    print(json.dumps(run(a.variant), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
