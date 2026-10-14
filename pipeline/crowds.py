"""Cheap stage: crowds from the existing person tracks (heuristic, no new model).

Idea (user, 14 Oct 2026): several people whose boxes overlap and are about the same size form a crowd.
Same size = similar distance from the camera, so overlapping boxes of similar size are people standing
together (not a near person in front of a far one).
Method : 1. per analysis frame, people = tracked persons that pass the viewer defaults (track max conf >= 0.6,
            track >= 1 s), minus pictured people / operator (depicted.py).
         1b. the camera operator (arm, T-shirt) is tracked as huge "people" in a fixed direction: tracks with median
            box height >= OP_ANG_H deg in the operator sector (or >= OP_ANG_H_ANY anywhere) are dropped; people
            farther than MAX_DIST_M are dropped (beyond ~12 m similar box size no longer means similar distance);
            duplicate boxes of one person (IoU > DUP_IOU, or a box >= 75% inside another) count once.
         2. link two people if their boxes touch or overlap (yaw gap <= GAP_FRAC x mean box width, pitch ranges
            overlap) AND their box heights are similar (smaller / larger >= SIZE_RATIO).
         3. crowd in a frame = connected group of >= min_people people (default 4, user decision 14 Oct).
         4. crowds are followed over time: a group continues an existing crowd if they share >= 1 person (best
            Jaccard); gaps <= 1 s allowed; crowds seen < MIN_DUR_F frames are dropped.
         4b. linked people must also stand within LINK_M on the ground (distance + direction), so groups cannot chain
            across the street; crowds that share people, or lie within MERGE_M and less than 5 s apart, are merged
            (a group that split for a moment keeps one ID).
         5. per crowd: size (median / most at once / unique people over its life), map position (members' map
            positions from motion.py, else camera position + ground distance along the bearing), speed mean and
            std over members (motion.py: camera motion removed, optical flow + geometry), age group and
            perceived gender (attributes.py; smoothed per linked person by link_persons.py when present; a person split
            into several tracks counts once in people_unique and the label tallies), time span, route position, view yaw.
Output : data/processed/<walk>/crowds<_stab>.json (each crowd has a persistent crowd_id "C<walk>-<nn>", numbered in
         time order), and the crowds are appended to observations<_stab>.geojson by export_overlays.py

usage:  python pipeline/crowds.py [walk1 walk2] [--variant stab] [--min-people 3]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, WALKS

CONF, MIN_TRACK_F = 0.6, 5
GAP_FRAC = 0.25            # boxes may be apart by up to 25% of their mean width and still touch
SIZE_RATIO = 0.7           # smaller box height / larger box height
MIN_DUR_F = 5              # crowd must last >= 1 s
MAX_GAP_F = 5              # a crowd may disappear for <= 1 s
OP_ANG_H, OP_ANG_H_ANY, OP_SECTOR = 48.0, 58.0, 60.0   # operator: box height (deg); +-sector around the operator direction
MAX_DIST_M = 8.0          # beyond ~8 m crowd members are too small to count by eye (user feedback 14 Oct; was 12)
DUP_IOU = 0.5
LINK_M = 2.0           # linked people must also stand within 2 m of each other on the ground (stops chaining)
MERGE_GAP_F, MERGE_M = 25, 4.0   # crowds sharing people, or within 4 m, less than 5 s apart = one crowd
DUP_CONTAIN = 0.75     # a box >= 75% inside another (half-body box inside a full-body box) = same person


def operator_tracks(tr: pd.DataFrame) -> set:
    a = tr.groupby("track").agg(h=("ang_h", "median"), yaw=("yaw", "median"))
    big = a[a.h >= OP_ANG_H_ANY]
    if len(big) == 0:
        return set()
    r = np.radians(big.yaw.to_numpy())
    op_yaw = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))      # operator direction (circular mean)
    near = np.abs((a.yaw - op_yaw + 540) % 360 - 180) <= OP_SECTOR
    return set(a.index[(a.h >= OP_ANG_H_ANY) | ((a.h >= OP_ANG_H) & near)].astype(int))


def dedup(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("conf", ascending=False)
    keep = []
    for r in g.itertuples():
        dup = False
        for k in keep:
            dx = min(r.yaw_r, k.yaw_r) - max(r.yaw_l, k.yaw_l); dy = min(r.pitch_top, k.pitch_top) - max(r.pitch_bot, k.pitch_bot)
            if dx > 0 and dy > 0:
                inter = dx * dy
                a1 = (r.yaw_r - r.yaw_l) * (r.pitch_top - r.pitch_bot); a2 = (k.yaw_r - k.yaw_l) * (k.pitch_top - k.pitch_bot)
                if inter / (a1 + a2 - inter) > DUP_IOU or inter / max(1e-6, min(a1, a2)) > DUP_CONTAIN:   # same box / part-body box inside a full one
                    dup = True; break
        if not dup:
            keep.append(r)
    return g[g.track.isin([k.track for k in keep])]


def groups_in_frame(g: pd.DataFrame, min_people: int) -> list[list[int]]:
    ids = g.track.to_numpy()
    yl, yr = g.yaw_l.to_numpy(), g.yaw_r.to_numpy()
    w = (yr - yl) % 360
    yc = yl + w / 2
    pt, pb = g.pitch_top.to_numpy(), g.pitch_bot.to_numpy()
    h = pt - pb
    d = g.dist_m.to_numpy()
    gx, gy = d * np.sin(np.radians(yc)), d * np.cos(np.radians(yc))   # ground position relative to the camera (m)
    n = len(ids)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if min(h[i], h[j]) / max(h[i], h[j], 1e-6) < SIZE_RATIO:
                continue
            dc = abs((yc[i] - yc[j] + 540) % 360 - 180)                  # centre distance in yaw
            gap = dc - (w[i] + w[j]) / 2                                 # < 0 = overlap
            if gap > GAP_FRAC * (w[i] + w[j]) / 2:
                continue
            if min(pt[i], pt[j]) < max(pb[i], pb[j]):                    # pitch ranges do not overlap
                continue
            if np.hypot(gx[i] - gx[j], gy[i] - gy[j]) > LINK_M:          # too far apart on the ground
                continue
            parent[find(i)] = find(j)
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(int(ids[i]))
    return [sorted(c) for c in comp.values() if len(c) >= min_people]


def run(walk: str, variant: str, min_people: int) -> dict:
    sfx = f"_{variant}" if variant else ""
    tr = pd.read_parquet(INTERIM / walk / f"tracks{sfx}" / "persons.parquet")
    best, length = tr.groupby("track").conf.max(), tr.groupby("track").size()
    keep = set(best[(best >= CONF)].index) & set(length[length >= MIN_TRACK_F].index)
    dp = INTERIM / walk / f"depicted{sfx}" / "depicted.parquet"
    if dp.exists():
        d = pd.read_parquet(dp)
        keep -= set(d.track[d.category != "person"].astype(int))
    ops = operator_tracks(tr)
    keep -= ops
    tr = tr[tr.track.isin(keep) & (tr.dist_m <= MAX_DIST_M)]
    attr_p = INTERIM / walk / f"attributes{sfx}" / "attributes.parquet"
    attr = pd.read_parquet(attr_p).set_index("track") if attr_p.exists() else None
    link_p = INTERIM / walk / f"persons_linked{sfx}" / "links.parquet"           # linked persons: smoothed labels, person ids
    if link_p.exists():
        attr = pd.read_parquet(link_p).set_index("track")
    person = (lambda t: int(attr.at[t, "person"])) if attr is not None and "person" in attr.columns else (lambda t: t)
    mo_p = PROCESSED / walk / f"motion{sfx}.json"
    mo = json.loads(mo_p.read_text(encoding="utf-8")) if mo_p.exists() else {"tracks": {}, "id2g": {}}
    g = json.loads((PROCESSED / walk / "georef.json").read_text(encoding="utf-8"))
    lat, lon, b0 = np.array(g["lat"]), np.array(g["lon"]), np.array(g["yaw0_bearing_deg"])
    kx, ky = 111320 * np.cos(np.radians(lat.mean())), 110574
    cx, cy = (lon - lon.mean()) * kx, (lat - lat.mean()) * ky
    chain = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
    rows = {(int(r.track), int(r.frame)): r for r in tr.itertuples()}

    def motion_at(tid, f):
        gid = mo["id2g"].get(str(tid))
        m = mo["tracks"].get(str(gid)) if gid is not None else None
        if not m:
            return None
        i = f - m["f0"]
        if 0 <= i < len(m["lat"]) and m["lat"][i] is not None:
            return m["lat"][i], m["lon"][i], m["spd"][i]
        return None

    def position(tid, f):
        m = motion_at(tid, f)
        if m:
            return m[0], m[1], "motion"
        r = rows.get((tid, f))
        if r is None or not np.isfinite(r.dist_m) or f >= len(lat):
            return None
        b = np.radians(b0[f] + r.yaw_foot)
        return lat.mean() + (cy[f] + r.dist_m * np.cos(b)) / ky, lon.mean() + (cx[f] + r.dist_m * np.sin(b)) / kx, "camera+distance"

    # 1-3: groups per frame; 4: follow over time
    active, done, next_id = [], [], 1
    for f in sorted(tr.frame.unique()):
        groups = groups_in_frame(dedup(tr[tr.frame == f]), min_people)
        used = set()
        for grp in groups:
            gs, best_c, best_j = set(grp), None, 0.0
            for c in active:
                if id(c) in used or f - c["last"] > MAX_GAP_F:
                    continue
                inter = len(gs & c["members_last"])
                if inter:
                    j = inter / len(gs | c["members_last"])
                    if j > best_j:
                        best_c, best_j = c, j
            if best_c is None:
                best_c = {"id": next_id, "first": int(f), "frames": [], "members_last": set()}
                next_id += 1
                active.append(best_c)
            best_c["frames"].append((int(f), grp))
            best_c["members_last"] = gs
            best_c["last"] = int(f)
            used.add(id(best_c))
        still = []
        for c in active:
            (still if f - c["last"] <= MAX_GAP_F else done).append(c)
        active = still
    done += active

    # 5. merge crowds that are really one (a group that split briefly, or re-formed at the same spot)
    def centre(c):
        ps = [p for f, m in c["frames"] for p in (position(t, f) for t in m) if p]
        return (np.median([p[0] for p in ps]), np.median([p[1] for p in ps])) if ps else None
    for c in done:
        c["mem"] = {t for _, m in c["frames"] for t in m}
        c["f0"], c["f1"], c["ctr"] = c["frames"][0][0], c["frames"][-1][0], centre(c)
    parent = list(range(len(done)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, a in enumerate(done):
        for j in range(i + 1, len(done)):
            b = done[j]
            gap = max(a["f0"], b["f0"]) - min(a["f1"], b["f1"])            # < 0 = overlap in time
            if gap > MERGE_GAP_F:
                continue
            near = a["ctr"] and b["ctr"] and np.hypot((a["ctr"][1] - b["ctr"][1]) * kx, (a["ctr"][0] - b["ctr"][0]) * ky) <= MERGE_M
            if (a["mem"] & b["mem"]) or near:
                parent[find(i)] = find(j)
    groups = {}
    for i, c in enumerate(done):
        groups.setdefault(find(i), []).append(c)
    merged = []
    for cs in groups.values():
        per = {}
        for c in cs:
            for f, m in c["frames"]:
                per.setdefault(f, set()).update(m)
        merged.append({"frames": [(f, sorted(per[f])) for f in sorted(per)], "n_merged": len(cs)})
    done = merged

    crowds = []
    for c in done:
        if c["frames"][-1][0] - c["frames"][0][0] + 1 < MIN_DUR_F:
            continue
        sizes = [len(m) for _, m in c["frames"]]
        members = sorted({t for _, m in c["frames"] for t in m})
        f0, f1 = c["frames"][0][0], c["frames"][-1][0]
        fm, mm = max(c["frames"], key=lambda x: len(x[1]))               # fullest moment: easiest to count in the video
        pts, src = [], set()
        for f, m in c["frames"]:
            ps = [p for p in (position(t, f) for t in m) if p]
            if ps:
                pts.append((np.mean([p[0] for p in ps]), np.mean([p[1] for p in ps])))
                src |= {p[2] for p in ps}
        if not pts:
            continue
        clat, clon = float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))
        # spread: max member distance from the centre at the middle frame (m)
        pm = [p for p in (position(t, fm) for t in mm) if p]
        radius = max([np.hypot((p[1] - clon) * kx, (p[0] - clat) * ky) for p in pm] + [0.0])
        # speed per member over the crowd's life (camera motion removed)
        spd = []
        for t in members:
            v = [m[2] for f, mem in c["frames"] if t in mem for m in [motion_at(t, f)] if m and m[2] is not None]
            if v:
                spd.append(float(np.median(v)))
        pm1 = list({person(t) if attr is not None and t in attr.index else t: t for t in members}.values())   # one member per linked person
        ages = pd.Series([attr.at[t, "age"] if attr is not None and t in attr.index else "uncertain" for t in pm1]).value_counts().to_dict()
        sexes = pd.Series([attr.at[t, "sex"] if attr is not None and t in attr.index else "uncertain" for t in pm1]).value_counts().to_dict()
        yaws = [rows[(t, fm)].yaw for t in mm if (t, fm) in rows]
        view_yaw = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(yaws))), np.mean(np.cos(np.radians(yaws)))))) if yaws else 0.0
        crowds.append({
            "id": len(crowds) + 1, "walk": walk, "lat": round(clat, 7), "lon": round(clon, 7), "radius_m": round(float(radius), 1),
            "frame_start": f0, "frame_end": f1, "t_start_s": round(f0 / 5, 1), "t_end_s": round(f1 / 5, 1),
            "duration_s": round((f1 - f0 + 1) / 5, 1), "chainage_m": round(float(chain[min(fm, len(chain) - 1)]), 1),
            "people_median": int(np.median(sizes)), "people_max_at_once": int(max(sizes)), "people_unique": len(pm1),
            "speed_mean_m_s": round(float(np.mean(spd)), 2) if spd else None, "speed_std_m_s": round(float(np.std(spd)), 2) if spd else None,
            "speed_n": len(spd), "walking": int(sum(v >= 0.5 for v in spd)), "standing": int(sum(v <= 0.3 for v in spd)),
            "age_group": ages, "perceived_gender": sexes, "members": [f"P-{t:04d}" for t in members],
            "view_frame": int(fm), "view_yaw": round(view_yaw, 1), "position_from": sorted(src), "merged_from": c["n_merged"],
            "frames": [[int(f), [int(t) for t in m]] for f, m in c["frames"]],
        })
    crowds.sort(key=lambda c: c["frame_start"])                         # persistent IDs in time order along the walk
    wn = walk.replace("walk", "")
    for i, c in enumerate(crowds, 1):
        c["id"] = i
        c["crowd_id"] = f"C{wn}-{i:02d}"                                 # e.g. C1-03 = walk 1, crowd 3
    res = {"walk": walk, "variant": variant, "crowds": len(crowds), "operator_tracks_excluded": len(ops), "max_dist_m": MAX_DIST_M, "link_m": LINK_M, "merge_gap_s": MERGE_GAP_F / 5, "merge_m": MERGE_M, "min_people": min_people, "size_ratio": SIZE_RATIO, "gap_frac": GAP_FRAC,
           "min_duration_s": MIN_DUR_F / 5,
           "note": "Heuristic: overlapping person boxes of similar size (similar distance) = people together. Counts are approximate "
                   "(people hidden behind others are missed; one person can have two track IDs).",
           "crowds_list": crowds}
    (PROCESSED / walk / f"crowds{sfx}.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    sz = [c["people_max_at_once"] for c in crowds]
    return {"walk": walk, "operator_tracks": len(ops), "crowds": len(crowds), "max_at_once_pct": np.percentile(sz, [50, 90, 100]).tolist() if sz else [],
            "median_duration_s": float(np.median([c["duration_s"] for c in crowds])) if crowds else 0}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab")
    ap.add_argument("--min-people", type=int, default=4)               # user, 14 Oct: a crowd is 4 people or more
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.variant, a.min_people)))


if __name__ == "__main__":
    main()
