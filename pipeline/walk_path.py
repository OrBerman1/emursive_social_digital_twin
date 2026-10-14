"""Cheap stage: APPROXIMATE placement of a walk on the map, for when the shop table (and so georef.py) is missing.

The videos have no GPS. The walk's shape comes from the camera itself:
  heading psi(f)   gyro heading per analysis frame (same integration as stabilize.py); in the stabilised view,
                   view yaw y points to world yaw y + psi(f)
  forward f(f)     walking direction in view yaw (forward_dir.py)
  path             p(f) = P0 + v * sum dt [sin th, cos th] while moving, th = alpha + f + s*psi
                   (view yaw grows clockwise on the map, as the viewer engine assumes; only the gyro sign s is free)
Unknown alpha (rotation), s (gyro sign), P0 (start), v (pace) are fitted so the path runs along the walkable
OpenStreetMap streets, preferring the street the walk is named after. A straight street fits both ways round, so
the two directions are reported; --flip picks the other one.

Output: data/processed/<walk>/georef_approx.json  (same fields as georef.json: lat, lon, yaw0_bearing_deg per
        analysis frame; "approx": true). The viewer uses georef.json when it exists, this file otherwise.
usage:  python pipeline/walk_path.py [walk1 walk2] [--flip walk1]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import INTERIM, PROCESSED, RAW
from stabilize import fit_mapping, gauss, integrate

FPS = 5.0
OSM_XML = RAW.parent / "osm" / "hatikva_osm.xml"          # same file and area as georef.py on main
OSM_URL = "https://api.openstreetmap.org/api/0.6/map?bbox=34.7880,32.0500,34.7925,32.0548"
WALKABLE = {"pedestrian", "footway", "residential", "service", "living_street", "tertiary", "secondary",
            "unclassified", "path", "steps", "primary"}
STREET = {"walk1": "HaTikva", "walk2": "Mevaser"}           # name:en in OpenStreetMap
# what the video shows: walk 1 starts at a wide road with traffic lights, then runs between market stalls throughout
START_AT_SIGNALS = {"walk1"}
IN_MARKET = {"walk1"}
PACE, PACE_SIG = 0.8, 0.12      # walking pace (m/s): walk 1 fits 0.78 with free pace; both walks are the same walker
GYRO_SIGNS = (1, -1)   # tried both; same camera + pipeline, so both walks should end up with the same sign
LAT0, LON0 = 32.0524, 34.7902
MX, MY = 111320 * np.cos(np.radians(LAT0)), 110574        # metres per degree


def heading(walk: str) -> np.ndarray:
    """Smoothed gyro heading (rad) per analysis frame, as in stabilize.run (same fit, same timing shift)."""
    vis = pd.read_parquet(INTERIM / walk / "visrot.parquet")
    imu = pd.read_csv(INTERIM / walk / "imu.csv")
    t_ms = imu.t_ms.to_numpy()
    g_rad, acc = np.radians(imu[["gx", "gy", "gz"]].to_numpy()), imu[["ax", "ay", "az"]].to_numpy()
    m, lag, *_ = fit_mapping(vis, t_ms, g_rad)
    p = -m if np.linalg.det(-m) > 0 else m
    w_cam, a_cam = g_rad @ p.T, acc @ p.T
    if np.median(a_cam[:, 1]) < 0:
        a_cam = -a_cam
    t_cam = t_ms - lag * 1000
    R = integrate(t_cam, w_cam, a_cam)
    fwd = R.apply([0.0, 0.0, 1.0])
    head_s = gauss(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2])), 1.0 * 1000)
    st = json.loads((PROCESSED / walk / "stab.json").read_text(encoding="utf-8"))
    n = json.loads((INTERIM / walk / "frames" / "manifest.json").read_text(encoding="utf-8"))["n_frames"]
    idx = np.clip(np.searchsorted(t_cam, (np.arange(n) / FPS + st["timing_shift_s"]["analysis"]) * 1000), 0, len(t_cam) - 1)
    return head_s[idx]


def streets() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Walkable street segments (local metres) and the segments of each named street (+ '_signals', '_market')."""
    if not OSM_XML.exists():
        OSM_XML.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(OSM_URL, headers={"User-Agent": "shuk-hatikva-digital-twin"})
        OSM_XML.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    root = ET.parse(OSM_XML).getroot()
    nodes = {n.get("id"): ((float(n.get("lon")) - LON0) * MX, (float(n.get("lat")) - LAT0) * MY) for n in root.iter("node")}
    segs, named = [], {}
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        if tags.get("highway") not in WALKABLE:
            continue
        pts = [nodes[nd.get("ref")] for nd in w.iter("nd") if nd.get("ref") in nodes]
        s = [(*a, *b) for a, b in zip(pts, pts[1:])]
        segs += s
        if tags.get("name:en"):
            named.setdefault(tags["name:en"], []).extend(s)
    out = {k: np.array(v) for k, v in named.items()}
    out["_signals"] = np.array([nodes[n.get("id")] for n in root.iter("node") if any(t.get("k") == "highway" and t.get("v") == "traffic_signals" for t in n.iter("tag"))])
    for w in root.iter("way"):
        if any(t.get("k") == "amenity" and t.get("v") == "marketplace" for t in w.iter("tag")):
            pts = [nodes[nd.get("ref")] for nd in w.iter("nd") if nd.get("ref") in nodes]
            out["_market"] = np.array([(*a, *b) for a, b in zip(pts, pts[1:])])
    return np.array(segs), out


def seg_dist(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Distance from each point (N,2) to the nearest segment (M,4)."""
    a, b = S[None, :, :2], S[None, :, 2:]
    ab = b - a
    t = np.clip(((P[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(P[:, None] - (a + t[..., None] * ab), axis=-1).min(1)


def path(params, psi, fwd, moving):
    alpha, s, x0, y0, v = params
    th = alpha + fwd + s * psi
    step = v / FPS * moving
    x = x0 + np.concatenate([[0], np.cumsum(step * np.sin(th))[:-1]])
    y = y0 + np.concatenate([[0], np.cumsum(step * np.cos(th))[:-1]])
    return np.c_[x, y]


def run(walk: str, flip: bool) -> dict:
    psi = heading(walk)
    fw = json.loads((PROCESSED / walk / "forward.json").read_text(encoding="utf-8"))
    n = len(psi)
    fwd = np.radians(np.resize(np.array(fw["fwd_deg"]), n))
    moving = np.resize(np.array(fw["moving"], float), n)
    segs, named = streets()
    main_st = named[STREET[walk]]
    sub = np.arange(0, n, 5)
    ends = np.r_[main_st[:, :2], main_st[:, 2:]]
    hint = ends[np.argmin(seg_dist(ends, np.c_[named['_signals'], named['_signals']]))] if walk in START_AT_SIGNALS else None
    market = named.get('_market') if walk in IN_MARKET else None

    def terms(P, v):
        t = [np.minimum(seg_dist(P[sub], segs), 10), 0.55 * np.minimum(seg_dist(P[sub], main_st), 30), [(v - PACE) / PACE_SIG * 6]]
        if hint is not None:
            t.append([np.hypot(*(P[0] - hint)) / 4])
        if market is not None:
            t.append(0.4 * np.minimum(seg_dist(P[sub], market), 30))       # the market polygon hugs the street
        return np.concatenate(t)

    def cost(P, v):
        return float(np.mean(terms(P, v) ** 2))

    # coarse search: start on the named street, every rotation, both handedness, three paces
    starts = np.concatenate([np.linspace(s[:2], s[2:], max(2, int(np.hypot(*(s[2:] - s[:2])) // 6) + 1)) for s in main_st])
    if hint is not None:
        starts = starts[np.hypot(*(starts - hint).T) < 30]
    cands = []
    for s in GYRO_SIGNS:
        for v in (PACE - 0.1, PACE, PACE + 0.1):
            for alpha in np.radians(np.arange(0, 360, 6)):
                shape = path((alpha, s, 0, 0, v), psi, fwd, moving)
                for x0, y0 in (starts if hint is not None else starts[::2]):
                    cands.append((cost(shape + [x0, y0], v), alpha, s, x0, y0, v))
    cands.sort(key=lambda c: c[0])

    def refine(c):
        _, alpha, s, x0, y0, v = c
        def res(q):
            return terms(path((q[0], s, q[1], q[2], q[3]), psi, fwd, moving), q[3])
        r = least_squares(res, [alpha, x0, y0, v], bounds=([-np.inf, -np.inf, -np.inf, PACE - 0.1], [np.inf, np.inf, np.inf, PACE + 0.1]), loss="soft_l1", f_scale=2.0)
        q = (r.x[0], s, r.x[1], r.x[2], r.x[3])
        return cost(path(q, psi, fwd, moving), q[4]), q

    # best fit, and the best fit walking the other way round (start/end swapped)
    best = refine(cands[0])
    P_best = path(best[1], psi, fwd, moving)
    other = next((c for c in cands if np.hypot(c[3] - P_best[-1, 0], c[4] - P_best[-1, 1]) < 25 and np.cos(c[1] - best[1][0]) < -0.5), None)
    alt = refine(other) if other is not None else best
    chosen, rejected = (alt, best) if flip else (best, alt)
    q = chosen[1]
    P = path(q, psi, fwd, moving)
    yaw0 = np.degrees(q[0] + q[1] * psi) % 360              # map bearing of view yaw 0
    out = {
        "walk": walk, "approx": True, "method": "gyro heading + walking direction, fitted to OpenStreetMap streets (no shop anchors)",
        "street": STREET[walk], "start_hint": "traffic-light junction" if hint is not None else None, "pace_m_s": round(float(q[4]), 3), "gyro_sign": int(q[1]), "flipped": flip,
        "cost": round(chosen[0], 2), "cost_other_direction": round(rejected[0], 2),
        "lat": np.round(LAT0 + P[:, 1] / MY, 7).tolist(), "lon": np.round(LON0 + P[:, 0] / MX, 7).tolist(),
        "yaw0_bearing_deg": np.round(yaw0, 1).tolist(),
    }
    (PROCESSED / walk / "georef_approx.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return {k: out[k] for k in ("walk", "street", "pace_m_s", "gyro_sign", "flipped", "cost", "cost_other_direction")} | {
        "start": [out["lat"][0], out["lon"][0]], "end": [out["lat"][-1], out["lon"][-1]],
        "length_m": round(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=["walk1", "walk2"])
    ap.add_argument("--flip", nargs="*", default=[], help="walks to place the other way round")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, w in a.flip)))
