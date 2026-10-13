"""Cheap stage: put the walks on the map from store anchors (no GPS in the videos).

Inputs : store table (lon/lat, WGS84) data/raw/LABAI/Shops_In_hatikva_30_7_26.xlsx
         anchors   config/anchors.json  (frame, view yaw where the camera looks at a known store)
         heading   data/processed/<walk>/heading.json (smoothed gyro heading per analysis frame)
         streets   data/raw/osm/hatikva_osm.xml (OpenStreetMap streets + buildings; downloaded if missing)
Model (per walk, local metres east/north):
  bearing of view yaw y at frame f       B(f, y) = alpha + s*(y + psi(f))          s = +-1 (handedness, tested)
  walking direction at frame f           theta(f) = alpha + s*psi(f) + beta        (camera held at a fixed angle)
  camera position (dead reckoning)       p(f) = P0 + v * sum_{k<f} dt [sin theta, cos theta]
  unknowns: P0 (2), alpha, beta, v      fitted so every anchor's view direction points at its store
Stage 2 (map matching): smooth gyro-drift and pace corrections at knots every 20 s; the walker must stay on
  OSM walkable street centrelines (stores line both sides, so the person walks between them, never through
  buildings) and anchor stores must lie beside the path, not on it.
Outputs: data/processed/<walk>/georef.json (path lat/lon per frame, map heading, residuals)
         data/processed/shops.json        (all stores + best walk/frame/yaw to see each one)

usage:  python pipeline/georef.py
"""
from __future__ import annotations

import difflib
import json
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares

from common import PROCESSED, RAW, ROOT, WALKS

SHOPS_XLSX = RAW / "Shops_In_hatikva_30_7_26.xlsx"
ANCHORS = ROOT / "config" / "anchors.json"
OSM_XML = RAW.parent / "osm" / "hatikva_osm.xml"
OSM_URL = "https://api.openstreetmap.org/api/0.6/map?bbox=34.7880,32.0500,34.7925,32.0548"
WALKABLE = {"pedestrian", "footway", "residential", "service", "living_street", "tertiary", "secondary",
            "unclassified", "path", "steps", "primary"}
KNOT_S = 20.0          # stage 2: drift / pace knots every 20 s
STREET_TOL_M = 3.0     # free band around a street centreline (market streets are ~4-8 m wide; covered alleys may be unmapped)
POS_SIG_M = 1.5       # stage 2: stall / GIS point position uncertainty (m) in the anchor sight-ray residual
STORE_CLEAR_M = 2.0   # stage 2: stalls are obstacles; the walker passes at least this far from any store point
GRID_M = 0.5           # resolution of the building / street distance fields
FPS = 5.0
DT = 1 / FPS
NEAR_M = 22.0          # a store counts as "on the walk" if the path passes within this distance


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_shops() -> pd.DataFrame:
    df = pd.read_excel(SHOPS_XLSX)
    df = df[df.X.between(34.70, 34.90) & df.Y.between(31.95, 32.15)].copy()     # drop test points far away
    df["name"] = df.Shop_Name.fillna("No name").astype(str)
    return df.reset_index(drop=True)


def match_shop(shops: pd.DataFrame, key: str):
    gid = shops.GlobalID.astype(str).str.lower() == key.lower()
    if gid.any():
        return shops[gid].index[0]
    lower = shops.name.str.lower().tolist()
    m = difflib.get_close_matches(key.lower(), lower, n=1, cutoff=0.82)
    return lower.index(m[0]) if m else None


class Local:
    """Equirectangular local metric frame around the market (accurate to cm over a few hundred metres)."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = 111320.0 * np.cos(np.radians(lat0))
        self.ky = 110574.0

    def to_xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def to_ll(self, x, y):
        return self.lon0 + np.asarray(x) / self.kx, self.lat0 + np.asarray(y) / self.ky


def load_osm(loc: "Local"):
    """Walkable street segments (N x 4, local metres) and building polygons from OpenStreetMap."""
    if not OSM_XML.exists():
        OSM_XML.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(OSM_URL, headers={"User-Agent": "shuk-hatikvah-twin/0.1"})
            OSM_XML.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        except Exception:                                   # Python SSL sometimes fails here; curl works
            subprocess.run(["curl", "-s", "-L", "-A", "shuk-hatikvah-twin/0.1", "-o", str(OSM_XML), OSM_URL], check=True)
    root = ET.parse(OSM_XML).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    segs, blds = [], []
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        pts = [nodes[n.get("ref")] for n in w.iter("nd") if n.get("ref") in nodes]
        if len(pts) < 2:
            continue
        x, y = loc.to_xy(*zip(*pts))
        if t.get("highway") in WALKABLE:
            segs += [(x[i], y[i], x[i + 1], y[i + 1]) for i in range(len(x) - 1)]
        elif "building" in t and len(pts) >= 4:
            blds.append(np.c_[x, y])
    return np.array(segs), blds


def street_dist(px, py, segs):
    """Distance (m) from each point to the nearest walkable street centreline."""
    ax, ay, bx, by = (c[None, :] for c in segs.T)
    dx, dy = bx - ax, by - ay
    t = np.clip(((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(ax + t * dx - px[:, None], ay + t * dy - py[:, None]).min(axis=1)


class Fields:
    """Distance fields on a 0.5 m grid: depth inside a building (0 outside) and distance to a street centreline.
    Looked up with bilinear interpolation, so the fit sees smooth penalties."""

    def __init__(self, segs, blds, x0, x1, y0, y1, stores=None):
        self.x0, self.y0 = x0, y0
        nx, ny = int((x1 - x0) / GRID_M) + 1, int((y1 - y0) / GRID_M) + 1
        gx, gy = np.meshgrid(x0 + np.arange(nx) * GRID_M, y0 + np.arange(ny) * GRID_M)
        inside = in_buildings(gx.ravel(), gy.ravel(), blds).reshape(ny, nx)
        self.depth = distance_transform_edt(inside) * GRID_M                     # metres into a building
        line = np.zeros((ny, nx), bool)
        for ax, ay, bx, by in segs:
            m = max(2, int(np.hypot(bx - ax, by - ay) / (GRID_M / 2)))
            t = np.linspace(0, 1, m)
            ix = np.round((ax + t * (bx - ax) - x0) / GRID_M).astype(int)
            iy = np.round((ay + t * (by - ay) - y0) / GRID_M).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            line[iy[ok], ix[ok]] = True
        self.street = distance_transform_edt(~line) * GRID_M                    # metres to the nearest centreline
        pts = np.zeros((ny, nx), bool)
        if stores is not None:
            ix = np.round((stores[0] - x0) / GRID_M).astype(int); iy = np.round((stores[1] - y0) / GRID_M).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            pts[iy[ok], ix[ok]] = True
        self.store = distance_transform_edt(~pts) * GRID_M if pts.any() else np.full((ny, nx), 99.0)

    def _at(self, f, px, py):
        return map_coordinates(f, [(py - self.y0) / GRID_M, (px - self.x0) / GRID_M], order=1, mode="nearest")

    def depth_at(self, px, py):
        return self._at(self.depth, px, py)

    def street_at(self, px, py):
        return self._at(self.street, px, py)

    def store_at(self, px, py):
        return self._at(self.store, px, py)


def in_buildings(px, py, blds) -> np.ndarray:
    inside = np.zeros(len(px), bool)
    pts = np.c_[px, py]
    for b in blds:
        lo, hi = b.min(0), b.max(0)
        m = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
        if m.any():
            inside[m] |= MPath(b).contains_points(pts[m])
    return inside


S_HAND = 1            # view yaw and gyro heading are both clockwise-positive about "up" (derived, not fitted)
D_NEAR, D_FAR = 2.0, 15.0   # a store being looked at is beside the street: plausible camera-store distance (m)


def path(params, psi, fr):
    """Dead reckoning with piecewise-constant speed between anchors (speeds as log-values for positivity)."""
    x0, y0, alpha, beta = params[:4]
    logv = params[4:]
    th = alpha + S_HAND * psi + beta
    seg = np.searchsorted(np.sort(fr), np.arange(len(psi)), side="right")      # segment index of each frame
    v = np.exp(logv)[np.clip(seg, 0, len(logv) - 1)]
    dx = np.concatenate([[0], np.cumsum(v[:-1] * DT * np.sin(th[:-1]))])
    dy = np.concatenate([[0], np.cumsum(v[:-1] * DT * np.cos(th[:-1]))])
    return x0 + dx, y0 + dy


def residuals(params, psi, fr, yaw, sx, sy):
    px, py = path(params, psi, fr)
    alpha = params[2]
    b_obs = alpha + S_HAND * (yaw + psi[fr])
    b_geo = np.arctan2(sx - px[fr], sy - py[fr])
    r_bear = wrap(b_geo - b_obs) / np.radians(8)                      # ~8 deg measurement noise
    d = np.hypot(sx - px[fr], sy - py[fr])
    r_dist = np.maximum(0, d - D_FAR) / 3 + np.maximum(0, D_NEAR - d) / 1   # soft: store beside the street
    logv = params[4:]
    r_speed = (logv - np.log(0.9)) / 0.6                              # speeds around 0.9 m/s (0.5-1.6)
    r_smooth = np.diff(logv) / 0.5                                    # pace changes gradually
    return np.concatenate([r_bear, r_dist, r_speed, r_smooth])


def path2(params, psi, knots):
    """Stage 2: speed and heading correction interpolated between knots (frames)."""
    x0, y0, alpha, beta = params[:4]
    k = len(knots)
    f = np.arange(len(psi))
    v = np.exp(np.interp(f, knots, params[4:4 + k]))
    dh = np.interp(f, knots, params[4 + k:4 + 2 * k])
    th = alpha + S_HAND * psi + dh + beta
    dx = np.concatenate([[0], np.cumsum(v[:-1] * DT * np.sin(th[:-1]))])
    dy = np.concatenate([[0], np.cumsum(v[:-1] * DT * np.cos(th[:-1]))])
    return x0 + dx, y0 + dy, dh


def residuals2(params, psi, knots, fr, yaw, sx, sy, fld, samp):
    px, py, dh = path2(params, psi, knots)
    k = len(knots)
    b_obs = params[2] + S_HAND * (yaw + psi[fr]) + dh[fr]
    d = np.hypot(sx - px[fr], sy - py[fr])
    err = wrap(np.arctan2(sx - px[fr], sy - py[fr]) - b_obs)
    # miss distance of the store from the camera's sight ray (m): sideways miss + "store is behind the camera";
    # noise = stall/GIS position (POS_SIG_M) + 8 deg pointing error at that range
    sig = np.hypot(POS_SIG_M, d * np.radians(8))
    r_bear = np.r_[d * np.sin(err) / sig, np.maximum(0, -d * np.cos(err)) / sig]
    r_dist = np.maximum(0, d - D_FAR) / 3 + np.maximum(0, D_NEAR - d) / 0.5      # store beside the path, not on it
    w = 1 / np.sqrt(len(samp) / 40)
    r_bld = fld.depth_at(px[samp], py[samp]) / 0.3 * w                            # never inside a building
    r_street = np.maximum(0, fld.street_at(px[samp], py[samp]) - STREET_TOL_M) / 2.0 * w   # keep to the streets
    r_store = np.maximum(0, STORE_CLEAR_M - fld.store_at(px[samp], py[samp])) / 0.5 * w     # walk between stalls, not through
    logv, dhk = params[4:4 + k], params[4 + k:]
    r_speed = np.r_[(logv - np.log(0.9)) / 0.6, np.diff(logv) / 0.3]
    r_drift = np.r_[dhk / np.radians(12), np.diff(dhk) / np.radians(5)]
    return np.concatenate([r_bear, r_dist, r_bld, r_street, r_store, r_speed, r_drift])


def fit_walk(walk: str, anchors: list, shops: pd.DataFrame, loc: Local):
    psi = np.array(json.loads((PROCESSED / walk / "heading.json").read_text(encoding="utf-8"))["heading_rad"])
    used, skipped = [], []
    for a in anchors:
        i = match_shop(shops, a["shop"])
        (used if i is not None else skipped).append({**a, "shop_idx": i})
    used.sort(key=lambda a: a["frame"])
    fr = np.array([a["frame"] for a in used])
    yaw = np.radians([a["yaw"] for a in used])
    idx = [a["shop_idx"] for a in used]
    sx, sy = loc.to_xy(shops.loc[idx, "X"].to_numpy(), shops.loc[idx, "Y"].to_numpy())
    n_seg = len(fr) + 1
    best = None
    for alpha0 in np.radians(np.arange(0, 360, 15)):
        for beta0 in np.radians([-90, -45, 0, 45, 90, 180]):
            x0 = np.r_[sx[0], sy[0], alpha0, beta0, np.full(n_seg, np.log(0.9))]
            try:
                r = least_squares(residuals, x0, args=(psi, fr, yaw, sx, sy), loss="soft_l1", f_scale=1.0, max_nfev=600)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    r = best
    # ---- stage 2: map matching on OSM streets, gyro drift + pace knots ----
    segs, blds = load_osm(loc)
    px1, py1 = path(r.x, psi, fr)
    box = (np.minimum(segs[:, 0], segs[:, 2]) < px1.max() + 80) & (np.maximum(segs[:, 0], segs[:, 2]) > px1.min() - 80) \
        & (np.minimum(segs[:, 1], segs[:, 3]) < py1.max() + 80) & (np.maximum(segs[:, 1], segs[:, 3]) > py1.min() - 80)
    segs = segs[box]
    fld = Fields(segs, blds, px1.min() - 80, px1.max() + 80, py1.min() - 80, py1.max() + 80,
                 stores=loc.to_xy(shops.X.to_numpy(), shops.Y.to_numpy()))
    n = len(psi)
    knots = np.linspace(0, n - 1, max(2, int(round(n * DT / KNOT_S)) + 1))
    seg1 = np.searchsorted(np.sort(fr), np.arange(n), side="right")
    v1 = np.exp(r.x[4:])[np.clip(seg1, 0, len(r.x) - 5)]
    samp = np.arange(0, n, 2)
    best2 = None
    for db in np.radians([0, -20, 20]):
        x2 = np.r_[r.x[:3], r.x[3] + db, np.log(np.interp(knots, np.arange(n), v1)), np.zeros(len(knots))]
        r2 = least_squares(residuals2, x2, args=(psi, knots, fr, yaw, sx, sy, fld, samp), loss="soft_l1", f_scale=1.0, max_nfev=400)
        if best2 is None or r2.cost < best2.cost:
            best2 = r2
    r = best2
    px, py, dh = path2(r.x, psi, knots)
    alpha = r.x[2]
    k = len(knots)
    ds_all = street_dist(px, py, segs)
    inb, inb1 = in_buildings(px, py, blds), in_buildings(px1, py1, blds)
    dstore = fld.store_at(px, py)
    b_obs = alpha + S_HAND * (yaw + psi[fr]) + dh[fr]
    res_deg = np.degrees(wrap(np.arctan2(sx - px[fr], sy - py[fr]) - b_obs))
    dist = np.hypot(sx - px[fr], sy - py[fr])
    miss = np.where(np.abs(res_deg) < 90, dist * np.abs(np.sin(np.radians(res_deg))), dist)       # sight-ray miss (m)
    lon, lat = loc.to_ll(px, py)
    out = {
        "walk": walk, "alpha_deg": float(np.degrees(alpha) % 360), "beta_deg": float(np.degrees(r.x[3])),
        "speeds_m_s": np.round(np.exp(r.x[4:4 + k]), 2).tolist(),
        "drift_deg": np.round(np.degrees(r.x[4 + k:]), 1).tolist(),
        "street_dist_m": {"median": round(float(np.median(ds_all)), 1), "p90": round(float(np.percentile(ds_all, 90)), 1),
                          "max": round(float(ds_all.max()), 1)},
        "frac_in_buildings": round(float(inb.mean()), 3), "frac_within_1m_of_a_store": round(float((dstore < 1.0).mean()), 3), "frac_in_buildings_before_map_matching": round(float(inb1.mean()), 3), "path_length_m": float(np.sum(np.hypot(np.diff(px), np.diff(py)))),
        "anchors": [{"shop": a["shop"], "name": shops.at[a["shop_idx"], "name"], "frame": a["frame"],
                     "residual_deg": round(float(e), 1), "dist_m": round(float(dd), 1), "miss_m": round(float(mm), 1)}
                    for a, e, dd, mm in zip(used, res_deg, dist, miss)],
        "skipped_anchors": [a["shop"] for a in skipped],
        "median_abs_residual_deg": round(float(np.median(np.abs(res_deg))), 1),
        "median_miss_m": round(float(np.median(miss)), 1),
        "lat": np.round(lat, 7).tolist(), "lon": np.round(lon, 7).tolist(),
        "yaw0_bearing_deg": np.round(np.degrees(alpha + S_HAND * psi + dh) % 360, 2).tolist(),
    }
    return out, (px, py, psi + S_HAND * dh, S_HAND, alpha)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    shops = load_shops()
    loc = Local(shops.X.mean(), shops.Y.mean())
    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    sx, sy = loc.to_xy(shops.X, shops.Y)
    best = {i: None for i in range(len(shops))}
    for walk in WALKS:
        if walk not in anchors:
            continue
        g, (px, py, psi, s, alpha) = fit_walk(walk, anchors[walk], shops, loc)
        (PROCESSED / walk / "georef.json").write_text(json.dumps(g, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(json.dumps({k: v for k, v in g.items() if k not in ("lat", "lon", "yaw0_bearing_deg")}, ensure_ascii=False, indent=1))
        # closest approach of this walk to every store -> where to jump to see it
        d = np.hypot(sx[:, None] - px[None, :], sy[:, None] - py[None, :])
        f_best = d.argmin(axis=1)
        for i in range(len(shops)):
            f = int(f_best[i]); dm = float(d[i, f])
            if dm <= NEAR_M and (best[i] is None or dm < best[i]["dist_m"]):
                bearing = np.arctan2(sx[i] - px[f], sy[i] - py[f])
                view_yaw = np.degrees(wrap(s * (bearing - alpha) - psi[f]))
                best[i] = {"walk": walk, "frame": f, "yaw": round(float(view_yaw), 1), "dist_m": round(dm, 1)}
    cols = ["OBJECTID", "name", "Main_Trade", "Sub_Category", "Shop_Type", "Ownership", "Patronage", "Facade_Mat", "Skeleton_Mat", "GlobalID"]
    items = []
    for i, r in shops.iterrows():
        rec = {c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else r[c])) for c in cols}
        rec.update(lon=round(float(r.X), 7), lat=round(float(r.Y), 7), seen=best[i])
        items.append(rec)
    (PROCESSED / "shops.json").write_text(json.dumps({"shops": items}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_seen = sum(1 for b in best.values() if b)
    print(f"stores: {len(shops)} on the map; {n_seen} within {NEAR_M:.0f} m of a walk (clickable)")


if __name__ == "__main__":
    main()
