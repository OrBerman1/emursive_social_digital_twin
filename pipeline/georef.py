"""Cheap stage: put the walks on the map from store anchors (no GPS in the videos).

Inputs : store table (lon/lat, WGS84) data/raw/LABAI/Shops_In_hatikva_30_7_26.xlsx
         anchors   config/anchors.json  (frame, view yaw where the camera looks at a known store)
         heading   data/processed/<walk>/heading.json (smoothed gyro heading per analysis frame)
Model (per walk, local metres east/north):
  bearing of view yaw y at frame f       B(f, y) = alpha + s*(y + psi(f))          s = +-1 (handedness, tested)
  walking direction at frame f           theta(f) = alpha + s*psi(f) + beta        (camera held at a fixed angle)
  camera position (dead reckoning)       p(f) = P0 + v * sum_{k<f} dt [sin theta, cos theta]
  unknowns: P0 (2), alpha, beta, v      fitted so every anchor's view direction points at its store
Outputs: data/processed/<walk>/georef.json (path lat/lon per frame, map heading, residuals)
         data/processed/shops.json        (all stores + best walk/frame/yaw to see each one)

usage:  python pipeline/georef.py
"""
from __future__ import annotations

import difflib
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import PROCESSED, RAW, ROOT, WALKS

SHOPS_XLSX = RAW / "Shops_In_hatikva_30_7_26.xlsx"
ANCHORS = ROOT / "config" / "anchors.json"
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


S_HAND = 1            # view yaw and gyro heading are both clockwise-positive about "up" (derived, not fitted)
D_NEAR, D_FAR = 1.5, 15.0   # a store being looked at is beside the street: plausible camera-store distance (m)


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
    px, py = path(r.x, psi, fr)
    alpha = r.x[2]
    b_obs = alpha + S_HAND * (yaw + psi[fr])
    res_deg = np.degrees(wrap(np.arctan2(sx - px[fr], sy - py[fr]) - b_obs))
    dist = np.hypot(sx - px[fr], sy - py[fr])
    lon, lat = loc.to_ll(px, py)
    out = {
        "walk": walk, "alpha_deg": float(np.degrees(alpha) % 360), "beta_deg": float(np.degrees(r.x[3])),
        "speeds_m_s": np.round(np.exp(r.x[4:]), 2).tolist(), "path_length_m": float(np.sum(np.hypot(np.diff(px), np.diff(py)))),
        "anchors": [{"shop": a["shop"], "name": shops.at[a["shop_idx"], "name"], "frame": a["frame"],
                     "residual_deg": round(float(e), 1), "dist_m": round(float(dd), 1)} for a, e, dd in zip(used, res_deg, dist)],
        "skipped_anchors": [a["shop"] for a in skipped],
        "median_abs_residual_deg": round(float(np.median(np.abs(res_deg))), 1),
        "lat": np.round(lat, 7).tolist(), "lon": np.round(lon, 7).tolist(),
        "yaw0_bearing_deg": np.round(np.degrees(alpha + S_HAND * psi) % 360, 2).tolist(),
    }
    return out, (px, py, psi, S_HAND, alpha)


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
