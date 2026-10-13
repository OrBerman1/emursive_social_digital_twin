"""Cheap stage: put every tracked person on the map and measure how they move (camera motion removed).

Inputs : data/interim/<walk>/tracks<_stab>/persons.parquet   (per track + frame: yaw_foot, dist_m)
         data/interim/<walk>/attributes<_stab>/attributes.parquet (perceived gender, to veto re-links; optional)
         data/processed/<walk>/georef.json                   (camera lat/lon + map bearing of view yaw 0, per frame)
Method : 1. person position = camera position + dist_m along bearing (yaw0_bearing + yaw_foot); dist_m comes from
            the foot point and the camera height (flat ground). Only frames with 1 m <= dist <= MAX_DIST_M are used
            (beyond that one pixel of foot error is metres).
         2. re-link broken tracks on the map: track B continues track A if B starts <= 2 s after A ends, close to
            where A was heading (error <= 1 m + 0.4 m per second of gap), the match is unambiguous and the perceived
            gender does not conflict. The image tracker fragments people at occlusions; on the map they line up.
         3. per linked person: positions smoothed (Gaussian, sigma 1 s, gaps <= 2 s bridged); velocity = derivative
            of the smoothed position (sigma 1.5 s). Positions are map coordinates, so the camera's own motion is
            already removed: this is ego-motion-compensated motion ("optical flow of people" for a moving camera).
Output : data/processed/<walk>/motion<_stab>.json  {tracks: {gid: {f0, lat[], lon[], spd[], brg[], members, ...}},
         id2g: {track id: gid}, stats}

usage:  python pipeline/motion.py [walk1 walk2] [--variant stab]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from common import INTERIM, PROCESSED, WALKS

FPS = 5.0
MAX_DIST_M = 12.0      # beyond this the ground distance from the foot pitch is too uncertain
MIN_DIST_M = 1.0
MIN_FRAMES = 5         # at least 1 s of usable positions
GAP_FRAMES = 10        # bridge gaps of <= 2 s by interpolation (inside a track and across re-links)
SIG_POS_S, SIG_VEL_S = 1.0, 1.5
WALK_MS, STAND_MS = 0.5, 0.3
LINK_GAP_F = 10        # re-link: B starts at most 2 s after A ends
LINK_TOL_M, LINK_TOL_M_PER_S = 1.0, 0.4
LINK_AMBIG_M = 0.3     # skip if the second-best candidate is almost as good


def local(lat0, lon0):
    kx, ky = 111320.0 * np.cos(np.radians(lat0)), 110574.0
    return (lambda lat, lon: ((np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky),
            lambda x, y: (lat0 + np.asarray(y) / ky, lon0 + np.asarray(x) / kx))


def raw_positions(tr: pd.DataFrame, cx, cy, b0) -> dict:
    """Per track: frames and map positions (local metres) where the distance is usable."""
    out = {}
    for tid in sorted(tr.track.unique()):                     # explicit loop (groupby.apply drops columns here)
        t = tr[tr.track == tid].sort_values("frame")
        t = t[t.frame < len(cx)]
        d = t.dist_m.to_numpy()
        ok = np.isfinite(d) & (d >= MIN_DIST_M) & (d <= MAX_DIST_M)
        if ok.sum() < 2:
            continue
        f = t.frame.to_numpy().astype(int)[ok]
        b = np.radians(b0[f] + t.yaw_foot.to_numpy()[ok])
        out[int(tid)] = {"f": f, "x": cx[f] + d[ok] * np.sin(b), "y": cy[f] + d[ok] * np.cos(b), "d": d[ok], "n_total": len(t)}
    return out


def end_state(r, at_end: bool):
    """Position and velocity at the start or end of a raw track (least squares line over its last/first 2 s)."""
    k = min(len(r["f"]), 10)
    sl = slice(-k, None) if at_end else slice(0, k)
    f, x, y = r["f"][sl], r["x"][sl], r["y"][sl]
    if len(f) < 3 or f[-1] == f[0]:
        return np.array([x[-1 if at_end else 0], y[-1 if at_end else 0]]), np.zeros(2)
    A = np.c_[f - f.mean(), np.ones(len(f))]
    (vx, x0), (vy, y0) = np.linalg.lstsq(A, x, rcond=None)[0], np.linalg.lstsq(A, y, rcond=None)[0]
    fe = f[-1] if at_end else f[0]
    v = np.array([vx, vy]) * FPS
    sp = np.hypot(*v)
    if sp > 2.0:
        v *= 2.0 / sp
    return np.array([x0 + vx * (fe - f.mean()), y0 + vy * (fe - f.mean())]), v


def link(raw: dict, sex: dict) -> list[list[int]]:
    ids = sorted(raw)
    ends = {i: (raw[i]["f"][-1], *end_state(raw[i], True)) for i in ids}
    starts = {i: (raw[i]["f"][0], *end_state(raw[i], False)) for i in ids}
    cand = {}
    for a in ids:
        fa, pa, va = ends[a]
        opts = []
        for b in ids:
            fb, pb, vb = starts[b]
            gap = fb - fa
            if b == a or gap < 1 or gap > LINK_GAP_F:
                continue
            sa, sb = sex.get(a), sex.get(b)
            if sa in ("man", "woman") and sb in ("man", "woman") and sa != sb:
                continue
            gs = gap / FPS
            err = (np.hypot(*(pa + va * gs - pb)) + np.hypot(*(pb - vb * gs - pa))) / 2   # forward + backward prediction
            if err <= LINK_TOL_M + LINK_TOL_M_PER_S * gs:
                opts.append((err, b))
        opts.sort()
        if opts and (len(opts) == 1 or opts[1][0] - opts[0][0] >= LINK_AMBIG_M):
            cand[a] = opts[0]
    nxt, prv = {}, {}
    for a, (err, b) in sorted(cand.items(), key=lambda kv: kv[1][0]):           # greedy, best links first
        if a in nxt or b in prv:
            continue
        nxt[a], prv[b] = b, a
    groups = []
    for i in ids:
        if i in prv:
            continue
        g = [i]
        while g[-1] in nxt:
            g.append(nxt[g[-1]])
        groups.append(g)
    return groups


def record(members, raw, to_ll, cvx, cvy):
    f = np.concatenate([raw[m]["f"] for m in members])
    x = np.concatenate([raw[m]["x"] for m in members])
    y = np.concatenate([raw[m]["y"] for m in members])
    d = np.concatenate([raw[m]["d"] for m in members])
    o = np.argsort(f, kind="stable")
    f, x, y = f[o], x[o], y[o]
    f, iu = np.unique(f, return_index=True)
    x, y = x[iu], y[iu]
    if len(f) < MIN_FRAMES:
        return None
    F = np.arange(f[0], f[-1] + 1)
    X, Y = np.interp(F, f, x), np.interp(F, f, y)
    valid = np.zeros(len(F), bool)
    idx = np.searchsorted(F, f)
    valid[idx] = True
    for a, c in zip(idx[:-1], idx[1:]):
        if c - a - 1 <= GAP_FRAMES:
            valid[a:c + 1] = True
    Xs, Ys = gaussian_filter1d(X, SIG_POS_S * FPS), gaussian_filter1d(Y, SIG_POS_S * FPS)
    Xv, Yv = gaussian_filter1d(X, SIG_VEL_S * FPS), gaussian_filter1d(Y, SIG_VEL_S * FPS)
    vx, vy = np.gradient(Xv) * FPS, np.gradient(Yv) * FPS
    spd, brg = np.hypot(vx, vy), np.degrees(np.arctan2(vx, vy)) % 360
    lat, lon = to_ll(Xs, Ys)
    cam_s = np.hypot(cvx[F], cvy[F]).clip(1e-6)
    along = (vx * cvx[F] + vy * cvy[F]) / cam_s                     # + = same direction as the camera walks
    vs = spd[valid]
    med = float(np.median(vs))
    state = "walking" if med >= WALK_MS else "standing" if med <= STAND_MS else "slow"
    moving = vs >= WALK_MS
    rel = "—"
    if moving.mean() > 0.5:
        rel = "with the camera" if np.median(along[valid][moving]) > 0 else "against the camera"
    seg = np.hypot(np.diff(Xs[valid]), np.diff(Ys[valid]))
    R = lambda a, k: [round(float(v), k) if m else None for v, m in zip(a, valid)]
    return {"f0": int(F[0]), "lat": R(lat, 7), "lon": R(lon, 7), "spd": R(spd, 2), "brg": R(brg, 0),
            "len_m": round(float(seg[seg < 3].sum()), 1), "dur_s": round(len(F) / FPS, 1), "med_spd": round(med, 2),
            "state": state, "rel": rel, "med_dist_m": round(float(np.median(d)), 1), "members": [int(m) for m in members]}, vs


def run(walk: str, variant: str) -> dict:
    sfx = f"_{variant}" if variant else ""
    tr = pd.read_parquet(INTERIM / walk / f"tracks{sfx}" / "persons.parquet")
    ap = INTERIM / walk / f"attributes{sfx}" / "attributes.parquet"
    sex = dict(zip(*pd.read_parquet(ap)[["track", "sex"]].to_numpy().T)) if ap.exists() else {}
    sex = {int(k): v for k, v in sex.items()}
    g = json.loads((PROCESSED / walk / "georef.json").read_text(encoding="utf-8"))
    lat_c, lon_c, b0 = np.array(g["lat"]), np.array(g["lon"]), np.array(g["yaw0_bearing_deg"])
    to_xy, to_ll = local(lat_c.mean(), lon_c.mean())
    cx, cy = to_xy(lat_c, lon_c)
    cvx, cvy = np.gradient(gaussian_filter1d(cx, 5)) * FPS, np.gradient(gaussian_filter1d(cy, 5)) * FPS   # camera velocity

    raw = raw_positions(tr, cx, cy, b0)
    groups = link(raw, sex)
    out, id2g, speeds = {}, {}, []
    for members in groups:
        res = record(members, raw, to_ll, cvx, cvy)
        if res is None:
            continue
        rec, vs = res
        gid = members[0]
        out[gid] = rec
        for m in members:
            id2g[m] = gid
        speeds.append(vs)
    sp = np.concatenate(speeds) if speeds else np.zeros(1)
    durs = np.array([r["dur_s"] for r in out.values()]) if out else np.zeros(1)
    res = {"walk": walk, "variant": variant, "fps": FPS, "max_dist_m": MAX_DIST_M, "walk_m_s": WALK_MS, "stand_m_s": STAND_MS,
           "n_tracks": int(tr.track.nunique()), "n_tracks_with_position": len(raw), "n_people_after_relinking": len(out),
           "n_relinks": int(sum(len(r["members"]) - 1 for r in out.values())),
           "states": pd.Series([r["state"] for r in out.values()]).value_counts().to_dict(),
           "duration_s_pct": {str(q): round(float(np.percentile(durs, q)), 1) for q in (25, 50, 75, 90)},
           "speed_pct_m_s": {str(q): round(float(np.percentile(sp, q)), 2) for q in (10, 25, 50, 75, 90)}}
    (PROCESSED / walk / f"motion{sfx}.json").write_text(json.dumps({**res, "tracks": out, "id2g": id2g}, separators=(",", ":")), encoding="utf-8")
    return res


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.variant), ensure_ascii=False))


if __name__ == "__main__":
    main()
