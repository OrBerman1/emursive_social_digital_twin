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
         0. camera ego-motion from the video itself: every static ground object (parked motorcycle, chair, crate,
            table, bin, stool...) implies camera velocity = -(its velocity relative to the camera); per frame the
            geometric median over static objects within +-2.4 s (map-fit velocity where there are too few). The map
            fit alone was up to ~2x too fast and ~15 deg off in places, which made walkers look static and
            static things look like walkers.
         1b. tracker jumps (an ID sliding onto another person or the operator) are cut: consecutive positions that
            imply > 1 m + 3 m/s are split and only the longest piece is kept.
         3. per linked person: position and velocity from a local linear fit around every frame (Gaussian weights,
            sigma 1 s / 1.5 s), gaps <= 2 s bridged. A local fit keeps the true speed at the start and end of a
            track (plain Gaussian smoothing mirrors the data there and flattened short tracks to ~0 m/s). Positions are map coordinates, so the camera's own motion is
            already removed: this is ego-motion-compensated motion ("optical flow of people" for a moving camera).
         4. velocity = average of two independent estimates: (a) the geometry above, (b) optical flow of the feet
            minus the ground right beside them (person_flow.py, 2 s rolling median). Held-out static objects read
            0.29 m/s median with the average (0.33 geometry, 0.42 flow alone): that is the noise floor.
            Direction is flagged reliable ("ok") only when 0.35 <= speed <= 2.5 m/s and the two estimates point the
            same way (within 60 deg); the viewer draws arrows only for those. (Check: for clearly moving people the two
            estimates agree within 45 deg in 80% of frames, peak at 0 deg: no sign error.)
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
JUMP_M, JUMP_M_PER_S = 1.0, 3.0   # a person cannot move further than this between two detections
ARROW_MS, MAX_MS = 0.35, 2.5   # direction shown from 0.35 m/s; faster than 2.5 m/s in a market = measurement error
DIR_AGREE_DEG = 60.0   # direction is 'reliable' when geometry and optical flow agree within this
STATIC = {"motorcycle", "chair", "potted plant", "dining table", "display table", "stack of crates", "trash bin", "stool",
          "plastic crate", "cardboard box", "plastic chair", "bench"}
EGO_CHUNK_F = 25       # static-object track pieces of <= 5 s
EGO_WIN_F = 12         # camera velocity = geometric median over static pieces centred within +-2.4 s
EGO_MIN_PIECES = 3


def local(lat0, lon0):
    kx, ky = 111320.0 * np.cos(np.radians(lat0)), 110574.0
    return (lambda lat, lon: ((np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky),
            lambda x, y: (lat0 + np.asarray(y) / ky, lon0 + np.asarray(x) / kx))


def geomedian(P, iters=30):
    """Geometric median of 2-D points (Weiszfeld): robust centre of a cloud of velocity estimates."""
    m = np.median(P, axis=0)
    for _ in range(iters):
        d = np.hypot(*(P - m).T).clip(1e-3)
        m = (P / d[:, None]).sum(0) / (1 / d).sum()
    return m


def static_tracks(walk: str, sfx: str, h: float, n: int, keep=None):
    """Static ground objects (parked motorcycles, chairs, crates, tables, bins, stools...) with ground distances."""
    objs = pd.concat([pd.read_parquet(INTERIM / walk / f"tracks{sfx}" / f"{k}.parquet").assign(src=k) for k in ("objects", "level2")
                      if (INTERIM / walk / f"tracks{sfx}" / f"{k}.parquet").exists()])
    objs = objs[objs.cls.isin(STATIC) & (objs.frame < n)].copy()
    objs["track"] = objs.groupby(["src", "track"]).ngroup()
    if keep is not None:
        objs = objs[objs.track.map(keep)]
    pb = objs.pitch_bot.to_numpy()
    objs["dist_m"] = np.where(pb < -1, h / np.tan(np.radians(-np.clip(pb, None, -1))), np.nan)
    return objs


def ego_motion(walk: str, sfx: str, cx, cy, b0, h: float, keep=None):
    """Camera ground velocity measured in the video: every static object implies camera velocity = -(its velocity
    relative to the camera). Per frame: geometric median over static-object pieces within +-EGO_WIN_F, smoothed;
    where there are too few static objects, the map-fit velocity is used. Same distance scale as the people, so
    camera-height errors scale speeds but do not create fake motion. Returns (ex, ey, n_frames_visual, n_pieces)."""
    n = len(cx)
    gx, gy = gaussian_filter1d(cx, 5), gaussian_filter1d(cy, 5)
    gvx, gvy = np.gradient(gx) * FPS, np.gradient(gy) * FPS
    objs = static_tracks(walk, sfx, h, n, keep)
    mids, vel = [], []
    for tid in objs.track.unique():
        t = objs[objs.track == tid].sort_values("frame")
        d, f = t.dist_m.to_numpy(), t.frame.to_numpy().astype(int)
        ok = np.isfinite(d) & (d >= 1.5) & (d <= 10)
        f, d, yw = f[ok], d[ok], t.yaw_foot.to_numpy()[ok]
        b = np.radians(b0[f] + yw)
        rx, ry = d * np.sin(b), d * np.cos(b)
        for a0 in range(0, len(f), EGO_CHUNK_F // 2):                      # overlapping pieces of <= 5 s
            sl = slice(a0, a0 + EGO_CHUNK_F)
            if len(f[sl]) < 8 or f[sl][-1] - f[sl][0] < 6:
                continue
            fs = f[sl] - f[sl].mean()
            v = -np.array([np.polyfit(fs, rx[sl], 1)[0], np.polyfit(fs, ry[sl], 1)[0]]) * FPS
            if np.hypot(*v) < 3.0:
                mids.append(f[sl].mean()); vel.append(v)
    mids, vel = np.array(mids), np.array(vel).reshape(-1, 2)
    V = np.c_[gvx, gvy].copy()
    vis = np.zeros(n, bool)
    for i in range(n):
        m = np.abs(mids - i) <= EGO_WIN_F
        if m.sum() >= EGO_MIN_PIECES:
            V[i] = geomedian(vel[m]); vis[i] = True
    V = np.c_[gaussian_filter1d(V[:, 0], 5), gaussian_filter1d(V[:, 1], 5)]
    ex = cx[0] + np.concatenate([[0], np.cumsum(V[:-1, 0] / FPS)])
    ey = cy[0] + np.concatenate([[0], np.cumsum(V[:-1, 1] / FPS)])
    return ex, ey, int(vis.sum()), len(vel)


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
        x, y, dd = cx[f] + d[ok] * np.sin(b), cy[f] + d[ok] * np.cos(b), d[ok]
        # cut at impossible jumps (ID switches), keep the longest piece
        jump = np.hypot(np.diff(x), np.diff(y)) > JUMP_M + JUMP_M_PER_S * np.diff(f) / FPS
        cuts = np.r_[0, np.where(jump)[0] + 1, len(f)]
        a, c = max(zip(cuts[:-1], cuts[1:]), key=lambda ac: ac[1] - ac[0])
        if c - a < 2:
            continue
        out[int(tid)] = {"f": f[a:c], "x": x[a:c], "y": y[a:c], "d": dd[a:c], "n_total": len(t), "n_cut": int(len(f) - (c - a))}
    return out


def loclin(t, v, T, sigma):
    """Local linear fit of samples (t, v) around each T: value and slope (per frame). Gaussian weights within 3 sigma."""
    dt = t[None, :] - T[:, None]
    w = np.exp(-0.5 * (dt / sigma) ** 2) * (np.abs(dt) <= 3 * sigma)
    s0, s1, s2 = w.sum(1), (w * dt).sum(1), (w * dt * dt).sum(1)
    r0, r1 = (w * v[None, :]).sum(1), (w * dt * v[None, :]).sum(1)
    det = s0 * s2 - s1 * s1
    good = det > 1e-6
    slope = np.where(good, (s0 * r1 - s1 * r0) / np.where(good, det, 1), 0.0)
    value = np.where(s0 > 0, (r0 - slope * s1) / np.where(s0 > 0, s0, 1), np.nan)
    return value, slope


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


def flow_series(walk: str, sfx: str, b0) -> dict:
    """Per person track: optical-flow ground velocity (world axes), 2 s rolling median, indexed by frame."""
    fp = INTERIM / walk / f"person_flow{sfx}" / "person_flow.parquet"
    if not fp.exists():
        return {}
    fl = pd.read_parquet(fp)
    fl = fl[fl.kind == "person"]
    out = {}
    for tid in fl.track.unique():
        t = fl[fl.track == tid].sort_values("frame")
        a = np.radians(b0[t.frame.to_numpy()])
        wx = t.vx.to_numpy() * np.cos(a) + t.vy.to_numpy() * np.sin(a)        # camera-yaw frame -> world (bearing b0)
        wy = -t.vx.to_numpy() * np.sin(a) + t.vy.to_numpy() * np.cos(a)
        rx = pd.Series(wx).rolling(10, center=True, min_periods=4).median().to_numpy()
        ry = pd.Series(wy).rolling(10, center=True, min_periods=4).median().to_numpy()
        out[int(tid)] = dict(zip(t.frame.to_numpy().astype(int), zip(rx, ry)))
    return out


def record(members, raw, to_ll, cvx, cvy, pin, flow=None):
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
    valid = np.zeros(len(F), bool)
    idx = np.searchsorted(F, f)
    valid[idx] = True
    for a, c in zip(idx[:-1], idx[1:]):
        if c - a - 1 <= GAP_FRAMES:
            valid[a:c + 1] = True
    Xs, _ = loclin(f, x, F, SIG_POS_S * FPS)
    Ys, _ = loclin(f, y, F, SIG_POS_S * FPS)
    _, vx = loclin(f, x, F, SIG_VEL_S * FPS)
    _, vy = loclin(f, y, F, SIG_VEL_S * FPS)
    vx, vy = vx * FPS, vy * FPS
    # combine with optical flow (feet vs ground beside them) where available; flag reliable directions
    ok_dir = np.zeros(len(F), bool)
    if flow:
        fx, fy = np.full(len(F), np.nan), np.full(len(F), np.nan)
        for m in members:
            for fr, (a, b) in (flow.get(m) or {}).items():
                if F[0] <= fr <= F[-1] and np.isfinite(a):
                    fx[fr - F[0]], fy[fr - F[0]] = a, b
        has = np.isfinite(fx)
        gvx, gvy = vx.copy(), vy.copy()
        vx = np.where(has, (gvx + fx) / 2, gvx)
        vy = np.where(has, (gvy + fy) / 2, gvy)
        sp = np.hypot(vx, vy)
        dang = np.abs((np.degrees(np.arctan2(gvx, gvy) - np.arctan2(np.nan_to_num(fx), np.nan_to_num(fy))) + 540) % 360 - 180)
        ok_dir = has & (sp >= ARROW_MS) & (sp <= MAX_MS) & (dang <= DIR_AGREE_DEG)
    fm = F[len(F) // 2]                                       # ego-frame path pinned to the map-fit camera position
    Xs, Ys = Xs + pin[0][fm], Ys + pin[1][fm]
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
            "ok": [int(o) if m else None for o, m in zip(ok_dir, valid)], "ok_share": round(float(ok_dir[valid].mean()), 2),
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
    h = json.loads((INTERIM / walk / f"tracks{sfx}" / "summary.json").read_text())["camera_height_used_m"]
    ex, ey, n_vis, n_pieces = ego_motion(walk, sfx, cx, cy, b0, h)
    cvx, cvy = np.gradient(ex) * FPS, np.gradient(ey) * FPS   # camera velocity on the image distance scale
    pin = (cx - ex, cy - ey)                                  # ego frame -> map frame offset per frame

    raw = raw_positions(tr, ex, ey, b0)
    flow = flow_series(walk, sfx, b0)
    groups = link(raw, sex)
    out, id2g, speeds = {}, {}, []
    for members in groups:
        res = record(members, raw, to_ll, cvx, cvy, pin, flow)
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
           "ego_frames_from_video": n_vis, "ego_static_pieces": n_pieces, "camera_speed_median_m_s": round(float(np.median(np.hypot(cvx, cvy))), 2),
           "flow_tracks": len(flow), "noise_floor_static_m_s": 0.29,
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
