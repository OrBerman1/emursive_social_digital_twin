"""Cheap stage (always recomputed from the cached detections): clean, de-duplicate and track.

  1. confidence filter + remove operator / finger artefacts (handheld camera, operator in frame)
  2. merge duplicates of one object seen in overlapping tiles (NMS in yaw/pitch space)
  3. track people across frames (Hungarian matching on angular IoU) -> persistent IDs P-####
  4. drop the operator track; estimate camera height from people (assumes ~1.7 m mean height)
  5. ground distance per detection: d = cam_h / tan(-pitch_foot)

usage:  python pipeline/tracks.py walk1 [--person-conf 0.35] [--det det_yolo11x]
"""
from __future__ import annotations

import argparse
import json

import lap
import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, WALKS
from spherical import wrap180

TRACK_CLASSES = {"person"}
OBJECT_CLASSES = {"bicycle", "motorcycle", "car", "truck", "bus", "bench", "chair", "umbrella",
                  "handbag", "backpack", "suitcase", "potted plant", "dining table"}
PERSON_H = 1.70


def angular_iou(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """IoU of spherical boxes (yaw_l..yaw_r x pitch_bot..pitch_top), yaw wrap-aware. Returns |a| x |b|."""
    ac, bc = a["yaw"].to_numpy()[:, None], b["yaw"].to_numpy()[None, :]
    aw = (wrap180(a["yaw_r"].to_numpy() - a["yaw_l"].to_numpy()) % 360)[:, None]
    bw = (wrap180(b["yaw_r"].to_numpy() - b["yaw_l"].to_numpy()) % 360)[None, :]
    d = np.abs(wrap180(ac - bc))
    ix = np.clip((aw + bw) / 2 - d, 0, None)
    ix = np.minimum(ix, np.minimum(aw, bw))
    at, ab = a["pitch_top"].to_numpy()[:, None], a["pitch_bot"].to_numpy()[:, None]
    bt, bb = b["pitch_top"].to_numpy()[None, :], b["pitch_bot"].to_numpy()[None, :]
    iy = np.clip(np.minimum(at, bt) - np.maximum(ab, bb), 0, None)
    inter = ix * iy
    union = aw * (at - ab) + bw * (bt - bb) - inter
    return inter / np.maximum(union, 1e-6)


def tile_centrality(df: pd.DataFrame, size: int = 1024) -> np.ndarray:
    """1 at the tile centre, 0 when the box touches the tile edge (prefer un-truncated views)."""
    m = np.minimum.reduce([df.x1.to_numpy(), df.y1.to_numpy(), size - df.x2.to_numpy(), size - df.y2.to_numpy()])
    return np.clip(m / (size / 4), 0, 1)


def dedup_frame(g: pd.DataFrame, iou_thr: float = 0.35) -> pd.DataFrame:
    if len(g) <= 1:
        return g
    score = g["conf"].to_numpy() * (0.5 + 0.5 * tile_centrality(g))
    order = np.argsort(-score)
    g = g.iloc[order]
    iou = angular_iou(g, g)
    keep, removed = [], np.zeros(len(g), bool)
    for i in range(len(g)):
        if removed[i]:
            continue
        keep.append(i)
        removed |= iou[i] > iou_thr
    return g.iloc[keep]


def similarity(pred: pd.DataFrame, g: pd.DataFrame) -> np.ndarray:
    """max(angular IoU, Gaussian of centre distance scaled by box height): robust for small/far people
    whose boxes jump more than their own width between frames."""
    iou = angular_iou(pred, g)
    pmid = ((pred.pitch_top + pred.pitch_bot) / 2).to_numpy()[:, None]
    gmid = ((g.pitch_top + g.pitch_bot) / 2).to_numpy()[None, :]
    dy = wrap180(pred.yaw.to_numpy()[:, None] - g.yaw.to_numpy()[None, :])
    scale = 0.6 * np.maximum((pred.pitch_top - pred.pitch_bot).to_numpy()[:, None], 1.0)
    dn2 = (dy ** 2 + (pmid - gmid) ** 2) / scale ** 2
    hr = (g.pitch_top - g.pitch_bot).to_numpy()[None, :] / np.maximum((pred.pitch_top - pred.pitch_bot).to_numpy()[:, None], 1e-3)
    size_ok = (hr > 0.6) & (hr < 1.65)
    return np.maximum(iou, np.exp(-dn2) * size_ok)


def track(dets: pd.DataFrame, sim_min: float = 0.3, max_age: int = 10, min_hits: int = 3, do_stitch: bool = True) -> pd.DataFrame:
    """Frame-by-frame Hungarian assignment (IoU + scaled centre distance) with constant-velocity yaw prediction."""
    active: list[dict] = []
    next_id, out = 1, []
    for frame, g in dets.groupby("frame", sort=True):
        g = g.reset_index(drop=True)
        if active:
            pred = pd.DataFrame([{**t["last"], **{k: t["last"][k] + t["vyaw"] * (frame - t["frame"]) for k in ("yaw", "yaw_l", "yaw_r")}}
                                 for t in active])
            sim = similarity(pred, g)
            cost = 1 - sim
            cost[sim < sim_min] = 1e3
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1 - sim_min + 1e-9)
        else:
            x = np.array([], int)
        assigned = set()
        for ti, di in enumerate(x):
            if di < 0:
                continue
            t, row = active[ti], g.iloc[di].to_dict()
            t["vyaw"] = 0.7 * t["vyaw"] + 0.3 * float(wrap180(row["yaw"] - t["last"]["yaw"])) / max(frame - t["frame"], 1)
            t.update(last=row, frame=frame, hits=t["hits"] + 1)
            out.append({**row, "track": t["id"]})
            assigned.add(di)
        for di in range(len(g)):
            if di not in assigned:
                row = g.iloc[di].to_dict()
                active.append({"id": next_id, "last": row, "frame": frame, "hits": 1, "vyaw": 0.0})
                out.append({**row, "track": next_id})
                next_id += 1
        active = [t for t in active if frame - t["frame"] <= max_age]
    df = pd.DataFrame(out)
    if do_stitch:
        df = stitch(df)
    hits = df.groupby("track")["frame"].transform("count")
    return df[hits >= min_hits]


def stitch(df: pd.DataFrame, max_gap: int = 15, max_dn: float = 1.2) -> pd.DataFrame:
    """Join tracklets: a track that ends and another that starts within max_gap frames, near the
    extrapolated position and with a similar angular size, are the same entity (occlusions, misses)."""
    if df.empty:
        return df
    g = df.sort_values("frame").groupby("track")
    heads, tails = g.head(1).set_index("track"), g.tail(1).set_index("track")
    first2 = g.head(2).groupby("track")
    last2 = g.tail(2).groupby("track")
    vel = {}
    for tid, s in last2:
        vel[tid] = float(wrap180(s.yaw.iloc[-1] - s.yaw.iloc[0])) / max(s.frame.iloc[-1] - s.frame.iloc[0], 1) if len(s) > 1 else 0.0
    starts = heads.sort_values("frame")
    parent = {}
    taken = set()
    for tid, end in tails.sort_values("frame").iterrows():
        h = end.pitch_top - end.pitch_bot
        cand = starts[(starts.frame > end.frame) & (starts.frame <= end.frame + max_gap)]
        best, best_d = None, max_dn
        for sid, st in cand.iterrows():
            if sid in taken or sid == tid:
                continue
            gap = st.frame - end.frame
            dy = float(wrap180(st.yaw - (end.yaw + vel.get(tid, 0.0) * gap)))
            dp = (st.pitch_top + st.pitch_bot) / 2 - (end.pitch_top + end.pitch_bot) / 2
            hr = (st.pitch_top - st.pitch_bot) / max(h, 1e-3)
            dn = np.hypot(dy, dp) / max(h, 1.0)
            if 0.6 < hr < 1.65 and dn < best_d:
                best, best_d = sid, dn
        if best is not None:
            parent[best] = tid
            taken.add(best)
    def root(t):
        while t in parent:
            t = parent[t]
        return t
    df = df.copy()
    df["track"] = df["track"].map(root)
    return df


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase (forward + backward) Gaussian smoothing; offline, so no lag."""
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def smooth_tracks(df: pd.DataFrame, max_gap: int = 5, sigma_pos: float = 1.5, sigma_size: float = 3.0) -> pd.DataFrame:
    """Per track: fill gaps up to max_gap frames by interpolation, then smooth centre (sigma_pos frames)
    and box size (sigma_size frames, sizes change slowly). Removes size jumps, tile-switch jitter and
    one-frame dropouts. Rows added by interpolation are flagged interp=True."""
    out = []
    for tid, g in df.groupby("track"):
        g = g.sort_values("frame").drop_duplicates("frame")
        seg = (g["frame"].diff().fillna(1) > max_gap + 1).cumsum()
        for _, s in g.groupby(seg):
            f = s["frame"].to_numpy()
            fr = np.arange(f[0], f[-1] + 1)
            yaw = np.degrees(np.unwrap(np.radians(s["yaw"].to_numpy())))
            width = wrap180(s["yaw_r"].to_numpy() - s["yaw_l"].to_numpy()) % 360
            pm = ((s["pitch_top"] + s["pitch_bot"]) / 2).to_numpy()
            h = (s["pitch_top"] - s["pitch_bot"]).to_numpy()
            yaw_s = _gauss(np.interp(fr, f, yaw), sigma_pos)
            pm_s = _gauss(np.interp(fr, f, pm), sigma_pos)
            w_s = _gauss(np.interp(fr, f, width), sigma_size)
            h_s = _gauss(np.interp(fr, f, h), sigma_size)
            conf = np.interp(fr, f, s["conf"].to_numpy())
            yc = wrap180(yaw_s)
            out.append(pd.DataFrame({
                "frame": fr, "t": fr / 5.0, "cls": s["cls"].iloc[0], "conf": conf, "track": tid,
                "yaw": yc, "yaw_l": wrap180(yc - w_s / 2), "yaw_r": wrap180(yc + w_s / 2), "yaw_foot": yc,
                "pitch": pm_s, "pitch_top": pm_s + h_s / 2, "pitch_bot": pm_s - h_s / 2, "ang_h": h_s,
                "interp": ~np.isin(fr, f),
            }))
    return pd.concat(out, ignore_index=True) if out else df.assign(interp=False)


def track_per_class(d: pd.DataFrame, id_offset: int = 0) -> pd.DataFrame:
    """Track each object class separately (a crate never becomes a chair), then smooth; IDs unique across classes."""
    parts, next_id = [], id_offset
    for _, g in d.groupby("cls"):
        t = track(g.sort_values("frame"), sim_min=0.25, max_age=8, min_hits=3)
        if len(t):
            t = smooth_tracks(t)
            t["track"] = t["track"].rank(method="dense").astype(int) + next_id
            next_id = int(t["track"].max())
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else d.iloc[:0].assign(track=pd.Series(dtype=int), interp=pd.Series(dtype=bool))


def operator_like(walk: str, p: pd.DataFrame, variant: str) -> np.ndarray:
    """True for detections of the handheld camera's operator: inside the operator's fixed CAMERA-frame sector
    and very close (big box or feet far below the horizon). On stabilised frames the box is first rotated
    back to camera directions with that frame's stabilisation rotation (d_cam = Q d_stab)."""
    from stabilize import operator_mask_yaw
    op_c, _ = operator_mask_yaw(walk)
    yaw, pbot = p["yaw"].to_numpy(), p["pitch_bot"].to_numpy()
    if variant:
        q = np.array(json.loads((PROCESSED / walk / "stab.json").read_text(encoding="utf-8"))["q_analysis"]).reshape(-1, 3, 3)
        Q = q[np.clip(p["frame"].to_numpy(), 0, len(q) - 1)]
        def to_cam(yw, pt):
            a, b = np.radians(yw), np.radians(pt)
            v = np.stack([np.cos(b) * np.sin(a), np.sin(b), np.cos(b) * np.cos(a)], -1)
            c = np.einsum("nij,nj->ni", Q, v)
            return np.degrees(np.arctan2(c[:, 0], c[:, 2])), np.degrees(np.arcsin(np.clip(c[:, 1], -1, 1)))
        yaw, _ = to_cam(p["yaw"].to_numpy(), (p["pitch_top"].to_numpy() + pbot) / 2)
        _, pbot = to_cam(p["yaw"].to_numpy(), pbot)
    in_sector = np.abs(wrap180(yaw - op_c)) < 45
    close = (pbot < -40) | (p["ang_h"].to_numpy() > 40)
    return in_sector & close


def run(walk: str, det: str, person_conf: float, object_conf: float, cam_h: float | None, variant: str = "") -> dict:
    src = INTERIM / walk / (det + variant) / "detections.parquet"
    d = pd.read_parquet(src)
    n_frames = json.loads((INTERIM / walk / ("frames" + variant) / "manifest.json").read_text(encoding="utf-8"))["n_frames"]
    d["ang_h"] = d["pitch_top"] - d["pitch_bot"]

    # --- people ---
    p = d[(d.cls == "person") & (d.conf >= person_conf)].copy()
    n0 = len(p)
    p = p[(p.pitch_top > -35)]                 # fingers / hand at the bottom of the frame
    p = p[(p.ang_h < 75)]                      # operator body: huge, right next to the lens
    p = p[~operator_like(walk, p, variant)]    # operator in the camera-fixed sector (works on stabilised frames too)
    p = pd.concat([dedup_frame(g) for _, g in p.groupby("frame")])
    tr = track(p.sort_values("frame"))
    # operator track(s): present most of the walk and very large on average
    stats = tr.groupby("track").agg(n=("frame", "count"), ang_h=("ang_h", "median"))
    operator = stats[(stats.n > 0.3 * n_frames) & (stats.ang_h > 35)].index.tolist()
    tr = tr[~tr.track.isin(operator)]
    tr = smooth_tracks(tr)

    # --- camera height from people (median of implied heights for well-framed, mid-distance boxes) ---
    ok = tr[(~tr.interp) & (tr.pitch_bot < -3) & (tr.pitch_bot > -45) & (tr.pitch_top > tr.pitch_bot + 3) & (tr.conf > 0.6)]
    tb, tt = np.tan(np.radians(-ok.pitch_bot)), np.tan(np.radians(ok.pitch_top))
    implied = PERSON_H * tb / (tt + tb)
    cam_est = float(np.median(implied)) if len(implied) > 20 else 1.4
    h = cam_h or cam_est
    tr["dist_m"] = np.where(tr.pitch_bot < -1, h / np.tan(np.radians(-tr.pitch_bot.clip(upper=-1))), np.nan)
    tr["pid"] = tr.track.map(lambda i: f"P-{int(i):04d}")

    # --- level-1 objects: de-duplicated, tracked per class and smoothed like people ---
    o = d[d.cls.isin(OBJECT_CLASSES) & (d.conf >= object_conf) & (d.ang_h < 60)]
    o = pd.concat([dedup_frame(g) for _, g in o.groupby(["frame", "cls"])]) if len(o) else o
    o = track_per_class(o, id_offset=100000)

    out = INTERIM / walk / ("tracks" + variant)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(out / "persons.parquet", index=False)
    o.to_parquet(out / "objects.parquet", index=False)
    per_frame = tr.groupby("frame").size().reindex(range(n_frames), fill_value=0)
    summary = {
        "walk": walk, "det": det, "person_conf": person_conf, "frames": n_frames,
        "person_boxes_raw": int(n0), "person_boxes_tracked": int(len(tr)),
        "unique_tracks": int(tr.track.nunique()), "operator_tracks_removed": operator,
        "camera_height_est_m": round(cam_est, 2), "camera_height_used_m": round(h, 2),
        "people_per_frame": {"mean": round(float(per_frame.mean()), 1), "max": int(per_frame.max())},
        "objects": o.cls.value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_level2(walk: str, det: str, conf: float, cam_h: float, variant: str = "") -> dict:
    """Level-2 (open-vocabulary) objects: per-class de-dup, ground distance, tracking -> persistent O-#### IDs."""
    src = INTERIM / walk / (det + variant) / "detections.parquet"
    if not src.exists():
        return {"walk": walk, "level2": "not run"}
    d = pd.read_parquet(src)
    d = d[(d.conf >= conf) & (d.pitch_top > -50)].copy()
    d["ang_h"] = d["pitch_top"] - d["pitch_bot"]
    d = d[d.ang_h < 60]                                            # the operator's arm/body read as "objects"
    d = pd.concat([dedup_frame(g) for _, g in d.groupby(["frame", "cls"])]) if len(d) else d
    l2 = track_per_class(d, id_offset=0)
    l2["dist_m"] = np.where(l2.pitch_bot < -1, cam_h / np.tan(np.radians(-l2.pitch_bot.clip(upper=-1))), np.nan)
    l2["oid"] = l2.track.map(lambda i: f"O-{int(i):04d}")
    l2.to_parquet(INTERIM / walk / ("tracks" + variant) / "level2.parquet", index=False)
    per_cls = l2.groupby("cls")["track"].nunique().sort_values(ascending=False)
    return {"walk": walk, "level2_unique_objects": per_cls.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--det", default="det_yolo11x")
    ap.add_argument("--person-conf", type=float, default=0.35)
    ap.add_argument("--object-conf", type=float, default=0.35)
    ap.add_argument("--cam-h", type=float, default=None, help="override the estimated camera height (m)")
    ap.add_argument("--det2", default="det_yolov8x-worldv2_level2", help="level-2 detections folder")
    ap.add_argument("--l2-conf", type=float, default=0.15)
    ap.add_argument("--variant", default="", choices=["", "stab"], help="stab = use offline-stabilised frames/detections")
    a = ap.parse_args()
    v = "_stab" if a.variant == "stab" else ""
    for w in a.walks:
        s = run(w, a.det, a.person_conf, a.object_conf, a.cam_h, v)
        print(json.dumps(s, indent=2))
        print(json.dumps(run_level2(w, a.det2, a.l2_conf, s["camera_height_used_m"], v), indent=2))


if __name__ == "__main__":
    main()
