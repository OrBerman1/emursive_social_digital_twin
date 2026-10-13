"""Stage `stab`: gyro-based stabilisation + horizon levelling, applied at render time (no re-encode).

  1. visual rotation between consecutive 5 fps frames (ORB on the equirect sphere + RANSAC Kabsch),
     masking the operator and the poles                                    -> cached visrot.parquet
  2. fit the IMU->camera axis mapping (signed permutation) and gyro/video time offset to those rotations
  3. integrate the 1 kHz gyro into camera orientation R(t) (cam -> world), levelled with the accelerometer
  4. smooth the heading; correction Q(t) = R(t)^T * Ry(smoothed heading)   (d_cam = Q d_view)
Output data/processed/<walk>/stab.json: Q per video frame (29.97 fps) and per analysis frame (5 fps).
The viewer rotates both the video and the overlays with Q, so they stay locked together.

usage:  python pipeline/stabilize.py walk1 [--sigma-heading 1.0]
"""
from __future__ import annotations

import argparse
import itertools
import json

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from common import INTERIM, PROCESSED, WALKS, Stage

VERSION = 1
FPS_A = 5.0
VIDEO_FPS = 30000 / 1001


# ---------- 1. visual rotation between frames ----------
def unit(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)], -1)


def kabsch(a, b):
    """R minimising |R a - b| for unit-vector sets (N,3)."""
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, d]) @ u.T


def ransac_rotation(a, b, iters=300, thr_deg=0.6, rng=np.random.default_rng(0)):
    best, best_in = np.eye(3), np.zeros(len(a), bool)
    cos_thr = np.cos(np.radians(thr_deg))
    for _ in range(iters):
        idx = rng.choice(len(a), 3, replace=False)
        r = kabsch(a[idx], b[idx])
        inl = np.einsum("ij,ij->i", (r @ a.T).T, b) > cos_thr
        if inl.sum() > best_in.sum():
            best, best_in = r, inl
    if best_in.sum() >= 6:
        best = kabsch(a[best_in], b[best_in])
    return best, int(best_in.sum())


def operator_mask_yaw(walk: str) -> tuple[float, float]:
    d = pd.read_parquet(INTERIM / walk / "det_yolo11x" / "detections.parquet", columns=["cls", "conf", "yaw", "pitch_top", "pitch_bot"])
    op = d[(d.cls == "person") & (d.conf > 0.5) & ((d.pitch_top - d.pitch_bot) > 75) & (d.pitch_bot < -50)]
    if len(op) < 20:
        return (0.0, 0.0)
    # the operator sits in a fixed camera direction: take the most frequent yaw (10 deg bins), fixed +-50 deg
    hist, edges = np.histogram(op.yaw, bins=36, range=(-180, 180))
    c = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    return (float(c), 50.0)


def visual_rotations(walk: str, width: int = 1920) -> pd.DataFrame:
    fdir = INTERIM / walk / "frames"
    index = json.loads((fdir / "index.json").read_text(encoding="utf-8"))
    op_c, op_w = operator_mask_yaw(walk)
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=12)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    h = width // 2
    yy, xx = np.mgrid[0:h, 0:width]
    yaw_px = (xx + 0.5) / width * 360 - 180
    pit_px = 90 - (yy + 0.5) / h * 180
    keep = (np.abs(pit_px) < 60) & (np.abs(((yaw_px - op_c + 180) % 360) - 180) > op_w)
    mask = keep.astype(np.uint8) * 255
    prev, rows = None, []
    for fr in index:
        img = cv2.imread(str(fdir / fr["file"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        kp, des = orb.detectAndCompute(img, mask)
        pts = np.array([k.pt for k in kp]) if kp else np.zeros((0, 2))
        cur = (pts, des)
        if prev is not None and prev[1] is not None and des is not None and len(pts) > 20:
            m = bf.match(prev[1], des)
            if len(m) >= 30:
                pa = prev[0][[x.queryIdx for x in m]]
                pb = pts[[x.trainIdx for x in m]]
                va = unit(pa[:, 0] / width * 360 - 180, 90 - pa[:, 1] / h * 180)
                vb = unit(pb[:, 0] / width * 360 - 180, 90 - pb[:, 1] / h * 180)
                r, n_in = ransac_rotation(va, vb)
                rv = Rot.from_matrix(r).as_rotvec()
                rows.append({"frame": fr["frame"], "t0": fr["t"] - 1 / FPS_A, "t1": fr["t"], "rx": rv[0], "ry": rv[1], "rz": rv[2], "inliers": n_in, "matches": len(m)})
        prev = cur
    return pd.DataFrame(rows)


# ---------- 2. axis mapping + time offset ----------
def signed_perms():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                m[i, j] = s
            yield m


def gyro_window_mean(t_ms, g, t0, t1):
    """Mean gyro (rad/s) over [t0, t1] seconds for many windows, via cumulative sums."""
    cs = np.vstack([np.zeros(3), np.cumsum(g, 0)])
    i0 = np.searchsorted(t_ms, t0 * 1000)
    i1 = np.searchsorted(t_ms, t1 * 1000)
    n = np.maximum(i1 - i0, 1)[:, None]
    return (cs[i1] - cs[i0]) / n


def fit_mapping(vis: pd.DataFrame, t_ms, g_rad):
    good = vis[vis.inliers >= 60]
    w_vis = good[["rx", "ry", "rz"]].to_numpy() / (good.t1 - good.t0).to_numpy()[:, None]
    best = (np.inf, None, None)
    for lag in np.arange(-0.5, 0.501, 0.01):
        w_g = gyro_window_mean(t_ms, g_rad, good.t0.to_numpy() + lag, good.t1.to_numpy() + lag)
        for m in signed_perms():
            err = np.median(np.linalg.norm(w_vis - w_g @ m.T, axis=1))
            if err < best[0]:
                best = (err, m, lag)
    err, m, lag = best
    w_g = gyro_window_mean(t_ms, g_rad, good.t0.to_numpy() + lag, good.t1.to_numpy() + lag) @ m.T
    corr = [float(np.corrcoef(w_vis[:, k], w_g[:, k])[0, 1]) for k in range(3)]
    return m, float(lag), float(err), corr, len(good)


# ---------- 3./4. orientation, levelling, smoothing ----------
def integrate(t_ms, w_cam, a_cam, gain=2e-4):
    """R(t) cam->world by integrating camera-frame angular velocity; slowly pulled so measured 'up' is world up."""
    up0 = a_cam[: 1000].mean(0)
    up0 /= np.linalg.norm(up0)
    r = Rot.align_vectors([[0, 1, 0]], [up0])[0]
    out = np.empty((len(t_ms), 4))
    dt = np.diff(t_ms, prepend=t_ms[0]) / 1000.0
    a_n = a_cam / np.maximum(np.linalg.norm(a_cam, axis=1, keepdims=True), 1e-6)
    for i in range(len(t_ms)):
        r = r * Rot.from_rotvec(w_cam[i] * dt[i])
        up_w = r.apply(a_n[i])                                   # measured 'up' expressed in world
        err = np.cross(up_w, [0.0, 1.0, 0.0])                    # rotate world so it aligns with +y
        if np.linalg.norm(a_cam[i]) < 11.5:                      # skip strong accelerations (steps, bumps)
            r = Rot.from_rotvec(err * gain) * r
        out[i] = r.as_quat()
    return Rot.from_quat(out)


def gauss(x, sigma):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def video_rotations(walk: str, starts_frac=(0.25, 0.5, 0.75), seconds: float = 8.0, w: int = 960) -> pd.DataFrame:
    """Visual rotation between consecutive frames of the encoded viewer video (30 fps), a few segments.
    Used to fine-tune the timing the viewer actually needs (5 fps fits are too coarse for walking shake)."""
    import subprocess
    h = w // 2
    src = PROCESSED / walk / "video" / "equirect.mp4"
    dur = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(src)],
                                          capture_output=True, text=True, check=True).stdout)["format"]["duration"])
    orb = cv2.ORB_create(3000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    yy, _ = np.mgrid[0:h, 0:w]
    mask = (np.abs(90 - (yy + .5) / h * 180) < 60).astype(np.uint8) * 255
    rows = []
    for sf in starts_frac:
        t0 = round(dur * sf, 2)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-i", str(src), "-t", str(seconds),
                              "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
        feats = [orb.detectAndCompute(f, mask) for f in fr]
        for i in range(len(fr) - 1):
            (ka, da), (kb, db) = feats[i], feats[i + 1]
            if da is None or db is None:
                continue
            m = bf.match(da, db)
            if len(m) < 40:
                continue
            pa = np.array([ka[x.queryIdx].pt for x in m]); pb = np.array([kb[x.trainIdx].pt for x in m])
            r, n = ransac_rotation(unit(pa[:, 0] / w * 360 - 180, 90 - pa[:, 1] / h * 180),
                                   unit(pb[:, 0] / w * 360 - 180, 90 - pb[:, 1] / h * 180), iters=200, thr_deg=0.8)
            if n >= 40:
                rv = Rot.from_matrix(r).as_rotvec()
                rows.append({"n": int(round(t0 * VIDEO_FPS)) + i, "rx": rv[0], "ry": rv[1], "rz": rv[2], "inliers": n})
    return pd.DataFrame(rows)


def residual_deg(qa: np.ndarray, qb: np.ndarray, rv: np.ndarray) -> float:
    """Median rotation left between consecutive stabilised frames: E = Qb^T R_vis Qa."""
    E = np.einsum("nji,njk,nkl->nil", qb, Rot.from_rotvec(rv).as_matrix(), qa)
    return float(np.median(np.degrees(Rot.from_matrix(E).magnitude())))


def run(walk: str, sigma_heading: float, force: bool) -> dict:
    out_dir = PROCESSED / walk
    vis_path = INTERIM / walk / "visrot.parquet"
    fm = json.loads((INTERIM / walk / "frames" / "manifest.json").read_text(encoding="utf-8"))
    st = Stage("visrot", VERSION, INTERIM / walk / "visrot", {"width": 1920}, {"frames_key": fm["key"]})
    if st.done() and vis_path.exists() and not force:
        vis = pd.read_parquet(vis_path)
    else:
        st.start()
        vis = visual_rotations(walk)
        vis.to_parquet(vis_path, index=False)
        st.finish({"pairs": len(vis)})

    imu = pd.read_csv(INTERIM / walk / "imu.csv")
    t_ms = imu.t_ms.to_numpy()
    g_rad = np.radians(imu[["gx", "gy", "gz"]].to_numpy())
    acc = imu[["ax", "ay", "az"]].to_numpy()
    m, lag, err, corr, n_pairs = fit_mapping(vis, t_ms, g_rad)
    p = -m                                                       # vectors: IMU axes -> camera axes
    if np.linalg.det(p) < 0:                                     # should not happen for a real IMU
        p = m
    w_cam = g_rad @ p.T
    a_cam = acc @ p.T
    if np.median(a_cam[:, 1]) < 0:        # sensor reports gravity (down); frames are upright, so 'up' must be ~+y
        a_cam = -a_cam
    t_cam = t_ms - lag * 1000                                     # IMU time -> video time
    R = integrate(t_cam, w_cam, a_cam)

    fwd = R.apply([0.0, 0.0, 1.0])
    head = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    head_s = gauss(head, sigma_heading * 1000)                    # samples are ~1 ms apart

    def q_at(times_s):
        idx = np.clip(np.searchsorted(t_cam, times_s * 1000), 0, len(t_cam) - 1)
        S = Rot.from_euler("y", head_s[idx][:, None])
        return (R[idx].inv() * S).as_matrix()

    # --- fine timing: pick the time shift that leaves the least frame-to-frame rotation, separately for the
    #     30 fps viewer video and the 5 fps analysis frames (each measured on its own images) ---
    vv_path = INTERIM / walk / "visrot_video.parquet"
    if vv_path.exists() and not force:
        vv = pd.read_parquet(vv_path)
    else:
        vv = video_rotations(walk)
        vv.to_parquet(vv_path, index=False)
    shifts = np.arange(-0.2, 0.2001, 0.005)
    nv, rvv = vv["n"].to_numpy(), vv[["rx", "ry", "rz"]].to_numpy()
    res_v = [residual_deg(q_at(nv / VIDEO_FPS + s), q_at((nv + 1) / VIDEO_FPS + s), rvv) for s in shifts]
    shift_v = float(shifts[int(np.argmin(res_v))])
    good = vis[vis.inliers >= 60]
    fa, rva = good["frame"].to_numpy(), good[["rx", "ry", "rz"]].to_numpy()
    res_a = [residual_deg(q_at((fa - 1) / FPS_A + s), q_at(fa / FPS_A + s), rva) for s in shifts]
    shift_a = float(shifts[int(np.argmin(res_a))])
    raw_v = float(np.median(np.degrees(np.linalg.norm(rvv, axis=1))))
    raw_a = float(np.median(np.degrees(np.linalg.norm(rva, axis=1))))

    n_video = int(round(fm["video_duration_s"] * VIDEO_FPS))
    qv = q_at(np.arange(n_video) / VIDEO_FPS + shift_v)
    qa = q_at(np.arange(fm["n_frames"]) / FPS_A + shift_a)
    # quality: residual shake = rotation left between consecutive stabilised analysis frames
    tilt = np.degrees(np.arccos(np.clip(R.apply([0.0, 1.0, 0.0])[:, 1], -1, 1)))
    out = {
        "walk": walk, "video_fps": VIDEO_FPS, "analysis_fps": FPS_A,
        "imu_to_cam": p.tolist(), "gyro_lag_s": lag, "fit_median_err_rad_s": round(err, 4),
        "fit_corr_xyz": [round(c, 3) for c in corr], "pairs_used": n_pairs,
        "camera_tilt_deg": {"median": round(float(np.median(tilt)), 1), "p95": round(float(np.percentile(tilt, 95)), 1)},
        "sigma_heading_s": sigma_heading,
        "timing_shift_s": {"video": round(shift_v, 3), "analysis": round(shift_a, 3)},
        "check_deg_per_frame": {
            "video_30fps": {"raw": round(raw_v, 3), "stabilised": round(min(res_v), 3), "at_shift_0": round(res_v[int(np.argmin(np.abs(shifts)))], 3)},
            "analysis_5fps": {"raw": round(raw_a, 3), "stabilised": round(min(res_a), 3)},
        },
        "q_video": np.round(qv.reshape(len(qv), 9), 4).tolist(),
        "q_analysis": np.round(qa.reshape(len(qa), 9), 4).tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stab.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return {k: v for k, v in out.items() if not k.startswith("q_")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--sigma-heading", type=float, default=1.0, help="heading smoothing (s); larger = steadier")
    ap.add_argument("--force", action="store_true", help="recompute the cached visual rotations")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.sigma_heading, a.force), indent=2))


if __name__ == "__main__":
    main()
