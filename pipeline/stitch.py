"""Calibrated dual-fisheye -> equirectangular stitch with a blended seam (replaces ffmpeg v360 for projects that have
config/stitch.json; the original project keeps its v360 stitch unchanged).

Why: v360 cuts each lens hard at 90 deg with one shared FOV (190) and no lens centring, so at the two seams (yaw +-90)
a strip of the scene is duplicated or missing and the two lens exposures meet in a visible line.
Model (per lens k): equidistant fisheye, image-circle centre (cx, cy) in its half of the frame, focal f (px / rad);
lens 0 looks along +z (camera frame: x right, y down in the lens-0 image), lens 1 = Ry(180) * small rotation.
Calibration: the ring both lenses see (about 80-100 deg from the lens axes) is rendered from each lens, SIFT matches
between the two renders give pairs of raw pixels that see the same point, and robust least squares fits
(f0, f1, cx0, cy0, cx1, cy1, rotation of lens 1) so the pairs agree (distant points dominate; the operator's
close-up parallax is down-weighted by the robust loss). Repeated on frames of both walks.
Orientation G: the camera frame is rotated to the old v360 output frame (matched features on the lens interiors),
so yaw / pitch in the new frames mean the same as before (anchor points stay valid).
Stitch: each output pixel takes lens 0 and lens 1 through precomputed remap tables and blends them over
+-BLEND_DEG around the 90 deg line; a per-frame gain (from the overlap ring, smoothed in time) evens out the two
exposures.

usage:  python pipeline/stitch.py calibrate [walk1 walk2]      -> CONFIG/stitch.json (+ residual statistics)
        python pipeline/stitch.py preview walk1 60              -> data/interim/<walk>/stitch_preview_60.jpg (v360 vs new)
"""
from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HALF_W, RAW_H = 1984, 1920
BLEND_DEG = 6.0
deg = np.pi / 180


# ---------------------------------------------------------------- model
def default_params() -> dict:
    return {"f": [611.0, 611.0], "c": [[990.0, 972.0], [990.0, 972.0]], "rot1": [0.0, 0.0, 0.0],
            "G": np.eye(3).tolist(), "r_max": 1060.0, "blend_deg": BLEND_DEG}


def r1_matrix(p: dict) -> np.ndarray:
    return Rot.from_euler("y", 180, degrees=True).as_matrix() @ Rot.from_rotvec(p["rot1"]).as_matrix()


def lens_to_dir(k: int, u, v, p: dict) -> np.ndarray:
    """raw pixel (u, v) in lens k's half -> unit direction in the camera frame."""
    cx, cy = p["c"][k]; f = p["f"][k]
    du, dv = np.asarray(u, float) - cx, np.asarray(v, float) - cy
    r = np.hypot(du, dv); th = r / f; ph = np.arctan2(dv, du)
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    return d if k == 0 else d @ r1_matrix(p).T


def dir_to_lens(k: int, d: np.ndarray, p: dict):
    """camera-frame directions -> raw pixel (u, v) in lens k's half, and the angle from that lens' axis (rad)."""
    v = d if k == 0 else d @ r1_matrix(p)              # R1^T d  (row vectors)
    th = np.arccos(np.clip(v[..., 2], -1, 1)); ph = np.arctan2(v[..., 1], v[..., 0])
    cx, cy = p["c"][k]; f = p["f"][k]
    return cx + f * th * np.cos(ph), cy + f * th * np.sin(ph), th


# ---------------------------------------------------------------- raw frames
def raw_frames(src, times: list[float]) -> list[np.ndarray]:
    out = []
    for t in times:
        b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
        out.append(np.frombuffer(b, np.uint8).reshape(RAW_H, 2 * HALF_W, 3))
    return out


def duration(src) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    return float(r.stdout.strip())


# ---------------------------------------------------------------- calibration
RING_DEG, RING_RES = 14.0, 0.1                        # ring = +-14 deg around the 90 deg line, 0.1 deg / px


def ring_dirs():
    a = np.arange(0, 360, RING_RES) * deg; b = np.arange(RING_DEG, -RING_DEG, -RING_RES) * deg
    A, B = np.meshgrid(a, b)
    return np.stack([np.cos(B) * np.cos(A), np.cos(B) * np.sin(A), np.sin(B)], -1)


def render(img_half: np.ndarray, k: int, d: np.ndarray, p: dict) -> np.ndarray:
    u, v, _ = dir_to_lens(k, d, p)
    return cv2.remap(img_half, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def ring_matches(frame: np.ndarray, p: dict, sift, clahe) -> tuple[np.ndarray, np.ndarray]:
    """pairs of raw pixels (lens 0, lens 1) that see the same point, found in the overlap ring."""
    d = ring_dirs()
    g = [clahe.apply(cv2.cvtColor(render(frame[:, k * HALF_W:(k + 1) * HALF_W], k, d, p), cv2.COLOR_BGR2GRAY)) for k in (0, 1)]
    k0, d0 = sift.detectAndCompute(g[0], None); k1, d1 = sift.detectAndCompute(g[1], None)
    if d0 is None or d1 is None or len(k0) < 10 or len(k1) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2))
    m = cv2.BFMatcher().knnMatch(d0, d1, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    P0, P1 = [], []
    W = g[0].shape[1]
    for mm in good:
        x0, y0 = k0[mm.queryIdx].pt; x1, y1 = k1[mm.trainIdx].pt
        dx = (x1 - x0 + W / 2) % W - W / 2
        if abs(dx) * RING_RES > 3 or abs(y1 - y0) * RING_RES > 3:        # same point within 3 deg
            continue
        P0.append((x0, y0)); P1.append((x1, y1))
    if not P0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    def to_dir(P):
        P = np.asarray(P); a = P[:, 0] * RING_RES * deg; b = (RING_DEG - P[:, 1] * RING_RES) * deg
        return np.stack([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)], -1)
    u0, v0, _ = dir_to_lens(0, to_dir(P0), p); u1, v1, _ = dir_to_lens(1, to_dir(P1), p)
    return np.c_[u0, v0], np.c_[u1, v1]


def fit(R0: np.ndarray, R1: np.ndarray, p: dict) -> tuple[dict, dict]:
    """Physical constraints (the overlap ring alone cannot separate them): one focal scale for both (identical lenses),
    centres held near the measured image-circle centres (prior, ~5 px), free rotation of lens 1."""
    cm = np.asarray(p["c_meas"])
    def unpack(x):
        q = dict(p); q["f"] = [x[0], x[0]]; q["c"] = [[cm[0, 0] + x[1], cm[0, 1] + x[2]], [cm[1, 0] + x[3], cm[1, 1] + x[4]]]; q["rot1"] = list(x[5:8]); return q
    def res(x):
        q = unpack(x); a = lens_to_dir(0, R0[:, 0], R0[:, 1], q); b = lens_to_dir(1, R1[:, 0], R1[:, 1], q)
        return np.r_[(a - b).ravel(), x[1:5] * 0.0005 * np.sqrt(len(R0) / 50)]
    x0 = np.r_[p["f"][0], np.asarray(p["c"]).ravel() - cm.ravel(), p["rot1"]]
    r = least_squares(res, x0, loss="soft_l1", f_scale=0.004)
    q = unpack(r.x)
    e = np.linalg.norm((lens_to_dir(0, R0[:, 0], R0[:, 1], q) - lens_to_dir(1, R1[:, 0], R1[:, 1], q)), axis=1) / deg
    e0 = np.linalg.norm((lens_to_dir(0, R0[:, 0], R0[:, 1], p) - lens_to_dir(1, R1[:, 0], R1[:, 1], p)), axis=1) / deg
    return q, {"pairs": len(e), "median_deg_before": round(float(np.median(e0)), 3), "median_deg_after": round(float(np.median(e)), 3),
               "p90_deg_after": round(float(np.percentile(e, 90)), 3)}


# ---------------------------------------------------------------- output orientation (match the old v360 frames)
def equi_dirs(W: int, H: int) -> np.ndarray:
    """output pixel grid -> direction in the output frame: yaw 0 = centre (spherical.py conventions), but as a
    right-handed frame like the camera's (x right, y DOWN, z forward), so output and camera differ by a rotation only."""
    yaw = ((np.arange(W) + .5) / W * 360 - 180) * deg; pit = (90 - (np.arange(H) + .5) / H * 180) * deg
    Y, P = np.meshgrid(yaw, pit)
    return np.stack([np.cos(P) * np.sin(Y), -np.sin(P), np.cos(P) * np.cos(Y)], -1)


def fit_orientation(frames, v360_frames, p: dict) -> tuple[np.ndarray, dict]:
    """rotation G (output = G @ camera) from features matched between the new stitch (G = I) and the v360 frame."""
    sift = cv2.SIFT_create(4000); A, B = [], []
    W, H = 1920, 960
    D = equi_dirs(W, H)
    for fr, vf in zip(frames, v360_frames):
        q = dict(p); q["G"] = np.eye(3).tolist()
        mine = stitch_once(fr, q, W, H)
        ref = cv2.resize(vf, (W, H), interpolation=cv2.INTER_AREA)
        k0, d0 = sift.detectAndCompute(cv2.cvtColor(mine, cv2.COLOR_BGR2GRAY), None)
        k1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
        for a, b in (x for x in cv2.BFMatcher().knnMatch(d0, d1, k=2) if len(x) == 2):
            if a.distance < 0.7 * b.distance:
                x0, y0 = k0[a.queryIdx].pt; x1, y1 = k1[a.trainIdx].pt
                c = D[int(y0), int(x0)]                                        # camera dir (G = I)
                if abs(c[2]) < np.cos(55 * deg):                                # only near a lens centre (v360 reliable there)
                    continue
                A.append(c); B.append(D[int(y1), int(x1)])
    A, B = np.asarray(A), np.asarray(B)
    best, inl = None, None
    rng = np.random.default_rng(0)
    for _ in range(400):                                                   # RANSAC Kabsch
        i = rng.choice(len(A), 3, replace=False)
        R, _ = Rot.align_vectors(B[i], A[i]); M = R.as_matrix()
        ok = np.linalg.norm(A @ M.T - B, axis=1) < 1.5 * deg
        if inl is None or ok.sum() > inl.sum():
            best, inl = M, ok
    R, _ = Rot.align_vectors(B[inl], A[inl]); M = R.as_matrix()
    e = np.linalg.norm(A[inl] @ M.T - B[inl], axis=1) / deg
    return M, {"matches": int(len(A)), "inliers": int(inl.sum()), "median_deg": round(float(np.median(e)), 3)}


# ---------------------------------------------------------------- stitch
class Stitcher:
    """precomputed remap tables + blend weights for one output size; call with a raw dual-fisheye BGR frame."""

    def __init__(self, p: dict, W: int, H: int):
        self.p, self.W, self.H = p, W, H
        d = equi_dirs(W, H) @ np.asarray(p["G"])                          # output dir -> camera dir (G^T d, row vectors)
        self.maps, th, raw_t = [], [], []
        for k in (0, 1):
            u, v, t = dir_to_lens(k, d, p)
            inside = (u >= 0) & (u < HALF_W - 1) & (v >= 0) & (v < RAW_H - 1) & (np.hypot(u - p["c"][k][0], v - p["c"][k][1]) < p["r_max"])
            th.append(np.where(inside, t, np.inf)); raw_t.append(t)
            # clamp into the lens' own half: directions no lens covers (sensor crop at zenith / nadir) repeat the
            # nearest edge pixels like v360 did, instead of black, and the two lenses never bleed into each other
            uc, vc = np.clip(u, 0, HALF_W - 1) + k * HALF_W, np.clip(v, 0, RAW_H - 1)
            self.maps.append(cv2.convertMaps(uc.astype(np.float32), vc.astype(np.float32), cv2.CV_16SC2))
        b = p.get("blend_deg", BLEND_DEG) * deg
        w0 = np.clip((np.pi / 2 + b - th[0]) / (2 * b), 0, 1)              # lens 0 fades out across 90 +- b
        w0 = np.where(np.isinf(th[1]), 1.0, np.where(np.isinf(th[0]), 0.0, w0))
        none = np.isinf(th[0]) & np.isinf(th[1])
        w0 = np.where(none, (raw_t[0] <= raw_t[1]).astype(float), w0)        # uncovered: the lens whose axis is closer
        self.uncovered = float(none.mean())
        self.w0 = cv2.merge([w0.astype(np.float32)] * 3); self.w1 = 1.0 - self.w0
        self.ring = (np.abs(th[0] - np.pi / 2) < b) & np.isfinite(th[0]) & np.isfinite(th[1])   # overlap for the gain
        self.gain = None

    def __call__(self, raw: np.ndarray, smooth: float = 0.15) -> np.ndarray:
        a = cv2.remap(raw, *self.maps[0], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        b = cv2.remap(raw, *self.maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
        m0, m1 = a[self.ring].mean(0) + 1, b[self.ring].mean(0) + 1            # per-channel exposure ratio in the overlap
        g = np.sqrt(m1 / m0)
        self.gain = g if self.gain is None else (1 - smooth) * self.gain + smooth * g
        out = a * (self.w0 * self.gain) + b * (self.w1 / self.gain)
        return np.clip(out, 0, 255).astype(np.uint8)


def stream(src, p: dict, W: int, H: int, fps: float | None = None):
    """yield stitched frames of a whole .insv (decoded by ffmpeg into a pipe; fps=None keeps the source rate)."""
    st = Stitcher(p, W, H)
    cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an", "-sn"] + (["-vf", f"fps={fps}"] if fps else []) +           ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    n = RAW_H * 2 * HALF_W * 3
    try:
        while True:
            buf = pr.stdout.read(n)
            if len(buf) < n:
                break
            yield st(np.frombuffer(buf, np.uint8).reshape(RAW_H, 2 * HALF_W, 3))
    finally:
        pr.stdout.close(); pr.wait()


def stitch_once(raw: np.ndarray, p: dict, W: int, H: int) -> np.ndarray:
    return Stitcher(p, W, H)(raw)


def load(config_dir) -> dict | None:
    f = config_dir / "stitch.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


# ---------------------------------------------------------------- CLI
def circle_centres(frames) -> list[list[float]]:
    """image-circle centre of each lens (least-squares circle through the dark-border edge, median over frames)."""
    C = [[], []]
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for k in (0, 1):
            h = g[:, k * HALF_W:(k + 1) * HALF_W]; pts = []
            for y in range(300, RAW_H - 300, 20):
                br = np.nonzero(h[y] > 25)[0]
                if len(br) and br[0] > 2: pts.append((br[0], y))
                if len(br) and br[-1] < HALF_W - 3: pts.append((br[-1], y))
            if len(pts) < 20: continue
            P = np.asarray(pts, float); A = np.c_[2 * P, np.ones(len(P))]
            cx, cy, _ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]; C[k].append((cx, cy))
    return [np.median(np.asarray(c), 0).tolist() for c in C]


def calibrate(walks: list[str]) -> dict:
    from common import CONFIG, WALKS
    p = default_params()
    sift, clahe = cv2.SIFT_create(6000), cv2.createCLAHE(2.0, (8, 8))
    frames, v360 = [], []
    for w in walks:
        src = WALKS[w]; T = duration(src)
        times = list(np.linspace(8, T - 8, 12))
        frames += raw_frames(src, times)
        for t in times[::3]:
            b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-vf",
                                "v360=input=dfisheye:output=e:ih_fov=190:iv_fov=190:roll=180:w=1920:h=960", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               capture_output=True, check=True).stdout
            v360.append(np.frombuffer(b, np.uint8).reshape(960, 1920, 3))
    p["c_meas"] = circle_centres(frames); p["c"] = [list(c) for c in p["c_meas"]]
    print(json.dumps({"circle_centres": p["c_meas"]}), flush=True)
    stats = []
    for it in range(4):                                         # re-render the ring with the improved model, rematch, refit
        R0, R1 = [], []
        for fr in frames:
            a, b = ring_matches(fr, p, sift, clahe); R0.append(a); R1.append(b)
        R0, R1 = np.concatenate(R0), np.concatenate(R1)
        p, s = fit(R0, R1, p); s["iteration"] = it + 1; stats.append(s); print(json.dumps(s), flush=True)
    G, so = fit_orientation(frames[::3], v360, p)
    p["G"] = G.tolist(); print(json.dumps({"orientation": so}), flush=True)
    p["calibration"] = {"walks": walks, "frames": len(frames), "ring": stats[-1], "orientation_vs_v360": so, "history": stats}
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "stitch.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    return p


def preview(walk: str, t: float) -> None:
    from common import CONFIG, INTERIM, WALKS
    p = load(CONFIG); src = WALKS[walk]
    raw = raw_frames(src, [t])[0]
    new = stitch_once(raw, p, 1920, 960)
    b = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-vf",
                        "v360=input=dfisheye:output=e:ih_fov=190:iv_fov=190:roll=180:w=1920:h=960", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True, check=True).stdout
    old = np.frombuffer(b, np.uint8).reshape(960, 1920, 3)
    out = INTERIM / walk / f"stitch_preview_{int(t)}.jpg"; out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([old, new]), [cv2.IMWRITE_JPEG_QUALITY, 90]); print(out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "calibrate":
        calibrate(sys.argv[2:] or ["walk1", "walk2"])
    elif cmd == "preview":
        preview(sys.argv[2], float(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
