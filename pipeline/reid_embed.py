"""Stage `reid_embed`: appearance embedding per person track (person re-identification), cached, GPU.

Why: link_persons.py proposes links between broken person tracks from geometry and time only; an appearance score
lets the links be decided automatically (see link_persons.py --reid).
Model : OSNet-AIN x1.0 trained on MSMT17 (torchreid model zoo, models/osnet_ain_x1_0_msmt17.pth). OSNet is a small
        person-ReID network (2.2 M parameters); the AIN variant (instance normalisation) is the torchreid model built
        for cross-domain use, i.e. for footage it was not trained on. The network is defined here (minimal copy of
        torchreid's osnet_ain.py, no torchreid install); weights load strictly. Output = 512-d feature after the fc
        layer, L2-normalised.
Crops : per track, non-interpolated boxes with conf >= 0.4, >= MIN_ANG_H deg tall, feet above -60 deg pitch, top below
        +60 deg (not cut off); score = height x conf x (1 - overlap with other person boxes in that frame).
        Best box in each of N_SPREAD equal time bins of the track + up to N_END best boxes in the first / last 2 s
        (track ends are where links happen and where tracker ID switches happen, so ends get their own crops).
        Undistorted perspective crop 256 x 128 (H x W) around the box (height x 1.1), everything outside the person's
        own box greyed out (like attributes.py). Crops live only in memory.
Speed : the cost is decoding 4K JPEGs: crops are grouped by frame (each frame decoded once), frames are decoded and
        cropped in a thread pool (cv2 releases the GIL), and the GPU (fp16, batches of BATCH) runs in the main thread
        while the pool keeps decoding (bounded producer / consumer).
Output: data/interim/<walk>/reid<_stab>/crops.parquet (track, frame, role spread/start/end, ang_h, conf, overlap, e0..e511),
        tracks.parquet (track, n_crops, centred mean embedding, start / end means; cheap, always rebuilt),
        manifest.json

usage:  python pipeline/reid_embed.py [walk1 walk2] [--variant stab] [--force] [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import INTERIM, ROOT, WALKS, Stage
from spherical import Tile, tile_to_sphere

VERSION = 1
WEIGHTS = ROOT / "models" / "osnet_ain_x1_0_msmt17.pth"
H, W = 256, 128
N_SPREAD, N_END, END_F = 8, 3, 10   # spread crops; crops in the first / last END_F frames (2 s at 5 fps)
MIN_ANG_H, MIN_CONF = 4.0, 0.4
BATCH = 128
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


# ---------------- OSNet-AIN (minimal copy of torchreid/models/osnet_ain.py, inference only) ----------------
class ConvLayer(nn.Module):
    def __init__(self, i, o, k, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(i, o, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(o, affine=True) if IN else nn.BatchNorm2d(o)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, i, o, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, i, o, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(o) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv1 = nn.Conv2d(i, o, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(o, o, 3, stride=1, padding=1, bias=False, groups=o)
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, i, o, depth):
        super().__init__()
        self.layers = nn.Sequential(LightConv3x3(i, o), *[LightConv3x3(o, o) for _ in range(depth - 1)])

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    def __init__(self, i, o, reduction=4, T=4, IN=False):
        super().__init__()
        mid = o // reduction
        self.conv1 = Conv1x1(i, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, o, bn=not IN)
        self.downsample = Conv1x1Linear(i, o) if i != o else None
        self.IN = nn.InstanceNorm2d(o, affine=True) if IN else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = 0
        for s in self.conv2:
            x2 = x2 + self.gate(s(x1))
        x3 = self.conv3(x2)
        if self.IN is not None:
            x3 = self.IN(x3)
        idt = self.downsample(x) if self.downsample is not None else x
        return F.relu(x3 + idt)


class OSNetAIN(nn.Module):
    """osnet_ain_x1_0: channels 64-256-384-512, blocks [[IN, IN], [-, IN], [IN, -]], conv1 with instance norm."""

    def __init__(self, num_classes=4101, channels=(64, 256, 384, 512), feature_dim=512):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(OSBlock(c[0], c[1], IN=True), OSBlock(c[1], c[1], IN=True))
        self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(OSBlock(c[1], c[2]), OSBlock(c[2], c[2], IN=True))
        self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(OSBlock(c[2], c[3], IN=True), OSBlock(c[3], c[3]))
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):                                                   # eval: the 512-d feature (torchreid "v")
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        return self.fc(self.global_avgpool(x).flatten(1))


def load_model(dev: str) -> nn.Module:
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)        # weights only: no code from the file runs
    sd = sd.get("state_dict", sd)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    m = OSNetAIN(num_classes=sd["classifier.weight"].shape[0])
    m.load_state_dict(sd, strict=True)
    return m.eval().to(dev)


# ---------------- crops ----------------
def reid_crop(equi: np.ndarray, r) -> np.ndarray:
    """Undistorted 256 x 128 crop around a spherical person box (height x 1.1), outside the box greyed out."""
    pm = (r.pitch_top + r.pitch_bot) / 2
    span = float(np.clip((r.pitch_top - r.pitch_bot) * 1.1, 3, 110))    # vertical field of view of the crop
    t = Tile(0, float(r.yaw), float(pm), span, H)                       # square H x H tile; keep the central W columns
    uu, vv = np.meshgrid(np.arange((H - W) // 2, (H + W) // 2) + .5, np.arange(H) + .5)
    yaw, pitch = tile_to_sphere(t, uu, vv)
    h, w = equi.shape[:2]
    mx = (((yaw + 180) / 360 * w - .5) % w).astype(np.float32)
    my = np.clip((90 - pitch) / 180 * h - .5, 0, h - 1).astype(np.float32)
    out = cv2.remap(equi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    dy = ((yaw - r.yaw + 180) % 360) - 180
    half_w = (((r.yaw_r - r.yaw_l) % 360) / 2) * 1.08
    inside = (np.abs(dy) <= half_w) & (pitch <= r.pitch_top + 0.5) & (pitch >= r.pitch_bot - 0.5)
    out[~inside] = 127
    return out


def overlap_share(fr: pd.DataFrame) -> np.ndarray:
    """Per box: largest share of its area covered by another person box in the same frame (yaw wrap-safe)."""
    yl, yr = fr.yaw_l.to_numpy(), fr.yaw_r.to_numpy()
    pt, pb = fr.pitch_top.to_numpy(), fr.pitch_bot.to_numpy()
    wdt = (yr - yl) % 360
    out = np.zeros(len(fr))
    for i in range(len(fr)):
        best = 0.0
        for j in range(len(fr)):
            if i == j:
                continue
            d = ((yl[j] - yl[i]) + 180) % 360 - 180                           # left edge of j relative to i
            ox = max(0.0, min(wdt[i], d + wdt[j]) - max(0.0, d))
            oy = max(0.0, min(pt[i], pt[j]) - max(pb[i], pb[j]))
            best = max(best, ox * oy / max(wdt[i] * (pt[i] - pb[i]), 1e-6))
        out[i] = best
    return out


def pick_crops(tr: pd.DataFrame) -> pd.DataFrame:
    """Rows to embed: best box per time bin (spread) + best boxes at the start / end of each track."""
    real = tr[~tr.interp].copy()
    real["overlap"] = 0.0
    for f, g in real.groupby("frame"):                                      # explicit loop
        if len(g) > 1:
            real.loc[g.index, "overlap"] = overlap_share(g)
    ok = real[(real.conf >= MIN_CONF) & (real.ang_h >= MIN_ANG_H) & (real.pitch_bot > -60) & (real.pitch_top < 60)].copy()
    ok["score"] = ok.ang_h * ok.conf * (1 - ok.overlap.clip(0, 1))
    out = []
    span = tr.groupby("track").frame.agg(["min", "max"])
    for tid, g in ok.groupby("track"):
        f0, f1 = span.at[tid, "min"], span.at[tid, "max"]
        bins = np.minimum(((g.frame - f0) / max(f1 - f0 + 1, 1) * N_SPREAD).astype(int), N_SPREAD - 1)
        sp = g.loc[g.groupby(bins).score.idxmax()].assign(role="spread")
        st = g[g.frame < f0 + END_F].nlargest(N_END, "score").assign(role="start")
        en = g[g.frame > f1 - END_F].nlargest(N_END, "score").assign(role="end")
        out += [sp, st, en]
    return pd.concat(out, ignore_index=True) if out else ok.head(0).assign(role="")


def track_embeddings(out_dir) -> pd.DataFrame:
    """Cheap step (always rerun): per-track embeddings from crops.parquet -> tracks.parquet.
    Crop embeddings are centred on the walk's mean crop embedding, then L2-normalised: OSNet features are
    non-negative (after ReLU), so raw cosines of unrelated people are already ~0.7; centring removes the shared
    'any person in this market footage' direction (validation: AUC same-track halves vs co-occurring people within
    15 deg 0.67 -> 0.81, METHODS.txt section 32). mean = all crops of the track, start / end = crops of the first /
    last 2 s (None if the track has none there)."""
    crops = pd.read_parquet(out_dir / "crops.parquet")
    ecols = [c for c in crops.columns if c[0] == "e" and c[1:].isdigit()]
    X = crops[ecols].to_numpy(np.float64)
    X = X - X.mean(0)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    rows = []
    for tid, g in crops.groupby("track"):
        r = {"track": int(tid), "n_crops": int(g.frame.nunique())}
        for name, sel in (("mean", g.drop_duplicates("frame")), ("start", g[g.role == "start"]), ("end", g[g.role == "end"])):
            if len(sel):
                m = X[sel.index].mean(0)
                r[name] = (m / max(np.linalg.norm(m), 1e-9)).astype(np.float32).tolist()
            else:
                r[name] = None
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "tracks.parquet", index=False)
    return df


def run(walk: str, variant: str, force: bool, workers: int) -> dict:
    sfx = f"_{variant}" if variant else ""
    tp = INTERIM / walk / f"tracks{sfx}" / "persons.parquet"
    out_dir = INTERIM / walk / f"reid{sfx}"
    params = {"weights": WEIGHTS.name, "h": H, "w": W, "n_spread": N_SPREAD, "n_end": N_END, "end_f": END_F,
              "min_ang_h": MIN_ANG_H, "min_conf": MIN_CONF, "isolate": True}
    st = Stage("reid_embed", VERSION, out_dir, params, {"tracks": hashlib.sha256(tp.read_bytes()).hexdigest()[:16]})
    if st.done() and not force:
        print(f"[reid_embed] {walk}{sfx}: cached")
        track_embeddings(out_dir)                                           # cheap, always recomputed
        return json.loads(st.manifest_path.read_text(encoding="utf-8"))
    st.start()
    t0 = time.time()
    tr = pd.read_parquet(tp)
    picks = pick_crops(tr)
    t_pick = time.time() - t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(dev)
    half = dev == "cuda"
    if half:
        model = model.half()
    dt = torch.float16 if half else torch.float32
    mean_t = torch.tensor(MEAN, dtype=dt, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, dtype=dt, device=dev).view(1, 3, 1, 1)
    frames_dir = INTERIM / walk / f"frames{sfx}"
    groups = [(int(f), g) for f, g in picks.groupby("frame")]
    t_dec = [0.0]
    lock = threading.Lock()

    def work(item):                                                         # thread: decode once, cut all crops of the frame
        f, g = item
        a = time.perf_counter()
        equi = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        cs = [cv2.cvtColor(reid_crop(equi, r), cv2.COLOR_BGR2RGB) for r in g.itertuples()]
        with lock:
            t_dec[0] += time.perf_counter() - a
        return g.index.to_numpy(), cs

    q: queue.Queue = queue.Queue(maxsize=workers * 4)                       # bounded: decoding runs ahead of the GPU, not unboundedly

    def producer():
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for it in groups:
                futs.append(ex.submit(work, it))
                if len(futs) >= workers * 2:                                # keep a window of frames in flight
                    q.put(futs.pop(0).result())
            for fu in futs:
                q.put(fu.result())
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    t1 = time.time()
    th.start()
    idx, embs, buf_i, buf_c = [], [], [], []
    t_gpu, t_wait = 0.0, 0.0

    def flush():
        nonlocal t_gpu
        if not buf_c:
            return
        a = time.perf_counter()
        xt = torch.from_numpy(np.stack(buf_c)).to(dev, non_blocking=True)    # uint8 upload; normalise on the GPU
        xt = xt.permute(0, 3, 1, 2).to(torch.float16 if half else torch.float32).div_(255)
        xt = (xt - mean_t) / std_t
        with torch.no_grad():
            e = model(xt).float()
            e = F.normalize(e, dim=1).cpu().numpy()
        embs.append(e); idx.extend(buf_i)
        buf_i.clear(); buf_c.clear()
        t_gpu += time.perf_counter() - a

    while True:
        a = time.perf_counter()
        item = q.get()
        t_wait += time.perf_counter() - a
        if item is None:
            break
        ii, cs = item
        buf_i.extend(ii.tolist()); buf_c.extend(cs)
        if len(buf_c) >= BATCH:
            flush()
    flush()
    th.join()
    t_loop = time.time() - t1
    E = np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)
    meta = picks.loc[idx, ["track", "frame", "role", "ang_h", "conf", "overlap"]].reset_index(drop=True)
    crops = pd.concat([meta, pd.DataFrame(E.astype(np.float32), columns=[f"e{i}" for i in range(E.shape[1])])], axis=1)
    crops.to_parquet(st.out_dir / "crops.parquet", index=False)                # embeddings only, no images
    rows = track_embeddings(st.out_dir)
    timing = {"pick_s": round(t_pick, 1), "decode_crop_thread_s": round(t_dec[0], 1), "gpu_s": round(t_gpu, 1),
              "gpu_wait_for_decode_s": round(t_wait, 1), "wall_decode_embed_s": round(t_loop, 1), "workers": workers,
              "frames": len(groups), "crops": len(crops), "tracks_with_crops": len(rows), "tracks": int(tr.track.nunique())}
    st.finish(timing)
    print(f"[reid_embed] {walk}{sfx}: {json.dumps(timing)}")
    return timing


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab", choices=["", "stab"])
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 2))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.walks:
        run(w, a.variant, a.force, a.workers)


if __name__ == "__main__":
    main()
