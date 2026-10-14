"""Stage `bake`: write the stabilisation INTO the pixels (offline), so the viewer just plays a steady video.

  video : data/processed/<walk>/video/equirect.mp4 (30 fps)  -> data/processed/<walk>/video_stab/equirect.mp4
  frames: data/interim/<walk>/frames/*.jpg (5 fps, 3840 px)  -> data/interim/<walk>/frames_stab/*.jpg
Each output pixel's direction d_stab is rotated into the camera frame, d_cam = Q(t) d_stab (Q from stab.json),
and the stitched panorama is resampled there (GPU, torch grid_sample). Detection then runs on frames_stab.

usage:  python pipeline/bake_stab.py walk1 [--what video frames] [--seconds N (test)] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import INTERIM, PROCESSED, WALKS, Stage

VERSION = 1
PAD = 16  # columns of horizontal wrap padding


DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if DEV == "cuda":     # NVIDIA hardware encoder (original Windows setup)
    ENC_ARGS = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "24", "-b:v", "12M", "-maxrate", "20M"]
elif sys.platform == "darwin":   # Apple hardware encoder
    ENC_ARGS = ["-c:v", "h264_videotoolbox", "-b:v", "12M", "-maxrate", "20M"]
else:
    ENC_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]


def directions(w: int, h: int, dev) -> torch.Tensor:
    yaw = (torch.arange(w, device=dev) + 0.5) / w * 2 * np.pi - np.pi
    pit = np.pi / 2 - (torch.arange(h, device=dev) + 0.5) / h * np.pi
    p, y = torch.meshgrid(pit, yaw, indexing="ij")
    return torch.stack([torch.cos(p) * torch.sin(y), torch.sin(p), torch.cos(p) * torch.cos(y)], -1)  # (h,w,3)


def remap(img: torch.Tensor, q: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """img (3,H,W) float on GPU; q (3,3) with d_cam = q d_stab. Returns stabilised (3,H,W)."""
    _, h, w = img.shape
    d = dirs @ q.T                                                  # (h,w,3) camera-frame directions
    lon = torch.atan2(d[..., 0], d[..., 2])
    lat = torch.asin(d[..., 1].clamp(-1, 1))
    x = (lon + np.pi) / (2 * np.pi) * w - 0.5 + PAD                 # pixel coords in the padded image
    y = (np.pi / 2 - lat) / np.pi * h - 0.5
    wp = w + 2 * PAD
    grid = torch.stack([(2 * x + 1) / wp - 1, (2 * y + 1) / h - 1], -1)[None]
    padded = torch.cat([img[:, :, -PAD:], img, img[:, :, :PAD]], 2)[None]
    return F.grid_sample(padded, grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


def stab_key(walk: str) -> str:
    return hashlib.sha256((PROCESSED / walk / "stab.json").read_bytes()).hexdigest()[:16]


def bake_video(walk: str, seconds: float | None, force: bool) -> None:
    src = PROCESSED / walk / "video" / "equirect.mp4"
    out_dir = PROCESSED / walk / "video_stab"
    st_json = json.loads((PROCESSED / walk / "stab.json").read_text(encoding="utf-8"))
    vm = json.loads((PROCESSED / walk / "video" / "manifest.json").read_text(encoding="utf-8"))
    st = Stage("video_stab", VERSION, out_dir, {"seconds": seconds}, {"video_key": vm["key"], "stab": stab_key(walk)})
    if st.done() and not force:
        print(f"[bake] {walk} video: cached"); return
    st.start()
    w, h = vm["params"]["width"], vm["params"]["height"]
    q = torch.tensor(np.array(st_json["q_video"]).reshape(-1, 3, 3), dtype=torch.float32, device=DEV)
    dirs = directions(w, h, DEV)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), *(["-t", str(seconds)] if seconds else []),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out = out_dir / "equirect.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", "30000/1001", "-i", "-", "-i", str(src), "-map", "0:v", "-map", "1:a?",
                            *ENC_ARGS,
                            "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    n, t0, nbytes = 0, time.time(), w * h * 3
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        img = torch.frombuffer(bytearray(buf), dtype=torch.uint8).to(DEV).view(h, w, 3).permute(2, 0, 1).float()
        res = remap(img, q[min(n, len(q) - 1)], dirs)
        enc.stdin.write(res.round().clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
        n += 1
        if n % 300 == 0:
            print(f"\r[bake] {walk} video: {n} frames  {n / (time.time() - t0):.0f} fps", end="", flush=True)
    enc.stdin.close(); enc.wait(); dec.wait()
    print()
    st.finish({"frames": n, "file": out.name})
    print(f"[bake] {walk} video: {n} frames -> {out}")


def bake_frames(walk: str, force: bool) -> None:
    src_dir, out_dir = INTERIM / walk / "frames", INTERIM / walk / "frames_stab"
    fm = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
    st_json = json.loads((PROCESSED / walk / "stab.json").read_text(encoding="utf-8"))
    st = Stage("frames", VERSION, out_dir, {**fm["params"], "stabilised": True}, {"frames_key": fm["key"], "stab": stab_key(walk)})
    if st.done() and not force:
        print(f"[bake] {walk} frames: cached"); return
    st.start()
    q = torch.tensor(np.array(st_json["q_analysis"]).reshape(-1, 3, 3), dtype=torch.float32, device=DEV)
    index = json.loads((src_dir / "index.json").read_text(encoding="utf-8"))
    dirs, t0 = None, time.time()
    for fr in index:
        bgr = cv2.imread(str(src_dir / fr["file"]))
        img = torch.from_numpy(bgr).to(DEV).permute(2, 0, 1).float()
        if dirs is None:
            dirs = directions(img.shape[2], img.shape[1], DEV)
        res = remap(img, q[fr["frame"]], dirs).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(str(out_dir / fr["file"]), res, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if fr["frame"] % 100 == 0:
            print(f"\r[bake] {walk} frames: {fr['frame']}/{len(index)}  {fr['frame'] / max(time.time() - t0, 1e-6):.1f} fps", end="", flush=True)
    print()
    shutil.copy(src_dir / "index.json", out_dir / "index.json")
    # same manifest fields as frames/ so detect.py and tracks.py can treat this folder identically
    st.finish({"n_frames": fm["n_frames"], "video_duration_s": fm["video_duration_s"]})
    print(f"[bake] {walk} frames: {len(index)} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--what", nargs="+", default=["video", "frames"], choices=["video", "frames"])
    ap.add_argument("--seconds", type=float, default=None, help="video only: bake just the first N seconds (test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.walks:
        if "video" in a.what:
            bake_video(w, a.seconds, a.force)
        if "frames" in a.what:
            bake_frames(w, a.force)


if __name__ == "__main__":
    main()
