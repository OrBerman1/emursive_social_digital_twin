"""Stage `video`: stitched equirectangular MP4 for the viewer (cached).

Uses the same stitch as extract_frames.py (dual fisheye, 190 deg lenses, roll 180) at the full frame rate,
so overlay timestamps (frame index / 5 fps) line up with video time. Encodes on the GPU (NVENC) if available.
Projects with config/stitch.json: the calibrated blended stitch (pipeline/stitch.py) is piped into the same encoder.

usage:  python pipeline/encode_video.py [walk1 walk2] [--width 2880] [--force]
"""
from __future__ import annotations

import argparse
import subprocess

from common import CONFIG, PROCESSED, WALKS, Stage, file_fingerprint

VERSION = 1


def has_nvenc() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "h264_nvenc" in out


def run(walk: str, width: int, lens_fov: float, force: bool) -> None:
    src = WALKS[walk]
    out_dir = PROCESSED / walk
    nvenc = has_nvenc()
    import stitch
    cal = stitch.load(CONFIG)
    params = {"width": width, "height": width // 2, "lens_fov": lens_fov, "roll": 180, "codec": "h264_nvenc" if nvenc else "libx264"}
    if cal:
        params = {"width": width, "height": width // 2, "stitch": "calibrated", "stitch_params": {k: cal[k] for k in ("f", "c", "rot1", "G", "r_max", "blend_deg")},
                  "codec": params["codec"]}
    st = Stage("video", VERSION, out_dir / "video", params, {"video": src.name, "fingerprint": file_fingerprint(src)})
    if st.done() and not force:
        print(f"[video] {walk}: cached")
        return
    st.start()
    out = out_dir / "video" / "equirect.mp4"
    vf = f"v360=input=dfisheye:output=e:ih_fov={lens_fov}:iv_fov={lens_fov}:roll=180:w={width}:h={width // 2}:interp=cubic"
    enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "24", "-b:v", "12M", "-maxrate", "20M"]
           if nvenc else ["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
    if cal:                                                              # stitched in Python, piped into the encoder
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "csv=p=0", str(src)],
                           capture_output=True, text=True).stdout.strip().split(",")
        rate, n_est = r[0], int(r[1]) if len(r) > 1 and r[1].isdigit() else 0
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{width // 2}", "-r", rate, "-i", "-",
               "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", *enc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
               "-shortest", "-movflags", "+faststart", str(out)]
        print(f"[video] {walk}: calibrated stitch, encoding -> {out}", flush=True)
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i, im in enumerate(stitch.stream(src, cal, width, width // 2)):
            pr.stdin.write(im.tobytes())
            if i % 150 == 0:
                print(f"[video] {walk} {i}/{n_est}", flush=True)
        pr.stdin.close()
        if pr.wait() != 0:
            raise SystemExit("encoder failed")
    else:
        cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf,
               *enc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)]
        print(f"[video] {walk}: encoding -> {out}")
        subprocess.run(cmd, check=True)
    st.finish({"file": out.name, "bytes": out.stat().st_size})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--width", type=int, default=2880, help="2880x1440 plays smoothly in browsers")
    ap.add_argument("--lens-fov", type=float, default=190.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.walks:
        run(w, a.width, a.lens_fov, a.force)


if __name__ == "__main__":
    main()
