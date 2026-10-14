"""Stage `frames`: raw Insta360 dual-fisheye .insv -> upright equirectangular JPEG frames (cached).

Stitch settings were tuned by eye on walk 1 (12 Oct 2026): the Insta360 ONE RS stores both lenses side by
side (3968x1920, each ~1984 px wide); ffmpeg's dual-fisheye model with a 190 deg lens FOV lines up the seams,
and roll=180 turns the upside-down handheld footage upright. Hebrew signage reads correctly (not mirrored).
This is a fast approximate stitch; Insta360 Studio / MediaSDK give cleaner seams if needed later.
Projects with config/stitch.json use the calibrated, blended stitch instead (pipeline/stitch.py: no missing strip and
no hard line at the lens seams; same frame times, same orientation as the v360 frames).

usage:  python pipeline/extract_frames.py [walk1 walk2] [--fps 5] [--force]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import CONFIG, INTERIM, WALKS, Stage, file_fingerprint

VERSION = 1


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def run(walk: str, fps: float, width: int, lens_fov: float, quality: int, force: bool) -> Path:
    src = WALKS[walk]
    out_dir = INTERIM / walk / "frames"
    import stitch
    cal = stitch.load(CONFIG)
    params = {"fps": fps, "width": width, "height": width // 2, "lens_fov": lens_fov, "roll": 180, "jpeg_q": quality}
    if cal:
        params = {"fps": fps, "width": width, "height": width // 2, "stitch": "calibrated", "stitch_params": {k: cal[k] for k in ("f", "c", "rot1", "G", "r_max", "blend_deg")}, "jpeg": 95}
    st = Stage("frames", VERSION, out_dir, params, {"video": src.name, "fingerprint": file_fingerprint(src)})
    if st.done() and not force:
        print(f"[frames] {walk}: cached ({out_dir})")
        return out_dir
    st.start()
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    if cal:                                                              # calibrated blended stitch (pipeline/stitch.py)
        import cv2
        n_est = int(ffprobe_duration(src) * fps)
        print(f"[frames] {walk}: calibrated stitch at {fps} fps -> {out_dir}", flush=True)
        for i, im in enumerate(stitch.stream(src, cal, width, width // 2, fps)):
            cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if i % 25 == 0:
                print(f"[frames] {walk} {i}/{n_est}", flush=True)
    else:
        vf = (
            f"fps={fps},"
            f"v360=input=dfisheye:output=e:ih_fov={lens_fov}:iv_fov={lens_fov}:roll=180"
            f":w={width}:h={width // 2}:interp=cubic"
        )
        cmd = [
            "ffmpeg", "-v", "error", "-stats", "-y", "-i", str(src), "-an", "-sn",
            "-vf", vf, "-q:v", str(quality), "-start_number", "0", str(out_dir / "%06d.jpg"),
        ]
        print(f"[frames] {walk}: extracting at {fps} fps -> {out_dir}")
        subprocess.run(cmd, check=True)

    frames = sorted(out_dir.glob("*.jpg"))
    duration = ffprobe_duration(src)
    # frame i is sampled at t = i / fps seconds of the source video
    index = [{"frame": i, "file": f.name, "t": round(i / fps, 3)} for i, f in enumerate(frames)]
    (out_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    st.finish({"n_frames": len(frames), "video_duration_s": round(duration, 3)})
    print(f"[frames] {walk}: {len(frames)} frames in {st.out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--fps", type=float, default=5.0, help="frames per second to sample (5 keeps tracking stable)")
    ap.add_argument("--width", type=int, default=3840, help="equirect width (height = width/2)")
    ap.add_argument("--lens-fov", type=float, default=190.0)
    ap.add_argument("--quality", type=int, default=2, help="ffmpeg JPEG quality, 2 = high")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    a = ap.parse_args()
    for w in a.walks:
        if w not in WALKS:
            sys.exit(f"unknown walk {w!r}; choose from {list(WALKS)}")
        run(w, a.fps, a.width, a.lens_fov, a.quality, a.force)


if __name__ == "__main__":
    main()
