"""Run the levels 1-2 pipeline end to end. Expensive stages skip themselves when cached.

usage (from the project root, with the venv):
    .venv\\Scripts\\python pipeline\\run_all.py            # both walks
    .venv\\Scripts\\python pipeline\\run_all.py walk1
then:  .venv\\Scripts\\python pipeline\\serve.py  and open http://localhost:8000/viewer/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"          # weights are downloaded/kept here


def step(*args: str, cwd: Path = HERE) -> None:
    print(f"\n=== {' '.join(args)}")
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def main() -> None:
    walks = sys.argv[1:] or ["walk1", "walk2"]
    MODELS.mkdir(exist_ok=True)
    step(str(HERE / "extract_frames.py"), *walks)                                          # cached
    step(str(HERE / "encode_video.py"), *walks)                                            # cached
    step(str(HERE / "detect.py"), *walks, "--model", "yolo11x.pt", cwd=MODELS)             # cached, GPU
    step(str(HERE / "detect.py"), *walks, "--model", "yolov8x-worldv2.pt", "--prompts", "level2", cwd=MODELS)
    imu_py = HERE.parent / ".venv-imu" / "Scripts" / "python.exe"                          # telemetry-parser needs py3.9
    print(f"\n=== imu_extract.py {' '.join(walks)}")
    subprocess.run([str(imu_py), str(HERE / "imu_extract.py"), *walks], check=True)         # cached
    step(str(HERE / "stabilize.py"), *walks)                                               # visual rotations cached
    step(str(HERE / "tracks.py"), *walks)                                                  # cheap (unstabilised variant)
    step(str(HERE / "export_overlays.py"), *walks)                                         # cheap
    # offline stabilisation: bake into video + frames, then detect/track/export on the stabilised frames
    step(str(HERE / "bake_stab.py"), *walks)                                               # cached, GPU
    step(str(HERE / "detect.py"), *walks, "--model", "yolo11x.pt", "--frames", "frames_stab", cwd=MODELS)
    step(str(HERE / "detect.py"), *walks, "--model", "yolov8x-worldv2.pt", "--prompts", "level2", "--frames", "frames_stab", cwd=MODELS)
    step(str(HERE / "tracks.py"), *walks, "--variant", "stab")
    step(str(HERE / "attributes.py"), *walks, "--variant", "stab", cwd=MODELS)                # cached, GPU (CLIP)
    step(str(HERE / "ocr_signs.py"), *walks, "--variant", "stab")                          # cached: docTR (GPU) + Tesseract
    step(str(HERE / "export_overlays.py"), *walks, "--variant", "stab")
    step(str(HERE / "make_validation.py"))                                                 # hand-count sample


if __name__ == "__main__":
    main()
