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
    step(str(HERE / "detect.py"), *walks, "--model", "yolov8x-oiv7.pt", "--frames", "frames_stab", cwd=MODELS)          # food (Open Images V7)
    step(str(HERE / "detect.py"), *walks, "--model", "yolov8x-worldv2.pt", "--prompts", "waste", "--frames", "frames_stab", cwd=MODELS)
    step(str(HERE / "tracks.py"), *walks, "--variant", "stab")
    step(str(HERE / "attributes.py"), *walks, "--variant", "stab", cwd=MODELS)                # cached, GPU (CLIP)
    step(str(HERE / "ocr_signs.py"), *walks, "--variant", "stab")                          # cached: docTR (GPU) + Tesseract
    step(str(HERE / "depicted.py"), *walks, "--variant", "stab", cwd=MODELS)                  # pictured people -> possible_rabi_sign (CLIP cached)
    step(str(HERE / "food.py"), *walks, "--variant", "stab", cwd=MODELS)                      # food displays: where (OIV7+COCO), what (CLIP cached)
    step(str(HERE / "georef.py"))                                                          # cheap: walks on the map from store anchors
    step(str(HERE / "waste.py"), *walks, "--variant", "stab", cwd=MODELS)                     # floor masks + CLIP cached (needs georef); not shown
    step(str(HERE / "stall_materials.py"), "--variant", "stab", cwd=MODELS)                   # stall crops + CLIP cached
    step(str(HERE / "frontage.py"), "--variant", "stab")                                   # frontage materials as locations (shown, experimental)
    step(str(HERE / "export_overlays.py"), *walks, "--variant", "stab")
    step(str(HERE / "person_flow.py"), *walks, "--variant", "stab")                        # cached: optical flow, feet vs ground (~2 min/walk)
    step(str(HERE / "motion.py"), *walks, "--variant", "stab")                             # cheap: people on the map + motion
    step(str(HERE / "make_validation.py"))                                                 # hand-count sample


if __name__ == "__main__":
    main()
