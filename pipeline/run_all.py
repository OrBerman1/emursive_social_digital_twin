"""Run the whole pipeline for one project; every expensive stage skips itself when cached (run once, offline).

Only stages whose inputs exist run; the others are reported as skipped (with the reason):
  - a walk's video stages need its .insv
  - map stages (georef and everything on the map) need the store table AND anchor points for that walk
  - sign text by Claude (OCR) and the Claude reviews need a Claude key (llm.py: Anthropic or Bedrock); without it the automatic fallbacks run
Progress is written to <processed>/job.json (stage states, per-stage %, overall %) for the projects page.

usage (from the project root, with the venv):
    .venv\\Scripts\\python pipeline\\run_all.py                       # active project (default shuk_hatikvah), all walks
    .venv\\Scripts\\python pipeline\\run_all.py walk1                 # some walks
    .venv\\Scripts\\python pipeline\\run_all.py --project my_area     # another project
then:  .venv\\Scripts\\python pipeline\\serve.py  and open http://localhost:8000/viewer/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"          # weights are downloaded/kept here
IMU_PY = HERE.parent / ".venv-imu" / "Scripts" / "python.exe"   # Windows: telemetry-parser needs py3.9
if not IMU_PY.exists():
    IMU_PY = Path(sys.executable)                                # macOS/Linux: built into the main venv (ui_alex / mac_compatibility)


def plan(walks: list[str], C) -> list[dict]:
    """Ordered stage list for this project; each stage has a 'skip' reason or None."""
    have = [w for w in walks if C.WALKS[w].exists()]
    missing = [w for w in walks if w not in have]
    anchors = json.loads((C.CONFIG / "anchors.json").read_text(encoding="utf-8")) if (C.CONFIG / "anchors.json").exists() else {}
    store = C.STORE_TABLE is not None and C.STORE_TABLE.exists()
    mapped = [w for w in have if store and anchors.get(w)]
    import llm
    api = llm.available()                                            # Anthropic or Bedrock key (llm.py)
    legacy = C.PROJECT_ID == C.DEFAULT_PROJECT
    no_map = ("no store table in the project" if not store else "no anchor points for these walks") if not mapped else None
    S = []

    def add(key, label, args, weight, skip=None, cwd=HERE, py=None, walks_=None):
        S.append({"key": key, "label": label, "args": args, "weight": weight, "skip": skip, "cwd": str(cwd),
                  "py": py, "walks": walks_ if walks_ is not None else have})

    v = ["--variant", "stab"]
    novideo = "no 360 video in the project" if not have else None
    add("frames", "Stitch 360 video, extract frames", ["extract_frames.py", *have], 4, novideo)
    add("encode", "Encode viewer video", ["encode_video.py", *have], 4, novideo)
    add("det_raw", "Detect people (for stabilisation)", ["detect.py", *have, "--model", "yolo11x.pt"], 11, novideo, cwd=MODELS)
    if legacy:
        add("det_raw_l2", "Detect market items (unstabilised)", ["detect.py", *have, "--model", "yolov8x-worldv2.pt", "--prompts", "level2"], 11, novideo, cwd=MODELS)
    add("imu", "Read gyro log", ["imu_extract.py", *have], 0.3, novideo, py=str(IMU_PY))
    add("stabilize", "Stabilisation fit", ["stabilize.py", *have], 3, novideo)
    if legacy:
        add("tracks_raw", "Track people (unstabilised)", ["tracks.py", *have], 1, novideo)
        add("export_raw", "Export overlays (unstabilised)", ["export_overlays.py", *have], 0.2, novideo)
    add("bake", "Stabilise video and frames", ["bake_stab.py", *have], 8, novideo)
    add("det", "Detect people and objects", ["detect.py", *have, "--model", "yolo11x.pt", "--frames", "frames_stab"], 11, novideo, cwd=MODELS)
    add("det_l2", "Detect market items", ["detect.py", *have, "--model", "yolov8x-worldv2.pt", "--prompts", "level2", "--frames", "frames_stab"], 11, novideo, cwd=MODELS)
    add("det_food", "Detect food", ["detect.py", *have, "--model", "yolov8x-oiv7.pt", "--frames", "frames_stab"], 13, novideo, cwd=MODELS)
    add("det_waste", "Detect waste candidates", ["detect.py", *have, "--model", "yolov8x-worldv2.pt", "--prompts", "waste", "--frames", "frames_stab"], 13, novideo, cwd=MODELS)
    add("tracks", "Track people and objects", ["tracks.py", *have, *v], 1, novideo)
    add("attributes", "Age group and perceived gender", ["attributes.py", *have, *v], 3, novideo, cwd=MODELS)
    add("ocr", "Find and read signs (Tesseract)", ["ocr_signs.py", *have, *v], 15, novideo)
    if api and not legacy:
        add("ocr_api", "Read signs (Claude)", ["ocr_vlm.py", "api", *have], 10, novideo)
    add("depicted", "Pictured people on signs" + (" (Claude review)" if api and not legacy else ""),
        ["depicted.py", *have, *v] + (["--api"] if api and not legacy else []), 3, novideo, cwd=MODELS)
    add("food", "Name food displays", ["food.py", *have, *v], 4, novideo, cwd=MODELS)
    add("georef", "Place walks on the map", ["georef.py"], 1, no_map or novideo, walks_=mapped)
    add("waste", "Waste on the floor (hidden layer)", ["waste.py", *mapped, *v], 4, no_map or novideo, cwd=MODELS, walks_=mapped)
    sm_skip = no_map or novideo or (None if (api or legacy) else "needs a Claude key (stall materials are read by Claude)")
    add("stalls", "Stall-front photos" + (" + Claude review" if api and not legacy else ""), ["stall_materials.py", *v] + (["--api"] if api and not legacy else []), 2, sm_skip, cwd=MODELS, walks_=mapped)
    add("frontage", "Frontage materials on the map", ["frontage.py", *v], 0.1, sm_skip, walks_=mapped)
    # motion -> person linking -> crowds -> export: the overlays and observations carry the person ids, smoothed labels and crowds
    add("flow", "Optical flow of people", ["person_flow.py", *have, *v], 2, novideo)
    add("motion", "People on the map + motion", ["motion.py", *mapped, *v], 0.3, no_map or novideo, walks_=mapped)
    add("link_persons", "Link broken tracks into persons", ["link_persons.py", *have, *v], 0.2, novideo)
    # people ReID (user, 15 Oct: the default links): appearance per track (GPU, cached), then appearance-decided links
    reid_skip = novideo or (None if (MODELS / "osnet_ain_x1_0_msmt17.pth").exists() else "OSNet weights missing (run setup.bat)")
    add("reid", "People ReID: appearance per track", ["reid_embed.py", *have, *v], 1, reid_skip)
    add("link_reid", "People ReID: link persons by appearance", ["link_persons.py", *have, *v, "--reid"], 0.2, reid_skip)
    add("crowds", "Crowds", ["crowds.py", *mapped, *v], 0.2, no_map or novideo, walks_=mapped)
    add("export", "Export overlays for the viewer", ["export_overlays.py", *have, *v], 0.3, novideo)
    add("sign_names", "Clean place names for the viewer", ["sign_names.py", *have], 0.05, novideo)      # ui_alex "Passed on the walk"
    if legacy:
        add("validation", "Hand-count sample", ["make_validation.py"], 0.1, novideo)
    for s in S:
        if s["skip"] is None and not s["walks"] and s["key"] not in ("georef",):
            s["skip"] = "nothing to do"
    if missing:
        S.insert(0, {"key": "missing", "label": f"Missing video: {', '.join(missing)}", "args": [], "weight": 0,
                     "skip": "file not found", "cwd": str(HERE), "py": None, "walks": []})
    return S


class Job:
    def __init__(self, path: Path, stages: list[dict], project: str):
        self.path, self.stages = path, stages
        self.state = {"project": project, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None,
                      "error": None, "stages": [{"key": s["key"], "label": s["label"], "state": "skipped" if s["skip"] else "pending",
                                                 "note": s["skip"] or "", "pct": 0, "weight": s["weight"]} for s in stages]}
        self._last = 0.0
        self.write(force=True)

    def overall(self) -> float:
        st = [s for s in self.state["stages"] if s["state"] != "skipped"]
        tot = sum(s["weight"] for s in st) or 1
        return round(100 * sum(s["weight"] * (1 if s["state"] == "done" else s["pct"] / 100) for s in st) / tot, 1)

    def write(self, force: bool = False) -> None:
        if not force and time.time() - self._last < 1.0:
            return
        self._last = time.time()
        self.state["overall_pct"] = self.overall()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


PROG = re.compile(r"(\d+)\s*/\s*(\d+)")
FFTIME = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")          # ffmpeg -stats (stitching, encoding, baking)
WALKRE = re.compile(r"\b(walk\d+)\b")
_dur = {}


def duration(path) -> float:
    """Video length in seconds (ffprobe), for ffmpeg's time= progress; 0 if unknown."""
    if path not in _dur:
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True, timeout=60)
            _dur[path] = float(r.stdout.strip() or 0)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            _dur[path] = 0.0
    return _dur[path]


def run_stage(job: Job, i: int, s: dict, log, videos: dict) -> None:
    js = job.state["stages"][i]
    js["state"], js["pct"] = "running", 0
    job.write(force=True)
    cmd = [s["py"] or sys.executable, str(HERE / s["args"][0]), *s["args"][1:]]
    log.write(f"\n=== {' '.join(s['args'])}\n"); log.flush()
    print(f"\n=== {' '.join(s['args'])}", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")   # Hebrew sign text in the children's output
    p = subprocess.Popen(cmd, cwd=s["cwd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    walks = s["walks"] or [""]
    wi = 0                                                          # walk being processed (from the stage's own lines)
    tail = []
    for line in p.stdout:
        for part in line.replace("\r", "\n").splitlines():
            if not part.strip():
                continue
            log.write(part + "\n")
            tail = (tail + [part])[-12:]
            mw = WALKRE.search(part)
            if mw and mw.group(1) in walks:
                wi = walks.index(mw.group(1))
            m, mt = PROG.findall(part), FFTIME.search(part)
            frac = None
            if mt and walks[wi] in videos:
                d = duration(videos[walks[wi]])
                frac = min(1.0, (int(mt[1]) * 3600 + int(mt[2]) * 60 + float(mt[3])) / d) if d else None
            elif m:
                a, b = map(int, m[-1])
                frac = a / b if b > 1 and a <= b else None
            if frac is not None:
                js["pct"] = round(100 * (wi + frac) / len(walks), 1)
                job.write()
        sys.stdout.write(line)
    log.flush()
    if p.wait() != 0:
        js["state"], js["note"] = "failed", "\n".join(tail[-6:])
        raise RuntimeError(f"stage {s['key']} failed (exit {p.returncode})")
    js["state"], js["pct"] = "done", 100
    job.write(force=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*")
    ap.add_argument("--project", default=None)
    a = ap.parse_args()
    if a.project:
        os.environ["SHUK_PROJECT"] = a.project                      # children read the same project
    sys.path.insert(0, str(HERE))
    import common as C
    walks = a.walks or list(C.WALKS)
    unknown = [w for w in walks if w not in C.WALKS]
    if unknown:
        raise SystemExit(f"unknown walks {unknown} (project {C.PROJECT_ID}: {list(C.WALKS)})")
    MODELS.mkdir(exist_ok=True)
    if any(s["key"] == "imu" and not s["skip"] for s in plan(walks, C)) and IMU_PY.exists():
        try:                                                         # a helper environment from another computer does not start
            ok = subprocess.run([str(IMU_PY), "-c", "import telemetry_parser"], capture_output=True, timeout=120).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if not ok:
            raise SystemExit("The gyro helper (.venv-imu) does not work on this computer. Run setup.bat again "
                             "(it rebuilds it), then run this again.")
    import prepare_data                                              # saved sign readings (git) -> cache, if the cache has none
    prepare_data.restore_signs(C.PROJECT_ID)
    stages = plan(walks, C)
    job = Job(C.PROCESSED / "job.json", stages, C.PROJECT_ID)
    with open(C.PROCESSED / "job.log", "w", encoding="utf-8") as log:
        try:
            for i, s in enumerate(stages):
                if s["skip"]:
                    print(f"\n--- skip {s['key']}: {s['skip']}", flush=True)
                    continue
                run_stage(job, i, s, log, C.WALKS)
            job.state["status"] = "done"
        except Exception as e:                                       # noqa: BLE001 - reported to the projects page
            job.state["status"], job.state["error"] = "failed", str(e)
        finally:
            job.state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            job.write(force=True)
    if job.state["status"] != "done":
        raise SystemExit(job.state["error"])


if __name__ == "__main__":
    main()
