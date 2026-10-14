"""Files attached to specific people (interviews, notes, photos) - used by the viewer's person card through serve.py.

A person = one track of one walk in one overlay variant (track ids differ between the stabilised and the original
detections). Each entry also keeps the frame and view direction where the file was attached, so the person can be
found again if tracking is ever recomputed and the ids change.
Storage (per project, under data/ = not in git: interviews are personal data):
    <processed>/attachments/<walk><_variant>/T<track>/<file>
    <processed>/attachments.json   {"people": [{walk, variant, track, frame, yaw, pitch, files: [{name, size, added}]}]}
Only document / image / audio / video types are accepted (no HTML or scripts: the files are served from this origin).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

from common import ROOT
from projects_api import ApiError, _safe_name, get_project, receive

ALLOWED = (".pdf", ".txt", ".md", ".rtf", ".doc", ".docx", ".odt", ".xls", ".xlsx", ".csv", ".ppt", ".pptx",
           ".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".tif", ".tiff",
           ".mp3", ".m4a", ".wav", ".ogg", ".aac", ".flac", ".mp4", ".mov", ".webm", ".m4v")
MAX_FILE = 2 << 30
WALK_RE = re.compile(r"^walk\d{1,3}$")
_lock = threading.Lock()


def _paths(pid: str) -> tuple[Path, Path]:
    proc = ROOT / get_project(pid)["paths"]["processed"]
    return proc / "attachments.json", proc / "attachments"


def _load(index: Path) -> dict:
    try:
        return json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"people": []}


def _save(index: Path, data: dict) -> None:
    index.parent.mkdir(parents=True, exist_ok=True)
    tmp = index.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, index)


def _key(q: dict) -> tuple[str, str, int]:
    walk, variant = q.get("walk", ""), q.get("variant", "")
    if not WALK_RE.match(walk) or variant not in ("", "stab"):
        raise ApiError(400, "bad walk or variant")
    try:
        track = int(q.get("track", ""))
    except ValueError:
        raise ApiError(400, "bad track id")
    return walk, variant, track


def _folder(walk: str, variant: str, track: int) -> str:
    return f"{walk}{'_' + variant if variant else ''}/T{track}"


def listing(pid: str) -> dict:
    index, _ = _paths(pid)
    data = _load(index)
    proc_url = "/" + index.parent.relative_to(ROOT).as_posix()
    for p in data["people"]:
        p["dir"] = f"{proc_url}/attachments/{_folder(p['walk'], p['variant'], p['track'])}"   # + "/" + encoded file name
    return data


def add(pid: str, q: dict, rfile, length: int) -> dict:
    walk, variant, track = _key(q)
    fn = _safe_name(q.get("filename", ""), ALLOWED)
    index, root = _paths(pid)
    dest = root / _folder(walk, variant, track) / fn
    if dest.exists():                                                  # keep both: name (2).ext
        stem, ext, i = dest.stem, dest.suffix, 2
        while dest.exists():
            dest = dest.with_name(f"{stem} ({i}){ext}"); i += 1
    if length == 0:                                                    # an empty note is still a note
        dest.parent.mkdir(parents=True, exist_ok=True); dest.touch()
    else:
        receive(rfile, length, dest, MAX_FILE)
    num = lambda k: round(float(q[k]), 2) if re.fullmatch(r"-?\d+(\.\d+)?", q.get(k, "")) else None
    with _lock:
        data = _load(index)
        p = next((p for p in data["people"] if (p["walk"], p["variant"], p["track"]) == (walk, variant, track)), None)
        if p is None:
            fr = num("frame")
            p = {"walk": walk, "variant": variant, "track": track, "frame": int(fr) if fr is not None else None,
                 "yaw": num("yaw"), "pitch": num("pitch"), "files": []}
            data["people"].append(p)
        p["files"].append({"name": dest.name, "size": dest.stat().st_size, "added": time.strftime("%Y-%m-%d %H:%M")})
        _save(index, data)
    return {"ok": True, "name": dest.name}


def remove(pid: str, body: dict) -> dict:
    walk, variant, track = _key({k: str(v) for k, v in body.items()})
    name = Path(str(body.get("name", ""))).name
    index, root = _paths(pid)
    with _lock:
        data = _load(index)
        p = next((p for p in data["people"] if (p["walk"], p["variant"], p["track"]) == (walk, variant, track)), None)
        if p is None or not any(f["name"] == name for f in p["files"]):
            raise ApiError(404, "no such file")
        (root / _folder(walk, variant, track) / name).unlink(missing_ok=True)
        p["files"] = [f for f in p["files"] if f["name"] != name]
        if not p["files"]:
            data["people"].remove(p)
            try:
                (root / _folder(walk, variant, track)).rmdir()             # only if empty
            except OSError:
                pass
        _save(index, data)
    return {"ok": True}
