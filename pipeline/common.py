"""Shared helpers: project paths, walks, and the compute-once cache (see CLAUDE.md).

Projects: the active project is SHUK_PROJECT (env var, default "shuk_hatikvah"); its project.json gives the
paths, the walks (raw .insv file + label) and the optional store table.

Every expensive stage writes into its own output folder together with a manifest.json that records
the inputs, parameters and stage version. A stage is skipped when the manifest matches, so stitching,
detection etc. run once and later iterations reuse the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = ROOT / "projects"
DEFAULT_PROJECT = "shuk_hatikvah"

# The original project keeps its data where it always was (data/, config/); new projects live in projects/<id>/.
LEGACY = {
    "id": DEFAULT_PROJECT, "name": "Shuk HaTikvah",
    "paths": {"raw": "data/raw/LABAI", "interim": "data/interim", "processed": "data/processed", "config": "config", "osm": "data/raw/osm"},
    "walks": {"walk1": {"file": "1_Hatikvah St.insv", "label": "Hatikvah St"},
              "walk2": {"file": "2_Hamevasser St.insv", "label": "Hamevasser St"}},
    "store_table": "Shops_In_hatikva_30_7_26.xlsx",
}


def project_paths(pid: str) -> dict:
    base = f"projects/{pid}"
    return {"raw": f"{base}/data/raw", "interim": f"{base}/data/interim", "processed": f"{base}/data/processed",
            "config": f"{base}/config", "osm": f"{base}/data/interim/osm"}


def load_project(pid: str) -> dict:
    """project.json of a project (the original project works even without one)."""
    pj = PROJECTS_DIR / pid / "project.json"
    if pj.exists():
        proj = json.loads(pj.read_text(encoding="utf-8"))
    elif pid == DEFAULT_PROJECT:
        proj = dict(LEGACY)
    else:
        raise SystemExit(f"unknown project '{pid}' (no {pj})")
    proj.setdefault("paths", project_paths(pid))
    proj.setdefault("walks", {})
    return proj


def viewer_default_project() -> str:
    """Project the viewers open when the link names none (config/viewer.json; falls back to the original project)."""
    try:
        pid = json.loads((ROOT / "config" / "viewer.json").read_text(encoding="utf-8")).get("default_project") or DEFAULT_PROJECT
    except (OSError, ValueError):
        return DEFAULT_PROJECT
    return pid if pid == DEFAULT_PROJECT or (PROJECTS_DIR / pid / "project.json").exists() else DEFAULT_PROJECT


def list_projects() -> list[dict]:
    ids = {DEFAULT_PROJECT} | ({p.parent.name for p in PROJECTS_DIR.glob("*/project.json")} if PROJECTS_DIR.exists() else set())
    return [load_project(i) for i in sorted(ids)]


PROJECT_ID = os.environ.get("SHUK_PROJECT", DEFAULT_PROJECT)
PROJECT = load_project(PROJECT_ID)
_P = {k: ROOT / v for k, v in PROJECT["paths"].items()}
RAW, INTERIM, PROCESSED, CONFIG = _P["raw"], _P["interim"], _P["processed"], _P["config"]
OSM_DIR = _P.get("osm", INTERIM / "osm")

# walk id -> raw Insta360 file; walk id -> street / label shown in the viewer
WALKS = {w: RAW / v["file"] for w, v in PROJECT["walks"].items()}
WALK_LABELS = {w: v.get("label", w) for w, v in PROJECT["walks"].items()}
STORE_TABLE = (RAW / PROJECT["store_table"]) if PROJECT.get("store_table") else None


def file_fingerprint(path: Path, chunk: int = 8 << 20) -> str:
    """Fast identity of a large file: size + mtime + hash of its first and last 8 MB."""
    st = path.stat()
    h = hashlib.sha256(f"{st.st_size}:{int(st.st_mtime)}".encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def cache_key(stage: str, version: int, params: dict, inputs: dict) -> str:
    blob = json.dumps({"stage": stage, "version": version, "params": params, "inputs": inputs}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Stage:
    """Context for one cached stage run.

    usage:
        st = Stage("frames", 1, out_dir, params, inputs)
        if st.done() and not force: return
        ... do work into out_dir ...
        st.finish(extra={...})
    """

    def __init__(self, name: str, version: int, out_dir: Path, params: dict, inputs: dict):
        self.name, self.version, self.out_dir = name, version, Path(out_dir)
        self.params, self.inputs = params, inputs
        self.key = cache_key(name, version, params, inputs)
        self.manifest_path = self.out_dir / "manifest.json"
        self.t0 = time.time()

    def done(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return m.get("key") == self.key and m.get("complete") is True

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self.manifest_path.unlink()  # an interrupted run must not look complete

    def finish(self, extra: dict | None = None) -> None:
        m = {
            "stage": self.name, "version": self.version, "key": self.key, "complete": True,
            "params": self.params, "inputs": self.inputs,
            "seconds": round(time.time() - self.t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            m.update(extra)
        self.manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
