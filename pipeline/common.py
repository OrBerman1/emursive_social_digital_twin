"""Shared helpers: project paths, walks, and the compute-once cache (see CLAUDE.md).

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
RAW = ROOT / "data" / "raw" / "LABAI"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# walk id -> raw Insta360 file
WALKS = {
    "walk1": RAW / "1_Hatikvah St.insv",
    "walk2": RAW / "2_Hamevasser St.insv",
}


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
