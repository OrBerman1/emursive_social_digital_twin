"""Prepare the raw data for both Shuk HaTikvah projects (called by process_data.bat / process_data_mac.command).

  1. checks that the raw files are in data/raw/LABAI/ and complete (the two videos have known sizes)
  2. gives the restitched project "Shuk HaTikvah" (projects/shuk_hatikvah_v2) the same files in its own raw folder:
     a hard link (no extra disk space) where possible, otherwise a copy
  3. puts back the saved sign readings (sign text read by Claude, kept in git under <config>/signs_read/<walk>/)
     into the pipeline cache, so a new computer gets the same sign text without a Claude key or new Claude calls

usage:  python pipeline/prepare_data.py            check + link + restore (exit code 1 = data missing)
        python pipeline/prepare_data.py --save     copy this computer's sign readings into <config>/signs_read/ (to commit)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from common import DEFAULT_PROJECT, ROOT, load_project

SOURCE = ROOT / "data" / "raw" / "LABAI"
SIZES = {"1_Hatikvah St.insv": 2627240283, "2_Hamevasser St.insv": 2564140748}      # complete downloads (bytes)
STORE = "Shops_In_hatikva_30_7_26.xlsx"
PROJECTS = [DEFAULT_PROJECT, "shuk_hatikvah_v2"]
SIGN_FILES = ["lines.parquet", "line_boxes.parquet", "signs.parquet", "sign_boxes.parquet"]   # + sheets/*.json (the readings)
VARIANT = "_stab"


def paths(pid: str) -> dict:
    return {k: ROOT / v for k, v in load_project(pid)["paths"].items()}


def check_source() -> bool:
    ok = True
    for name in [*SIZES, STORE]:
        f = SOURCE / name
        if not f.exists():
            print(f"  [MISSING]    {f.relative_to(ROOT)}")
            ok = False
        elif name in SIZES and f.stat().st_size != SIZES[name]:
            print(f"  [INCOMPLETE] {name}: {f.stat().st_size / 1e9:.2f} GB, should be {SIZES[name] / 1e9:.2f} GB - download it again")
            ok = False
        else:
            print(f"  [OK]         {name}")
    return ok


def link_into(pid: str) -> None:
    raw = paths(pid)["raw"]
    if raw.resolve() == SOURCE.resolve():
        return
    raw.mkdir(parents=True, exist_ok=True)
    for name in [*SIZES, STORE]:
        src, dst = SOURCE / name, raw / name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        if dst.exists():
            dst.unlink()                                                    # an old or broken copy
        try:
            os.link(src, dst)                                               # same file, no extra space
            how = "linked"
        except OSError:
            print(f"  copying {name} (a few minutes) ...", flush=True)
            shutil.copy2(src, dst)
            how = "copied"
        print(f"  {how} {name} -> {dst.relative_to(ROOT)}")


def restore_signs(pid: str) -> None:
    """Saved sign readings -> pipeline cache, only where the cache has none (never overwrites)."""
    p = paths(pid)
    for saved in sorted((p["config"] / "signs_read").glob("walk*")):
        out = p["interim"] / saved.name / f"ocr_vlm{VARIANT}"
        if (out / "signs.parquet").exists():
            continue
        (out / "sheets").mkdir(parents=True, exist_ok=True)
        for f in SIGN_FILES:
            if (saved / f).exists():
                shutil.copy2(saved / f, out / f)
        for j in (saved / "sheets").glob("*.json"):
            shutil.copy2(j, out / "sheets" / j.name)
        print(f"  restored saved sign text: {load_project(pid)['name']}, {saved.name}")


def save_signs(pid: str) -> None:
    p = paths(pid)
    for out in sorted(p["interim"].glob(f"walk*/ocr_vlm{VARIANT}")):
        if not (out / "signs.parquet").exists():
            continue
        saved = p["config"] / "signs_read" / out.parent.name
        (saved / "sheets").mkdir(parents=True, exist_ok=True)
        for f in SIGN_FILES:
            if (out / f).exists():
                shutil.copy2(out / f, saved / f)
        for j in (out / "sheets").glob("*.json"):
            shutil.copy2(j, saved / "sheets" / j.name)
        print(f"  saved {saved.relative_to(ROOT)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    if a.save:
        for pid in PROJECTS:
            save_signs(pid)
        return
    print(f"Checking the raw data in {SOURCE.relative_to(ROOT)}:")
    if not check_source():
        print("\nPut the missing files into the folder above (README.md, section 2), then run this again.")
        raise SystemExit(1)
    for pid in PROJECTS:
        link_into(pid)
        restore_signs(pid)
    print("Data ready.")


if __name__ == "__main__":
    main()
