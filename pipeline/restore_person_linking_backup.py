"""Undo the person linking (link_persons.py) in the data: copy the backup taken before it back into place.

Backup (made 14 Oct 2026, before the linking): data/backup/before_person_linking_20261014/
  interim/<walk>/{tracks_stab, attributes_stab, depicted_stab, det_yolo11x_stab}/
  processed/<walk>/{overlays_stab.json, motion_stab.json, crowds_stab.json, observations_stab.geojson}, processed/jeeves/
Every backed-up file is copied back over data/interim and data/processed (only files that differ). The linking output
data/interim/<walk>/persons_linked_stab/ is renamed to persons_linked_stab.undone (not deleted), so a later
export_overlays.py / crowds.py run does not pick it up again. The backup itself is never changed.
The code stays; to stop the stage re-linking on the next run_all.py, also switch back to a branch without it.

usage:  .venv\\Scripts\\python pipeline\\restore_person_linking_backup.py [--dry-run]
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "data" / "backup" / "before_person_linking_20261014"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only list what would be copied / renamed")
    a = ap.parse_args()
    if not BACKUP.exists():
        raise SystemExit(f"no backup at {BACKUP}")
    n = 0
    for part in ("interim", "processed"):
        for src in sorted((BACKUP / part).rglob("*")):
            if not src.is_file():
                continue
            dst = ROOT / "data" / part / src.relative_to(BACKUP / part)
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                continue
            print(("would copy " if a.dry_run else "copy ") + str(dst.relative_to(ROOT)))
            n += 1
            if not a.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    for walk in sorted(p.name for p in (BACKUP / "interim").iterdir() if p.is_dir()):
        out = ROOT / "data" / "interim" / walk / "persons_linked_stab"
        if out.exists():
            print(("would rename " if a.dry_run else "rename ") + f"{out.relative_to(ROOT)} -> persons_linked_stab.undone")
            if not a.dry_run:
                tgt = out.with_name("persons_linked_stab.undone")
                if tgt.exists():
                    shutil.rmtree(tgt)                                       # an older undo of the same output
                out.rename(tgt)
    print(f"{'would restore' if a.dry_run else 'restored'} {n} files from {BACKUP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
