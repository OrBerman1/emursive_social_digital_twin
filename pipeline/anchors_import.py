"""Cheap stage: import anchor points from a text file into the project's config/anchors.json.

Text format (the one the team used), entries separated by ';' or new lines:
    Hasaluf f104 yaw -8; Rosita Bar f163 yaw 150; 28d0c5e2-7a38-49bc-810c-fdb8da897945 f103 yaw -169
    optional: pitch P and fov F after the yaw, e.g.  Garinei Afula f174 yaw 2 pitch 7 fov 120
  - the store is its Shop_Name or GlobalID from the store table
  - fN is the analysis frame (5 fps) shown in the viewer's top-left box; yaw is the viewer's "view yaw"
An entry for the same walk + frame + store replaces the old one; others are added.

usage:  python pipeline/anchors_import.py <walk> <file.txt>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from common import CONFIG, WALKS

ENTRY = re.compile(r"^\s*(?P<shop>.+?)\s+f(?P<frame>\d+)\s+yaw\s+(?P<yaw>-?\d+(?:\.\d+)?)"
                   r"(?:\s+pitch\s+(?P<pitch>-?\d+(?:\.\d+)?))?(?:\s+fov\s+(?P<fov>\d+(?:\.\d+)?))?\s*$", re.I)


def parse(text: str) -> tuple[list[dict], list[str]]:
    text = text.strip().strip("[]")
    ok, bad = [], []
    for part in re.split(r"[;\n]+", text):
        part = part.strip().strip(",")
        if not part:
            continue
        m = ENTRY.match(part)
        if not m:
            bad.append(part)
            continue
        a = {"shop": m["shop"].strip(), "frame": int(m["frame"]), "yaw": float(m["yaw"])}
        if m["pitch"]:
            a["pitch"] = float(m["pitch"])
        if m["fov"]:
            a["fov"] = float(m["fov"])
        ok.append(a)
    return ok, bad


def run(walk: str, path: Path) -> dict:
    if walk not in WALKS:
        raise SystemExit(f"unknown walk '{walk}' (project walks: {', '.join(WALKS)})")
    new, bad = parse(path.read_text(encoding="utf-8-sig"))
    out = CONFIG / "anchors.json"
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    cur = data.get(walk, [])
    keyof = lambda a: (a["frame"], a["shop"].lower())
    merged = {keyof(a): a for a in cur}
    for a in new:
        merged[keyof(a)] = a
    data[walk] = sorted(merged.values(), key=lambda a: a["frame"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"walk": walk, "imported": len(new), "unreadable": bad, "anchors_now": len(data[walk])}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(run(sys.argv[1], Path(sys.argv[2])), ensure_ascii=False))
