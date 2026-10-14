"""Cheap stage: clean, display-worthy sign names for the viewer ("Passed on the walk"), from the OCR results.

Most OCR'd text on market signs is noise. A name is kept only if:
  - it was not changed by the spell-checker (raw == corrected after punctuation is stripped): corrections tend to
    turn garbage into real but wrong words,
  - it is one script (Hebrew or Latin; mixed lines are almost always noise) with >= 3 letters,
  - Hebrew: its words are in the 100k Hebrew frequency list (pipeline/ocr/frequency_lists) and the OCR confidence
    is high (>= 80, or >= 70 when every word is known and there are >= 2 words),
  - Latin: confidence >= 85 and only alphabetic words of >= 4 letters.
Duplicates are merged (first sighting kept).

Output: data/processed/<walk>/sign_names.json  {sign_id: name}
usage:  python pipeline/sign_names.py [walk1 walk2]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from common import PROCESSED, WALKS

WORDS = Path(__file__).parent / "ocr" / "frequency_lists" / "hebrew_100000.txt"
HE, LAT = re.compile(r"[֐-׿]"), re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[^A-Za-z֐-׿]")


def clean(t: str) -> str:
    t = re.sub(r"[‎‏]", "", t)
    toks = [re.sub(r"^[^\w֐-׿(]+|[^\w֐-׿)]+$", "", x) for x in t.split()]
    return " ".join(x for x in toks if len(LETTER.sub("", x)) >= 2)


def name_of(text: str, raw: str, conf: float, words: set[str]) -> str | None:
    c = clean(text)
    if c != clean(raw) or len(LETTER.sub("", c)) < 3:
        return None
    toks = c.split()
    he = [re.sub(r"[^֐-׿]", "", x) for x in toks if HE.search(x)]
    la = [x for x in toks if LAT.search(x) and not HE.search(x)]
    if he and la:
        return None
    if he:
        known = sum(w in words for w in he) / len(he)
        ok = (conf >= 80 and known >= 0.5) or (conf >= 70 and known == 1 and len(he) >= 2)
        return c if ok else None
    return c if conf >= 85 and la and all(len(x) >= 4 and x.isalpha() for x in la) else None


def run(walk: str) -> dict:
    words = {ln.split()[0].lstrip("﻿") for ln in WORDS.read_text(encoding="utf-8").splitlines() if ln.strip()}
    d = json.loads((PROCESSED / walk / "overlays_stab.json").read_text(encoding="utf-8"))
    out, seen = {}, set()
    first = {}
    for f, F in enumerate(d["frames"]):
        for s in F.get("s", []):
            first.setdefault(str(s[0]), f)
    for sid in sorted(d["signs"], key=lambda k: first.get(k, 1 << 30)):
        text, _lang, conf, readable, raw = d["signs"][sid]
        if not readable:
            continue
        n = name_of(text, raw, conf, words)
        if n and n not in seen:
            seen.add(n)
            out[sid] = n
    (PROCESSED / walk / "sign_names.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {"walk": walk, "readable": sum(1 for v in d["signs"].values() if v[3]), "names": len(out), "list": list(out.values())}


if __name__ == "__main__":
    for w in (sys.argv[1:] or list(WALKS)):
        print(json.dumps(run(w), ensure_ascii=False))
