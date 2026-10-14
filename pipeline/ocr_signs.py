"""OCR of signs (Tesseract only, Hebrew-first + English), two cached stages on the stabilised frames.

  stage `text_det` (GPU): docTR db_resnet50 text detection on 2 rings of undistorted tiles
                          (eye level for price cards/stalls, tilted up for shop signs), every 2nd frame;
                          word boxes merged into text lines; saved with tile + spherical boxes.
  stage `ocr`      (CPU): lines grouped into SIGNS across frames (same tracker as people) -> best 2 crops per
                          sign -> upscale + CLAHE/unsharp (from C:\\Projects\\OCR) -> Tesseract heb+eng (psm 7,
                          normal + inverted) -> vote -> HebrewFinalFixer + SpellChecker (symspell + Hebrew list).
Below the confidence threshold a sign is "unreadable" (not guessed).

usage:  python pipeline/ocr_signs.py walk1 walk2 [--variant stab] [--force-det] [--force-ocr]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import shutil

import cv2
import numpy as np
import pandas as pd

from common import INTERIM, WALKS, Stage
from spherical import Tile, TileSampler, box_to_sphere

VERSION_DET, VERSION_OCR = 1, 1
TESS = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
HEB = re.compile(r"[\u0590-\u05FF]")
ARA = re.compile(r"[\u0600-\u06FF]")
LAT = re.compile(r"[A-Za-z]")


def text_tiles() -> list[Tile]:
    tiles = [Tile(i, -180 + i * 45, -10.0, 90.0, 1024) for i in range(8)]          # eye level: stalls, price cards
    tiles += [Tile(8 + i, -157.5 + i * 45, 25.0, 90.0, 1024) for i in range(8)]   # tilted up: shop signs
    return tiles


def merge_lines(words: np.ndarray) -> list[list[float]]:
    """Merge word boxes (x1,y1,x2,y2,score) on the same text line into line boxes."""
    if len(words) == 0:
        return []
    words = words[np.argsort(words[:, 0])]
    lines: list[list[float]] = []
    for x1, y1, x2, y2, s in words:
        h = y2 - y1
        for L in lines:
            lh = L[3] - L[1]
            v_overlap = min(y2, L[3]) - max(y1, L[1])
            gap = max(x1 - L[2], L[0] - x2, 0)
            if v_overlap > 0.5 * min(h, lh) and gap < 1.5 * max(h, lh):
                L[0], L[1], L[2], L[3] = min(L[0], x1), min(L[1], y1), max(L[2], x2), max(L[3], y2)
                L[4], L[5] = max(L[4], s), L[5] + 1
                break
        else:
            lines.append([x1, y1, x2, y2, s, 1])
    return lines


def detect_text(walk: str, variant: str, step: int, force: bool) -> pd.DataFrame:
    fdir = INTERIM / walk / f"frames{variant}"
    fm = json.loads((fdir / "manifest.json").read_text(encoding="utf-8"))
    out_dir = INTERIM / walk / f"text_det{variant}"
    tiles = text_tiles()
    st = Stage("text_det", VERSION_DET, out_dir, {"step": step, "model": "docTR db_resnet50", "tiles": [t.__dict__ for t in tiles]}, {"frames_key": fm["key"]})
    if st.done() and not force:
        print(f"[text_det] {walk}{variant}: cached")
        return pd.read_parquet(out_dir / "text_lines.parquet")
    st.start()
    from doctr.models import detection_predictor
    det = detection_predictor(arch="db_resnet50", pretrained=True, assume_straight_pages=True, batch_size=16)
    import torch
    if torch.cuda.is_available():
        det = det.cuda()          # docTR predictors default to CPU (~80 min/walk); GPU is minutes
    index = json.loads((fdir / "index.json").read_text(encoding="utf-8"))[::step]
    first = cv2.imread(str(fdir / index[0]["file"]))
    sampler = TileSampler(tiles, first.shape[1], first.shape[0])
    rows = []
    for k, fr in enumerate(index):
        equi = cv2.imread(str(fdir / fr["file"]))
        imgs = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in sampler.cut(equi)]
        res = det(imgs)
        for tile, r in zip(tiles, res):
            words = r["words"] if isinstance(r, dict) else np.asarray(r)
            words = np.asarray(words, dtype=float).reshape(-1, words.shape[-1] if np.ndim(words) > 1 else 5)
            if len(words) == 0:
                continue
            xy = words[:, :4] * tile.size
            for x1, y1, x2, y2, s, n in merge_lines(np.c_[xy, words[:, 4]]):
                if (y2 - y1) < 12 or (x2 - x1) < 24:
                    continue
                rows.append({"frame": fr["frame"], "tile": tile.idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": s,
                             "n_words": n, "cls": "text", **box_to_sphere(tile, x1, y1, x2, y2)})
        if k % 50 == 0:
            print(f"\r[text_det] {walk}{variant}: {k}/{len(index)} frames  {len(rows)} lines", end="", flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "text_lines.parquet", index=False)
    st.finish({"frames": len(index), "lines": len(df)})
    return df


# ---------------- OCR ----------------
def prep(img: np.ndarray) -> np.ndarray:
    """Upscale to ~64 px text height, grey, CLAHE + unsharp (steps from C:\\Projects\\OCR preprocessor)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = float(np.clip(64 / max(g.shape[0], 1), 1, 6))
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 3)
    g = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
    return cv2.copyMakeBorder(g, 16, 16, 16, 16, cv2.BORDER_REPLICATE)


def tesseract_best(img: np.ndarray) -> tuple[str, float, str]:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESS
    best = ("", -1.0, "")
    for name, im in (("normal", img), ("inverted", 255 - img)):
        d = pytesseract.image_to_data(im, lang="heb+eng", config="--psm 7", output_type=pytesseract.Output.DICT)
        words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if str(w).strip() and float(c) >= 0]
        if not words:
            continue
        text = " ".join(w for w, _ in words)
        conf = float(np.mean([c for _, c in words]))
        letters = sum(ch.isalpha() for ch in text)
        score = conf * min(1.0, letters / 3)
        if score > best[1]:
            best = (text, score, name)
    return best


def lang_of(s: str) -> str:
    he, ar, en = bool(HEB.search(s)), bool(ARA.search(s)), bool(LAT.search(s))
    if he and en:
        return "mixed"
    return "he" if he else "ar" if ar else "en" if en else "none"


def run_ocr(walk: str, variant: str, lines: pd.DataFrame, min_conf: float, force: bool) -> pd.DataFrame:
    from tracks import track
    out_dir = INTERIM / walk / f"ocr{variant}"
    lkey = hashlib.sha256(pd.util.hash_pandas_object(lines[["frame", "tile", "x1", "y1"]], index=False).values.tobytes()).hexdigest()[:16]
    st = Stage("ocr", VERSION_OCR, out_dir, {"engine": "tesseract heb+eng psm7 normal/inverted", "crops": 2, "min_conf": min_conf}, {"lines": lkey})
    if st.done() and not force:
        print(f"[ocr] {walk}{variant}: cached")
        return pd.read_parquet(out_dir / "signs.parquet")
    st.start()
    # group text lines into signs across frames (frames are every 2nd analysis frame -> allow gaps)
    t = lines.copy()
    t["ang_h"] = t.pitch_top - t.pitch_bot
    t = track(t.sort_values("frame"), sim_min=0.2, max_age=6, min_hits=2)
    t["px_h"] = t.y2 - t.y1
    signs = t.groupby("track")
    picks = t.sort_values(["px_h", "conf"], ascending=False).groupby("track").head(2)
    picks = picks[picks.px_h >= 16]
    fdir = INTERIM / walk / f"frames{variant}"
    tiles = {tl.idx: tl for tl in text_tiles()}
    first = cv2.imread(str(next(fdir.glob("*.jpg"))))
    sampler = TileSampler(list(tiles.values()), first.shape[1], first.shape[0])
    jobs = []
    for f, g in picks.groupby("frame"):
        cut = sampler.cut(cv2.imread(str(fdir / f"{int(f):06d}.jpg")))
        by_idx = {tl.idx: im for tl, im in zip(sampler.tiles, cut)}
        for r in g.itertuples():
            im = by_idx[int(r.tile)]
            ph, pw = (r.y2 - r.y1) * 0.25, (r.x2 - r.x1) * 0.06
            y1, y2 = int(max(0, r.y1 - ph)), int(min(im.shape[0], r.y2 + ph))
            x1, x2 = int(max(0, r.x1 - pw)), int(min(im.shape[1], r.x2 + pw))
            jobs.append((int(r.track), int(f), prep(im[y1:y2, x1:x2])))
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(lambda j: (j[0], j[1], *tesseract_best(j[2])), jobs))
    res = pd.DataFrame(results, columns=["track", "frame", "raw", "score", "variant"])
    res.to_parquet(out_dir / "tesseract_raw.parquet", index=False)   # keep raw reads even if post-correction fails
    # post-correction from the user's earlier OCR project
    from ocr.spellchecker import HebrewFinalFixer, SpellChecker
    hdict = __import__("pathlib").Path(__file__).parent / "ocr" / "frequency_lists" / "hebrew_100000.txt"
    fixer = HebrewFinalFixer(hdict, min_count=200)
    spell = SpellChecker(language="hebrew_english", hebrew_dict_path=hdict)
    best = res.sort_values("score", ascending=False).groupby("track").head(1).set_index("track")
    rows = []
    for tid, g in signs:
        b = g.loc[g.px_h.idxmax()]
        row = {"track": int(tid), "sign_id": f"T{int(tid):04d}", "frame_first": int(g.frame.min()), "frame_last": int(g.frame.max()),
               "n_det": len(g), "best_frame": int(b.frame), "max_px_h": float(g.px_h.max()),
               "yaw": float(b.yaw), "yaw_l": float(b.yaw_l), "yaw_r": float(b.yaw_r), "pitch_top": float(b.pitch_top), "pitch_bot": float(b.pitch_bot),
               "raw": "", "text": "", "conf": 0.0, "lang": "none", "readable": False, "fixes": ""}
        if tid in best.index:
            r = best.loc[tid]
            raw = str(r.raw)
            fixed = spell.correct(fixer.fix(raw))
            row.update(raw=raw, text=fixed, conf=float(r.score), lang=lang_of(fixed),
                       readable=bool(r.score >= min_conf and lang_of(fixed) != "none"),
                       fixes=("final letters/spellcheck" if fixed != raw else ""))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(out_dir / "signs.parquet", index=False)
    t[["frame", "track", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot", "px_h"]].to_parquet(out_dir / "sign_boxes.parquet", index=False)
    st.finish({"signs": len(out), "ocr_crops": len(jobs), "readable": int(out.readable.sum())})
    return out


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):               # Windows console is cp1252: Hebrew output would crash
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab", choices=["", "stab"])
    ap.add_argument("--step", type=int, default=2, help="use every Nth analysis frame for text detection")
    ap.add_argument("--min-conf", type=float, default=55.0, help="Tesseract mean word confidence to call a sign readable")
    ap.add_argument("--force-det", action="store_true")
    ap.add_argument("--force-ocr", action="store_true")
    a = ap.parse_args()
    v = "_stab" if a.variant == "stab" else ""
    for w in a.walks:
        lines = detect_text(w, v, a.step, a.force_det)
        s = run_ocr(w, v, lines, a.min_conf, a.force_ocr)
        rd = s[s.readable]
        print(json.dumps({"walk": w + v, "text_lines": len(lines), "signs": len(s), "readable": len(rd),
                          "lang": rd.lang.value_counts().to_dict(),
                          "examples": rd.sort_values("conf", ascending=False).text.head(12).tolist()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
