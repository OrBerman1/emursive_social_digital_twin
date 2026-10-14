"""OCR of signs by a vision-language model reading numbered crop sheets (no API key needed when the
reading is done by Claude in the Claude Code session; any VLM could fill the same JSON files).

  prep  : text lines (cached docTR detections) -> operator mask -> strict per-LINE grouping across frames
          -> one crop per SIGN (all its lines, best frame) -> numbered contact sheets
          data/interim/<walk>/ocr_vlm<variant>/crops/S#####.png, sheets/sheet_###.jpg, lines.parquet
  read  : (outside this script) each sheet is read and written as sheets/sheet_###.json:
          {"S00012": {"text": "...", "lang": "he|en|ar|ru|mixed|none", "legible": true|false}, ...}
  merge : sheet JSONs -> ocr_vlm<variant>/signs.parquet + sign_boxes.parquet (same schema the export reads)
  api   : prep (if needed) + each unread sheet sent to Claude through the API (needs ANTHROPIC_API_KEY; vlm.py)
          + merge. Sheets already answered are not sent again. Used for new projects from the projects page.

usage:  python pipeline/ocr_vlm.py prep  walk1 walk2
        python pipeline/ocr_vlm.py merge walk1 walk2
        python pipeline/ocr_vlm.py api   walk1 walk2
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import cv2
import numpy as np
import pandas as pd

from common import INTERIM, WALKS
from ocr_signs import text_tiles
from spherical import TileSampler

PER_SHEET = 16            # 4 columns x 4 rows
COLS, ROWS = 4, 4
CELL_W, CELL_H = 420, 270
MIN_PX_H = 18             # signs whose largest line is smaller stay "detected, too small"


def prep(walk: str, variant: str) -> dict:
    """One crop per SIGN: lines grouped across frames (loose grouping, seen >= 2 times); at the sign's best frame,
    all of its lines in the same tile are cropped together, so multi-line signs are read as one."""
    from tracks import operator_like, track
    lines = pd.read_parquet(INTERIM / walk / f"text_det{variant}" / "text_lines.parquet")
    lines["ang_h"] = lines.pitch_top - lines.pitch_bot
    lines = lines[~operator_like(walk, lines, variant)]                     # operator's T-shirt etc.
    t = track(lines.sort_values("frame"), sim_min=0.2, max_age=6, min_hits=2)
    t["px_h"] = t.y2 - t.y1
    out = INTERIM / walk / f"ocr_vlm{variant}"
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    # best frame per sign = the frame where its lines are largest in total
    score = t.groupby(["track", "frame"]).px_h.sum().reset_index()
    bestf = score.sort_values("px_h", ascending=False).groupby("track").head(1)
    keep = t.groupby("track").px_h.max()
    bestf = bestf[bestf.track.map(keep) >= MIN_PX_H].sort_values(["frame", "track"]).reset_index(drop=True)
    bestf["sign_id"] = [f"S{i:05d}" for i in range(len(bestf))]
    fdir = INTERIM / walk / f"frames{variant}"
    tiles = text_tiles()
    first = cv2.imread(str(next(fdir.glob("*.jpg"))))
    sampler = TileSampler(tiles, first.shape[1], first.shape[0])
    rows = []
    for f, g in bestf.groupby("frame"):
        cut = dict(zip([tl.idx for tl in tiles], sampler.cut(cv2.imread(str(fdir / f"{int(f):06d}.jpg")))))
        for r in g.itertuples():
            ls = t[(t.track == r.track) & (t.frame == f)]
            main = ls.loc[ls.px_h.idxmax()]
            ls = ls[ls.tile == main.tile]
            im = cut[int(main.tile)]
            x1, y1, x2, y2 = ls.x1.min(), ls.y1.min(), ls.x2.max(), ls.y2.max()
            ph, pw = max(8, (y2 - y1) * 0.15), max(8, (x2 - x1) * 0.06)
            cy1, cy2 = int(max(0, y1 - ph)), int(min(im.shape[0], y2 + ph))
            cx1, cx2 = int(max(0, x1 - pw)), int(min(im.shape[1], x2 + pw))
            cv2.imwrite(str(out / "crops" / f"{r.sign_id}.png"), im[cy1:cy2, cx1:cx2])
            rows.append({"sign_id": r.sign_id, "track": int(r.track), "frame": int(f), "px_h": float(main.px_h),
                         "n_lines": len(ls), "frame_first": int(t[t.track == r.track].frame.min()),
                         "frame_last": int(t[t.track == r.track].frame.max()),
                         "yaw": float(main.yaw), "yaw_l": float(ls.yaw_l.min()), "yaw_r": float(ls.yaw_r.max()),
                         "pitch_top": float(ls.pitch_top.max()), "pitch_bot": float(ls.pitch_bot.min())})
    best = pd.DataFrame(rows)
    n_sheets = math.ceil(len(best) / PER_SHEET)
    for k in range(n_sheets):
        part = best.iloc[k * PER_SHEET:(k + 1) * PER_SHEET]
        sheet = np.full((CELL_H * ROWS, CELL_W * COLS, 3), 30, np.uint8)
        for j, r in enumerate(part.itertuples()):
            c = cv2.imread(str(out / "crops" / f"{r.sign_id}.png"))
            s = min((CELL_W - 10) / c.shape[1], (CELL_H - 40) / c.shape[0], 4.0)
            c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
            x0, y0 = (j % COLS) * CELL_W, (j // COLS) * CELL_H
            sheet[y0 + 32:y0 + 32 + c.shape[0], x0 + 5:x0 + 5 + c.shape[1]] = c
            cv2.putText(sheet, r.sign_id, (x0 + 6, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.rectangle(sheet, (x0, y0), (x0 + CELL_W - 1, y0 + CELL_H - 1), (90, 90, 90), 1)
        cv2.imwrite(str(out / "sheets" / f"sheet_{k:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "sheets" / f"sheet_{k:03d}.ids.json").write_text(json.dumps(part.sign_id.tolist()), encoding="utf-8")
    best.to_parquet(out / "lines.parquet", index=False)
    t.merge(best[["track", "sign_id"]], on="track")[["frame", "track", "sign_id", "yaw", "yaw_l", "yaw_r", "pitch_top", "pitch_bot", "px_h"]] \
        .to_parquet(out / "line_boxes.parquet", index=False)
    return {"walk": walk + variant, "sign_tracks": int(t.track.nunique()), "signs_to_read": len(best), "sheets": n_sheets}


KINDS = ["shop_sign", "banner", "price_sign", "product_packaging", "other_text", "no_text", "operator_shirt"]
PROMPT = """This contact sheet shows numbered crops of text found in a 360-degree walk through a street market in Israel
(signs are mostly Hebrew, some English, Arabic or Russian). The crops come from a camera held upside down by an operator,
so some crops show the operator's T-shirt print. For EVERY id on the sheet give:
- text: the text exactly as written (keep Hebrew as Hebrew; separate lines with " / "; "" if nothing can be read)
- lang: he, en, ar, ru, mixed or none
- legible: true only if you can read the text with confidence
- kind: shop_sign (shop name / trade sign), banner, price_sign, product_packaging, other_text,
  no_text (the crop shows no text) or operator_shirt (print on the camera operator's T-shirt)
Do not guess words you cannot see."""


def api(walk: str, variant: str) -> dict:
    """Send every unread sheet to Claude (vlm.py); answers are cached as sheet_###.json like a manual reading."""
    import vlm
    out = INTERIM / walk / f"ocr_vlm{variant}"
    if not (out / "lines.parquet").exists():
        print(json.dumps(prep(walk, variant), ensure_ascii=False), flush=True)
    item = {"type": "object", "additionalProperties": False, "required": ["text", "lang", "legible", "kind"],
            "properties": {"text": {"type": "string"}, "lang": {"type": "string", "enum": ["he", "en", "ar", "ru", "mixed", "none"]},
                           "legible": {"type": "boolean"}, "kind": {"type": "string", "enum": KINDS}}}
    ids_files = sorted((out / "sheets").glob("sheet_*.ids.json"))
    sent, failed = 0, []
    for k, idf in enumerate(ids_files):
        res = idf.with_name(idf.name.replace(".ids.json", ".json"))
        if not res.exists():
            ids = json.loads(idf.read_text(encoding="utf-8"))
            jpg = idf.with_name(idf.name.replace(".ids.json", ".jpg")).read_bytes()
            try:
                ans = vlm.ask(jpg, PROMPT, vlm.id_map_schema(ids, item))
                res.write_text(json.dumps(ans, ensure_ascii=False), encoding="utf-8")
                sent += 1
            except Exception as e:                                     # noqa: BLE001 - one bad sheet stays unread
                failed.append(f"{idf.name}: {e}")
        print(f"[ocr_vlm api] {walk} sheet {k + 1}/{len(ids_files)}", flush=True)
    if failed and len(failed) == len(ids_files):
        raise SystemExit("Claude could not read any sheet: " + failed[0])
    return {"walk": walk + variant, "sheets": len(ids_files), "sent": sent, "failed": failed[:5], **merge(walk, variant)}


def merge(walk: str, variant: str) -> dict:
    out = INTERIM / walk / f"ocr_vlm{variant}"
    best = pd.read_parquet(out / "lines.parquet")
    reads = {}
    missing = []
    for ids_file in sorted((out / "sheets").glob("sheet_*.ids.json")):
        res = ids_file.with_name(ids_file.name.replace(".ids.json", ".json"))
        if not res.exists():
            missing.append(res.name)
            continue
        reads.update(json.loads(res.read_text(encoding="utf-8")))
    rows = []
    for r in best.itertuples():
        rd = reads.get(r.sign_id, {})
        text = (rd.get("text") or "").strip()
        kind = rd.get("kind", "unread" if not rd else "other_text")
        legible = bool(rd.get("legible")) and bool(text) and kind not in ("no_text", "operator_shirt")
        rows.append({"track": int(r.track), "sign_id": r.sign_id, "frame_first": int(r.frame_first), "frame_last": int(r.frame_last),
                     "n_det": int(r.n_lines), "best_frame": int(r.frame), "max_px_h": float(r.px_h),
                     "yaw": r.yaw, "yaw_l": r.yaw_l, "yaw_r": r.yaw_r, "pitch_top": r.pitch_top, "pitch_bot": r.pitch_bot,
                     "raw": text, "text": text, "conf": 90.0 if legible else 0.0, "lang": rd.get("lang", "none") if legible else "none",
                     "readable": legible, "fixes": "", "kind": kind, "engine": "VLM (Claude) reading crop sheets"})
    signs = pd.DataFrame(rows)
    signs.to_parquet(out / "signs.parquet", index=False)
    drop = set(signs.loc[signs.kind.isin(["no_text", "operator_shirt"]), "track"])   # not signs: hide their boxes
    boxes = pd.read_parquet(out / "line_boxes.parquet").drop(columns=["sign_id"])
    boxes = boxes[~boxes.track.isin(drop)]
    # same treatment as people/objects: one box per sign per frame, gaps <= 1 s filled, zero-phase smoothing
    from tracks import smooth_tracks
    boxes = boxes.sort_values("px_h", ascending=False).drop_duplicates(["track", "frame"])
    boxes = smooth_tracks(boxes.assign(cls="text", conf=1.0), max_gap=5, sigma_pos=1.5, sigma_size=3.0)
    boxes.to_parquet(out / "sign_boxes.parquet", index=False)
    return {"walk": walk + variant, "lines": len(signs), "read": int(signs.readable.sum()), "sheets_missing": len(missing),
            "lang": signs[signs.readable].lang.value_counts().to_dict(), "kind": signs.kind.value_counts().to_dict()}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prep", "merge", "api"])
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab", choices=["", "stab"])
    a = ap.parse_args()
    v = "_stab" if a.variant == "stab" else ""
    for w in a.walks:
        print(json.dumps({"prep": prep, "merge": merge, "api": api}[a.action](w, v), ensure_ascii=False))


if __name__ == "__main__":
    main()
