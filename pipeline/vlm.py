"""Claude vision calls for the review stages (used only when a Claude key is saved: Anthropic API or Amazon
Bedrock, see llm.py). One request = one numbered contact sheet + a question; the answer is JSON forced by a schema.

  sheet(images, labels)        -> one JPEG contact sheet, each cell labelled (sign id, track id, store id)
  ask(jpeg, prompt, schema)    -> dict parsed from Claude's JSON answer (structured outputs)

Model: llm.model() (Claude Sonnet 5.5), effort medium; server-side fallback on a refusal where available. Every answer is written to the stage's
cache by the caller, so a sheet is sent once per project (compute once, cache everything).
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np

import llm

MODEL = llm.model()


def available() -> bool:
    return llm.available()


def sheet(images: list, labels: list[str], cols: int = 4, cell_w: int = 420, cell_h: int = 300) -> bytes:
    """Contact sheet: images (BGR arrays) scaled into cells, label in yellow at the top-left of each cell."""
    rows = max(1, -(-len(images) // cols))
    out = np.full((cell_h * rows, cell_w * cols, 3), 30, np.uint8)
    for j, (im, lab) in enumerate(zip(images, labels)):
        s = min((cell_w - 10) / im.shape[1], (cell_h - 40) / im.shape[0], 4.0)
        c = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
        x0, y0 = (j % cols) * cell_w, (j // cols) * cell_h
        out[y0 + 32:y0 + 32 + c.shape[0], x0 + 5:x0 + 5 + c.shape[1]] = c
        cv2.putText(out, lab, (x0 + 6, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.rectangle(out, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (90, 90, 90), 1)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def ask(jpeg: bytes, prompt: str, schema: dict) -> dict:
    """One sheet + one question -> JSON (validated by the API against `schema`)."""
    r = llm.create(
        max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpeg).decode()}},
            {"type": "text", "text": prompt}]}])
    if r.stop_reason == "refusal":
        raise RuntimeError("Claude declined this sheet")
    if r.stop_reason == "max_tokens":
        raise RuntimeError("answer cut off (max_tokens)")
    out = json.loads(next(b.text for b in r.content if b.type == "text"))
    if schema.get("title") == "id_map":                                  # list of {id, ...} -> {id: answer}
        wrap = "value" in schema["properties"]["items"]["items"]["properties"]
        want = set(schema["properties"]["items"]["description"].split(": ", 1)[1].split(","))
        out = {it["id"]: (it["value"] if wrap else {k: v for k, v in it.items() if k != "id"}) for it in out["items"] if it["id"] in want}
    return out


def id_map_schema(ids: list[str], item: dict) -> dict:
    """Answer for each id on the sheet. Sent as a LIST of {id, ...} items (one small grammar, repeated: a schema with
    one property per id is too large a grammar for Bedrock); ask() turns it back into {"<id>": item, ...}."""
    if item.get("type") == "object":
        it = {**item, "properties": {"id": {"type": "string"}, **item["properties"]}, "required": ["id", *item["required"]]}
    else:
        it = {"type": "object", "additionalProperties": False, "required": ["id", "value"], "properties": {"id": {"type": "string"}, "value": item}}
    return {"title": "id_map", "type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "items": it, "description": "one item for each of these ids: " + ",".join(ids)}}}
