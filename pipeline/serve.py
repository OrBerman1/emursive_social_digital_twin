"""Local static server with HTTP Range support (needed for seeking in the 360 video).

usage:  python pipeline/serve.py [--port 8000]   then open http://localhost:8000/viewer/

Also: POST /api/comments saves the viewer's timeline comments to data/processed/comments.json (validated, local only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from common import PROCESSED, ROOT

COMMENTS = PROCESSED / "comments.json"
MAX_BODY = 1 << 20                   # 1 MB is plenty for text comments


def clean_comments(data) -> list:
    """Keep only well-formed comments: {id, walk, t, yaw, pitch, text, color, created, updated}."""
    out = []
    for c in (data or {}).get("comments", [])[:5000]:
        if not isinstance(c, dict) or c.get("walk") not in ("walk1", "walk2"):
            continue
        text = str(c.get("text", ""))[:2000]
        color = str(c.get("color", "#E69F00"))
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            color = "#E69F00"
        try:
            out.append({"id": str(c.get("id", ""))[:40], "walk": c["walk"], "t": round(float(c.get("t", 0)), 2),
                        "yaw": round(float(c.get("yaw", 0)), 1), "pitch": round(float(c.get("pitch", 0)), 1),
                        "text": text, "color": color, "created": str(c.get("created", ""))[:40], "updated": str(c.get("updated", ""))[:40]})
        except (TypeError, ValueError):
            continue
    return out


class RangeHandler(SimpleHTTPRequestHandler):
    # HTTP/1.1 keep-alive: on Windows, closing the socket right after a large write (HTTP/1.0)
    # intermittently truncates the response; with Content-Length + keep-alive the client reads it all.
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        # the viewer and its data change often during development: always revalidate, never serve stale copies
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/comments":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            self.send_error(413)
            return
        try:
            comments = clean_comments(json.loads(self.rfile.read(n).decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400)
            return
        COMMENTS.parent.mkdir(parents=True, exist_ok=True)
        tmp = COMMENTS.with_suffix(".tmp")
        tmp.write_text(json.dumps({"comments": comments}, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, COMMENTS)                                  # atomic: never a half-written file
        body = json.dumps({"saved": len(comments)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        start = int(m.group(1)) if m and m.group(1) else 0
        end = int(m.group(2)) if m and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (ConnectionResetError, BrokenPipeError):
                break
            remaining -= len(chunk)
        self._remaining = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    handler = partial(RangeHandler, directory=str(ROOT))
    print(f"serving {ROOT} at http://localhost:{a.port}/viewer/")
    ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()


if __name__ == "__main__":
    main()
