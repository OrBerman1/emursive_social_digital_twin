"""Local static server with HTTP Range support (needed for seeking in the 360 video).

usage:  python pipeline/serve.py [--port 8000]   then open http://localhost:8000/viewer/

Also (local only, all validated):
  POST /api/comments?project=<id>      timeline comments -> <project processed>/comments.json
  GET  /api/projects                   projects with their walks, inputs and last analysis
  POST /api/projects {name}            new project
  PUT  /api/upload?project=&kind=video|store|anchors&filename=[&label=][&walk=]   streamed upload
  POST /api/run?project=<id>           start the analysis (one at a time);  GET /api/job?project=<id> = progress
  GET  /api/settings; POST /api/settings/apikey {key}; POST /api/settings/apikey/check
"""
from __future__ import annotations

import argparse
import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import projects_api as API
from common import DEFAULT_PROJECT, ROOT

MAX_BODY = 1 << 20                   # 1 MB is plenty for text comments


def clean_comments(data) -> list:
    """Keep only well-formed comments: {id, walk, t, yaw, pitch, text, color, created, updated}."""
    out = []
    for c in (data or {}).get("comments", [])[:5000]:
        if not isinstance(c, dict) or not re.fullmatch(r"walk\d{1,3}", str(c.get("walk", ""))):
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

    def _json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            raise API.ApiError(413, "body too large or empty")
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise API.ApiError(400, "bad JSON")

    def _api(self, method: str) -> bool:
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return False
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if method == "GET" and u.path == "/api/projects":
                return self._json({"projects": API.list_info()}) or True
            if method == "GET" and u.path == "/api/job":
                return self._json(API.job_state(q.get("project", DEFAULT_PROJECT))) or True
            if method == "GET" and u.path == "/api/settings":
                return self._json(API.settings()) or True
            if method == "POST" and u.path == "/api/projects":
                return self._json(API.create_project(self._body_json().get("name", ""))) or True
            if method == "POST" and u.path == "/api/run":
                return self._json(API.start_job(q.get("project", ""))) or True
            if method == "POST" and u.path == "/api/settings/apikey":
                return self._json(API.set_key(self._body_json().get("key", ""))) or True
            if method == "POST" and u.path == "/api/settings/apikey/check":
                return self._json(API.check_key()) or True
            if method == "PUT" and u.path == "/api/upload":
                n = int(self.headers.get("Content-Length") or 0)
                return self._json(API.upload(q.get("project", ""), q.get("kind", ""), q, self.rfile, n)) or True
            if method == "POST" and u.path == "/api/comments":
                proj = API.get_project(q.get("project", DEFAULT_PROJECT))
                comments = clean_comments(self._body_json())
                target = ROOT / proj["paths"]["processed"] / "comments.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps({"comments": comments}, ensure_ascii=False, indent=1), encoding="utf-8")
                os.replace(tmp, target)                                # atomic: never a half-written file
                return self._json({"saved": len(comments)}) or True
            self._json({"error": "not found"}, 404)
        except API.ApiError as e:
            self.close_connection = method == "PUT"                    # an unread upload body must not poison keep-alive
            self._json({"error": e.msg}, e.status)
        except Exception as e:                                         # noqa: BLE001 - report instead of dropping the connection
            self.close_connection = True
            self._json({"error": f"server error: {e}"}, 500)
        return True

    def do_GET(self):
        if not self._api("GET"):
            super().do_GET()

    def do_HEAD(self):
        super().do_HEAD()

    def do_POST(self):
        if not self._api("POST"):
            self.send_error(404)

    def do_PUT(self):
        if not self._api("PUT"):
            self.send_error(404)

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
