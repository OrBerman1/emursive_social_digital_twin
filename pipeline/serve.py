"""Local static server with HTTP Range support (needed for seeking in the 360 video).

usage:  python pipeline/serve.py [--port 8000]   then open http://localhost:8000/viewer/
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from common import ROOT


class RangeHandler(SimpleHTTPRequestHandler):
    # HTTP/1.1 keep-alive: on Windows, closing the socket right after a large write (HTTP/1.0)
    # intermittently truncates the response; with Content-Length + keep-alive the client reads it all.
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        # the viewer and its data change often during development: always revalidate, never serve stale copies
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

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
