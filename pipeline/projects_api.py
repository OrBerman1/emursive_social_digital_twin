"""Projects API for the local server (serve.py): list / create projects, upload data, run the analysis, API key.

All paths come from the project registry (common.py). Uploads are streamed to disk and validated:
  video     -> Insta360 ONE RS .insv (MP4 container) -> a new walk in the project
  store     -> .xlsx store table with at least Shop_Name, X, Y          -> the project's store table
  anchors   -> .txt in the team's anchor format, for one walk           -> merged into config/anchors.json
The analysis runs pipeline/run_all.py --project <id> in the background (one job at a time); its progress is the
job.json it writes. The Claude API key is stored outside the repo (%LOCALAPPDATA%/ShukTwin/anthropic_api_key) and is
only passed to the analysis process; the API never returns it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

from common import DEFAULT_PROJECT, LEGACY, PROJECTS_DIR, ROOT, list_projects, load_project, project_paths

HERE = Path(__file__).resolve().parent
KEY_FILE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ShukTwin" / "anthropic_api_key"
BEDROCK_FILE = KEY_FILE.with_name("bedrock_api_key")                 # Amazon Bedrock API key (llm.py picks the provider)
ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
_job = {"proc": None, "project": None}
_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


def _paths(proj: dict) -> dict:
    return {k: ROOT / v for k, v in proj["paths"].items()}


def _save_project(proj: dict) -> None:
    pj = PROJECTS_DIR / proj["id"] / "project.json"
    pj.parent.mkdir(parents=True, exist_ok=True)
    tmp = pj.with_suffix(".tmp")
    tmp.write_text(json.dumps(proj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, pj)


def get_project(pid: str) -> dict:
    if not ID_RE.match(pid or ""):
        raise ApiError(400, "bad project id")
    try:
        return load_project(pid)
    except SystemExit:
        raise ApiError(404, f"no project '{pid}'")


def _url(p: Path) -> str:
    return "/" + p.relative_to(ROOT).as_posix()


def project_info(proj: dict) -> dict:
    P = _paths(proj)
    anchors = {}
    af = P["config"] / "anchors.json"
    if af.exists():
        try:
            anchors = {w: len(v) for w, v in json.loads(af.read_text(encoding="utf-8")).items() if isinstance(v, list)}
        except ValueError:
            pass
    st = proj.get("store_table")
    job = None
    jf = P["processed"] / "job.json"
    if jf.exists():
        try:
            j = json.loads(jf.read_text(encoding="utf-8"))
            job = {"status": j.get("status"), "overall_pct": j.get("overall_pct"), "finished": j.get("finished")}
        except ValueError:
            pass
    running = _job["proc"] is not None and _job["proc"].poll() is None and _job["project"] == proj["id"]
    return {"id": proj["id"], "name": proj.get("name", proj["id"]), "created": proj.get("created"),
            "walks": {w: {"label": v.get("label", w), "file": v["file"], "has_video": (P["raw"] / v["file"]).exists(),
                          "analysed": (P["processed"] / w / "overlays_stab.json").exists(), "anchors": anchors.get(w, 0)}
                      for w, v in proj["walks"].items()},
            "store_table": st, "has_store_table": bool(st) and (P["raw"] / st).exists(),
            "urls": {"processed": _url(P["processed"]), "interim": _url(P["interim"])},
            "job": job, "running": running}


def list_info() -> list[dict]:
    return [project_info(p) for p in list_projects()]


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:30] or "project"
    base, i = s, 2
    while (PROJECTS_DIR / s).exists() or s == DEFAULT_PROJECT:
        s = f"{base}_{i}"; i += 1
    return s


def create_project(name: str) -> dict:
    name = (name or "").strip()[:80]
    if not name:
        raise ApiError(400, "a project needs a name")
    pid = slugify(name)
    proj = {"id": pid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "paths": project_paths(pid), "walks": {}, "store_table": None}
    for p in _paths(proj).values():
        p.mkdir(parents=True, exist_ok=True)
    _save_project(proj)
    return project_info(proj)


def _materialise(proj: dict) -> dict:
    """The original project has no project.json until it is first changed: write the built-in definition."""
    if proj["id"] == DEFAULT_PROJECT and not (PROJECTS_DIR / DEFAULT_PROJECT / "project.json").exists():
        proj = json.loads(json.dumps(LEGACY))
    return proj


def _safe_name(fn: str, exts: tuple) -> str:
    fn = Path(fn or "").name
    if not fn.lower().endswith(exts):
        raise ApiError(400, f"expected a {' / '.join(exts)} file")
    fn = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", fn).strip(" .")
    if not fn:
        raise ApiError(400, "bad file name")
    return fn


def receive(rfile, length: int, dest: Path, max_bytes: int) -> None:
    if length <= 0 or length > max_bytes:
        raise ApiError(413, "file too large" if length > max_bytes else "empty upload")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    left = length
    with open(tmp, "wb") as f:
        while left > 0:
            chunk = rfile.read(min(8 << 20, left))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    if left:
        tmp.unlink(missing_ok=True)
        raise ApiError(400, "upload interrupted")
    os.replace(tmp, dest)


def upload(pid: str, kind: str, q: dict, rfile, length: int) -> dict:
    if _job["proc"] is not None and _job["proc"].poll() is None and _job["project"] == pid:
        raise ApiError(409, "the analysis of this project is running; wait until it finishes")
    proj = _materialise(get_project(pid))
    P = _paths(proj)
    if kind == "video":
        fn = _safe_name(q.get("filename", ""), (".insv",))
        dest = P["raw"] / fn
        if dest.exists():
            raise ApiError(409, f"{fn} is already in the project")
        receive(rfile, length, dest, 20 << 30)
        with open(dest, "rb") as f:
            head = f.read(12)
        if head[4:8] != b"ftyp":
            dest.unlink(missing_ok=True)
            raise ApiError(400, "this is not an Insta360 .insv video (no MP4 header)")
        nums = [int(m.group(1)) for w in proj["walks"] for m in [re.match(r"walk(\d+)$", w)] if m]
        wid = f"walk{max(nums, default=0) + 1}"
        proj["walks"][wid] = {"file": fn, "label": (q.get("label") or Path(fn).stem)[:60]}
        _save_project(proj)
        return {"ok": True, "walk": wid, "message": f"video added as {wid}"}
    if kind == "store":
        fn = _safe_name(q.get("filename", ""), (".xlsx",))
        dest = P["raw"] / fn
        receive(rfile, length, dest.with_name(dest.name + ".new"), 50 << 20)
        try:
            import pandas as pd
            df = pd.read_excel(dest.with_name(dest.name + ".new"))
        except Exception as e:                                    # noqa: BLE001
            dest.with_name(dest.name + ".new").unlink(missing_ok=True)
            raise ApiError(400, f"cannot read the Excel file: {e}")
        need = {"Shop_Name", "X", "Y"}
        if not need <= set(df.columns):
            dest.with_name(dest.name + ".new").unlink(missing_ok=True)
            raise ApiError(400, f"the store table needs columns {sorted(need)}; it has {list(df.columns)[:12]}")
        os.replace(dest.with_name(dest.name + ".new"), dest)
        proj["store_table"] = fn
        _save_project(proj)
        return {"ok": True, "message": f"store table set: {len(df)} stores"}
    if kind == "anchors":
        walk = q.get("walk", "")
        if walk not in proj["walks"]:
            raise ApiError(400, "choose the walk these anchors belong to")
        fn = _safe_name(q.get("filename", "anchors.txt"), (".txt",))
        dest = P["config"] / "anchor_files" / f"{time.strftime('%Y%m%d_%H%M%S')}_{walk}_{fn}"
        receive(rfile, length, dest, 1 << 20)
        _save_project(proj)
        r = subprocess.run([sys.executable, str(HERE / "anchors_import.py"), walk, str(dest)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=dict(os.environ, SHUK_PROJECT=pid, PYTHONIOENCODING="utf-8"))
        if r.returncode != 0:
            raise ApiError(400, (r.stderr or r.stdout).strip()[-400:])
        res = json.loads(r.stdout.strip().splitlines()[-1])
        msg = f"{res['imported']} anchors imported for {walk} ({res['anchors_now']} in total)"
        if res["unreadable"]:
            msg += f"; could not read: {'; '.join(res['unreadable'][:5])}"
        return {"ok": True, "message": msg, **res}
    raise ApiError(400, "unknown upload kind")


def start_job(pid: str) -> dict:
    proj = get_project(pid)
    with _lock:
        if _job["proc"] is not None and _job["proc"].poll() is None:
            raise ApiError(409, f"an analysis is already running (project {_job['project']})")
        if not any((_paths(proj)["raw"] / v["file"]).exists() for v in proj["walks"].values()):
            raise ApiError(400, "add a 360 video first")
        env = dict(os.environ, SHUK_PROJECT=pid, PYTHONIOENCODING="utf-8")
        key = read_key()
        if key:
            env["ANTHROPIC_API_KEY"] = key                             # (llm.py also reads the saved Bedrock key itself)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        _job["proc"] = subprocess.Popen([sys.executable, str(HERE / "run_all.py"), "--project", pid], cwd=str(ROOT), env=env,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        _job["project"] = pid
    return {"ok": True, "message": "analysis started"}


def job_state(pid: str) -> dict:
    proj = get_project(pid)
    jf = _paths(proj)["processed"] / "job.json"
    st = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else {"status": "never run", "stages": []}
    running = _job["proc"] is not None and _job["proc"].poll() is None and _job["project"] == pid
    if not running and st.get("status") == "running":
        st["status"] = "stopped"                                   # the process ended without finishing the file
    st["running"] = running
    return st


def read_key() -> str:
    try:
        return KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def set_key(key: str) -> dict:
    """Anthropic key (sk-ant-...) or Amazon Bedrock API key (ABSK...); empty = remove both."""
    key = (key or "").strip()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not key:
        KEY_FILE.unlink(missing_ok=True); BEDROCK_FILE.unlink(missing_ok=True)
    elif re.fullmatch(r"sk-ant-[A-Za-z0-9_\-]{20,}", key):
        KEY_FILE.write_text(key, encoding="utf-8")
    elif re.fullmatch(r"ABSK[A-Za-z0-9+/=]{20,}", key):
        BEDROCK_FILE.write_text(key, encoding="utf-8")
    else:
        raise ApiError(400, "not a Claude key: expected an Anthropic key (sk-ant-...) or an Amazon Bedrock API key (ABSK...)")
    import llm
    llm._client = None                                                 # pick up the new key
    return settings()


def check_key() -> dict:
    import llm
    prov, key = llm.provider()
    if prov == "bedrock":
        try:
            import httpx2
            r = httpx2.get(f"https://bedrock.{llm.region()}.amazonaws.com/inference-profiles?maxResults=100",
                           headers={"Authorization": f"Bearer {key}"}, timeout=30)
            ok = r.status_code == 200 and "claude-sonnet-5-5" in r.text
            return {"ok": ok, "message": f"Bedrock key works ({llm.region()}, {llm.model()})" if ok else f"Bedrock answered HTTP {r.status_code}"}
        except Exception as e:                                         # noqa: BLE001
            return {"ok": False, "message": f"could not reach Bedrock: {e}"}
    if not key:
        return {"ok": False, "message": "no API key saved"}
    # curl (Python's SSL is unreliable on this machine); the key is passed on stdin, not on the command line
    r = subprocess.run(["curl", "-s", "-o", os.devnull, "-w", "%{http_code}", "-H", "@-", "https://api.anthropic.com/v1/models"],
                       input=f"x-api-key: {key}\nanthropic-version: 2023-06-01\n", capture_output=True, text=True, timeout=30)
    code = r.stdout.strip()
    ok = code == "200"
    return {"ok": ok, "message": "key works" if ok else ("key rejected" if code in ("401", "403") else f"could not reach the API (HTTP {code or 'error'})")}


def settings() -> dict:
    import llm
    prov = llm.provider()[0]
    return {"api_key_set": bool(prov), "provider": prov, "model": llm.model() if prov else None}
