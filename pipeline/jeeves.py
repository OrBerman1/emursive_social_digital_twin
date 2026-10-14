"""Ask Jeeves: a question-answering and data-analysis agent for one project (Claude Sonnet 5.5, see llm.py).

The agent is not blank: its system prompt carries the project's facts (walks, totals, what each number means and how
reliable it is), and it has four skills over the ANALYSED data and the video frames:
  run_sql          read-only SQL over the project's tables (people, people_frames, motion, objects, signs, food,
                   pictured_people, crowds, stores, walks, comments, attachments, classifications)
  view_people      a numbered sheet of up to 30 person crops (to look at clothing, activity, ...)
  view_scene       the 360 frame at a moment, rendered as a normal view with the people labelled
  classify_people  one visual question over many people at once ("blue shirt?" yes / no / unclear): crops in sheets
                   of 20, each sheet one separate Claude call (not inside the chat), answers cached per question, and
                   saved as table classification_<n> for counting and joining
Answers point to the video with tokens the viewer turns into jump buttons: {{p:walk1:310}} (a person) and
{{t:walk1:45.2}} or {{t:walk1:45.2:120}} (a moment, optional view yaw).
Chats are saved per project in <processed>/jeeves/chats/<id>.json (under data/ = not in git). Person crops are made
in memory and sent to Claude; they are never stored.
"""
from __future__ import annotations

import base64
import json
import re
import sqlite3
import threading
import time
import uuid
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import llm
from common import ROOT

MAX_TURNS = 40
SHEET_N, CLASSIFY_N = 30, 20
R = namedtuple("R", "yaw yaw_l yaw_r pitch_top pitch_bot")


# ---------------------------------------------------------------- project data -> tables
class Project:
    def __init__(self, proj: dict):
        self.proj = proj
        self.id = proj["id"]
        P = {k: ROOT / v for k, v in proj["paths"].items()}
        self.interim, self.processed = P["interim"], P["processed"]
        self.jdir = self.processed / "jeeves"
        self.walks = {w: v.get("label", w) for w, v in proj["walks"].items() if (self.processed / w / "overlays_stab.json").exists()}
        self.tables: dict[str, pd.DataFrame] = {}
        self.ov, self.persons = {}, {}
        self._build()

    def _json(self, p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _build(self) -> None:
        T = {k: [] for k in ("walks", "people", "people_frames", "objects", "signs", "food", "pictured_people", "crowds")}
        att = self._json(self.processed / "attachments.json") or {"people": []}
        files_of = {(a["walk"], a["track"]): len(a["files"]) for a in att["people"] if a.get("variant") == "stab"}
        for w, label in self.walks.items():
            d = self._json(self.processed / w / "overlays_stab.json")
            self.ov[w] = d
            fps = d["fps"]
            mo = self._json(self.processed / w / "motion_stab.json") or {"tracks": {}, "id2g": {}}
            id2g = {int(k): int(v) for k, v in mo.get("id2g", {}).items()}
            people = []
            for tid, t in d["tracks"].items():
                tid = int(tid); g = id2g.get(tid); m = mo["tracks"].get(str(g)) if g is not None else None
                people.append({"walk": w, "track": tid, "person_id": f"P-{tid:04d}", "t_start_s": round(t[0] / fps, 1),
                               "t_end_s": round(t[1] / fps, 1), "duration_s": round((t[1] - t[0]) / fps, 1), "frames_seen": t[2],
                               "max_conf": t[3], "median_dist_m": t[4], "gender_est": t[5] or "uncertain", "age_group_est": t[6] or "uncertain",
                               "motion_state": m["state"] if m else None, "median_speed_m_s": m["med_spd"] if m else None,
                               "path_len_m": m["len_m"] if m else None, "relative_to_camera": m["rel"] if m else None,
                               "same_person_group": g, "attached_files": files_of.get((w, tid), 0),
                               # linked person (link_persons.py): one id per real person; gender_est / age_group_est are
                               # smoothed over it, the *_track_est columns keep the track's own estimate
                               "linked_person": f"P-{(t[7] if len(t) > 7 else tid):04d}",
                               "gender_track_est": (t[8] if len(t) > 8 else t[5]) or "uncertain", "age_group_track_est": (t[9] if len(t) > 9 else t[6]) or "uncertain",
                               "in_viewer_count": bool(t[2] >= 5 and t[3] >= 0.6)})
            T["people"] += people
            pp = self.interim / w / "tracks_stab" / "persons.parquet"
            if pp.exists():
                pf = pd.read_parquet(pp)
                pf = pf[pf.track.isin([p["track"] for p in people])].copy()
                self.persons[w] = pf
                T["people_frames"].append(pd.DataFrame({"walk": w, "frame": pf.frame, "t_s": (pf.frame / fps).round(1), "track": pf.track,
                                                        "yaw": pf.yaw.round(1), "pitch_mid": ((pf.pitch_top + pf.pitch_bot) / 2).round(1),
                                                        "box_height_deg": (pf.pitch_top - pf.pitch_bot).round(1), "dist_m": pf.dist_m.round(1),
                                                        "conf": pf.conf.round(2), "interpolated": pf.interp}))
            objs, signs_first, food = {}, {}, {}
            for fi, F in enumerate(d["frames"]):
                for lvl, key, ci, ti in (("level1 (COCO)", "o", 0, 7), ("level2 (open vocabulary)", "l2", 0, 8)):
                    for o in F.get(key, []):
                        r = objs.setdefault((key, o[ti]), {"walk": w, "track": o[ti], "level": lvl, "class": o[ci], "first_frame": fi, "frames": 0, "max_conf": 0})
                        r["frames"] += 1; r["max_conf"] = max(r["max_conf"], o[1])
                for s in F.get("s", []):
                    signs_first.setdefault(s[0], fi)
                for o in F.get("fd", []):
                    r = food.setdefault(o[7], {"walk": w, "track": o[7], "food": o[0], "group": (d.get("food_group") or {}).get(o[0]), "first_frame": fi, "frames": 0})
                    r["frames"] += 1
            for r in list(objs.values()) + list(food.values()):
                r["t_first_s"] = round(r["first_frame"] / fps, 1)
            T["objects"] += list(objs.values()); T["food"] += list(food.values())
            for sid, s in (d.get("signs") or {}).items():
                f0 = signs_first.get(int(sid))
                T["signs"].append({"walk": w, "track": int(sid), "text": s[0], "lang": s[1], "readable": bool(s[3]),
                                   "t_first_s": round(f0 / fps, 1) if f0 is not None else None, "frames": (d.get("sign_len") or {}).get(sid)})
            for tid, t in (d.get("dp_tracks") or {}).items():
                T["pictured_people"].append({"walk": w, "track": int(tid), "t_start_s": round(t[0] / fps, 1), "t_end_s": round(t[1] / fps, 1), "decided_by": t[4]})
            cr = self._json(self.processed / w / "crowds_stab.json")
            for c in (cr or {}).get("crowds_list", []):
                T["crowds"].append({k: v for k, v in c.items() if not isinstance(v, (list, dict))})
            T["walks"].append({"walk": w, "label": label, "duration_s": round(d["n_frames"] / fps, 1), "analysis_fps": fps,
                               "frames": d["n_frames"], "people": len(people), "mapped": (self.processed / w / "georef.json").exists()})
        for k, v in T.items():
            if k == "people_frames":
                self.tables[k] = pd.concat(v, ignore_index=True) if v else pd.DataFrame()
            else:
                self.tables[k] = pd.DataFrame(v)
        shops = self._json(self.processed / "shops.json")
        if shops:
            rows = []
            for s in shops["shops"]:
                seen = s.get("seen") or {}
                rows.append({**{k: v for k, v in s.items() if k != "seen"}, "seen_walk": seen.get("walk"), "seen_t_s": round(seen["frame"] / 5, 1) if seen.get("frame") is not None else None,
                             "seen_view_yaw": seen.get("yaw"), "seen_dist_m": seen.get("dist_m")})
            self.tables["stores"] = pd.DataFrame(rows)
        cm = self._json(self.processed / "comments.json")
        if cm:
            self.tables["comments"] = pd.DataFrame([{k: c[k] for k in ("walk", "t", "text", "created")} for c in cm.get("comments", [])])
        self.tables["attachments"] = pd.DataFrame([{"walk": a["walk"], "track": a["track"], "file": f["name"], "added": f["added"]}
                                                   for a in att["people"] if a.get("variant") == "stab" for f in a["files"]])
        index = []
        for p in sorted((self.jdir / "classify").glob("*.json")) if (self.jdir / "classify").exists() else []:
            c = self._json(p)
            if c:
                self.tables[c["table"]] = pd.DataFrame(c["rows"])
                index.append({"table": c["table"], "question": c["question"], "labels": " | ".join(c["labels"]), "people": len(c["rows"]),
                              "walks": ",".join(sorted({r["walk"] for r in c["rows"]}))})
        # earlier visual answers (classify_people): reuse them instead of looking at the video again
        self.tables["classifications"] = pd.DataFrame(index, columns=["table", "question", "labels", "people", "walks"])

    # ------------------------------------------------------------ facts for the system prompt
    def facts(self) -> str:
        pe = self.tables["people"]
        lines = [f"Project: {self.proj.get('name', self.id)} (id {self.id})."]
        for w in self.tables["walks"].itertuples():
            p = pe[(pe.walk == w.walk) & pe.in_viewer_count]
            pp = p.drop_duplicates("linked_person")                         # one row per linked person
            g = pp.gender_est.value_counts().to_dict(); a = pp.age_group_est.value_counts().to_dict()
            st = p.motion_state.value_counts().to_dict()
            lines.append(f"- {w.walk} = {w.label}: {w.duration_s:.0f} s of 360 video; {len(pp)} people as counted in the viewer "
                         f"(distinct linked_person with an in_viewer_count track; {len(p)} such tracks, {w.people} tracks in total incl. short / low-confidence ones); "
                         f"perceived gender {g}; age group {a} (per person); motion {st} (per track); on the map: {'yes' if w.mapped else 'no'}.")
        for k in ("objects", "signs", "food", "crowds", "stores", "comments", "attachments"):
            if k in self.tables and len(self.tables[k]):
                lines.append(f"- {k}: {len(self.tables[k])} rows")
        return "\n".join(lines)

    def schema(self) -> str:
        out = []
        for k, df in self.tables.items():
            if len(df.columns):
                out.append(f"{k}({', '.join(df.columns)})  -- {len(df)} rows")
        return "\n".join(out)

    def sql(self, query: str, limit: int = 200) -> str:
        if not re.match(r"^\s*(select|with)\b", query, re.I):
            return "Only SELECT queries are allowed."
        con = sqlite3.connect(":memory:")
        for k, df in self.tables.items():
            if len(df.columns):
                df.to_sql(k, con, index=False)
        con.execute("PRAGMA query_only = ON")
        steps = [0]
        def guard():
            steps[0] += 1
            return 1 if steps[0] > 20000 else 0          # ~20M VM steps: stop runaway queries
        con.set_progress_handler(guard, 1000)
        try:
            cur = con.execute(query)
            cols = [c[0] for c in cur.description or []]
            rows = cur.fetchmany(limit + 1)
        except sqlite3.Error as e:
            return f"SQL error: {e}"
        finally:
            con.close()
        more = len(rows) > limit
        body = "\n".join(",".join("" if v is None else str(v) for v in r) for r in rows[:limit])
        return ",".join(cols) + "\n" + body + (f"\n... (more than {limit} rows; aggregate or add LIMIT)" if more else "")

    # ------------------------------------------------------------ images
    def _frame(self, w: str, f: int):
        return cv2.imread(str(self.interim / w / "frames_stab" / f"{int(f):06d}.jpg"))

    def _best(self, w: str, tracks: list[int]) -> pd.DataFrame:
        pf = self.persons.get(w)
        if pf is None:
            return pd.DataFrame()
        c = pf[pf.track.isin(tracks) & ~pf.interp].copy()
        # a whole, sharp view: confident, big but not a cut-off close-up (2-10 m preferred, size bonus capped at 45 deg)
        c["score"] = c.conf * c.ang_h.clip(upper=45) * np.where(c.dist_m.between(2, 10), 1.0, 0.6)
        return c.sort_values("score", ascending=False).groupby("track").head(1).set_index("track")

    def crops(self, w: str, tracks: list[int], isolate: bool, size: int = 256) -> list:
        from attributes import crop
        best = self._best(w, tracks)
        out, cache = [], {}
        for t in tracks:
            if t not in best.index:
                continue
            r = best.loc[t]
            f = int(r.frame)
            if f not in cache:
                cache[f] = self._frame(w, f)
            if cache[f] is None:
                continue
            out.append((t, crop(cache[f], R(r.yaw, r.yaw_l, r.yaw_r, r.pitch_top, r.pitch_bot), size=size, isolate=isolate)))
        return out

    def scene(self, w: str, t_s: float, yaw: float, pitch: float, fov: float) -> tuple[bytes, list[int]]:
        from spherical import Tile, TileSampler
        fps = self.ov[w]["fps"]
        f = int(round(t_s * fps)); f = max(0, min(f, self.ov[w]["n_frames"] - 1))
        equi = self._frame(w, f)
        tile = Tile(0, float(yaw), float(pitch), float(fov), 1024)
        img = TileSampler([tile], equi.shape[1], equi.shape[0]).cut(equi)[0].copy()
        shown = []
        pf = self.persons.get(w)
        if pf is not None:
            for r in pf[pf.frame == f].itertuples():
                pts = [project(tile, y, p) for y, p in ((r.yaw_l, r.pitch_top), (r.yaw_r, r.pitch_top), (r.yaw_l, r.pitch_bot), (r.yaw_r, r.pitch_bot))]
                if any(p is None for p in pts):
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                if x2 < 0 or y2 < 0 or x1 > 1024 or y1 > 1024:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(img, f"P-{int(r.track):04d}", (max(0, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                shown.append(int(r.track))
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), shown


def project(tile, yaw: float, pitch: float):
    """(yaw, pitch) in degrees -> pixel in the tile (inverse of spherical._rays); None if behind the camera."""
    y_, p_ = np.radians(yaw), np.radians(pitch)
    x2, y1, z2 = np.cos(p_) * np.sin(y_), np.sin(p_), np.cos(p_) * np.cos(y_)
    w, p = np.radians(tile.yaw), np.radians(tile.pitch)
    x = x2 * np.cos(w) - z2 * np.sin(w)
    z1 = x2 * np.sin(w) + z2 * np.cos(w)
    y = y1 * np.cos(p) - z1 * np.sin(p)
    z = y1 * np.sin(p) + z1 * np.cos(p)
    if z <= 0.05:
        return None
    f = tile.focal
    return x / z * f + tile.size / 2, -y / z * f + tile.size / 2


# ---------------------------------------------------------------- tools
TOOLS = [
    {"name": "run_sql", "description": "Run one read-only SQLite SELECT over the project tables (schema in the system prompt). "
     "Returns CSV text, at most 200 rows - aggregate in SQL for counts and statistics.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "view_people", "description": "Look at people: one image with up to 30 numbered crops (each person's largest view in the walk). "
     "Use it to inspect a few people or to check results; for a question over many people use classify_people.",
     "input_schema": {"type": "object", "properties": {"walk": {"type": "string"}, "tracks": {"type": "array", "items": {"type": "integer"}},
                                                        "isolate": {"type": "boolean", "description": "grey out everything outside the person's box (default true); false shows the surroundings"}},
                      "required": ["walk", "tracks"], "additionalProperties": False}},
    {"name": "view_scene", "description": "Look at the 360 video at a moment, rendered as a normal camera view (people boxed and labelled with their ids). "
     "yaw 0 = roughly the walking direction (stabilised video), +90 = right, -90 = left, 180 = behind.",
     "input_schema": {"type": "object", "properties": {"walk": {"type": "string"}, "t_s": {"type": "number"}, "yaw": {"type": "number"},
                                                        "pitch": {"type": "number", "description": "default -5"}, "fov": {"type": "number", "description": "degrees, default 100"}},
                      "required": ["walk", "t_s", "yaw"], "additionalProperties": False}},
    {"name": "classify_people", "description": "Ask ONE visual question about many people at once and get a label per person, e.g. "
     "'What colour is the shirt or top?' with labels [blue, not blue, unclear]. Each person is judged from their largest crop. "
     "Results are cached and saved as a new SQL table (name returned) with columns walk, track, label for counting and joining. "
     "Always include an 'unclear' label. Takes about 1 s per person.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 12},
                                                        "walks": {"type": "array", "items": {"type": "string"}, "description": "default: all walks"},
                                                        "tracks": {"type": "array", "items": {"type": "integer"}, "description": "optional subset (single walk)"}},
                      "required": ["question", "labels"], "additionalProperties": False}},
]

SYSTEM = """You are Jeeves, the research assistant of a digital twin of a street market. You answer questions and carry out
data-analysis tasks for urban researchers (planners, anthropologists) about this project, using its analysed data and its
360-degree walk videos. Be precise, brief and honest.

How the data was made (know the source and error of every number you give):
- 360 walks filmed with a handheld Insta360 camera, stabilised offline, analysed at 5 frames per second.
- People: YOLO11 detection + tracking. One "track" = one person while continuously visible; the same real person can be
  split into several tracks. linked_person joins pieces confirmed as the same person (link_persons.py, vision-reviewed;
  few links, conservative); same_person_group is motion.py's map re-link (not reviewed, about half are wrong). The camera operator and people pictured on
  signs/posters are excluded from people (pictured ones are in pictured_people).
- gender_est (man / woman / uncertain) and age_group_est are CLIP zero-shot ESTIMATES from body crops, smoothed over the
  linked person (gender_track_est / age_group_track_est = the track's own estimate), not validated:
  always call them "perceived/estimated" and mention the uncertain share. Never claim to know anyone's real gender or age.
- Motion: speed and state (walking / standing) from the foot point and camera height, camera motion removed; only for
  people within ~12 m; speed noise ~0.3 m/s.
- Objects: level1 = COCO classes (YOLO11); level2 = open-vocabulary market items (YOLO-World). Signs: text read by a
  vision model (or Tesseract when no key). Food: displays of a food type, not item counts. Crowds: groups of >= 4 people.
- Stores: the team's GIS store table; seen_* = where the walk passes closest.
- people_frames has one row per person per analysed frame (yaw: 0 = frame centre, + right; pitch_mid: + up).

Project facts:
{facts}

SQL tables (SQLite):
{schema}

How to work - DATA FIRST, video second:
- Always start from the analysed data (run_sql). Many questions are answered by the tables alone (counts, gender / age
  estimates, who walks or stands, crowds, signs, food, stores, where and when). Look at the video only for what the
  tables cannot tell (clothing, carrying, activities, checking a doubtful result).
- Even for a visual question, the data is the baseline: use it to define and scope who or what to look at (e.g. "people
  in blue shirts" = the people table WHERE in_viewer_count, maybe one walk or a time window), then classify only those,
  and join the visual answer back to the data for the final numbers and breakdowns (by walk, gender, motion, place).
- Check the classifications table first: an earlier classify_people run may already answer the question (same or
  similar question) - reuse its table instead of classifying again, and say so. If several match, use the one that
  covers the most people and mention how much the other one differs (that difference is a measure of uncertainty).
- Numbers come from run_sql. view_people / view_scene to look; classify_people to answer one visual question for many
  people, then run_sql on its table to count and join.
- Count people as COUNT(DISTINCT linked_person) WHERE in_viewer_count (seen >= 1 s, confidence >= 0.6), the same rule as
  the viewer's totals, unless the user asks otherwise. Say that a person split into tracks the linking did not join can
  still count twice. Give the method and its main uncertainty in one short line.
- Refer to people and moments with these tokens; the viewer turns them into jump buttons:
  {{{{p:WALK:TRACK}}}} for a person (e.g. {{{{p:walk1:310}}}}), {{{{t:WALK:SECONDS}}}} or {{{{t:WALK:SECONDS:YAW}}}} for a moment.
  Add the YAW to a moment whenever you know where to look (crowds.view_yaw, stores.seen_view_yaw, people_frames.yaw),
  so the jump also turns the view. Give a few examples when they help; do not list hundreds.
- If a task would need more than ~800 people classified, say so and ask before running it.
- Answer in the user's language. Use short paragraphs or a small list; no tables wider than 4 columns."""


class Jeeves:
    def __init__(self, proj: dict):
        self.P = Project(proj)

    def system_text(self) -> str:
        """Frozen per chat (saved with it): Claude's earlier reasoning in a chat is only valid with the exact same system
        prompt, so tables added later (classify_people) are announced in tool results instead of here."""
        return SYSTEM.format(facts=self.P.facts(), schema=self.P.schema())

    # ---- tools
    def tool(self, name: str, a: dict, step) -> list:
        P = self.P
        if name == "run_sql":
            step(f"Querying the data: {a['query'][:140]}")
            return [{"type": "text", "text": P.sql(a["query"])}]
        if name == "view_people":
            w = a["walk"]; tr = [int(t) for t in a["tracks"]][:SHEET_N]
            if w not in P.walks:
                return [{"type": "text", "text": f"unknown walk {w}; walks: {list(P.walks)}"}]
            step(f"Looking at {len(tr)} people ({w})")
            cs = P.crops(w, tr, a.get("isolate", True))
            if not cs:
                return [{"type": "text", "text": "no crops for these tracks"}]
            import vlm
            jpg = vlm.sheet([c for _, c in cs], [f"P-{t:04d}" for t, _ in cs], cols=6, cell_w=230, cell_h=270)
            return [{"type": "text", "text": f"{len(cs)} people, labelled P-<track> ({w})"}, img(jpg)]
        if name == "view_scene":
            w = a["walk"]
            if w not in P.walks:
                return [{"type": "text", "text": f"unknown walk {w}; walks: {list(P.walks)}"}]
            step(f"Looking at {w} at {fmt(a['t_s'])}, view {a['yaw']:.0f}°")
            jpg, shown = P.scene(w, a["t_s"], a["yaw"], a.get("pitch", -5), a.get("fov", 100))
            return [{"type": "text", "text": f"{w} t={a['t_s']} s, yaw {a['yaw']}; people in view: {', '.join(f'P-{t:04d}' for t in shown) or 'none'}"}, img(jpg)]
        if name == "classify_people":
            return [{"type": "text", "text": self.classify(a, step)}]
        return [{"type": "text", "text": f"unknown tool {name}"}]

    def classify(self, a: dict, step) -> str:
        import vlm
        P = self.P
        q, labels = a["question"].strip(), [str(x) for x in a["labels"]]
        walks = [w for w in (a.get("walks") or list(P.walks)) if w in P.walks]
        sel = {w: sorted(P.tables["people"].query("walk == @w and in_viewer_count").track.tolist()) for w in walks}
        if a.get("tracks") and len(walks) == 1:
            sel = {walks[0]: sorted(set(int(t) for t in a["tracks"]) & set(sel[walks[0]]))}
        key = re.sub(r"[^a-z0-9]+", "_", (q + "|" + "|".join(labels)).lower())[:60]
        cdir = P.jdir / "classify"; cdir.mkdir(parents=True, exist_ok=True)
        cpath = cdir / f"{uuid.uuid5(uuid.NAMESPACE_URL, q + '|' + '|'.join(labels)).hex[:10]}.json"
        cache = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else \
            {"question": q, "labels": labels, "table": f"classification_{len(list(cdir.glob('*.json'))) + 1}", "key": key, "rows": []}
        done = {(r["walk"], r["track"]) for r in cache["rows"]}
        todo = [(w, t) for w, ts in sel.items() for t in ts if (w, t) not in done]
        total = sum(len(v) for v in sel.values())
        step(f"Classifying {total} people: “{q}” ({len(todo)} new, {total - len(todo)} cached)")
        prompt = (f"Each numbered cell shows one person from a street-market video (everything outside the person's box is greyed out). "
                  f"For EVERY id answer this question about the person: {q}\nUse exactly one of the labels. Use 'unclear' when you cannot tell.")
        item = {"type": "string", "enum": labels}
        batches = []
        for w in walks:
            ts = [t for ww, t in todo if ww == w]
            batches += [(w, ts[i:i + CLASSIFY_N]) for i in range(0, len(ts), CLASSIFY_N)]
        lock, n_done, errors = threading.Lock(), [0], []

        def run(job):
            w, tr = job
            cs = P.crops(w, tr, True, size=224)
            if not cs:
                return
            ids = [f"P-{t:04d}" for t, _ in cs]
            try:
                ans = vlm.ask(vlm.sheet([c for _, c in cs], ids, cols=5, cell_w=230, cell_h=270), prompt, vlm.id_map_schema(ids, item))
            except Exception as e:                                       # noqa: BLE001 - reported, the rest continues
                errors.append(str(e)[:200]); return
            with lock:
                cache["rows"] += [{"walk": w, "track": int(i[2:]), "label": lab} for i, lab in ans.items()]
                n_done[0] += len(cs)
                cpath.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                step(f"Classified {n_done[0]} / {len(todo)} people", replace=True)

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(run, batches))
        rows = [r for r in cache["rows"] if r["track"] in sel.get(r["walk"], [])]
        P.tables[cache["table"]] = pd.DataFrame(cache["rows"])
        counts = pd.Series([r["label"] for r in rows]).value_counts().to_dict() if rows else {}
        by_walk = {w: pd.Series([r["label"] for r in rows if r["walk"] == w]).value_counts().to_dict() for w in walks}
        return (f"Saved as table {cache['table']}(walk, track, label). People asked: {total}; answered: {len(rows)}. "
                f"Counts: {counts}. By walk: {by_walk}." + (f" Errors in {len(errors)} batches: {errors[:2]}" if errors else ""))

    # ---- one question (agent loop)
    def ask(self, history: list, question: str, step, system_text: str) -> tuple[str, list]:
        msgs = history + [{"role": "user", "content": question}]
        system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        for _ in range(MAX_TURNS):
            r = llm.create(max_tokens=16000, system=system, tools=TOOLS, messages=msgs, output_config={"effort": "medium"})
            msgs.append({"role": "assistant", "content": [b.to_dict() for b in r.content]})
            if r.stop_reason == "refusal":
                return "I can't help with that request.", msgs
            if r.stop_reason != "tool_use":
                text = "\n".join(b.text for b in r.content if b.type == "text").strip()
                return text or "(no answer)", msgs
            results = []
            for b in r.content:
                if b.type == "tool_use":
                    try:
                        out = self.tool(b.name, b.input, step)
                    except Exception as e:                               # noqa: BLE001 - the model sees the error and can adapt
                        out = [{"type": "text", "text": f"tool error: {e}"}]
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            msgs.append({"role": "user", "content": results})
        return "I stopped after too many steps; please narrow the question.", msgs


def img(jpg: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpg).decode()}}


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ---------------------------------------------------------------- jobs + chats (used by serve.py)
_jobs: dict[str, dict] = {}
_agents: dict[str, tuple[float, Jeeves]] = {}
_lock = threading.Lock()


def _agent(proj: dict) -> Jeeves:
    proc = ROOT / proj["paths"]["processed"]
    stamp = max([p.stat().st_mtime for p in proc.rglob("*.json") if "jeeves" not in p.parts] or [0])
    with _lock:
        a = _agents.get(proj["id"])
        if a is None or a[0] != stamp:
            a = (stamp, Jeeves(proj)); _agents[proj["id"]] = a
    return a[1]


def chats_dir(proj: dict) -> Path:
    return ROOT / proj["paths"]["processed"] / "jeeves" / "chats"


def list_chats(proj: dict) -> list:
    d = chats_dir(proj)
    out = []
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []:
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            out.append({"id": c["id"], "title": c["title"], "updated": c["updated"], "turns": len(c["display"])})
        except (OSError, ValueError, KeyError):
            pass
    return out


def get_chat(proj: dict, cid: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{12}", cid or ""):
        raise ValueError("bad chat id")
    c = json.loads((chats_dir(proj) / f"{cid}.json").read_text(encoding="utf-8"))
    return {"id": c["id"], "title": c["title"], "display": c["display"]}


def start(proj: dict, question: str, cid: str | None) -> dict:
    if not llm.available():
        raise ValueError("no Claude key on this computer (projects page → Claude key)")
    question = (question or "").strip()[:4000]
    if not question:
        raise ValueError("empty question")
    d = chats_dir(proj); d.mkdir(parents=True, exist_ok=True)
    if cid:
        if not re.fullmatch(r"[a-f0-9]{12}", cid):
            raise ValueError("bad chat id")
        chat = json.loads((d / f"{cid}.json").read_text(encoding="utf-8"))
    else:
        chat = {"id": uuid.uuid4().hex[:12], "title": question[:70], "created": time.strftime("%Y-%m-%d %H:%M"), "messages": [], "display": []}
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "chat": chat["id"], "status": "running", "steps": [], "answer": None, "error": None, "started": time.time()}
    _jobs[jid] = job

    def step(text: str, replace: bool = False):
        if replace and job["steps"] and job["steps"][-1].startswith(text.split(" ")[0]):
            job["steps"][-1] = text
        else:
            job["steps"].append(text)

    def work():
        try:
            agent = _agent(proj)
            chat.setdefault("system", agent.system_text())
            answer, msgs = agent.ask(chat["messages"], question, step, chat["system"])
            chat["messages"] = msgs
            chat["display"].append({"q": question, "a": answer, "steps": job["steps"], "at": time.strftime("%Y-%m-%d %H:%M")})
            job["answer"], job["status"] = answer, "done"
        except Exception as e:                                           # noqa: BLE001 - shown in the panel
            job["error"], job["status"] = str(e)[:500], "failed"
            chat["display"].append({"q": question, "a": None, "error": job["error"], "steps": job["steps"], "at": time.strftime("%Y-%m-%d %H:%M")})
        chat["updated"] = time.strftime("%Y-%m-%d %H:%M")
        tmp = d / f"{chat['id']}.tmp"
        tmp.write_text(json.dumps(chat, ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / f"{chat['id']}.json")

    threading.Thread(target=work, daemon=True).start()
    return {"job": jid, "chat": chat["id"]}


def job(jid: str) -> dict:
    j = _jobs.get(jid)
    if j is None:
        raise KeyError("no such job")
    return {k: j[k] for k in ("id", "chat", "status", "steps", "answer", "error")} | {"elapsed_s": round(time.time() - j["started"])}


def delete_chat(proj: dict, cid: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{12}", cid or ""):
        raise ValueError("bad chat id")
    (chats_dir(proj) / f"{cid}.json").unlink(missing_ok=True)
    return {"ok": True}
