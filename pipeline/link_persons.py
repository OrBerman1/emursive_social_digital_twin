"""Cheap stage: link broken person tracks into persons and smooth perceived gender / age over each person.

Why: a person's track breaks at occlusions, when leaving the view or passing close to others; the pieces get new track
ids and separate labels (a far piece "uncertain", the near piece "woman") and the person is counted twice.
Track ids are NOT changed (attachments, comments and Jeeves classifications are keyed by them): this stage only adds
a person id per track.
Inputs : data/interim/<walk>/tracks<_stab>/persons.parquet, depicted<_stab>/depicted.parquet (only category "person"),
         attributes<_stab>/attributes.parquet + attributes_raw.parquet (per-track and per-crop probabilities),
         data/processed/<walk>/motion<_stab>.json (map re-links "id2g"; optional, only mapped walks)
Method : 1. links = motion.py's map re-links (consecutive members of each group) + gap bridges in the image:
            track B continues track A if B starts 1..GAP_F frames after A ends, at the direction A was heading
            (yaw, linear fit over the last / first 2 s, predicted forward and backward, wrap-safe) within
            YAW_TOL x box height, mid pitch within PITCH_TOL x box height, box height ratio and ground distance ratio
            within SIZE_RATIO, and the match is unambiguous both ways (second-best >= AMBIG worse).
            Never linked: tracks mostly closer than ~1.25 m (feet below CLOSE_PB) or >= BIG_H deg tall (the camera
            operator's arm / T-shirt when depicted.py misses it, partial boxes). A map re-link must also continue in
            the image (direction error <= MO_YAW_TOL deg, mid pitch within MO_PITCH_TOL deg).
         1b. a vision review decides (config/person_links_review.json: same / different / unclear; only "same" links).
            Unreviewed candidates that pass the gates are linked only with --auto: the visual check (14 Oct) confirmed
            only 2 of 6 gated links (1 wrong, 3 unclear), because track ends are where the tracker switches people.
            --review-sheet PATH writes contact sheets of the unreviewed candidates (scratch folder only).
         2. links are joined best first; a join is refused when the two sets overlap in time (any shared frame) or
            one holds a confident "man" and the other a confident "woman" (wrong merges are worse than missed links).
         3. person attributes = per-track probabilities averaged with weight = sum of the track's crop heights (deg):
            bigger / closer views and more crops count more; same rule as attributes.py (man / woman if >= SURE,
            age group if its probability >= 0.5, else uncertain). A one-track person keeps its own label.
Output : data/interim/<walk>/persons_linked<_stab>/links.parquet (track, person, source, n_tracks, smoothed and
         per-track labels and probabilities, weight), summary.json, manifest.json

ReID mode (--reid; reid_embed.py must have run): the same candidates and gates, but an unreviewed gated candidate is
         linked when its appearance agrees: cosine(OSNet mean of A's last-2 s crops, mean of B's first-2 s crops) >=
         REID_LOCAL and cosine(whole-track means) >= REID_MEAN (both; centred embeddings; a missing end embedding falls
         back to the track mean), neither track looks like the camera operator (REID_OP), and for the extra "wide"
         candidates (gap <= 10 s, cost <= 3: looser geometry) the pair is the mutual best appearance match. The review still wins
         (same -> link; different / unclear -> no link). Output goes to a SEPARATE folder
         persons_linked_reid<_stab>/ (same schema + candidates.parquet with every candidate's scores and decision), so
         the viewer keeps the reviewed links until export_overlays.py is pointed at it (--links persons_linked_reid).

usage:  python pipeline/link_persons.py [walk1 walk2] [--variant stab] [--auto] [--reid] [--review-sheet C:/tmp/links.jpg]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from common import CONFIG, INTERIM, PROCESSED, WALKS, Stage

VERSION = 1
GAP_F = 10            # B starts at most 2 s after A ends
FIT_F = 10            # yaw velocity from a line over the last / first 2 s of a track
MAX_VEL = 3.0         # deg/frame; faster predictions are clipped (fit noise on short tracks)
YAW_TOL = 0.35        # x box height (deg): predicted direction error
PITCH_TOL = 0.25      # x box height: mid-pitch difference
SIZE_RATIO = 1.3      # box height ratio and ground distance ratio within 1/1.3 .. 1.3
CLOSE_PB, CLOSE_SHARE = -42.0, 0.5   # tracks whose feet are mostly below -42 deg pitch (< ~1.25 m) ...
BIG_H = 40.0          # ... or whose median box is >= 40 deg tall are never linked: the camera operator's arm / body
                      # (tracked as "person" when depicted.py misses it) and partial boxes of people brushing the camera
MO_YAW_TOL, MO_PITCH_TOL = 15.0, 8.0   # a map re-link (motion.py) must also continue in the image: direction
                      # prediction error <= 15 deg and mid pitch within 8 deg (visual check: map-only links were ~50% right)
AMBIG = 0.15          # the second-best candidate must cost at least this much more (both directions)
SURE = 0.75           # = attributes.py --sure default
AGE_SURE = 0.5
REVIEW = CONFIG / "person_links_review.json"
REID_LOCAL = 0.50     # --reid: centred cosine at the junction (A's last 2 s vs B's first 2 s) ...
REID_MEAN = 0.30      # ... and of the whole tracks; both needed (validation in METHODS.txt section 32)
REID_GAP_F = 50       # --reid: extra "wide" gap bridges up to 10 s with geometric cost <= REID_COST (looser than the
REID_COST = 3.0       # 2 s gates) and no geometric ambiguity test; same appearance rule + mutual best match
REID_OP = 0.30        # --reid: tracks that look like the camera operator (centred cosine to the mean of depicted.py's
                      # operator tracks >= 0.30; 95th percentile of persons ~0.29, operator's arm tracks 0.31-0.51) are
                      # never linked by appearance: the arm matches itself across tracks
SEX = ("man", "woman")
AGES = ("child", "young adult 18-35", "middle adult 36-60", "older adult 61+")
wrap = lambda a: (np.asarray(a) + 540) % 360 - 180


def end_state(t: pd.DataFrame, at_end: bool) -> dict:
    """Yaw, yaw velocity (deg/frame), box height, mid pitch and distance at the end (or start) of a track."""
    t = t[~t.interp] if (~t.interp).sum() >= 2 else t
    s = t.tail(FIT_F) if at_end else t.head(FIT_F)
    f = s.frame.to_numpy().astype(float)
    y = s.yaw.to_numpy()
    y = y[0] + np.cumsum(np.r_[0, wrap(np.diff(y))])                        # unwrap across +-180
    v = float(np.polyfit(f - f.mean(), y, 1)[0]) if len(f) >= 3 and f[-1] > f[0] else 0.0
    r = s.iloc[-1] if at_end else s.iloc[0]
    k = min(3, len(s))
    e = s.tail(k) if at_end else s.head(k)                                  # sizes from the last / first 3 boxes
    return {"f": int(r.frame), "yaw": float(r.yaw), "v": float(np.clip(v, -MAX_VEL, MAX_VEL)), "h": float(e.ang_h.median()),
            "pm": float(((e.pitch_top + e.pitch_bot) / 2).median()), "d": float(e.dist_m.median())}


def yaw_err(e: dict, s: dict) -> float:
    """Mean of the forward (A's end -> B's start) and backward direction prediction errors (deg, wrap-safe)."""
    gap = s["f"] - e["f"]
    return float(abs(wrap(e["yaw"] + e["v"] * gap - s["yaw"])) + abs(wrap(s["yaw"] - s["v"] * gap - e["yaw"]))) / 2


def gap_candidates(ends: dict, starts: dict, gap_f: int = GAP_F, cost_max: float = 1.0, ambig: bool = True) -> list[tuple[float, int, int]]:
    """(cost, a, b) for every unambiguous gap bridge a -> b; cost <= 1 is a pass on every test."""
    cost = {}
    for a, e in ends.items():
        for b, s in starts.items():
            gap = s["f"] - e["f"]
            if b == a or gap < 1 or gap > gap_f:
                continue
            h = (e["h"] + s["h"]) / 2
            hr = max(e["h"], s["h"]) / max(min(e["h"], s["h"]), 1e-3)
            dr = max(e["d"], s["d"]) / max(min(e["d"], s["d"]), 1e-3) if np.isfinite(e["d"]) and np.isfinite(s["d"]) else 1.0
            c = max(yaw_err(e, s) / (YAW_TOL * h), abs(s["pm"] - e["pm"]) / (PITCH_TOL * h),
                    np.log(hr) / np.log(SIZE_RATIO), np.log(dr) / np.log(SIZE_RATIO))
            if c <= cost_max:
                cost[(a, b)] = float(c)
    if not ambig:                                                           # --reid wide candidates: appearance resolves ambiguity
        return sorted((c, a, b) for (a, b), c in cost.items())
    out = []
    for (a, b), c in cost.items():
        rivals = [c2 for (a2, b2), c2 in cost.items() if (a2 == a) != (b2 == b)]
        if all(c2 - c >= AMBIG for c2 in rivals):                         # unambiguous for A's successor and B's predecessor
            out.append((c, a, b))
    return sorted(out)


def reid_scores(walk: str, sfx: str, operator_ids: set) -> tuple[dict, dict]:
    """track -> {"mean", "start", "end"} centred, L2-normalised OSNet embeddings (reid_embed.py; start / end fall back
    to the mean), and track -> similarity to the camera operator (mean of the operator tracks; leave-one-out for them)."""
    p = INTERIM / walk / f"reid{sfx}" / "tracks.parquet"
    if not p.exists():
        raise SystemExit(f"--reid needs {p}: run pipeline/reid_embed.py {walk} first")
    out = {}
    for r in pd.read_parquet(p).itertuples():
        m = np.asarray(r.mean, float)
        out[int(r.track)] = {k: (np.asarray(v, float) if v is not None else m) for k, v in (("mean", r.mean), ("start", r.start), ("end", r.end))}
    ops = [t for t in operator_ids if t in out]
    tot = sum((out[t]["mean"] for t in ops), np.zeros(len(next(iter(out.values()))["mean"])) if out else np.zeros(1))
    op_sim = {}
    for t, e in out.items():
        q = tot - e["mean"] if t in ops else tot
        op_sim[t] = float(e["mean"] @ q / np.linalg.norm(q)) if len(ops) > (t in ops) else 0.0
    return out, op_sim


def appearance(E: dict, a: int, b: int) -> tuple[float, float]:
    """(junction cosine A end vs B start, whole-track cosine); NaN when a track has no usable crop."""
    if a not in E or b not in E:
        return float("nan"), float("nan")
    return float(E[a]["end"] @ E[b]["start"]), float(E[a]["mean"] @ E[b]["mean"])


def review_sheet(walk: str, sfx: str, tr: pd.DataFrame, pairs: list, path: str) -> None:
    """One row per candidate: 4 isolated crops from the last 2 s of A | 4 from the first 2 s of B. Written only to
    `path` (a scratch folder, never data/ or the repo): crops of people are not stored."""
    import cv2
    from attributes import crop
    rows, S = [], 160
    for i, (a, b, how) in enumerate(pairs):
        ims = []
        for t, end in ((a, True), (b, False)):
            s = tr[(tr.track == t) & ~tr.interp].sort_values("frame")
            s = s.tail(FIT_F) if end else s.head(FIT_F)
            for r in s.iloc[np.linspace(0, len(s) - 1, 4).astype(int)].itertuples():
                c = crop(cv2.imread(str(INTERIM / walk / f"frames{sfx}" / f"{int(r.frame):06d}.jpg")), r, S)
                cv2.putText(c, f"{t} f{int(r.frame)}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
                ims.append(c)
        lab = np.full((S, 70, 3), 255, np.uint8)
        cv2.putText(lab, f"#{i}", (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(lab, how, (4, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        rows.append(np.hstack([lab] + ims[:4] + [np.full((S, 8, 3), 0, np.uint8)] + ims[4:]))
    if rows:
        cv2.imwrite(path.replace(".jpg", f"_{walk}.jpg"), np.vstack(rows))
    print(f"[link_persons] {walk}: review sheet {len(rows)} candidates:", [(a, b) for a, b, _ in pairs])


def run(walk: str, variant: str, sheet: str | None = None, auto: bool = False, reid: bool = False) -> dict:
    sfx = f"_{variant}" if variant else ""
    tp, dp = INTERIM / walk / f"tracks{sfx}" / "persons.parquet", INTERIM / walk / f"depicted{sfx}" / "depicted.parquet"
    ap, rp = INTERIM / walk / f"attributes{sfx}" / "attributes.parquet", INTERIM / walk / f"attributes{sfx}" / "attributes_raw.parquet"
    mp = PROCESSED / walk / f"motion{sfx}.json"
    ep = INTERIM / walk / f"reid{sfx}" / "tracks.parquet"
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    params = {"gap_f": GAP_F, "fit_f": FIT_F, "yaw_tol": YAW_TOL, "pitch_tol": PITCH_TOL, "size_ratio": SIZE_RATIO, "ambig": AMBIG,
              "close_pb": CLOSE_PB, "close_share": CLOSE_SHARE, "big_h": BIG_H, "mo_yaw_tol": MO_YAW_TOL, "mo_pitch_tol": MO_PITCH_TOL,
              "sure": SURE, "age_sure": AGE_SURE, "auto": auto}
    inputs = {"tracks": h(tp), "depicted": h(dp), "attributes": h(ap), "attributes_raw": h(rp), "motion": h(mp), "review": h(REVIEW)}
    if reid:                                                                 # separate output: the reviewed links stay untouched
        params.update(reid_local=REID_LOCAL, reid_mean=REID_MEAN, reid_gap_f=REID_GAP_F, reid_cost=REID_COST, reid_op=REID_OP)
        inputs["reid"] = h(ep)
    st = Stage("link_persons" + ("_reid" if reid else ""), VERSION, INTERIM / walk / f"persons_linked{'_reid' if reid else ''}{sfx}", params, inputs)
    st.start()                                                               # cheap: always recomputed, manifest for provenance
    tr = pd.read_parquet(tp)
    all_ids = sorted(int(t) for t in tr.track.unique())
    operator_ids = set()
    if dp.exists():                                                          # operator and pictured people are never linked
        d = pd.read_parquet(dp)
        tr = tr[tr.track.isin(d.track[d.category == "person"])]
        operator_ids = set(d.track[d.category == "operator"].astype(int))
    pids = sorted(int(t) for t in tr.track.unique())                       # person tracks (they get a person id)
    a = tr.assign(close=tr.pitch_bot < CLOSE_PB).groupby("track").agg(close=("close", "mean"), h=("ang_h", "median"))
    unreliable = set(a.index[(a.close > CLOSE_SHARE) | (a.h >= BIG_H)].astype(int))
    tr = tr[~tr.track.isin(unreliable)]                                     # linkable tracks
    ok = set(tr.track.astype(int))
    att = pd.read_parquet(ap).set_index("track")
    raw = pd.read_parquet(rp)
    wgt = raw.groupby("track").ang_h.sum()                                  # crop quality: bigger / closer / more crops
    span = pd.read_parquet(tp, columns=["track", "frame"]).groupby("track").frame.agg(["min", "max"])   # all tracks: they keep a row
    sex0 = att.sex.to_dict()

    # 1. candidate links: motion.py map re-links (in time order), then image gap bridges (best first)
    ends, starts = {}, {}
    for tid in tr.track.unique():                                           # explicit loop (groupby.apply drops columns)
        t = tr[tr.track == tid].sort_values("frame")
        ends[int(tid)], starts[int(tid)] = end_state(t, True), end_state(t, False)
    cands = []                                                              # (cost, a, b, source, passes the automatic gates)
    mo = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"id2g": {}}
    groups = {}
    for t, g in mo.get("id2g", {}).items():
        if int(t) in span.index:
            groups.setdefault(g, []).append(int(t))
    for m in groups.values():                                               # the chain breaks at unlinkable members
        m = sorted(m, key=lambda t: span.at[t, "min"])
        for a, b in zip(m[:-1], m[1:]):
            if a in ok and b in ok:
                good = yaw_err(ends[a], starts[b]) <= MO_YAW_TOL and abs(ends[a]["pm"] - starts[b]["pm"]) <= MO_PITCH_TOL
                cands.append((-1.0, a, b, "motion", good))
    cands += [(c, a, b, "gap", True) for c, a, b in gap_candidates(ends, starts)]
    if reid:                                                                # wide bridges (longer gap / looser geometry), appearance decides
        have = {(a, b) for _, a, b, _, _ in cands}
        cands += [(c, a, b, "wide", True) for c, a, b in gap_candidates(ends, starts, REID_GAP_F, REID_COST, ambig=False) if (a, b) not in have]
    # vision review (config/person_links_review.json) decides where it exists; otherwise the automatic gates
    rv = json.loads(REVIEW.read_text(encoding="utf-8")).get(walk, {}) if REVIEW.exists() else {}
    verdict = {(int(a), int(b)): lab for lab in ("same", "different", "unclear") for a, b in rv.get(lab, [])}
    links, refused = [], {"motion_not_continuous_in_image": 0, "review_different_or_unclear": 0, "not_reviewed": 0, "overlap": 0, "sex_conflict": 0}
    E, op_sim = reid_scores(walk, sfx, operator_ids) if reid else ({}, {})
    crow = []                                                               # --reid: every candidate with its scores and decision
    if reid:
        refused["reid_low"] = refused["reid_not_mutual_best"] = refused["reid_operator_like"] = 0
        wide = [(a, b, *appearance(E, a, b)) for _, a, b, how, _ in cands if how == "wide"]
        best_a, best_b = {}, {}                                             # best junction similarity per A (successor) / per B
        for a, b, sl, sm in wide:
            if np.isfinite(sl):
                best_a[a] = max(best_a.get(a, -1), sl); best_b[b] = max(best_b.get(b, -1), sl)
    for c, a, b, how, good in cands:
        v = verdict.get((a, b))
        sl, sm = appearance(E, a, b) if reid else (np.nan, np.nan)
        rok = bool(sl >= REID_LOCAL and sm >= REID_MEAN)                    # NaN -> False
        opl = max(op_sim.get(a, 0.0), op_sim.get(b, 0.0)) if reid else 0.0
        mutual = how != "wide" or bool(np.isfinite(sl) and sl >= best_a.get(a, 9) and sl >= best_b.get(b, 9))
        crow.append({"a": a, "b": b, "source": how, "gap_f": int(span.at[b, "min"] - span.at[a, "max"]), "cost": c, "gated": good,
                     "sim_junction": sl, "sim_tracks": sm, "operator_sim": opl, "mutual_best": mutual, "reid_ok": rok, "review": v or ""})
        if span.at[a, "max"] >= span.at[b, "min"]:                         # motion.py may chain tracks that share frames
            refused["overlap"] += 1; continue
        if v == "same":
            links.append((c - 10, a, b, how + "+review"))                   # reviewed links are joined first
        elif v is not None:
            refused["review_different_or_unclear"] += 1
        elif good and reid and rok and opl >= REID_OP:
            refused["reid_operator_like"] += 1
        elif good and reid and rok and not mutual:
            refused["reid_not_mutual_best"] += 1
        elif good and reid and rok:
            links.append((c - (sl + sm), a, b, how + "+reid"))             # after reviewed links; stronger appearance first
        elif good and reid:
            refused["reid_low"] += 1
        elif good and auto:
            links.append((c, a, b, how))
        elif good:
            refused["not_reviewed"] += 1
        else:
            refused["motion_not_continuous_in_image"] += 1
    links.sort(key=lambda x: x[0])

    # 2. join with checks on whole sets (no shared frame, no confident man + confident woman)
    parent = {t: t for t in pids}
    members = {t: [t] for t in parent}
    srcs = {t: set() for t in parent}
    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]; t = parent[t]
        return t
    overlap = lambda A, B: any(span.at[x, "min"] <= span.at[y, "max"] and span.at[y, "min"] <= span.at[x, "max"] for x in A for y in B)
    for c, a, b, how in links:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        A, B = members[ra], members[rb]
        if overlap(A, B):
            refused["overlap"] += 1; continue
        sa, sb = {sex0.get(x) for x in A} & set(SEX), {sex0.get(y) for y in B} & set(SEX)
        if len(sa | sb) > 1:
            refused["sex_conflict"] += 1; continue
        r, o = min(ra, rb), max(ra, rb)
        parent[o] = r; members[r] = A + B; del members[o]
        srcs[r] = srcs[ra] | srcs[rb] | {how}
    if sheet and reid:                                                      # --reid: sheet of the automatic (unreviewed) ReID links
        todo = [(a, b, f"{how[:6]} {sl:.2f}/{sm:.2f}") for c, a, b, how in links if how.endswith("+reid")
                for sl, sm in [appearance(E, a, b)]]
        review_sheet(walk, sfx, tr, todo, sheet)
    elif sheet:                                                             # contact sheet of unreviewed candidates (review aid)
        todo = [(a, b, how) for c, a, b, how, good in cands if (a, b) not in verdict and not overlap([a], [b])]   # all of them, gated or not
        review_sheet(walk, sfx, tr, todo, sheet)
    if reid:
        joined = {(a, b) for c, a, b, how in links if find(a) == find(b)}
        cdf = pd.DataFrame(crow)
        if len(cdf):
            cdf["linked"] = [(r.a, r.b) in joined for r in cdf.itertuples()]
        cdf.to_parquet(st.out_dir / "candidates.parquet", index=False)

    # 3. smoothed attributes per person
    pcols = [f"p_{x}" for x in SEX + AGES]
    rows = []
    for root, ms in members.items():
        P = att.reindex(ms)[pcols]
        w = wgt.reindex(ms).fillna(0.0).to_numpy()
        has = P.notna().all(axis=1).to_numpy() & (w > 0)
        if has.any():
            p = (P.to_numpy()[has] * w[has, None]).sum(0) / w[has].sum()
            sx = "man" if p[0] >= SURE else "woman" if p[1] >= SURE else "uncertain"
            ag = AGES[int(np.argmax(p[2:]))] if p[2:].max() >= AGE_SURE else "uncertain"
        else:
            p, sx, ag = np.full(len(pcols), np.nan), "uncertain", "uncertain"
        if len(ms) == 1:                                                    # identical to attributes.py by construction
            sx, ag = att.sex.get(ms[0], "uncertain"), att.age.get(ms[0], "uncertain")
        for t in ms:
            rows.append({"track": int(t), "person": int(root), "n_tracks": len(ms), "source": "+".join(sorted(srcs[root])) or "single",
                         "sex": sx, "age": ag, "sex_track": att.sex.get(t, "uncertain"), "age_track": att.age.get(t, "uncertain"),
                         "weight": float(wgt.get(t, 0.0)), **{f"{c}_person": float(v) for c, v in zip(pcols, p)}})
    for t in all_ids:                                                        # operator / pictured: their own person, own label
        if t not in parent:
            rows.append({"track": t, "person": t, "n_tracks": 1, "source": "not_a_person", "sex": att.sex.get(t, "uncertain"),
                         "age": att.age.get(t, "uncertain"), "sex_track": att.sex.get(t, "uncertain"),
                         "age_track": att.age.get(t, "uncertain"), "weight": float(wgt.get(t, 0.0))})
    df = pd.DataFrame(rows).sort_values("track").reset_index(drop=True)
    df.to_parquet(st.out_dir / "links.parquet", index=False)

    pe = df[df.source != "not_a_person"]
    multi = pe[pe.n_tracks > 1]
    per = multi.groupby("person")
    conflict_before = int(sum(len(set(g.sex_track)) > 1 for _, g in per))
    ch = pe[pe.sex != pe.sex_track]
    summary = {"walk": walk, "variant": variant or "raw", "tracks": len(pe), "persons": int(pe.person.nunique()),
               "relinks": int(len(pe) - pe.person.nunique()), "multi_track_persons": int(multi.person.nunique()),
               "links_used": {k: int(v) for k, v in pe[pe.n_tracks > 1].drop_duplicates("person").source.value_counts().items()},
               "candidate_links": {"motion": sum(1 for x in cands if x[3] == "motion"), "gap": sum(1 for x in cands if x[3] == "gap"),
                                   "reviewed": sum(1 for x in cands if (x[1], x[2]) in verdict)},
               "refused": refused, "persons_with_differing_track_labels": conflict_before,
               "sex_changed_tracks": {f"{a}->{b}": int(n) for (a, b), n in ch.groupby(["sex_track", "sex"]).size().items()},
               "age_changed_tracks": int((pe.age != pe.age_track).sum()),
               "sex_persons": pe.drop_duplicates("person").sex.value_counts().to_dict(),
               "sex_tracks_before": pe.sex_track.value_counts().to_dict()}
    (st.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    st.finish({k: summary[k] for k in ("tracks", "persons", "relinks")})
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab", choices=["", "stab"])
    ap.add_argument("--review-sheet", default=None, help="write contact sheets of unreviewed candidate links to this .jpg path (scratch folder)")
    ap.add_argument("--auto", action="store_true", help="also link unreviewed candidates that pass the automatic gates (measured: ~1 in 3 confirmed)")
    ap.add_argument("--reid", action="store_true", help="decide unreviewed gated candidates by appearance (reid_embed.py); writes persons_linked_reid<_stab>/")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.variant, a.review_sheet, a.auto, a.reid)))


if __name__ == "__main__":
    main()
