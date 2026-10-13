"""Diagnostic: does the stabilisation reduce frame-to-frame rotation?

Visual rotation between analysis frames i -> i+1 is R_vis (d_cam[i+1] = R_vis d_cam[i]).
With render rule d_cam = Q d_view, the rotation left in the stabilised view is
    E_i = Q[i+1]^T  R_vis  Q[i]
which should be small (only the smooth heading change). Compare |E| with |R_vis|.

usage:  python pipeline/stab_check.py walk1
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from common import INTERIM, PROCESSED


def residual(q_list: np.ndarray, vis: pd.DataFrame) -> tuple[float, float]:
    rv = vis[["rx", "ry", "rz"]].to_numpy()
    f = vis["frame"].to_numpy()                          # pair (f-1 -> f)
    R = Rot.from_rotvec(rv).as_matrix()
    Qa, Qb = q_list[f - 1], q_list[f]
    E = np.einsum("nji,njk,nkl->nil", Qb, R, Qa)         # Qb^T R Qa
    ang_e = np.degrees(Rot.from_matrix(E).magnitude())
    ang_r = np.degrees(np.linalg.norm(rv, axis=1))
    return float(np.median(ang_r)), float(np.median(ang_e))


def main() -> None:
    walk = sys.argv[1] if len(sys.argv) > 1 else "walk1"
    vis = pd.read_parquet(INTERIM / walk / "visrot.parquet")
    vis = vis[vis.inliers >= 60]
    st = json.loads((PROCESSED / walk / "stab.json").read_text(encoding="utf-8"))
    q = np.array(st["q_analysis"]).reshape(-1, 3, 3)
    raw, res = residual(q, vis)
    res_t, _ = residual(np.transpose(q, (0, 2, 1)), vis)[::-1], None
    print(f"{walk}: median rotation between frames  raw {raw:.2f} deg | stabilised (Q) {res:.2f} deg | "
          f"stabilised (Q^T) {residual(np.transpose(q, (0, 2, 1)), vis)[1]:.2f} deg")


if __name__ == "__main__":
    main()
