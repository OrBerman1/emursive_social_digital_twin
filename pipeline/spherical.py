"""Equirectangular <-> perspective-tile geometry.

Conventions (degrees): yaw/lon in [-180, 180), 0 = centre column of the equirect frame, positive to the right;
pitch/lat in [-90, 90], positive up. Equirect pixel x = (yaw + 180) / 360 * W, y = (90 - pitch) / 180 * H.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Tile:
    idx: int
    yaw: float      # centre direction of the tile
    pitch: float
    fov: float      # horizontal = vertical field of view (square tile)
    size: int       # output pixels

    @property
    def focal(self) -> float:
        return (self.size / 2) / np.tan(np.radians(self.fov) / 2)


def default_tiles(n_yaw: int = 8, pitch: float = -15.0, fov: float = 100.0, size: int = 1024) -> list[Tile]:
    """n_yaw tiles around the horizon, tilted slightly down (people's feet are below the camera)."""
    return [Tile(i, -180 + i * 360 / n_yaw, pitch, fov, size) for i in range(n_yaw)]


def _rays(tile: Tile, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tile pixel coords -> (yaw, pitch) in degrees."""
    f = tile.focal
    x = (u - tile.size / 2) / f
    y = -(v - tile.size / 2) / f
    z = np.ones_like(x)
    p, w = np.radians(tile.pitch), np.radians(tile.yaw)
    y1 = y * np.cos(p) + z * np.sin(p)            # rotate around x (pitch)
    z1 = -y * np.sin(p) + z * np.cos(p)
    x2 = x * np.cos(w) + z1 * np.sin(w)           # rotate around y (yaw)
    z2 = -x * np.sin(w) + z1 * np.cos(w)
    yaw = np.degrees(np.arctan2(x2, z2))
    pitch = np.degrees(np.arctan2(y1, np.hypot(x2, z2)))
    return yaw, pitch


def tile_to_sphere(tile: Tile, u, v):
    return _rays(tile, np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))


class TileSampler:
    """Precomputes cv2.remap grids once per (tile, frame size) so cutting tiles costs a few ms."""

    def __init__(self, tiles: list[Tile], width: int, height: int):
        self.tiles, self.maps = tiles, []
        for t in tiles:
            uu, vv = np.meshgrid(np.arange(t.size) + 0.5, np.arange(t.size) + 0.5)
            yaw, pitch = _rays(t, uu, vv)
            mx = ((yaw + 180) / 360 * width - 0.5) % width
            my = np.clip((90 - pitch) / 180 * height - 0.5, 0, height - 1)
            self.maps.append(cv2.convertMaps(mx.astype(np.float32), my.astype(np.float32), cv2.CV_16SC2))

    def cut(self, equi: np.ndarray) -> list[np.ndarray]:
        return [cv2.remap(equi, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) for m1, m2 in self.maps]


def box_to_sphere(tile: Tile, x1, y1, x2, y2) -> dict:
    """Tile-pixel box -> spherical extent. Foot point = bottom centre (used for ground distance later)."""
    xc = (x1 + x2) / 2
    us = np.array([xc, xc, x1, x2, xc])
    vs = np.array([(y1 + y2) / 2, y2, (y1 + y2) / 2, (y1 + y2) / 2, y1])
    yaw, pitch = tile_to_sphere(tile, us, vs)
    return {
        "yaw": float(yaw[0]), "pitch": float(pitch[0]),
        "yaw_l": float(yaw[2]), "yaw_r": float(yaw[3]),
        "pitch_top": float(pitch[4]), "pitch_bot": float(pitch[1]), "yaw_foot": float(yaw[1]),
    }


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0
