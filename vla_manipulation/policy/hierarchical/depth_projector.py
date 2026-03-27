"""Pixel + depth → 3D point in robot base frame. Canonical location: vla_manipulation/policy/hierarchical/depth_projector.py"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from vla_manipulation.policy.hierarchical.config import CameraIntrinsics, CameraExtrinsics

log = logging.getLogger(__name__)


class DepthLookupUnreliable(Exception):
    """Raised when depth sampling finds no reliable measurement at a pixel."""


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


class DepthProjector:
    """Back-projects pixels to 3D robot-frame positions using a depth map."""

    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics):
        self.K    = intrinsics
        self.ext  = extrinsics

    @classmethod
    def from_mujoco_fov(
        cls,
        fov_deg:    float,
        width:      int,
        height:     int,
        T_cam2base: np.ndarray,
    ) -> "DepthProjector":
        """Build from MuJoCo vertical FOV (degrees) and image dimensions."""
        fov_rad = np.deg2rad(fov_deg)
        fy = (height / 2.0) / np.tan(fov_rad / 2.0)
        fx = fy  # MuJoCo cameras are square-pixel
        cx = width  / 2.0
        cy = height / 2.0
        intrinsics  = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
        extrinsics  = CameraExtrinsics(T_cam2base=T_cam2base)
        return cls(intrinsics, extrinsics)

    def project(self, pixel: Tuple[int, int], depth_map: np.ndarray) -> Point3D:
        """Project one pixel to a 3D point in robot base frame."""
        u, v   = int(pixel[0]), int(pixel[1])
        h, w   = depth_map.shape[:2]
        # Sample a 3×3 patch for robustness
        u0, u1 = max(0, u - 1), min(w, u + 2)
        v0, v1 = max(0, v - 1), min(h, v + 2)
        patch  = depth_map[v0:v1, u0:u1]
        valid  = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            raise DepthLookupUnreliable(f"No valid depth at pixel ({u}, {v})")
        d = float(np.median(valid))

        # Back-project to camera frame
        x_c = (u - self.K.cx) * d / self.K.fx
        y_c = (v - self.K.cy) * d / self.K.fy
        p_cam = np.array([x_c, y_c, d, 1.0], dtype=np.float64)

        # Transform to robot base frame
        p_base = self.ext.T_cam2base @ p_cam
        return Point3D(x=float(p_base[0]), y=float(p_base[1]), z=float(p_base[2]))

    def project_batch(
        self,
        pixels:    List[Tuple[int, int]],
        depth_map: np.ndarray,
    ) -> List[Point3D]:
        results = []
        for px in pixels:
            try:
                results.append(self.project(px, depth_map))
            except DepthLookupUnreliable as e:
                log.warning("DepthLookupUnreliable: %s — using NaN point", e)
                results.append(Point3D(x=float('nan'), y=float('nan'), z=float('nan')))
        return results
