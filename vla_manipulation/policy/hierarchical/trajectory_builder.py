"""Waypoints → dense interpolated trajectory. Canonical location: vla_manipulation/policy/hierarchical/trajectory_builder.py"""
from __future__ import annotations
import logging
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

from vla_manipulation.policy.hierarchical.config import ActionType, ActionOffsets
from vla_manipulation.policy.hierarchical.gemini_planner import SemanticWaypoint
from vla_manipulation.policy.hierarchical.depth_projector import Point3D

log = logging.getLogger(__name__)

# vla_robot sibling repo (optional SO101 kinematics)
_VLA_ROBOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'vla_robot')
)
if os.path.isdir(_VLA_ROBOT) and _VLA_ROBOT not in sys.path:
    sys.path.insert(0, _VLA_ROBOT)


@dataclass
class TrajectoryPoint:
    x:       float
    y:       float
    z:       float
    gripper: float  # 0=open, 1=closed


class TrajectoryBuilder:
    """Converts semantic waypoints + 3D positions into a dense trajectory."""

    def __init__(self, offsets: ActionOffsets | None = None, steps_per_segment: int = 20):
        self.offsets = offsets or ActionOffsets()
        self.steps   = steps_per_segment

    def build(
        self,
        waypoints: List[SemanticWaypoint],
        positions: List[Point3D],
    ) -> List[TrajectoryPoint]:
        """Return a list of keyframe TrajectoryPoints."""
        keyframes: List[TrajectoryPoint] = []
        for wp, pos in zip(waypoints, positions):
            if np.isnan(pos.x):
                continue
            gripper = wp.gripper_state

            # Pre-approach lift for grasp/place
            if wp.action_type in (ActionType.GRASP, ActionType.PLACE):
                keyframes.append(TrajectoryPoint(
                    x=pos.x, y=pos.y,
                    z=pos.z + self.offsets.pre_grasp_z,
                    gripper=0.0,
                ))

            keyframes.append(TrajectoryPoint(x=pos.x, y=pos.y, z=pos.z, gripper=gripper))

            # Post-grasp lift
            if wp.action_type == ActionType.GRASP:
                keyframes.append(TrajectoryPoint(
                    x=pos.x, y=pos.y,
                    z=pos.z + self.offsets.post_grasp_z,
                    gripper=1.0,
                ))
        return keyframes

    def interpolate(self, keyframes: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """Linearly interpolate between keyframes."""
        if len(keyframes) < 2:
            return keyframes
        dense: List[TrajectoryPoint] = []
        for i in range(len(keyframes) - 1):
            a, b = keyframes[i], keyframes[i + 1]
            for t in np.linspace(0.0, 1.0, self.steps, endpoint=False):
                dense.append(TrajectoryPoint(
                    x       = a.x       + t * (b.x       - a.x),
                    y       = a.y       + t * (b.y       - a.y),
                    z       = a.z       + t * (b.z       - a.z),
                    gripper = a.gripper + t * (b.gripper  - a.gripper),
                ))
        dense.append(keyframes[-1])
        return dense
