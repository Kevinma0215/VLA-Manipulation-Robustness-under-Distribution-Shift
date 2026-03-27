"""Hierarchical policy configuration types. Canonical location: vla_manipulation/policy/hierarchical/config.py"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ActionType(str, Enum):
    MOVE   = "move"
    GRASP  = "grasp"
    PLACE  = "place"
    OPEN   = "open"


@dataclass
class ActionOffsets:
    pre_grasp_z:  float = 0.10
    post_grasp_z: float = 0.10
    pre_place_z:  float = 0.10


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width:  int
    height: int


@dataclass
class CameraExtrinsics:
    T_cam2base: np.ndarray  # (4,4) rigid transform


@dataclass
class PIDGains:
    kp: float = 2.0
    ki: float = 0.0
    kd: float = 0.05
    max_vel: float = 0.5
