"""EpisodeResult dataclass. Canonical location: vla_manipulation/common/types.py"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EpisodeResult:
    condition:      str
    seed:           int
    success:        bool
    episode_length: int
    ee_x_initial:   float
    failure_type:   str
    policy:         str
    timestamp:      str
