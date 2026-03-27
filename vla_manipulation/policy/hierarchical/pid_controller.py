"""Cartesian PID controller. Canonical location: vla_manipulation/policy/hierarchical/pid_controller.py"""
from __future__ import annotations
import numpy as np
from vla_manipulation.policy.hierarchical.config import PIDGains


class CartesianPIDController:
    """Simple proportional-integral-derivative controller in Cartesian space."""

    def __init__(self, gains: PIDGains | None = None):
        self.gains = gains or PIDGains()
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def reset(self):
        self._integral[:] = 0.0
        self._prev_error[:] = 0.0

    def step(self, error: np.ndarray, dt: float) -> np.ndarray:
        self._integral += error * dt
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error.copy()
        vel = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )
        norm = np.linalg.norm(vel)
        if norm > self.gains.max_vel:
            vel = vel * self.gains.max_vel / norm
        return vel
