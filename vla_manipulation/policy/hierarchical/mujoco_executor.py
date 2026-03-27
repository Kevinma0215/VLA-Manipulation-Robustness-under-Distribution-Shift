"""IK + PID executor for MuJoCo. Canonical location: vla_manipulation/policy/hierarchical/mujoco_executor.py"""
from __future__ import annotations
import time
import logging
from typing import List

import numpy as np

from vla_manipulation.envs.ik import solve_ik
from vla_manipulation.envs.transforms import rpy2r
from vla_manipulation.policy.hierarchical.pid_controller import CartesianPIDController
from vla_manipulation.policy.hierarchical.config import PIDGains
from vla_manipulation.policy.hierarchical.trajectory_builder import TrajectoryPoint

log = logging.getLogger(__name__)

JOINT_NAMES     = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
R_EE_TARGET     = rpy2r(np.deg2rad([90, -0., 90]))
DT              = 1.0 / 20.0      # 20 Hz control
MAX_IK_ERR      = 0.05


class MuJoCoExecutor:
    """Executes a dense trajectory in MuJoCo via damped least-squares IK + Cartesian PID."""

    def __init__(self, env, gains: PIDGains | None = None):
        self.env  = env
        self.pid  = CartesianPIDController(gains)

    def execute(self, trajectory: List[TrajectoryPoint]) -> None:
        """Blocking: drives env through each TrajectoryPoint in sequence."""
        self.pid.reset()
        for pt in trajectory:
            if not self.env.env.is_viewer_alive():
                break
            target = np.array([pt.x, pt.y, pt.z])
            self._move_to(target, pt.gripper)

    def _move_to(self, target: np.ndarray, gripper: float) -> None:
        q_init = self.env.last_q if hasattr(self.env, 'last_q') else np.zeros(6)
        q_solved, _, _ = solve_ik(
            env             = self.env.env,
            joint_names_for_ik = JOINT_NAMES,
            body_name_trgt  = 'tcp_link',
            q_init          = q_init,
            p_trgt          = target,
            R_trgt          = R_EE_TARGET,
            verbose_warning = False,
        )
        action = np.concatenate([q_solved, [gripper]])
        self.env.step(action)
        # Advance physics one step
        self.env.step_env()
        self.env.env.render()
