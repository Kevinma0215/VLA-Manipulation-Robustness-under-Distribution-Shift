"""
experiments/patched_env.py

Subclass of SimpleEnv2 that exposes a clean reset interface for EE position
shift experiments. Does NOT modify any existing project files.
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import random
import copy

import numpy as np

from robot.sim_env import SimpleEnv2
from env.ik import solve_ik
from env.transforms import rpy2r

# ── constants ─────────────────────────────────────────────────────────────────
_JOINT_NAMES  = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
_EE_NOMINAL_P = np.array([0.30, 0.0, 1.0])          # from y_env2.reset() line 65
_R_TARGET     = rpy2r(np.deg2rad([90, -0., 90]))     # from y_env2.reset() line 66
_IK_ERR_TH    = 0.01                                  # validated: all conditions < 0.009
_X_TOL        = 0.01                                  # 10 mm tolerance on EE x


class PatchedEnv(SimpleEnv2):
    """
    SimpleEnv2 with a clean reset interface for EE position shift experiments.

    The only public additions are:
        reset_with_offset(ee_offset_x, seed)
        get_ee_position() -> np.ndarray
    """

    def reset_with_offset(self, ee_offset_x: float = 0.0, seed: int = 0):
        """
        Reset the environment with the EE starting at x = 0.30 + ee_offset_x.

        Nominal x = 0.30.  Supported offsets: 0.0, -0.05, -0.10, -0.15
        (mapping to x = 0.30, 0.25, 0.20, 0.15).

        Steps:
          1. Seed Python random for reproducible instruction selection.
          2. Call parent reset(seed) — sets object poses, moves arm to nominal
             [0.3, 0.0, 1.0] via IK, runs 100-step warmup.
          3. Solve IK for shifted target p_trgt = [0.30+ee_offset_x, 0.0, 1.0].
          4. Apply via env.forward() to overwrite qpos without a physics step.
          5. Update self.q / self.last_q / self.p0 / self.R0 so subsequent
             step_env() calls hold the new pose.
          6. Run 100-step warmup at the new position to let physics settle.
          7. Verify IK error < 0.01 and actual EE x within 10 mm of target;
             print a warning (but do not raise) if either check fails.

        Args:
            ee_offset_x: x-axis shift in metres relative to nominal (0.30 m).
                         Use negative values to move toward the robot.
            seed:        Random seed for object layout and instruction selection.
        """
        # Step 1 — seed Python random so instruction (red/blue) is reproducible
        random.seed(seed)

        # Step 2 — parent reset: object placement, nominal IK, 100-step warmup
        super().reset(seed=seed)

        # If no shift requested, we are done (parent already solved nominal IK)
        if ee_offset_x == 0.0:
            return

        # Step 3 — build shifted target and solve IK
        p_trgt = _EE_NOMINAL_P.copy()
        p_trgt[0] += ee_offset_x

        q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
        q_solved, ik_err_stack, _ = solve_ik(
            env=self.env,
            joint_names_for_ik=_JOINT_NAMES,
            body_name_trgt='tcp_link',
            q_init=q_init,
            p_trgt=p_trgt,
            R_trgt=_R_TARGET,
            verbose_warning=False,
        )
        ik_err = np.linalg.norm(ik_err_stack)

        # Step 4 — overwrite qpos via forward kinematics (no physics step)
        self.env.forward(q=q_solved, joint_names=_JOINT_NAMES, increase_tick=False)

        # Step 5 — update SimpleEnv2 state variables so step_env() holds pose
        self.last_q = copy.deepcopy(q_solved)
        self.q      = np.concatenate([q_solved, np.array([0.0] * 4)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')

        # Step 6 — warmup at new position
        for _ in range(100):
            self.step_env()

        # Step 7 — verify
        p_actual, _ = self.env.get_pR_body(body_name='tcp_link')
        x_err = abs(p_actual[0] - p_trgt[0])

        if ik_err >= _IK_ERR_TH:
            print(
                f"[PatchedEnv WARNING] IK error {ik_err:.4f} >= {_IK_ERR_TH} "
                f"for p_trgt={np.round(p_trgt, 3)}"
            )
        if x_err > _X_TOL:
            print(
                f"[PatchedEnv WARNING] EE x error {x_err * 1000:.1f} mm > 10 mm "
                f"(target={p_trgt[0]:.3f}, actual={p_actual[0]:.4f})"
            )

    def get_ee_position(self) -> np.ndarray:
        """Return current EE xyz position (tcp_link body, world frame)."""
        p, _ = self.env.get_pR_body(body_name='tcp_link')
        return p.copy()
