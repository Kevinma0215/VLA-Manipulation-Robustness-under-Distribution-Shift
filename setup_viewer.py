"""
Interactive viewer for testing initial EE position setup.

Usage:
    conda run -n lerobot-mujoco-tutorial python setup_viewer.py

Keyboard controls (when viewer window is focused):
    1 / 2 / 3 / 4  — switch condition: nominal / mild / medium / strong
    R               — reset current condition
    ESC             — quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
import glfw
from mujoco_env.y_env2 import SimpleEnv2
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to test different conditions
# ──────────────────────────────────────────────────────────────────────────────
XML_PATH = './asset/example_scene_y2.xml'
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
R_TARGET = rpy2r(np.deg2rad([90, -0., 90]))

CONDITIONS = {
    '1_nominal': np.array([0.30, 0.00, 1.00]),
    '2_mild':    np.array([0.25, 0.00, 1.00]),
    '3_medium':  np.array([0.20, 0.00, 1.00]),
    '4_strong':  np.array([0.15, 0.00, 1.00]),
}
CONDITION_KEYS = list(CONDITIONS.keys())
SEED = 0
# ──────────────────────────────────────────────────────────────────────────────


def apply_condition(env, p_trgt):
    """Solve IK and hold the arm at p_trgt."""
    q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
    q_solved, ik_err_stack, _ = solve_ik(
        env=env.env,
        joint_names_for_ik=JOINT_NAMES,
        body_name_trgt='tcp_link',
        q_init=q_init,
        p_trgt=p_trgt,
        R_trgt=R_TARGET,
        verbose_warning=False,
    )
    ik_err = np.linalg.norm(ik_err_stack)
    env.env.forward(q=q_solved, joint_names=JOINT_NAMES, increase_tick=False)
    env.last_q = copy.deepcopy(q_solved)
    env.q = np.concatenate([q_solved, np.array([0.0] * 4)])
    return q_solved, ik_err


def overlay(env, loc, text1, text2):
    env.env.viewer_text_overlay(loc=loc, text1=text1, text2=text2)


def main():
    print("Initializing environment...")
    env = SimpleEnv2(XML_PATH, action_type='joint_angle')
    env.reset(seed=SEED)

    current_idx = 0
    condition_name = CONDITION_KEYS[current_idx]
    p_trgt = CONDITIONS[condition_name]
    q_solved, ik_err = apply_condition(env, p_trgt)
    print(f"Starting with condition: {condition_name}  p_trgt={p_trgt}  IK_err={ik_err:.4f}")
    print(__doc__)

    KEY_MAP = {
        glfw.KEY_1: 0,
        glfw.KEY_2: 1,
        glfw.KEY_3: 2,
        glfw.KEY_4: 3,
    }

    while env.env.is_viewer_alive():
        env.step_env()

        if env.env.loop_every(HZ=20):

            # ── keyboard handling ───────────────────────────────────────────
            for key, idx in KEY_MAP.items():
                if env.env.is_key_pressed_once(key=key):
                    current_idx = idx
                    condition_name = CONDITION_KEYS[current_idx]
                    p_trgt = CONDITIONS[condition_name]
                    env.reset(seed=SEED)
                    q_solved, ik_err = apply_condition(env, p_trgt)
                    print(f"Switched → {condition_name}  p_trgt={p_trgt}  IK_err={ik_err:.4f}")

            if env.env.is_key_pressed_once(key=glfw.KEY_R):
                env.reset(seed=SEED)
                q_solved, ik_err = apply_condition(env, p_trgt)
                print(f"Reset → {condition_name}  IK_err={ik_err:.4f}")

            # ── query state ─────────────────────────────────────────────────
            p_ee, R_ee = env.env.get_pR_body(body_name='tcp_link')
            rpy_ee     = r2rpy(R_ee, unit='deg')
            p_mug5     = env.env.get_p_body('body_obj_mug_5')
            p_mug6     = env.env.get_p_body('body_obj_mug_6')
            p_plate    = env.env.get_p_body('body_obj_plate_11')
            gripper    = env.env.get_qpos_joint('rh_r1')[0]

            # ── visual markers ──────────────────────────────────────────────
            env.env.plot_sphere(p=p_trgt, r=0.015, rgba=[0.1, 0.3, 1.0, 0.8], label='target')
            env.env.plot_sphere(p=p_ee,   r=0.012, rgba=[1.0, 0.2, 0.1, 0.9], label='EE')
            env.env.plot_T(p=p_ee, R=R_ee, axis_len=0.07, axis_width=0.005,
                           plot_axis=True, plot_sphere=False)
            env.env.plot_time()

            # ── text overlays (one call per line) ───────────────────────────
            overlay(env, 'top left',
                    'Keys: [1]nominal [2]mild [3]medium [4]strong',
                    '[R]reset  [ESC]quit')
            overlay(env, 'bottom left',
                    f'Condition: {condition_name}',
                    f'p_trgt={np.round(p_trgt,3)}')
            overlay(env, 'bottom',
                    f'p_ee={np.round(p_ee,4)}  rpy={np.round(rpy_ee,1)}deg  IK_err={ik_err:.4f}',
                    f'mug5={np.round(p_mug5,3)}  mug6={np.round(p_mug6,3)}  plate={np.round(p_plate,3)}  gripper={"OPEN" if gripper<0.1 else "CLOSED"}')

            # use env.env.render() directly — avoids rgb_ego AttributeError
            env.env.render()


if __name__ == '__main__':
    main()
