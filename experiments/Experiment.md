# Experiment Technical Reference

> For setup, commands, and repo structure see [README.md](../README.md).
> This document covers: configuration values, scene geometry, failure logic, code entry points, validation, and caveats.

---

## Experiment Configuration

Constants live at the top of [eval_runner.py](eval_runner.py).

| Parameter | Value | Notes |
|---|---|---|
| `N_EPISODES` | 20 | Episodes per condition |
| `SEEDS` | 0–19 | One seed per episode, deterministic |
| `MAX_WALL_SEC` | 60.0 s | Primary timeout (wall clock) |
| `MAX_STEPS` | 500 | Secondary timeout (policy steps) |
| `DEVICE` | `cuda` | GPU inference |
| `MUG_LIFT_Z` | 0.86 m | Mug must exceed this to count as "lifted" |
| `MUG_DROP_Z` | 0.83 m | Mug below this after lift = drop detected |

---

## Shift Conditions (Validated 2026-03-13)

| Condition | `ee_offset_x` | `p_trgt` | Actual EE pos | Pos error | IK error |
|---|---|---|---|---|---|
| nominal | +0.00 | `[0.30, 0.00, 1.00]` | `[0.3011, -0.0001, 1.0040]` | 4.1 mm | 0.0070 |
| mild    | -0.05 | `[0.25, 0.00, 1.00]` | `[0.2507, -0.0004, 1.0049]` | 4.9 mm | 0.0080 |
| medium  | -0.10 | `[0.20, 0.00, 1.00]` | `[0.2006, -0.0012, 1.0048]` | 5.0 mm | 0.0081 |
| strong  | -0.15 | `[0.15, 0.00, 1.00]` | `[0.1531, -0.0030, 1.0016]` | 4.6 mm | 0.0072 |

All four conditions passed IK convergence (error < 5 mm) and physical validity checks.

EE orientation is fixed at `rpy = [90, 0, 90] deg` across all conditions.

> IK consistently overshoots z by ~4–5 mm due to `ik_err_th=1e-2`. Acceptable for this task.

---

## Scene Geometry

Fixed every reset (validated: drift = 0.0 across 3 resets).

| Body | Position (x, y, z) |
|---|---|
| `body_obj_mug_5` (red, target) | `[0.325, 0.010, 0.838]` |
| `body_obj_mug_6` (blue, distractor) | `[0.295, 0.200, 0.838]` |
| `body_obj_plate_11` | `[0.300, -0.250, 0.818]` |
| Cameras | XML-baked, never drift |

Cameras: `agentview`, `topview`, `sideview`, `egocentric`.

---

## Failure Classification

Failure types are **behavioral** — timeouts are loop guards, not failure modes.

| Type | Condition |
|---|---|
| `no_grasp` | Gripper never closed during episode |
| `drop` | Gripper closed + mug lifted (`z > MUG_LIFT_Z`), then fell (`z < MUG_DROP_Z`) |
| `wrong_place` | Gripper closed, no drop — approached but failed to place |

**Success criterion** (`check_success()`): mug within 0.1 m of plate XY, gripper open (`rh_r1 < 0.1`), TCP z > 0.9 m.

---

## Video Format

Filename: `{condition}_seed{seed:02d}_{outcome}.mp4`

Outcome values: `success`, `fail_no_grasp`, `fail_drop`, `fail_wrong_place`

Each video is a **512×256 side-by-side composite** (agentview left, egocentric right) at 20 fps with a HUD line: `[condition] seed=NN  step=NNN  ee_x=N.NNN`.

---

## Code Entry Points

| Component | File | Symbol |
|---|---|---|
| Env class | [robot/sim_env.py](../robot/sim_env.py) | `class SimpleEnv2` |
| Reset + IK | [robot/sim_env.py](../robot/sim_env.py) | `reset()` |
| EE offset reset | [experiments/patched_env.py](patched_env.py) | `PatchedEnv.reset_with_offset()` |
| Camera images | [robot/sim_env.py](../robot/sim_env.py) | `grab_image()` |
| Joint state | [robot/sim_env.py](../robot/sim_env.py) | `get_joint_state()` |
| EE position | [env/mujoco_parser.py](../env/mujoco_parser.py) | `get_pR_body('tcp_link')` |
| Policy loading | [policy/smolvla.py](../policy/smolvla.py) | `load_policy()` |
| Failure logic | [experiments/eval_runner.py](eval_runner.py) | `classify_failure()` |
| Success logic | [robot/sim_env.py](../robot/sim_env.py) | `check_success()` |

---

## Pre-Experiment Validation Checklist

Run before any new experiment session. Re-run after any change to `robot/sim_env.py` or `asset/example_scene_y2.xml`.

| Check | What | Status |
|---|---|---|
| A | IK converges to all shift targets (< 5 mm error) | PASS |
| B | EE orientation fixed across shifts (< 0.1 deg) | PASS |
| C | Scene elements (objects, plate, cameras) fixed across resets | PASS |
| D | No EE–object overlap, joints within limits, gripper open | PASS |
| E | `get_qpos_joints()` matches `q_solved`, FK consistent | PASS |
| F | Robot stable for 100 hold steps; mug settles ~15 mm (normal) | PASS |

Camera check requires a live GLFW viewer — validate visually on the first rollout.

---

## Known Behaviors and Caveats

1. **Mug settles ~15 mm at reset** — free-floating bodies drop onto the table. The 100-step warmup in `reset()` handles this; the first observation is already post-settle.

2. **IK z-overshoot of ~5 mm** — consistent across all conditions, not a bug.

3. **`grab_image()` requires a live viewer** — `get_fixed_cam_rgb()` uses `self.viewer.ctx`. No headless path. Always run with the viewer open.

4. **`p_trgt` is not a parameter of `SimpleEnv2`** — EE shifting is done entirely in `PatchedEnv.reset_with_offset()`, which overwrites qpos via `env.forward()` without modifying `sim_env.py`.

5. **Object positions are quasi-fixed** — `sample_xyzs` ranges are `±0.01 m`, so positions are effectively deterministic per seed.

6. **SmolVLA runs at ~3–5 Hz** — the simulation runs at 500 Hz internally; policy is gated at 20 Hz via `loop_every()`. Wall-clock timeout (60 s) is the primary episode guard.

7. **`setup_viewer.py` calls `env.env.render()` directly** — bypasses `SimpleEnv2.render()` to avoid `AttributeError: rgb_ego` on the first frame before `grab_image()` is called.
