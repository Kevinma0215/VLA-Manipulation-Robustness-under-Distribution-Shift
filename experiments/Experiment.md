# Experiment Manual — SmolVLA Monolithic VLA Baseline

## Overview

- **Policy**: pretrained SmolVLA (`Jeongeun/omy_pnp_smolvla`)
- **Task**: pick-and-place (mug → plate)
- **Main variable**: initial end-effector (EE) position shift along x-axis
- **Fixed**: base, objects (mug_5, mug_6), plate, all cameras
- **EE orientation**: always fixed at `rpy = [90, 0, 90] deg`

---

## Project File Map

| File | Purpose |
|---|---|
| `mujoco_env/y_env2.py` | Core simulation env (`SimpleEnv2`) — do not modify |
| `mujoco_env/mujoco_parser.py` | MuJoCo wrapper (viewer, FK, cameras) — do not modify |
| `asset/example_scene_y2.xml` | MuJoCo scene (robot + objects + cameras) |
| `experiments/patched_env.py` | `PatchedEnv` subclass — adds `reset_with_offset()` |
| `experiments/eval_runner.py` | Batch evaluation runner across all conditions |
| `experiments/analysis.py` | Post-hoc plots from `experiments/results.csv` |
| `setup_viewer.py` | Interactive viewer for checking EE positions |
| `Experiment.md` | This document |

---

## Running Commands

### Activate env
```bash
conda activate lerobot-mujoco-tutorial
```

### Interactive viewer (check EE positions before running)

```bash
 python setup_viewer.py
```

Keyboard controls:
- `1` / `2` / `3` / `4` — switch condition: nominal / mild / medium / strong
- `R` — reset current condition
- `ESC` — quit

### Batch evaluation

```bash
# Run all 4 conditions × 20 episodes
python experiments/eval_runner.py --condition all

# Run a single condition
python experiments/eval_runner.py --condition nominal
python experiments/eval_runner.py --condition mild
python experiments/eval_runner.py --condition medium
python experiments/eval_runner.py --condition strong
```

Results append to `experiments/results.csv` after every episode. If interrupted, re-running automatically resumes from where it left off (skips already-completed seeds).

### Analysis plots

```bash
python experiments/analysis.py
```
For video record:
```bash
# Review all failures for one condition
ls experiments/media/videos/strong_*fail*

# Review all successes
ls experiments/media/videos/*success*

# Play any episode
vlc experiments/media/videos/nominal_seed00_success.mp4
```

Outputs four PNG files to `experiments/media/`:
- `success_rate.png` — bar chart with Wilson 95% CI, nominal reference line
- `failure_breakdown.png` — stacked bar: no_grasp / drop / wrong_place
- `episode_length.png` — box + strip plot per condition
- `degradation_summary.png` — 2-panel hero figure (success rate + failure breakdown)

---

## Outputs

| Path | Contents |
|---|---|
| `experiments/results.csv` | One row per episode (condition, seed, success, length, EE pos, failure type, timestamp) |
| `experiments/media/videos/` | MP4 per episode: `{condition}_seed{NN}_{outcome}.mp4` |
| `experiments/media/*.png` | Analysis plots |

### Video filename convention

`{condition}_seed{seed:02d}_{outcome}.mp4`

Outcome values: `success`, `fail_no_grasp`, `fail_drop`, `fail_wrong_place`

Example: `mild_seed03_fail_no_grasp.mp4`

Each video is a 512×256 side-by-side composite (agentview left, egocentric right) at 20 fps with a HUD line showing condition, seed, step, and EE x position.

---

## Experiment Configuration (eval_runner.py)

| Parameter | Value | Notes |
|---|---|---|
| `N_EPISODES` | 20 | Episodes per condition |
| `SEEDS` | 0–19 | Deterministic per seed |
| `MAX_WALL_SEC` | 60.0 s | Primary timeout guard (wall clock) |
| `MAX_STEPS` | 500 | Secondary timeout guard (policy steps) |
| `DEVICE` | `cuda` | GPU inference |
| `POLICY_HUB` | `Jeongeun/omy_pnp_smolvla` | HuggingFace model |
| `MUG_LIFT_Z` | 0.86 m | Mug must exceed this to count as "lifted" |
| `MUG_DROP_Z` | 0.83 m | Mug below this after lift = drop detected |

---

## Shift Conditions

| Condition | `ee_offset_x` | `p_trgt` | Actual EE pos | Pos error | IK error |
|---|---|---|---|---|---|
| nominal | +0.00 | `[0.30, 0.00, 1.00]` | `[0.3011, -0.0001, 1.0040]` | 4.1 mm | 0.0070 |
| mild    | -0.05 | `[0.25, 0.00, 1.00]` | `[0.2507, -0.0004, 1.0049]` | 4.9 mm | 0.0080 |
| medium  | -0.10 | `[0.20, 0.00, 1.00]` | `[0.2006, -0.0012, 1.0048]` | 5.0 mm | 0.0081 |
| strong  | -0.15 | `[0.15, 0.00, 1.00]` | `[0.1531, -0.0030, 1.0016]` | 4.6 mm | 0.0072 |

All four conditions passed IK convergence (error < 5 mm) and physical validity checks (validated 2026-03-13).

> **Note**: IK consistently overshoots z by ~4–5 mm due to `ik_err_th=1e-2`. Acceptable for this task.

---

## Failure Classification

Failure types are behavioral — time limits are loop guards only, not failure modes.

| Type | Condition |
|---|---|
| `no_grasp` | Gripper never closed during episode |
| `drop` | Gripper closed + mug lifted (z > 0.86), then fell (z < 0.83) |
| `wrong_place` | Gripper closed, no drop — approached but failed to place |

Success criterion (`check_success()`): mug within 0.1 m of plate XY, gripper open (< 0.1), TCP z > 0.9 m.

---

## Environment Entry Points

| Component | File | Location |
|---|---|---|
| Env class | `mujoco_env/y_env2.py` | `class SimpleEnv2` |
| Reset | `mujoco_env/y_env2.py` | `reset()` line 53 |
| Initial arm pose (IK) | `mujoco_env/y_env2.py` | line 60–67 |
| Object positions | `mujoco_env/y_env2.py` | line 71–95 |
| EE pose query | `mujoco_env/mujoco_parser.py` | `get_pR_body('tcp_link')` |
| Observation | `mujoco_env/y_env2.py` | `grab_image()` + `get_joint_state()` |
| Patched reset with offset | `experiments/patched_env.py` | `PatchedEnv.reset_with_offset()` |

---

## Object and Scene Positions

Fixed every reset (validated: drift = 0.0 across 3 resets).

| Body | Position (x, y, z) |
|---|---|
| `body_obj_mug_5` (red) | `[0.325, 0.010, 0.838]` |
| `body_obj_mug_6` (blue) | `[0.295, 0.200, 0.838]` |
| `body_obj_plate_11` | `[0.300, -0.250, 0.818]` |
| Cameras | XML-baked, never drift |

Cameras available: `agentview`, `topview`, `sideview`, `egocentric`.

---

## Pre-Experiment Validation Checklist

Run before any experiment session. All checks were validated on 2026-03-13.

| Check | What | Status |
|---|---|---|
| A | IK converges to all shift targets (< 5 mm error) | PASS |
| B | EE orientation fixed across shifts (< 0.1 deg) | PASS |
| C | All scene elements (objects, plate, cameras) fixed across resets | PASS |
| D | No EE–object overlap, joints within limits, gripper open | PASS |
| E | `get_qpos_joints()` matches `q_solved`, FK is consistent | PASS |
| F | Robot stable for 100 hold steps; mug settles ~15 mm (normal) | PASS |

**Camera observation check**: requires a live GLFW viewer context. Cannot be run headlessly. Validate visually during the first rollout.

Headless validation script:

```bash
python /tmp/test_mujoco_env_validation.py
```

Re-run after any changes to `y_env2.py` or `example_scene_y2.xml`.

---

## Known Behaviors and Caveats

1. **Mug falls ~15 mm at each reset** — free-floating bodies settle onto the table. The 100-step warmup in `reset()` handles this. The first observation after `reset()` is already post-settle.

2. **IK z-overshoot of ~5 mm** — consistent across all conditions, not a bug. Acceptable for this task.

3. **`grab_image()` requires a live viewer** — `get_fixed_cam_rgb()` uses `self.viewer.ctx`. No headless image path. Always run with the viewer open.

4. **`p_trgt` has no public parameter in `SimpleEnv2`** — shifting the EE uses `PatchedEnv.reset_with_offset()`, which solves IK and overwrites qpos without touching `y_env2.py`.

5. **Object randomization is quasi-fixed** — `sample_xyzs` ranges are `±0.01 m`, so positions are effectively deterministic per seed.

6. **SmolVLA inference is ~3–5 Hz** — the simulation loop runs at 500 Hz internally; the policy runs at 20 Hz. Wall-clock timeout (60 s) is the primary episode guard.

7. **`setup_viewer.py` calls `env.env.render()`** — uses `MuJoCoParserClass.render()` directly (not `SimpleEnv2.render()`) to avoid `AttributeError: rgb_ego` on the first frame before `grab_image()` is called.
