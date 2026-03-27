# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
conda activate lerobot-mujoco-tutorial
```

All scripts must be run from the **repo root**. Scripts call `os.chdir(repo_root)` internally, but `asset/` paths in XML are relative to repo root.

## Common Commands

```bash
# Verify EE positions interactively before any eval run
python scripts/setup_viewer.py

# Run batch evaluation (appends to experiments/results.csv, auto-resumes)
python scripts/eval_runner.py --condition all
python scripts/eval_runner.py --condition nominal   # or mild / medium / strong

# Run hierarchical VLA evaluation (appends to experiments/results_hvla.csv)
python scripts/eval_hvla.py --condition all --api-key $GEMINI_KEY
python scripts/eval_hvla.py --condition nominal --api-key $GEMINI_KEY
python scripts/eval_hvla.py --dry-run   # skip Gemini + execution

# Generate analysis plots from results.csv (and results_hvla.csv if present)
python scripts/analysis.py

# Regenerate the 4×3 result grid GIF (edit SEED_SELECTION at top of file first)
python scripts/make_grid_gif.py
```

## Architecture

```
vla_manipulation/           ← installable package (pip install -e .)
  envs/                     ← MuJoCo layer (env + robot merged)
    mujoco_parser.py        ←   GLFW viewer, FK, camera capture
    ik.py                   ←   damped least-squares IK solver
    transforms.py           ←   SO(3)/SE(3) math
    utils.py                ←   misc helpers
    sim_env.py              ←   SimpleEnv2 tabletop environment
  simulation/
    patched_env.py          ←   PatchedEnv: reset_with_offset, get_depth
  common/
    types.py                ←   EpisodeResult dataclass
  evaluation/
    metrics.py              ←   check_success, classify_failure
    logger.py               ←   EpisodeLogger CSV writer
  policy/
    monolithic/
      smolvla.py            ←   SmolVLA policy loader
    hierarchical/
      config.py             ←   ActionType, CameraIntrinsics, PIDGains
      gemini_planner.py     ←   Gemini API → SemanticWaypoint list
      depth_projector.py    ←   pixel + depth → Point3D
      trajectory_builder.py ←   waypoints → dense trajectory
      pid_controller.py     ←   Cartesian PID
      mujoco_executor.py    ←   IK + PID → MuJoCo joint commands
scripts/                    ← thin entry points, no logic
  eval_runner.py            ←   monolithic batch evaluation
  eval_hvla.py              ←   hierarchical batch evaluation
  analysis.py               ←   post-hoc plots + comparison
  setup_viewer.py           ←   interactive EE position viewer
  make_grid_gif.py          ←   GIF grid assembler
asset/                      ← scene XML and mesh files
demo_data_example/          ← SmolVLA policy metadata
```

**`vla_manipulation/envs/mujoco_parser.py`** — `MuJoCoParserClass` is the only MuJoCo interface. Everything else (`ik.py`, `transforms.py`, `utils.py`) is stateless math. `get_pR_body('tcp_link')` is the canonical way to read EE pose.

**`vla_manipulation/envs/sim_env.py`** — `SimpleEnv2` wraps `MuJoCoParserClass`. Key state: `self.q` (10-dim: 6 joints + 4 gripper), `self.p0`/`self.R0` (EE target for IK mode), `self.last_q`. The main loop pattern is `step_env()` at full physics rate, gated by `env.env.loop_every(HZ=20)` for policy steps. `grab_image()` must be called before `render()` — it populates `self.rgb_agent` and `self.rgb_ego` which `render()` reads. **Do not call `SimpleEnv2.render()` before `grab_image()`** — use `env.env.render()` directly if needed (see `scripts/setup_viewer.py`).

**`vla_manipulation/simulation/patched_env.py`** — `PatchedEnv(SimpleEnv2)` adds `reset_with_offset(ee_offset_x, seed)`, `get_ee_position()`, and `get_depth(camera_name)`. This is the only place EE x-shift is applied — it solves IK for the shifted target and overwrites `qpos` via `env.forward()` without touching `sim_env.py`.

**`vla_manipulation/policy/monolithic/smolvla.py`** — `load_policy(device)` loads `Jeongeun/omy_pnp_smolvla` from HuggingFace. Metadata is read from `demo_data_example/` (falls back to `omy_pnp_language/`). Policy expects observations keyed as `observation.state` (6-dim joints), `observation.image` (agentview), `observation.wrist_image` (egocentric), `task` (list of strings). Images must be 256×256 float tensors via `get_img_transform()`.

**`scripts/eval_runner.py`** — monolithic (SmolVLA) batch runner. Config constants (`N_EPISODES`, `CONDITIONS`, `MAX_WALL_SEC`, etc.) are at the top of the file. Results append to `experiments/results.csv` after every episode; re-running skips already-completed seeds automatically.

**`scripts/eval_hvla.py`** — hierarchical VLA batch runner. Uses `GeminiPlanner → DepthProjector → TrajectoryBuilder → MuJoCoExecutor`. Results append to `experiments/results_hvla.csv`.

## Monolithic Path

**Runner**: `scripts/eval_runner.py`
**Policy**: SmolVLA (`vla_manipulation/policy/monolithic/smolvla.py`) — end-to-end joint-angle prediction
**Observation loop**: `grab_image()` → resize to 256×256 → `policy.select_action()` → `env.step(action)`
**Results CSV**: `experiments/results.csv`

The monolithic pipeline runs a standard RL-style action loop at 20 Hz, gated by `env.env.loop_every(HZ=20)`. Each policy step produces a 7-dim action (6 joints + gripper). Success is checked via `env.check_success()` before every policy step.

## Hierarchical Path

**Runner**: `scripts/eval_hvla.py`
**Requires**: `export GEMINI_KEY=your_key` or `--api-key` flag
**Gemini model**: `gemini-2.5-flash` (default in `GeminiPlanner`)
**Results CSV**: `experiments/results_hvla.csv`

Five-stage pipeline per episode:

```
RGB image + command
    │
    ▼  Stage 1 — GeminiPlanner
SemanticWaypoint list  (action_type, pixel_u, pixel_v, gripper_state)
    │
    ▼  Stage 2 — DepthProjector
Point3D list  (robot base frame, metres)
    │
    ▼  Stage 3 — TrajectoryBuilder
Dense TrajectoryPoint list  (interpolated Cartesian path)
    │
    ▼  Stage 4 — CartesianPIDController  (inside MuJoCoExecutor)
Cartesian velocity commands
    │
    ▼  Stage 5 — solve_ik() + mj_step()
MuJoCo joint positions
```

**Depth access**: `PatchedEnv.get_depth(camera_name='agentview')`
Uses MuJoCo offscreen renderer. NDC→metric conversion:
```
depth = near / (1 - depth_raw * (1 - near/far))
```
Pixels at the far clipping plane → `np.nan`.

**Known limitation**: `MuJoCoExecutor` uses damped least-squares IK (`vla_manipulation/envs/ik.py` `solve_ik()`). Near kinematic singularities at table height, XY drift ~10 cm observed during contact descent (paper Section 4.2.2).

## Analysis

```bash
# Generate plots (auto-detects results_hvla.csv if present)
python scripts/analysis.py
```

Outputs to `experiments/plots/`:

| File | Content |
|------|---------|
| `success_rate.png` | Monolithic success rate vs EE offset |
| `failure_breakdown.png` | Monolithic failure mode distribution |
| `episode_length.png` | Monolithic episode length distribution |
| `degradation_summary.png` | Monolithic degradation hero figure |
| `success_rate_comparison.png` | Both systems — written only if `results_hvla.csv` exists |

`load_data()` in `scripts/analysis.py` accepts `results_path` and `hvla_path` keyword args. When `hvla_path` is absent, all rows are tagged `policy='monolithic'` and the comparison plot is skipped.

## Key Constraints

- **Viewer required**: `get_fixed_cam_rgb()` uses `self.viewer.ctx` — no headless path exists.
- **`action_type='joint_angle'`** is used for eval (7-dim: 6 joints + gripper). Action `[-1]` is gripper (0=open, 1=closed); internally fanned out to 4 gripper joints with `gripper_cmd[[1,3]] *= 0.8`.
- **IK z-overshoot ~5 mm** is consistent and expected across all conditions.
- The scene XML is `asset/example_scene_y2.xml`. `example_scene_y.xml` is unused.

**C-NEW-1**: `vla_manipulation/` has no dependency on `scripts/`. Data flows one way: `scripts/` imports `vla_manipulation/`, never the reverse.

**C-NEW-2**: `get_depth()` must be called after `grab_image()` in the same step — both use `self.env.cams[cam_idx]` which requires the viewer context to be active.

**C-NEW-3**: `MuJoCoExecutor.execute()` is blocking. Do not call it from a thread that also needs to update the viewer — call `env.env.render()` from the main thread after each `mj_step`.
