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
python setup_viewer.py

# Run batch evaluation (appends to experiments/results.csv, auto-resumes)
python experiments/eval_runner.py --condition all
python experiments/eval_runner.py --condition nominal   # or mild / medium / strong

# Generate analysis plots from results.csv
python experiments/analysis.py

# Regenerate the 4×3 result grid GIF (edit SEED_SELECTION at top of file first)
python experiments/make_grid_gif.py
```

## Architecture

The codebase has four layers, each with a strict direction of dependency (lower layers never import higher ones):

```
env/          ← pure MuJoCo wrapper — do not modify
robot/        ← task env built on top of env/
policy/       ← policy loading, no env dependency
experiments/  ← experiment logic built on robot/ + policy/
```

**`env/`** — `MuJoCoParserClass` in `mujoco_parser.py` is the only MuJoCo interface. Everything else (`ik.py`, `transforms.py`, `utils.py`) is stateless math. `get_pR_body('tcp_link')` is the canonical way to read EE pose.

**`robot/sim_env.py`** — `SimpleEnv2` wraps `MuJoCoParserClass`. Key state: `self.q` (10-dim: 6 joints + 4 gripper), `self.p0`/`self.R0` (EE target for IK mode), `self.last_q`. The main loop pattern is `step_env()` at full physics rate, gated by `env.env.loop_every(HZ=20)` for policy steps. `grab_image()` must be called before `render()` — it populates `self.rgb_agent` and `self.rgb_ego` which `render()` reads. **Do not call `SimpleEnv2.render()` before `grab_image()`** — use `env.env.render()` directly if needed (see `setup_viewer.py`).

**`experiments/patched_env.py`** — `PatchedEnv(SimpleEnv2)` adds `reset_with_offset(ee_offset_x, seed)`. This is the only place EE x-shift is applied — it solves IK for the shifted target and overwrites `qpos` via `env.forward()` without touching `sim_env.py`.

**`policy/smolvla.py`** — `load_policy(device)` loads `Jeongeun/omy_pnp_smolvla` from HuggingFace. Metadata is read from `demo_data_example/` (falls back to `omy_pnp_language/`). Policy expects observations keyed as `observation.state` (6-dim joints), `observation.image` (agentview), `observation.wrist_image` (egocentric), `task` (list of strings). Images must be 256×256 float tensors via `get_img_transform()`.

**`experiments/eval_runner.py`** — batch runner. Config constants (`N_EPISODES`, `CONDITIONS`, `MAX_WALL_SEC`, etc.) are at the top of the file. Results append to `experiments/results.csv` after every episode; re-running skips already-completed seeds automatically.

## Key Constraints

- **Viewer required**: `get_fixed_cam_rgb()` uses `self.viewer.ctx` — no headless path exists.
- **`action_type='joint_angle'`** is used for eval (7-dim: 6 joints + gripper). Action `[-1]` is gripper (0=open, 1=closed); internally fanned out to 4 gripper joints with `gripper_cmd[[1,3]] *= 0.8`.
- **IK z-overshoot ~5 mm** is consistent and expected across all conditions.
- The scene XML is `asset/example_scene_y2.xml`. `example_scene_y.xml` is unused.
