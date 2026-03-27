# VLA Manipulation — Architecture

## 1. Overview

This project evaluates two robotic manipulation systems — a monolithic SmolVLA policy and a hierarchical Gemini+IK pipeline — on a tabletop pick-and-place task under systematic EE position shift conditions. Both systems operate in MuJoCo simulation on the same scene and are compared using shared evaluation infrastructure.

## 2. Design Principles

**Fair comparison**: Both pipelines share the same `PatchedEnv` reset logic, the same `CONDITIONS` table (four EE offset levels), the same success criteria (`metrics.py`), and the same CSV schema (`EpisodeLogger`). The only difference is the policy component being evaluated.

**Separation of concerns**: `vla_manipulation/` cleanly separates experiment-layer concerns (simulation extension, metrics, logging, policy components) from thin entry-point scripts that contain no logic of their own.

**Clean architecture**: Dependency direction is strictly one-way — `vla_manipulation/` never imports from `scripts/`. Within `vla_manipulation/`, lower subpackages (`envs/`) never import from higher ones (`evaluation/`, `policy/`). `scripts/` imports `vla_manipulation/`, never the reverse (C-NEW-1).

**Additive, not invasive**: `PatchedEnv` extends `SimpleEnv2` without modifying it. All experiment extensions live in `vla_manipulation/simulation/patched_env.py`. The monolithic eval continues to run identically after the extension.

## 3. Repository Structure

```
lerobot-mujoco-tutorial/
├── vla_manipulation/           # installable package (pip install -e .)
│   ├── envs/                   # MuJoCo layer — env + robot merged
│   │   ├── mujoco_parser.py    # MuJoCoParserClass — sole MuJoCo interface
│   │   ├── ik.py               # damped least-squares IK (solve_ik)
│   │   ├── transforms.py       # stateless SO(3) math
│   │   ├── utils.py            # misc helpers
│   │   └── sim_env.py          # SimpleEnv2: tabletop scene
│   ├── simulation/
│   │   └── patched_env.py      # PatchedEnv: reset_with_offset, get_depth
│   ├── common/
│   │   └── types.py            # EpisodeResult dataclass
│   ├── evaluation/
│   │   ├── metrics.py          # check_success, classify_failure
│   │   └── logger.py           # EpisodeLogger: CSV append + skip-check
│   └── policy/
│       ├── monolithic/
│       │   └── smolvla.py      # load_policy(), get_img_transform()
│       └── hierarchical/
│           ├── config.py             # ActionType, CameraIntrinsics, PIDGains
│           ├── gemini_planner.py     # GeminiPlanner: RGB+cmd → waypoints
│           ├── depth_projector.py    # DepthProjector: pixel+depth → Point3D
│           ├── trajectory_builder.py # TrajectoryBuilder: waypoints → path
│           ├── pid_controller.py     # PIDController, CartesianPIDController
│           └── mujoco_executor.py    # MuJoCoExecutor: path → joint cmds
├── scripts/                    # thin entry points — no logic lives here
│   ├── eval_runner.py          # monolithic (SmolVLA) batch evaluation
│   ├── eval_hvla.py            # hierarchical VLA batch evaluation
│   ├── analysis.py             # post-hoc plots + comparison
│   ├── setup_viewer.py         # interactive EE position viewer
│   └── make_grid_gif.py        # GIF grid assembler
├── asset/                      # MuJoCo scene XML and robot meshes
├── demo_data_example/          # SmolVLA policy metadata
└── pyproject.toml              # pip install -e . support
```

## 4. Component Responsibilities

### 4.1 vla_manipulation/envs/

**`mujoco_parser.py`** — `MuJoCoParserClass` is the sole interface to MuJoCo. All other modules (`ik.py`, `transforms.py`, `utils.py`) are stateless math. `get_pR_body('tcp_link')` is the canonical EE pose accessor. This layer is never modified.

**`sim_env.py`** — `SimpleEnv2` wraps `MuJoCoParserClass` for the tabletop scene. Manages 10-dim joint state `q` (6 arm + 4 gripper), EE target `p0`/`R0`, physics stepping, image capture, and success checking.

**`asset/`** — Scene XML (`example_scene_y2.xml`), robot URDF meshes, and object models. The `agentview` camera is defined in `asset/tabletop/object/object_table.xml` at body `pos="0.8 0.0 1.2"`, `xyaxes="0 1 0 -0.5 0 0.707"`, `fovy=60`.

### 4.2 vla_manipulation/common/

**`types.py`** — `EpisodeResult` dataclass with fields: `condition`, `seed`, `success`, `episode_length`, `ee_x_initial`, `failure_type`, `policy`, `timestamp`. Used by both eval runners and `EpisodeLogger`.

### 4.3 vla_manipulation/simulation/

**`patched_env.py`** — `PatchedEnv(SimpleEnv2)` adds three methods to `SimpleEnv2`:

- `reset_with_offset(ee_offset_x, seed)` — applies EE x-shift by solving IK for `p_trgt = [0.30 + ee_offset_x, 0.0, 1.0]`, overwrites `qpos` via `env.forward()`, runs 100-step warmup
- `get_ee_position()` — returns `tcp_link` body position via `get_pR_body()`
- `get_depth(camera_name)` — renders depth via MuJoCo offscreen buffer, converts NDC to metric metres using `depth = near / (1 - depth_raw * (1 - near/far))`

### 4.4 vla_manipulation/evaluation/

**`metrics.py`** — Shared thresholds and evaluation functions:
- `PLATE_CENTER = [0.300, -0.250, 0.818]`, `PLATE_RADIUS_XY = 0.10`, `MUG_LIFT_Z = 0.86`, `MUG_DROP_Z = 0.83`
- `check_success(mug_trajectory, plate_center, gripper_open_at_end)` — four criteria: lifted, XY within radius, gripper open, stable
- `classify_failure(mug_trajectory, plate_center, gripper_closed_ever)` — returns `'no_grasp'` | `'drop'` | `'wrong_place'`

**`logger.py`** — `EpisodeLogger` wraps CSV append with O(1) skip-check:
- `already_done(condition, seed, policy)` — checked against `_done_set` loaded at init
- `log(result)` — appends one `EpisodeResult` row, creates header if absent, flushes immediately

### 4.5 vla_manipulation/policy/monolithic/

**`smolvla.py`** — `load_policy(device)` loads `Jeongeun/omy_pnp_smolvla` from HuggingFace. `get_img_transform()` returns the preprocessing pipeline (resize to 256×256, ToTensor). Used by `scripts/eval_runner.py`.

### 4.6 vla_manipulation/policy/hierarchical/

**`config.py`** — Dataclasses shared across pipeline stages:
- `ActionType` enum: `APPROACH | PRE_GRASP | GRASP | LIFT | MOVE | PLACE | RETREAT | HOME`
- `CameraIntrinsics`, `CameraExtrinsics` — camera calibration containers
- `PIDGains` — `kp`, `ki`, `kd`, `max_integral`, `output_min`, `output_max`
- `ActionOffsets` — per-action geometric heights (safety, pre-grasp, grasp descent, lift, place, retreat)

**`gemini_planner.py`** — `GeminiPlanner` sends RGB image + natural language command to Gemini and parses the JSON response into a `SemanticWaypoint` list. Includes exponential back-off retry for rate limits (429) and transient errors.

**`depth_projector.py`** — `DepthProjector` converts `(u, v, depth_map)` to `Point3D` in robot base frame via pinhole back-projection and `T_cam→robot`. `from_mujoco_fov(fov_deg, width, height, T_cam2base)` derives intrinsics analytically: `fy = (h/2) / tan(fov/2 * π/180)`, `fx = fy`.

**`trajectory_builder.py`** — `TrajectoryBuilder` expands semantic waypoints into geometric keyframes (applying action offsets) then interpolates: sinusoidal velocity profile for long transit segments (≥0.08 m), cubic spline for precision segments. Uses `SO101Kinematics` from the sibling `vla_robot/` repo.

**`pid_controller.py`** — Discrete-time PID with anti-windup (back-calculation), output clamping, and derivative-kick suppression on first step. `CartesianPIDController` wraps three independent scalar PIDs for X, Y, Z axes.

**`mujoco_executor.py`** — `MuJoCoExecutor` executes a dense Cartesian trajectory on the OMY arm: for each `TrajectoryPoint`, runs a PID loop (`CartesianPIDController` → `solve_ik()` → `data.qpos` update → `mj_step`) until convergence or `max_iter`. Logs a warning per non-converged waypoint; does not raise.

### 4.7 scripts/

**`eval_runner.py`** — Monolithic batch evaluation. Loads SmolVLA, creates one `PatchedEnv` instance, loops over `CONDITIONS × SEEDS`, calls `run_episode()` per episode, appends to `experiments/results.csv`.

**`eval_hvla.py`** — Hierarchical batch evaluation. Constructs `GeminiPlanner`, `DepthProjector` (via `from_mujoco_fov()`), `TrajectoryBuilder`, `MuJoCoExecutor`. Loops over `CONDITIONS × SEEDS` using `EpisodeLogger.already_done()` for resume. Per episode: reset → capture → plan → project → build → execute → classify → log.

**`analysis.py`** — Loads `experiments/results.csv` (+ `experiments/results_hvla.csv` if present), generates five plots. The four monolithic plots (`success_rate.png`, `failure_breakdown.png`, `episode_length.png`, `degradation_summary.png`) use monolithic data only. `success_rate_comparison.png` is written only when both policies are present.

## 5. Data Flow

### 5.1 Monolithic Pipeline

```
env.reset_with_offset(ee_offset_x, seed)
        │
        ▼
while viewer_alive:
    env.step_env()                    # full physics rate
    if not env.env.loop_every(HZ=20): continue
    if env.check_success(): break     # success
    if timeout: break
    │
    ├── env.grab_image()              # → rgb_agent (agentview), rgb_ego (wrist)
    ├── env.get_joint_state()[:6]     # → 6-dim arm state
    │
    ▼
SmolVLAPolicy.select_action({
    'observation.state':       joint_state,
    'observation.image':       agentview_256x256,
    'observation.wrist_image': wrist_256x256,
    'task':                    [env.instruction],
})
        │
        ▼ 7-dim action [j1..j6, gripper]
env.step(action)                      # updates env.q → applied next step_env()
        │
        ▼
EpisodeResult → experiments/results.csv
```

### 5.2 Hierarchical Pipeline

```
env.reset_with_offset(ee_offset_x, seed)
        │
        ├── env.grab_image()          → RGB array (H×W×3 uint8)
        ├── env.get_depth('agentview') → depth map (H×W float32 metres)
        │
        ▼  Stage 1 — GeminiPlanner.plan(rgb, command)
SemanticWaypoint list  [action_type, (pixel_u, pixel_v), gripper_state, ...]
        │
        ▼  Stage 2 — DepthProjector.project_batch(pixels, depth_map)
Point3D list  [x, y, z in robot base frame]
        │           └─ on failure (no valid depth): None entry → builder skips
        ▼  Stage 3 — TrajectoryBuilder.build() + .interpolate()
TrajectoryPoint list  [position, gripper, action_type]  (dense, smooth)
        │
        ▼  Stage 4+5 — MuJoCoExecutor.execute(trajectory)  [blocking]
    for each TrajectoryPoint:
        PID loop → CartesianPIDController.update(error) → vel
        solve_ik(env, tcp_link, step_target, R_EE) → q_solved
        data.qpos[joint_idxs] += (q_solved - q_current)
        mujoco.mj_step(model, data)
    until converged or max_iter
        │
        ▼
env.check_success()
        │
        ▼
EpisodeResult → experiments/results_hvla.csv

Failure path: GeminiPlanner RuntimeError or DepthProjector returning all-None
→ EpisodeResult(failure_type='planning_error') via exception handling in main()
```

## 6. Shared Interfaces

Both pipelines share the following components, ensuring the comparison is fair:

| Component | What is shared |
|-----------|---------------|
| `PatchedEnv` | Same `reset_with_offset()`, same scene XML, same camera (`agentview`), same `check_success()` |
| `CONDITIONS` | Same 4 EE offset levels: nominal (0 cm), mild (5 cm), medium (10 cm), strong (15 cm) |
| `PLATE_CENTER` | `[0.300, -0.250, 0.818]` m — success target, defined once in `metrics.py` |
| `PLATE_RADIUS_XY` | 0.10 m — success radius, defined once in `metrics.py` |
| `EpisodeLogger` | Same CSV schema with `policy` column; same `already_done()` skip logic |
| `EpisodeResult` | Same dataclass; `policy` field distinguishes `'monolithic'` vs `'hierarchical'` |
| `N_EPISODES` | 20 seeds per condition (seeds 0–19) |

## 7. Key Design Decisions

### 7.1 Why vla_manipulation/ is a proper Python package

`env/` and `robot/` were upstream code that were absorbed into `vla_manipulation/envs/` during the final refactor. The package is installable via `pip install -e .` (pyproject.toml), making imports work from any working directory. `scripts/` contains only thin entry points — all logic lives in `vla_manipulation/`. The strict dependency rule: `scripts/` imports `vla_manipulation/`, never the reverse (C-NEW-1).

### 7.2 Why depth lives in PatchedEnv not SimpleEnv2

`SimpleEnv2` is the base environment. Adding `get_depth()` there would blur the boundary between the base task environment and experiment-specific instrumentation. `vla_manipulation/simulation/patched_env.py` is the designated experiment-layer extension point — it already adds `reset_with_offset()` and `get_ee_position()` for the same reason.

### 7.3 Why MuJoCoExecutor uses vla_manipulation/envs/ik.py

`vla_manipulation/envs/ik.py` already implements damped least-squares IK for the OMY arm and is validated against the same joint limits and robot model used in `vla_manipulation/envs/sim_env.py`. Reusing `solve_ik()` avoids duplicating kinematics code, ensures the IK solver is consistent between the monolithic reset path (`PatchedEnv.reset_with_offset`) and the hierarchical execution path (`MuJoCoExecutor`), and inherits all existing solver tuning.

### 7.4 Why EpisodeLogger uses a (condition, seed, policy) triple

The original `eval_runner.py` used per-condition row counts for resume detection, which breaks when a second policy adds rows to the same CSV. A unique triple per episode allows both pipelines to append to separate CSVs with O(1) skip-check, and `analysis.py` can merge them unambiguously using the `policy` column.

## 8. Known Limitations

**L-1**: `MuJoCoExecutor` XY drift near kinematic singularities (paper Section 4.2.2). Near table height, the damped least-squares IK in `vla_manipulation/envs/ik.py` produces ~10 cm XY drift during contact descent. The mocap+weld approach in the paper resolves this at the cost of MJCF changes.

**L-2**: `get_depth()` requires an active viewer context (C-NEW-2). The offscreen renderer shares `self.viewer.ctx` with the main viewer window — there is no headless depth path. The viewer must be open before calling `get_depth()`.

**L-3**: Gemini pixel coordinates are not validated against scene geometry before depth projection. If Gemini returns coordinates outside the valid depth range (zero, NaN, or beyond `depth_max_m = 2.0 m`), `DepthProjector.project()` returns `None` for that waypoint. `TrajectoryBuilder.build()` logs a warning and skips `None` entries, which may result in an incomplete trajectory.
