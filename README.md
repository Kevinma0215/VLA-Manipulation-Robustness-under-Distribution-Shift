# VLA Manipulation Robustness under Distribution Shift

![results grid](experiments/media/results_grid.gif)

We study how spatial distribution shift affects two VLA architectures
on a tabletop pick-and-place task. A monolithic policy (SmolVLA) maps
observations directly to joint commands; a hierarchical system
(Gemini + depth projection + IK) decomposes planning from execution.
Key finding: monolithic performance collapses sharply beyond ~10 cm
EE offset from the training distribution; hierarchical planning
remains coherent but execution is limited by singularity drift.

---

## Research Question

To what extent can a hierarchical VLA model maintain coherent reasoning
and task performance under progressively stronger shifts in spatial
initial conditions? We vary the robot's initial end-effector position
along the x-axis while keeping the scene, objects, and language
instruction fixed.

---

## Repository Structure

```
lerobot-mujoco-tutorial/
├── vla_manipulation/           # installable package
│   ├── envs/                   # MuJoCo layer (env + robot merged)
│   │   ├── mujoco_parser.py
│   │   ├── ik.py
│   │   ├── transforms.py
│   │   ├── utils.py
│   │   └── sim_env.py
│   ├── simulation/
│   │   └── patched_env.py      # PatchedEnv: shift + depth
│   ├── common/
│   │   └── types.py            # EpisodeResult
│   ├── evaluation/
│   │   ├── metrics.py          # success criteria
│   │   └── logger.py           # CSV logger
│   └── policy/
│       ├── monolithic/
│       │   └── smolvla.py
│       └── hierarchical/
│           ├── config.py
│           ├── gemini_planner.py
│           ├── depth_projector.py
│           ├── trajectory_builder.py
│           ├── pid_controller.py
│           └── mujoco_executor.py
├── scripts/                    # entry points
│   ├── eval_runner.py
│   ├── eval_hvla.py
│   ├── analysis.py
│   ├── setup_viewer.py
│   └── make_grid_gif.py
├── experiments/                # evaluation outputs (generated)
│   ├── results.csv             # monolithic results
│   ├── plots/                  # output PNGs
│   └── media/                  # episode videos and results GIF
├── asset/                      # scene XML and meshes
├── demo_data_example/          # SmolVLA metadata
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/Kevinma0215/VLA-Manipulation-Robustness-under-Distribution-Shift
cd VLA-Manipulation-Robustness-under-Distribution-Shift
conda activate lerobot-mujoco-tutorial
# Install dependencies + package
pip install -r requirements.txt && pip install -e .
```

For hierarchical VLA (Gemini API):
```bash
pip install google-genai
echo "GEMINI_KEY=your_key_here" > .env
```

---

## Usage

### Verify EE positions (interactive viewer)
```bash
python scripts/setup_viewer.py
```

Keyboard controls: `1`/`2`/`3`/`4` — switch condition, `R` — reset, `ESC` — quit

### Monolithic VLA (SmolVLA)
```bash
# All 4 conditions x 20 episodes
python scripts/eval_runner.py --condition all

# Single condition
python scripts/eval_runner.py --condition nominal
```

### Hierarchical VLA (Gemini + MuJoCo IK)
```bash
# Dry run — skip Gemini calls and execution
python scripts/eval_hvla.py --dry-run --condition nominal

# Full evaluation
python scripts/eval_hvla.py --condition all
python scripts/eval_hvla.py --condition nominal
```

### Analysis
```bash
python scripts/analysis.py
```

Outputs to `experiments/plots/`. Generates a comparison plot
automatically if `results_hvla.csv` is present.

---

## Shift Conditions

| Condition | EE offset | Target x | IK error |
|-----------|-----------|----------|----------|
| nominal   | 0.00 m    | 0.30 m   | 0.0070   |
| mild      | −0.05 m   | 0.25 m   | 0.0080   |
| medium    | −0.10 m   | 0.20 m   | 0.0081   |
| strong    | −0.15 m   | 0.15 m   | 0.0072   |

---

## Results

### Monolithic VLA (SmolVLA)

| Condition | Success Rate |
|-----------|-------------|
| nominal   | 100%        |
| mild      | 95%         |
| medium    | 20%         |
| strong    | 30%         |

Performance collapses sharply at medium shift (−10 cm).
All failures are `wrong_place` — grasping transfers but
language-conditioned placement does not.

### Hierarchical VLA (Gemini + MuJoCo IK)

| Condition | Success Rate |
|-----------|-------------|
| nominal   | TBD         |
| mild      | TBD         |
| medium    | TBD         |
| strong    | TBD         |

Results pending full evaluation run.

---

## Architecture

See [Architecture.md](Architecture.md) for full component
responsibilities, data flow diagrams, and design decisions.

| Stage      | Monolithic            | Hierarchical                  |
|------------|-----------------------|-------------------------------|
| Input      | RGB + joints          | RGB + depth + language        |
| Planning   | SmolVLA (end-to-end)  | GeminiPlanner (2D waypoints)  |
| Projection | —                     | DepthProjector (pixel → 3D)   |
| Trajectory | —                     | TrajectoryBuilder (dense path)|
| Execution  | Joint angle output    | MuJoCoExecutor (IK + PID)     |
| Output     | Joint commands        | Joint commands                |

---

## Known Limitations

- **Monolithic**: sharp performance cliff at ~10 cm EE offset
  (paper Section 4.1)
- **Hierarchical**: ~10 cm XY drift near kinematic singularities
  during contact descent (paper Section 4.2.2, Approach 1)
- **Viewer required**: `get_depth()` and `grab_image()` need an
  active GLFW context — no headless path exists

---

## Acknowledgements

- Upstream tutorial: [jeongeun980906/lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)
- Robot asset: [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
- MuJoCo parser: modified from [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3)
- Object assets: [Objaverse](https://objaverse.allenai.org/)
- Policy: [Jeongeun/omy_pnp_smolvla](https://huggingface.co/Jeongeun/omy_pnp_smolvla)

---

## License

MIT
