# SmolVLA Pick-and-Place Experiment

Batch evaluation of a pretrained [SmolVLA](https://huggingface.co/Jeongeun/omy_pnp_smolvla) policy on a MuJoCo pick-and-place task under initial end-effector position shifts.

**Task**: pick the red mug and place it on the plate.
**Variable**: initial EE x-position offset (0 / −5 / −10 / −15 cm from nominal).
**Policy**: `Jeongeun/omy_pnp_smolvla` (frozen, no fine-tuning).

See [experiments/Experiment.md](experiments/Experiment.md) for full experimental protocol, shift conditions, validation checklist, and known caveats.

---

## Repository Structure

```
env/                    # Pure MuJoCo simulation layer (do not modify)
  mujoco_parser.py      #   Viewer, FK, camera capture
  ik.py                 #   Numerical IK solver
  transforms.py         #   SO(3) math utilities
  utils.py              #   Misc helpers

robot/                  # Robot task environment
  sim_env.py            #   SimpleEnv2 — tabletop pick-and-place env

policy/                 # Policy loading & inference
  smolvla.py            #   load_policy(), get_img_transform()

experiments/            # Experiment execution
  patched_env.py        #   PatchedEnv — adds reset_with_offset()
  eval_runner.py        #   Batch evaluation runner
  analysis.py           #   Post-hoc analysis & plots
  Experiment.md         #   Full experimental protocol
  results.csv           #   One row per episode
  media/videos/         #   MP4 per episode
  plots/                #   PNG analysis figures

asset/                  # MuJoCo scene assets
  example_scene_y2.xml  #   Active scene (robot + objects + cameras)
  robotis_omy/          #   Robot URDF/meshes
  objaverse/            #   Mug and plate 3D models
  tabletop/             #   Table and floor

demo_data_example/      # Example dataset for policy metadata loading
setup_viewer.py         # Interactive viewer — check EE positions before eval
requirements.txt
```

---

## Installation

Tested on Python 3.10. Install with:

```bash
pip install -r requirements.txt
```

Requires MuJoCo **3.1.6**.

---

## Usage

### Activate environment

```bash
conda activate lerobot-mujoco-tutorial
```

### 1. Check EE positions (interactive viewer)

```bash
python setup_viewer.py
```

Keyboard controls:
- `1` / `2` / `3` / `4` — switch condition: nominal / mild / medium / strong
- `R` — reset current condition
- `ESC` — quit

### 2. Run batch evaluation

```bash
# All 4 conditions × 20 episodes
python experiments/eval_runner.py --condition all

# Single condition
python experiments/eval_runner.py --condition nominal
python experiments/eval_runner.py --condition mild
python experiments/eval_runner.py --condition medium
python experiments/eval_runner.py --condition strong
```

Results append to `experiments/results.csv` after every episode. Re-running resumes from where it left off.

### 3. Analyse results

```bash
python experiments/analysis.py
```

Outputs four plots to `experiments/plots/`:
- `success_rate.png` — success rate with Wilson 95% CI
- `failure_breakdown.png` — stacked bar: no_grasp / drop / wrong_place
- `episode_length.png` — box + strip plot per condition
- `degradation_summary.png` — two-panel hero figure

---

## Shift Conditions

| Condition | EE offset | Target x | IK error |
|-----------|-----------|----------|----------|
| nominal   | 0.00 m    | 0.30 m   | 0.0070   |
| mild      | −0.05 m   | 0.25 m   | 0.0080   |
| medium    | −0.10 m   | 0.20 m   | 0.0081   |
| strong    | −0.15 m   | 0.15 m   | 0.0072   |

---

## Outputs

| Path | Contents |
|------|----------|
| `experiments/results.csv` | One row per episode (condition, seed, success, length, EE pos, failure type, timestamp) |
| `experiments/media/videos/` | `{condition}_seed{NN}_{outcome}.mp4` — 512×256 side-by-side at 20 fps |
| `experiments/plots/` | Analysis PNG figures |

---

## Acknowledgements

- Robot asset: [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
- MuJoCo parser: modified from [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3)
- Object assets: [Objaverse](https://objaverse.allenai.org/)
- Policy: [Jeongeun/omy_pnp_smolvla](https://huggingface.co/Jeongeun/omy_pnp_smolvla) on HuggingFace
