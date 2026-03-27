"""Monolithic VLA (SmolVLA) batch evaluation runner.
Canonical location: scripts/eval_runner.py
"""

import argparse
import csv
import datetime
import os
import sys
import time

import cv2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

# Recommended for 6 GB GPU; set before torch import
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
from PIL import Image

from vla_manipulation.policy.monolithic.smolvla import load_policy, get_img_transform, SmolVLAPolicy
from vla_manipulation.simulation.patched_env import PatchedEnv

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
VIDEO_DIR   = 'experiments/media/videos'

CONDITIONS = [
    {"name": "nominal", "ee_offset_x":  0.00},
    {"name": "mild",    "ee_offset_x": -0.05},
    {"name": "medium",  "ee_offset_x": -0.10},
    {"name": "strong",  "ee_offset_x": -0.15},
]
N_EPISODES  = 20
SEEDS       = list(range(N_EPISODES))  # seed 0..19
MAX_STEPS   = 500          # policy steps before timeout (secondary guard)
MAX_WALL_SEC = 60.0        # wall-clock seconds per episode (primary guard)
                           # At ~3–5 Hz real inference, 60 s ≈ 180–300 policy steps
DEVICE      = 'cuda'
XML_PATH    = './asset/example_scene_y2.xml'
RESULTS_CSV = 'experiments/results.csv'

# Object z thresholds for drop detection (from validation: mug starts at z≈0.838)
MUG_LIFT_Z  = 0.86   # mug must exceed this to count as "lifted"
MUG_DROP_Z  = 0.83   # mug falling below this (after lift) = drop

CSV_FIELDS = [
    'condition', 'ee_offset_x', 'seed', 'success', 'episode_length',
    'final_ee_x', 'final_ee_y', 'final_ee_z', 'failure_type', 'timestamp',
]
# ──────────────────────────────────────────────────────────────────────────────


# ── video recorder ────────────────────────────────────────────────────────────

class EpisodeRecorder:
    """
    Accumulates RGB frames from one episode and writes an MP4 on save().

    Each frame is a 512×256 side-by-side composite of agentview (left) and
    egocentric (right) images, both resized to 256×256, with a one-line
    HUD overlay burned in before writing.
    """

    def __init__(self, fps: int = 20):
        self.frames: list = []
        self.fps = fps

    def add_frame(
        self,
        image: np.ndarray,       # agentview  H×W×3  uint8
        wrist_image: np.ndarray, # egocentric H×W×3  uint8
        step: int,
        condition: str,
        seed: int,
        ee_x: float,
    ):
        left  = cv2.resize(image,       (256, 256))
        right = cv2.resize(wrist_image, (256, 256))
        frame = np.concatenate([left, right], axis=1)          # 256×512×3 RGB

        hud = f"[{condition}] seed={seed:02d}  step={step:03d}  ee_x={ee_x:.3f}"
        cv2.putText(
            frame, hud,
            org=(6, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.45,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def save(self, condition: str, seed: int, outcome: str):
        """Write accumulated frames to VIDEO_DIR/{condition}_seed{seed:02d}_{outcome}.mp4."""
        if not self.frames:
            return
        os.makedirs(VIDEO_DIR, exist_ok=True)
        path = os.path.join(VIDEO_DIR, f"{condition}_seed{seed:02d}_{outcome}.mp4")
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h),
        )
        n = len(self.frames)
        for f in self.frames:
            writer.write(f)
        writer.release()
        self.frames.clear()
        print(f"  [recorder] saved {path}  ({n} frames)")

    def discard(self):
        self.frames.clear()


# policy loading is in policy/smolvla.py


# ── failure classification ─────────────────────────────────────────────────────

def classify_failure(
    gripper_ever_closed: bool,
    drop_detected: bool,
    steps: int,
    elapsed: float,
) -> str:
    """
    Classify failure based on robot behavior only.
    Time limits (MAX_WALL_SEC / MAX_STEPS) are loop guards, not failure modes.

      no_grasp    — gripper never closed during episode
      drop        — gripper closed, object lifted, then fell
      wrong_place — gripper closed, no drop (attempted but failed to place)
    """
    if not gripper_ever_closed:
        return "no_grasp"
    if drop_detected:
        return "drop"
    return "wrong_place"


# ── episode rollout ────────────────────────────────────────────────────────────

_OUTCOME_MAP = {
    "none":        "success",
    "no_grasp":    "fail_no_grasp",
    "drop":        "fail_drop",
    "wrong_place": "fail_wrong_place",
}


def run_episode(
    env: PatchedEnv,
    policy: SmolVLAPolicy,
    condition_name: str,
    ee_offset_x: float,
    seed: int,
    transform,
    device: str,
    episode_idx: int = 0,
):
    """
    Run one episode.  Returns (success, policy_steps, final_ee_pos, failure_type).

    Observation preprocessing:
      - grab_image() → (agentview rgb, egocentric rgb)
      - Each image: PIL.fromarray → resize(256,256) → ToTensor
      - observation.state: get_joint_state()[:6]
      - observation.image: agentview
      - observation.wrist_image: egocentric
      - task: env.instruction (set by set_instruction() in reset)
    """
    # ── reset ────────────────────────────────────────────────────────────────
    env.reset_with_offset(ee_offset_x=ee_offset_x, seed=seed)
    policy.reset()

    # ── video recorder ───────────────────────────────────────────────────────
    recorder = EpisodeRecorder(fps=20)

    # ── episode state ────────────────────────────────────────────────────────
    success             = False
    policy_steps        = 0
    gripper_ever_closed = False
    mug_max_z           = 0.0
    drop_detected       = False
    t_start             = time.time()

    # ── rollout loop ─────────────────────────────────────────────────────────
    while env.env.is_viewer_alive():
        env.step_env()                          # advance physics at full rate

        if not env.env.loop_every(HZ=20):       # policy runs at 20 Hz
            continue

        # success check (before policy step)
        if env.check_success():
            success = True
            break

        # wall-clock timeout (primary) — fires regardless of inference speed
        if time.time() - t_start > MAX_WALL_SEC:
            break

        # step-count timeout (secondary)
        if policy_steps >= MAX_STEPS:
            break

        # ── track failure signals ────────────────────────────────────────────
        gripper_q = env.env.get_qpos_joint('rh_r1')[0]
        if gripper_q > 0.5:
            gripper_ever_closed = True

        mug_z = env.env.get_p_body(env.obj_target)[2]
        if mug_z > mug_max_z:
            mug_max_z = mug_z
        if mug_max_z > MUG_LIFT_Z and mug_z < MUG_DROP_Z:
            drop_detected = True

        # ── observation (exact notebook preprocessing) ───────────────────────
        state = env.get_joint_state()[:6]                  # [j1..j6]

        # grab_image() must be called before render() (sets rgb_agent/rgb_ego)
        image_arr, wrist_arr = env.grab_image()

        # ── record frame ─────────────────────────────────────────────────────
        recorder.add_frame(
            image_arr, wrist_arr,
            step=policy_steps,
            condition=condition_name,
            seed=seed,
            ee_x=env.get_ee_position()[0],
        )

        img   = transform(Image.fromarray(image_arr).resize((256, 256)))
        wrist = transform(Image.fromarray(wrist_arr).resize((256, 256)))

        data = {
            'observation.state':       torch.tensor([state]).to(device),
            'observation.image':       img.unsqueeze(0).to(device),
            'observation.wrist_image': wrist.unsqueeze(0).to(device),
            'task':                    [env.instruction],
        }

        # ── policy inference ─────────────────────────────────────────────────
        with torch.no_grad():
            action = policy.select_action(data)
        action = action[0, :7].cpu().detach().numpy()

        env.step(action)      # sets env.q; applied on next step_env()
        env.render(idx=episode_idx)  # rgb_agent/rgb_ego already set above
        policy_steps += 1

    # ── collect results ──────────────────────────────────────────────────────
    elapsed      = time.time() - t_start
    final_ee     = env.get_ee_position()
    failure_type = "none" if success else classify_failure(
        gripper_ever_closed, drop_detected, policy_steps, elapsed)

    # ── save episode video ────────────────────────────────────────────────────
    outcome = _OUTCOME_MAP.get(failure_type, failure_type)
    recorder.save(condition_name, seed, outcome)

    return success, policy_steps, final_ee, failure_type


# ── summary printer ────────────────────────────────────────────────────────────

def print_summary(summary: dict, conditions: list):
    col_w = 60
    header = (
        f"{'Condition':<12} | {'N':>4} | {'Success Rate':>12} | {'Avg Length':>10}"
    )
    sep = "─" * len(header)
    print(f"\n{'═' * len(header)}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)
    for cond in conditions:
        name = cond["name"]
        if name not in summary or not summary[name]:
            continue
        rows    = summary[name]
        n       = len(rows)
        sr      = sum(r["success"] for r in rows) / n * 100
        avg_len = sum(r["episode_length"] for r in rows) / n
        print(f"{name:<12} | {n:>4} | {sr:>11.0f}% | {avg_len:>10.0f}")
    print(f"{'═' * len(header)}\n")

    # Failure breakdown
    print(f"{'Condition':<12} | {'no_grasp':>9} | {'drop':>6} | {'wrong_place':>12}")
    print(sep)
    for cond in conditions:
        name = cond["name"]
        if name not in summary or not summary[name]:
            continue
        rows = [r for r in summary[name] if not r["success"]]
        n    = len(rows) or 1
        counts = {ft: sum(1 for r in rows if r["failure_type"] == ft)
                  for ft in ("no_grasp", "drop", "wrong_place")}
        print(
            f"{name:<12} | {counts['no_grasp']:>9} | {counts['drop']:>6} | "
            f"{counts['wrong_place']:>12}"
        )
    print(f"{'═' * len(header)}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA batch evaluator — EE position shift conditions")
    parser.add_argument(
        "--condition", default="all",
        choices=["all", "nominal", "mild", "medium", "strong"],
        help="Run one condition or all (default: all)",
    )
    args = parser.parse_args()

    active_conditions = (
        CONDITIONS if args.condition == "all"
        else [c for c in CONDITIONS if c["name"] == args.condition]
    )

    # ── CSV setup ─────────────────────────────────────────────────────────────
    os.makedirs("experiments", exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    csv_file = open(RESULTS_CSV, "a", newline="", buffering=1)
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    # ── load policy ───────────────────────────────────────────────────────────
    print("Loading policy...")
    policy = load_policy(DEVICE)

    # ── create env (single instance, reused across all episodes) ─────────────
    print("Creating environment...")
    env       = PatchedEnv(XML_PATH, action_type="joint_angle")
    transform = get_img_transform()
    summary   = {}

    # ── condition loop ────────────────────────────────────────────────────────
    for cond in active_conditions:
        name        = cond["name"]
        ee_offset_x = cond["ee_offset_x"]
        summary[name] = []

        # ── resume logic: count valid existing rows for this condition ────────
        already_done = 0
        if os.path.exists(RESULTS_CSV):
            try:
                existing_df = pd.read_csv(RESULTS_CSV)
                existing_df = existing_df[existing_df["condition"] == name]
                # a valid row has a non-null success value
                valid = existing_df[existing_df["success"].notna()]
                already_done = len(valid)
            except Exception:
                already_done = 0

        episodes_remaining = N_EPISODES - already_done
        if episodes_remaining <= 0:
            print(f"Condition {name} already complete ({already_done} episodes). Skipping.")
            continue

        seeds_to_run = SEEDS[already_done:]
        if already_done > 0:
            print(
                f"Condition {name}: {already_done} episodes already done, "
                f"running {episodes_remaining} more "
                f"(seeds {already_done}–{SEEDS[-1]})"
            )

        p_trgt_x = 0.30 + ee_offset_x
        print(f"\n{'='*60}")
        print(f"Condition: {name}  |  ee_offset_x={ee_offset_x:+.2f}  |  p_trgt_x={p_trgt_x:.2f}")
        print(f"{'='*60}")

        for i, seed in enumerate(seeds_to_run):
            # bail out if viewer was closed
            if not env.env.is_viewer_alive():
                print("[eval_runner] Viewer closed — stopping evaluation.")
                csv_file.close()
                print_summary(summary, active_conditions)
                return

            success, length, final_ee, failure_type = run_episode(
                env, policy, name, ee_offset_x, seed, transform, DEVICE,
                episode_idx=already_done + i + 1)

            row = {
                "condition":      name,
                "ee_offset_x":    ee_offset_x,
                "seed":           seed,
                "success":        success,
                "episode_length": length,
                "final_ee_x":     round(float(final_ee[0]), 4),
                "final_ee_y":     round(float(final_ee[1]), 4),
                "final_ee_z":     round(float(final_ee[2]), 4),
                "failure_type":   failure_type,
                "timestamp":      datetime.datetime.now().isoformat(),
            }
            writer.writerow(row)
            csv_file.flush()
            summary[name].append(row)

            print(
                f"Condition {name} | Episode {already_done+i+1}/{N_EPISODES} | "
                f"success={success} | len={length} | failure={failure_type}"
            )

    csv_file.close()
    print_summary(summary, active_conditions)
    print(f"Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
