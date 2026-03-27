"""Hierarchical VLA (Gemini + MuJoCo IK) batch evaluation runner.
Canonical location: scripts/eval_hvla.py
"""
# At the top of eval_hvla.py, add:
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import numpy as np

from vla_manipulation.simulation.patched_env import PatchedEnv
from vla_manipulation.policy.hierarchical.gemini_planner import GeminiPlanner
from vla_manipulation.assets import get_scene_xml
from vla_manipulation.policy.hierarchical.depth_projector import DepthProjector
from vla_manipulation.policy.hierarchical.trajectory_builder import TrajectoryBuilder
from vla_manipulation.policy.hierarchical.mujoco_executor import MuJoCoExecutor
from vla_manipulation.evaluation.logger import EpisodeLogger
from vla_manipulation.common.types import EpisodeResult

log = logging.getLogger(__name__)


def _load_env_file() -> None:
    """
    Load a .env file from the repo root into os.environ.

    Looks for .env at the directory two levels above this file:
        experiments/eval_hvla.py → repo root = ../
    Silently does nothing if .env does not exist.
    Lines starting with # and blank lines are ignored.
    Does not overwrite variables already set in the environment.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CONDITIONS = [
    {"name": "nominal", "ee_offset_x":  0.00},
    {"name": "mild",    "ee_offset_x": -0.05},
    {"name": "medium",  "ee_offset_x": -0.10},
    {"name": "strong",  "ee_offset_x": -0.15},
]
N_EPISODES   = 20
SEEDS        = list(range(N_EPISODES))   # seeds 0..19
MAX_WALL_SEC = 60.0                      # wall-clock seconds per episode

SCENE_XML   = get_scene_xml()
RESULTS_CSV = 'experiments/results_hvla.csv'

# Camera parameters for agentview (from asset/tabletop/object/object_table.xml)
CAM_NAME    = 'agentview'
CAM_FOV_DEG = 60.0
CAM_WIDTH   = 640
CAM_HEIGHT  = 480

# Natural language task command passed to GeminiPlanner
LANGUAGE_CMD = 'pick up the mug and place it on the plate'

# Object z thresholds for drop detection (matches eval_runner.py)
MUG_LIFT_Z = 0.86   # mug must exceed this to count as "lifted"
MUG_DROP_Z = 0.83   # mug falling below this (after lift) = drop

# ──────────────────────────────────────────────────────────────────────────────


def _get_T_cam2base(scene_xml: str = SCENE_XML) -> np.ndarray:
    """
    Parse the MuJoCo scene XML (handling <include> recursively) to locate the
    'agentview' camera and compute its T_cam→base (world frame) transform.

    In MuJoCo MJCF the camera element's world position is the sum of its own
    `pos` (relative to parent body) and the parent body's `pos`.  We track the
    enclosing body position during traversal to accumulate the full offset.

    The MuJoCo 'xyaxes' attribute gives the camera x and y axes expressed in
    the parent frame.  z_cam = x_cam × y_cam after normalisation (right-hand
    rule).  The rotation matrix R has those three vectors as columns.

    Falls back to analytically-computed values from the known XML:
        body pos="0.8 0.0 1.2"   camera xyaxes="0 1 0 -0.5 0 0.707"

    Returns:
        T : (4, 4) float64 rigid transform, camera frame → robot base frame.
    """
    cam_pos: np.ndarray | None = None
    xyaxes:  np.ndarray | None = None

    def _parse_pos(elem: ET.Element) -> np.ndarray:
        s = elem.get('pos', '0 0 0')
        return np.array([float(v) for v in s.split()], dtype=np.float64)

    def _walk(elem: ET.Element, body_pos: np.ndarray, xml_dir: str) -> None:
        nonlocal cam_pos, xyaxes
        if cam_pos is not None:
            return

        for child in elem:
            if cam_pos is not None:
                return

            if child.tag == 'include':
                href = child.get('file', '')
                if href:
                    _search(os.path.join(xml_dir, href))
                continue

            if child.tag == 'body':
                child_body_pos = body_pos + _parse_pos(child)
                _walk(child, child_body_pos, xml_dir)
                continue

            if child.tag == 'camera' and child.get('name') == CAM_NAME:
                cam_offset = _parse_pos(child)
                cam_pos    = body_pos + cam_offset
                xy_str     = child.get('xyaxes', '0 1 0 -0.5 0 0.707')
                xyaxes     = np.array([float(v) for v in xy_str.split()], dtype=np.float64)
                return

            # Recurse for other container elements (worldbody, etc.)
            _walk(child, body_pos, xml_dir)

    def _search(xml_path: str) -> None:
        nonlocal cam_pos, xyaxes
        if cam_pos is not None:
            return
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except (ET.ParseError, FileNotFoundError, OSError):
            return
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        _walk(root, np.zeros(3, dtype=np.float64), xml_dir)

    _search(scene_xml)

    # Fallback: analytically-derived from known asset/tabletop/object/object_table.xml
    if cam_pos is None:
        log.warning("_get_T_cam2base: could not parse '%s'; using hardcoded values", scene_xml)
        cam_pos = np.array([0.8, 0.0, 1.2],            dtype=np.float64)
        xyaxes  = np.array([0.0, 1.0, 0.0, -0.5, 0.0, 0.707], dtype=np.float64)

    # Build orthonormal camera frame in world coordinates
    x_cam = xyaxes[:3] / np.linalg.norm(xyaxes[:3])
    y_cam = xyaxes[3:] / np.linalg.norm(xyaxes[3:])
    z_cam = np.cross(x_cam, y_cam)
    z_cam = z_cam / np.linalg.norm(z_cam)

    # Rotation matrix: columns = camera axes expressed in world (base) frame
    R = np.column_stack([x_cam, y_cam, z_cam])

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = cam_pos
    return T


def run_episode(
    env:       PatchedEnv,
    planner:   GeminiPlanner | None,
    projector: DepthProjector,
    builder:   TrajectoryBuilder,
    executor:  MuJoCoExecutor,
    condition: str,
    seed:      int,
    ee_x:      float,
    dry_run:   bool = False,
) -> EpisodeResult:
    """
    Run one hierarchical VLA episode.

    Pipeline:
      1. reset_with_offset(ee_x, seed)
      2. grab_image() → RGB;  get_depth() → depth map
      3. GeminiPlanner.plan() → semantic waypoints
      4. DepthProjector.project_batch() → 3D robot-frame positions
      5. TrajectoryBuilder.build() + interpolate() → dense trajectory
      6. MuJoCoExecutor.execute() — blocking physics execution
      7. check_success() + failure classification → EpisodeResult

    Args:
        env:       PatchedEnv instance (viewer already open, reused across episodes)
        planner:   GeminiPlanner.  Must be None when dry_run=True.
        projector: DepthProjector calibrated to CAM_NAME.
        builder:   TrajectoryBuilder for path generation.
        executor:  MuJoCoExecutor for low-level IK/PID control.
        condition: Condition label ('nominal', 'mild', 'medium', 'strong').
        seed:      Episode seed for reproducible object placement.
        ee_x:      EE x-axis offset in metres (ee_offset_x from CONDITIONS).
        dry_run:   Skip Gemini + execution; return a dummy result immediately.

    Returns:
        EpisodeResult with success/failure classification.
    """
    env.reset_with_offset(ee_offset_x=ee_x, seed=seed)
    ee_initial = env.get_ee_position()
    ts = datetime.datetime.now().isoformat()

    if dry_run:
        return EpisodeResult(
            condition      = condition,
            seed           = seed,
            success        = False,
            episode_length = 0,
            ee_x_initial   = float(ee_initial[0]),
            failure_type   = 'dry_run',
            policy         = 'hierarchical',
            timestamp      = ts,
        )

    t_start = time.time()

    # ── 1. Capture observation ────────────────────────────────────────────────
    image_arr, _ = env.grab_image()        # (H, W, 3) uint8 agentview RGB
    depth_map    = env.get_depth(CAM_NAME) # (H, W) float32 metres

    # ── 2. Semantic planning ──────────────────────────────────────────────────
    waypoints = planner.plan(image_arr, LANGUAGE_CMD)

    # ── 3. Depth projection ───────────────────────────────────────────────────
    pixels    = [wp.pixel_coords for wp in waypoints]
    positions = projector.project_batch(pixels, depth_map)

    # ── 4. Trajectory building ────────────────────────────────────────────────
    keyframes  = builder.build(waypoints, positions)
    trajectory = builder.interpolate(keyframes)

    # ── 5. Execute trajectory (blocking) ─────────────────────────────────────
    executor.execute(trajectory)

    # ── 6. Success + failure classification ───────────────────────────────────
    # Gripper-ever-closed: derived from trajectory commands (explicit in plan)
    gripper_closed_ever = any(pt.gripper > 0.5 for pt in trajectory)

    # Final mug z — used for drop heuristic (no step-by-step history available
    # because executor.execute() is a blocking call)
    mug_final_z = float(env.env.get_p_body(env.obj_target)[2])

    success = env.check_success()

    if success:
        failure_type = 'none'
    elif not gripper_closed_ever:
        failure_type = 'no_grasp'
    elif mug_final_z < MUG_DROP_Z:
        # Mug ended well below table height → likely dropped after being lifted
        failure_type = 'drop'
    else:
        failure_type = 'wrong_place'

    log.info(
        "[%s] seed=%02d  success=%s  len=%d  failure=%s  elapsed=%.1fs",
        condition, seed, success, len(trajectory), failure_type,
        time.time() - t_start,
    )

    return EpisodeResult(
        condition      = condition,
        seed           = seed,
        success        = success,
        episode_length = len(trajectory),
        ee_x_initial   = float(ee_initial[0]),
        failure_type   = failure_type,
        policy         = 'hierarchical',
        timestamp      = ts,
    )


def _print_summary(summary: dict, conditions: list) -> None:
    """Print per-condition success rates and failure breakdowns."""
    header = f"{'Condition':<12} | {'N':>4} | {'Success Rate':>12} | {'Avg Length':>10}"
    sep    = "─" * len(header)
    print(f"\n{'═' * len(header)}")
    print("  HIERARCHICAL VLA RESULTS SUMMARY")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)
    for cond in conditions:
        name = cond["name"]
        rows = summary.get(name, [])
        if not rows:
            continue
        n       = len(rows)
        sr      = sum(r.success for r in rows) / n * 100
        avg_len = sum(r.episode_length for r in rows) / n
        print(f"{name:<12} | {n:>4} | {sr:>11.0f}% | {avg_len:>10.0f}")
    print(f"{'═' * len(header)}\n")

    print(f"{'Condition':<12} | {'no_grasp':>9} | {'drop':>6} | {'wrong_place':>12}")
    print(sep)
    for cond in conditions:
        name = cond["name"]
        rows = [r for r in summary.get(name, []) if not r.success]
        if not rows:
            continue
        counts = {ft: sum(1 for r in rows if r.failure_type == ft)
                  for ft in ("no_grasp", "drop", "wrong_place")}
        print(
            f"{name:<12} | {counts['no_grasp']:>9} | {counts['drop']:>6} | "
            f"{counts['wrong_place']:>12}"
        )
    print(f"{'═' * len(header)}\n")


def main() -> None:
    _load_env_file()

    parser = argparse.ArgumentParser(
        description="Hierarchical VLA batch evaluator — EE position shift conditions"
    )
    parser.add_argument(
        "--condition", default="all",
        choices=["all", "nominal", "mild", "medium", "strong"],
        help="Run one condition or all (default: all)",
    )
    parser.add_argument(
        "--api-key", default=None,
        metavar="KEY",
        help="Google AI Studio API key (or set GEMINI_KEY in .env at repo root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip Gemini calls and trajectory execution; log dummy results",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: WARNING)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    api_key = args.api_key or os.environ.get("GEMINI_KEY")
    if not api_key and not args.dry_run:
        parser.error(
            "Gemini API key required. Pass --api-key KEY or "
            "set GEMINI_KEY=... in a .env file at the repo root."
        )

    active_conditions = (
        CONDITIONS if args.condition == "all"
        else [c for c in CONDITIONS if c["name"] == args.condition]
    )

    # ── camera transform + pipeline components ────────────────────────────────
    T_cam2base = _get_T_cam2base(SCENE_XML)

    planner   = GeminiPlanner(api_key=api_key) if not args.dry_run else None
    projector = DepthProjector.from_mujoco_fov(
        fov_deg    = CAM_FOV_DEG,
        width      = CAM_WIDTH,
        height     = CAM_HEIGHT,
        T_cam2base = T_cam2base,
    )
    builder = TrajectoryBuilder()

    # ── environment (single instance, reused across all episodes) ─────────────
    print("Creating environment...")
    env      = PatchedEnv(SCENE_XML, action_type="joint_angle")
    executor = MuJoCoExecutor(env)

    # ── logger ────────────────────────────────────────────────────────────────
    os.makedirs("experiments", exist_ok=True)
    logger  = EpisodeLogger(RESULTS_CSV)
    summary: dict = {}

    # ── condition loop ────────────────────────────────────────────────────────
    for cond in active_conditions:
        name        = cond["name"]
        ee_offset_x = cond["ee_offset_x"]
        summary[name] = []

        p_trgt_x = 0.30 + ee_offset_x
        print(f"\n{'='*60}")
        print(f"Condition: {name}  |  ee_offset_x={ee_offset_x:+.2f}  |  p_trgt_x={p_trgt_x:.2f}")
        print(f"{'='*60}")

        for seed in SEEDS:
            if logger.already_done(name, seed, 'hierarchical'):
                print(f"  [skip] {name} seed={seed:02d} already logged.")
                continue

            if not env.env.is_viewer_alive():
                print("[eval_hvla] Viewer closed — stopping evaluation.")
                _print_summary(summary, active_conditions)
                return

            result = run_episode(
                env       = env,
                planner   = planner,
                projector = projector,
                builder   = builder,
                executor  = executor,
                condition = name,
                seed      = seed,
                ee_x      = ee_offset_x,
                dry_run   = args.dry_run,
            )

            if not args.dry_run:
                logger.log(result)
            summary[name].append(result)

            print(
                f"Condition {name} | seed={seed:02d} | "
                f"success={result.success} | len={result.episode_length} | "
                f"failure={result.failure_type}"
            )

    _print_summary(summary, active_conditions)
    print(f"Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
