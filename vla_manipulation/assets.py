"""
Asset and experiment path resolver for vla_manipulation.

Single source of truth for all asset and output paths.  All scripts
import path constants from here — no script constructs paths itself.

Why a separate module (same pattern as metadata.py):
  - Scripts stay clean — no path manipulation logic
  - Moving assets/ or experiments/ means changing one file only
  - Works from any working directory
"""
from __future__ import annotations
from pathlib import Path

# vla_manipulation/assets.py
#   parent        → vla_manipulation/
#   parent.parent → repo root
_PKG_ROOT  = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent

# ── scene assets ──────────────────────────────────────────────────────────────
ASSETS_DIR = _PKG_ROOT / 'assets'
SCENE_XML  = ASSETS_DIR / 'example_scene_y2.xml'

# ── experiment output paths ───────────────────────────────────────────────────
_EXPERIMENTS_DIR = _REPO_ROOT / 'experiments'

MONOLITHIC_RESULTS_CSV   = _EXPERIMENTS_DIR / 'monolithic' / 'results.csv'
MONOLITHIC_PLOTS_DIR     = _EXPERIMENTS_DIR / 'monolithic' / 'plots'
MONOLITHIC_MEDIA_DIR     = _EXPERIMENTS_DIR / 'monolithic' / 'media'
MONOLITHIC_VIDEOS_DIR    = _EXPERIMENTS_DIR / 'monolithic' / 'media' / 'videos'

HIERARCHICAL_RESULTS_CSV = _EXPERIMENTS_DIR / 'hierarchical' / 'results.csv'
HIERARCHICAL_PLOTS_DIR   = _EXPERIMENTS_DIR / 'hierarchical' / 'plots'
HIERARCHICAL_MEDIA_DIR   = _EXPERIMENTS_DIR / 'hierarchical' / 'media'
HIERARCHICAL_VIDEOS_DIR  = _EXPERIMENTS_DIR / 'hierarchical' / 'media' / 'videos'

COMPARISON_PLOTS_DIR     = _EXPERIMENTS_DIR / 'comparison' / 'plots'


def ensure_output_dirs() -> None:
    """Create all experiment output directories (idempotent)."""
    for d in [
        MONOLITHIC_PLOTS_DIR, MONOLITHIC_VIDEOS_DIR,
        HIERARCHICAL_PLOTS_DIR, HIERARCHICAL_VIDEOS_DIR,
        COMPARISON_PLOTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_scene_xml(name: str = 'example_scene_y2.xml') -> str:
    """
    Return the absolute path to a scene XML file as a string.

    Args:
        name: filename inside vla_manipulation/assets/
              defaults to the active scene (example_scene_y2.xml)
    Returns:
        Absolute path string suitable for MuJoCoParserClass.
    Raises:
        FileNotFoundError: if the scene file does not exist.
    """
    path = ASSETS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Scene file not found: {path}\n"
            f"Assets directory: {ASSETS_DIR}"
        )
    return str(path)
