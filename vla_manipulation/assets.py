"""
Asset path resolver for vla_manipulation.

This is the single place in the codebase that knows where
vla_manipulation/assets/ lives. All scripts and modules that
need scene files import from here.

Why a separate module (same pattern as metadata.py):
  - Scripts stay clean — no path manipulation logic
  - Moving assets/ means changing one file only
  - Works from any working directory
"""
from __future__ import annotations
from pathlib import Path

# vla_manipulation/assets.py
#   parent       → vla_manipulation/
#   parent.parent → repo root
_PKG_ROOT  = Path(__file__).resolve().parent
ASSETS_DIR = _PKG_ROOT / 'assets'

# Scene files
SCENE_XML = ASSETS_DIR / 'example_scene_y2.xml'


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
