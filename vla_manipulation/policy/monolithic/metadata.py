"""
Adapter between demo_data_example/ (data) and the policy (code).

This is the single place in the codebase that knows where
demo_data_example/ lives. All other modules import from here
instead of resolving paths themselves.

Why a separate module:
  - smolvla.py stays focused on policy logic only
  - path resolution is tested and changed in one place
  - adding a second dataset means adding one function here,
    not editing policy code
"""
from __future__ import annotations
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

# Derive repo root from this file's location — works from any
# working directory:
#   vla_manipulation/policy/monolithic/metadata.py
#   parent       → monolithic/
#   parent.parent → policy/
#   parent.parent.parent → vla_manipulation/
#   parent.parent.parent.parent → repo root
_REPO_ROOT    = Path(__file__).resolve().parent.parent.parent.parent
DEMO_DATA_DIR = _REPO_ROOT / 'demo_data_example'


def load_omy_pnp_metadata() -> LeRobotDatasetMetadata:
    """
    Load LeRobotDatasetMetadata for the omy_pnp_language dataset.

    Reads from demo_data_example/ using an absolute path derived
    from this file's location. Works regardless of working directory.

    Returns:
        LeRobotDatasetMetadata with .features and .stats populated.

    Raises:
        FileNotFoundError: if demo_data_example/meta/info.json is
            missing. Run: git checkout HEAD -- demo_data_example/
    """
    meta_path = DEMO_DATA_DIR / 'meta' / 'info.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found at {meta_path}\n"
            f"Restore it with: git checkout HEAD -- demo_data_example/"
        )
    return LeRobotDatasetMetadata(
        "omy_pnp_language",
        root=str(DEMO_DATA_DIR),
    )
