"""
policy/smolvla.py

Load the pretrained SmolVLA policy and build the image transform.
"""

import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from torchvision import transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import dataset_to_policy_features

POLICY_HUB = 'Jeongeun/omy_pnp_smolvla'


def load_policy(device: str) -> SmolVLAPolicy:
    """Load SmolVLA from HuggingFace hub (Jeongeun/omy_pnp_smolvla)."""
    try:
        dataset_metadata = LeRobotDatasetMetadata(
            "omy_pnp_language", root='./demo_data_example')
    except Exception:
        dataset_metadata = LeRobotDatasetMetadata(
            "omy_pnp_language", root='./omy_pnp_language')

    features        = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features  = {k: v for k, v in features.items() if k not in output_features}

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=5,
        n_action_steps=5,
    )
    policy = SmolVLAPolicy.from_pretrained(
        POLICY_HUB, config=cfg, dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()
    print(f"Policy loaded on {device}.")
    return policy


def get_img_transform():
    """PIL [0-255] -> FloatTensor [0.0-1.0], C×H×W."""
    return transforms.Compose([
        transforms.ToTensor(),   # PIL [0–255] -> FloatTensor [0.0–1.0], C×H×W
    ])
