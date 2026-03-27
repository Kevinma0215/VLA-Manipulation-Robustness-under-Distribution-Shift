"""Success criteria and failure classification. Canonical location: vla_manipulation/evaluation/metrics.py"""

MUG_LIFT_Z = 0.86
MUG_DROP_Z = 0.83

def check_success(env) -> bool:
    """Return True if the mug is on the plate."""
    return env.check_success()

def classify_failure(
    gripper_ever_closed: bool,
    drop_detected: bool,
    steps: int,
    elapsed: float,
) -> str:
    if not gripper_ever_closed:
        return "no_grasp"
    if drop_detected:
        return "drop"
    return "wrong_place"
