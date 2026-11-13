"""RoboTwin checkpoint text recipe (robotwin_converter.py:create_meta_file)."""

CONDITIONING = "robotwin_scene_prefix_v1"
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


def robotwin_prompt(instruction):
    """Accept raw instructions or the converter's already-prefixed metadata."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("expected a nonempty RoboTwin instruction")
    return instruction if instruction.startswith(SCENE_PREFIX) else SCENE_PREFIX + instruction
