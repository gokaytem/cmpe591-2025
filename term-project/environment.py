from base_env_a import BaseEnvA
from base_env_b import BaseEnvB

def make_env(version="A", render_mode="gui", pick_pos=None, place_pos=None):
    """
    Creates an environment instance based on the specified version.

    Parameters:
        version (str): The version of the environment ("A" or "B"). Defaults to "A".
        render_mode (str): The rendering mode for the environment. Defaults to "gui".
        pick_pos (optional): The pick position for the environment. Defaults to None.
        place_pos (optional): The place position for the environment. Defaults to None.

    Returns:
        BaseEnvA or BaseEnvB: An instance of the specified environment version.

    Raises:
        ValueError: If an unknown environment version is provided.
    """
    version_upper = version.upper()
    if version_upper == "A":
        return BaseEnvA(render_mode=render_mode, pick_pos=pick_pos, place_pos=place_pos)
    elif version_upper == "B":
        return BaseEnvB(render_mode=render_mode, pick_pos=pick_pos, place_pos=place_pos)
    else:
        raise ValueError(f"Unknown environment version: {version}")
