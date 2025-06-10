import os
import random
from environment import make_env
from PIL import Image
import numpy as np

def save_env_image(env, filename):
    """
    Save an image of the environment's current state to a file.

    Parameters:
        env: An environment object with a `viewer` attribute. The `viewer` should support
             rendering via a `render()` method and optionally provide pixel data via a
             `read_pixels(depth=False)` method.
        raise RuntimeError("The environment viewer does not support rendering or pixel reading.")

    Raises:
        RuntimeError: If `env.viewer` does not support rendering.
    """
    env.step()
    if hasattr(env.viewer, "render") and hasattr(env.viewer, "read_pixels"):
        env.viewer.render()
        img = env.viewer.read_pixels(depth=False)
    elif hasattr(env.viewer, "render"):
        img = env.viewer.render()
    else:
        raise RuntimeError("Viewer does not support rendering.")
    if filename is not None:
        Image.fromarray(img).save(filename)
    return img

def randomize_env():
    """
    Generate randomized pick and place positions within a specified range.

    Returns:
        tuple: A tuple containing two lists:
            - pick_pos: [x, y, z] coordinates for the pick position.
            - place_pos: [x, y, z] coordinates for the place position.
    """
    pick_x = 0.7 + random.uniform(-0.1, 0.1)
    pick_y = 0.0 + random.uniform(-0.1, 0.1)
    place_x = 0.7 + random.uniform(-0.1, 0.1)
    place_y = 0.2 + random.uniform(-0.1, 0.1)
    pick_pos = [pick_x, pick_y, 1.02]
    place_pos = [place_x, place_y, 1.02]
    return pick_pos, place_pos

if __name__ == "__main__":
    n_episodes = 150  # Number of episodes to sample
    out_dir = "data/train_pick-and-place"
    os.makedirs(out_dir, exist_ok=True)
    idx_dict = {}

    # Choose which environment(s) to sample
    SAMPLE_VERSION = "A"  # options: "A", "B", "both"

    version_dirs = {}
    label_files = {}
    for version in ["A", "B"]:
        version_dir = os.path.join(out_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        version_dirs[version] = version_dir
        label_file_path = os.path.join(version_dir, "labels.txt")
        # Determine the next index by counting existing images
        existing_imgs = [f for f in os.listdir(version_dir) if f.endswith(".png")]
        if existing_imgs:
            max_idx = max([int(f.split("_")[1].split(".")[0]) for f in existing_imgs if "_" in f])
            idx_dict[version] = max_idx + 1
        else:
            idx_dict[version] = 0
        label_files[version] = open(label_file_path, "a")  # append mode

    try:
        for ep in range(n_episodes):
            pick_pos, place_pos = randomize_env()
            versions = []
            if SAMPLE_VERSION == "A":
                versions = ["A"]
            elif SAMPLE_VERSION == "B":
                versions = ["B"]
            elif SAMPLE_VERSION == "both":
                versions = ["A", "B"]
            else:
                raise ValueError("SAMPLE_VERSION must be 'A', 'B', or 'both'.")

            for version in versions:
                env = make_env(version=version, render_mode="gui", pick_pos=pick_pos, place_pos=place_pos)
                img_name = f"{version}_{idx_dict[version]:05d}.png"
                img_path = os.path.join(version_dirs[version], img_name)
                save_env_image(env, img_path)
                # Save pick and place positions and version as label
                label_files[version].write(f"{img_name} {pick_pos} {place_pos} {version}\n")
                idx_dict[version] += 1
                if hasattr(env, "viewer") and env.viewer is not None:
                    try:
                        env.viewer.close()
                    except Exception:
                        pass
                del env
    finally:
        for f in label_files.values():
            f.close()
