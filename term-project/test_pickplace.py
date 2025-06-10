import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
from PIL import Image
import torch
import argparse
from environment import make_env
from imitate_pick_and_place import PickPlaceCNN
from env_shared import pick_and_place
from sample_env import save_env_image, randomize_env

def preprocess_img(img):
    # Convert the image to RGB format and resize it to 256x256 pixels
    img = Image.fromarray(img).convert("RGB").resize((256,256))
    # Normalize pixel values to the range [0, 1] and convert to a NumPy array
    img = np.array(img).astype(np.float32) / 255.0
    # Convert the NumPy array to a PyTorch tensor, rearrange dimensions to (C, H, W), and add a batch dimension
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    # Return the processed image
    return img

def main():
    """
    Runs multiple episodes of the pick-and-place task using a trained PickPlaceCNN model.
    For each episode:
      - Randomizes the environment's pick and place positions.
      - Loads the environment and captures an image.
      - Preprocesses the image and predicts pick/place positions using the model.
      - Executes the pick-and-place action in the environment.
      - Tracks and prints the success rate over all episodes.

    Command-line arguments:
      --model_path: Path to the trained model weights (default: "pickplace_model.pt")
      --episodes: Number of episodes to run (default: 10)
      --sleep: Seconds to sleep between episodes (default: 0)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pickplace_model.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PickPlaceCNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model = model.to(device)

    success_count = 0

    for ep in range(args.episodes):
        # Randomize initial pick/place for environment
        pick_pos, place_pos = randomize_env()
        env = make_env(version="A", render_mode="gui", pick_pos=pick_pos, place_pos=place_pos)
        print(f"Episode {ep+1}: true pick_pos={pick_pos}, true place_pos={place_pos}")

        # Get current image from env
        img = save_env_image(env, filename=None)
        img_path = None  # Not saving to disk, use in-memory

        # Predict pick/place from image
        img_tensor = preprocess_img(img).to(device)
        with torch.no_grad():
            preds = model(img_tensor)
            preds = preds.squeeze(0).cpu().numpy()
            pred_pick_xy = preds[:2].tolist()
            pred_place_xy = preds[2:].tolist()
            pred_pick_pos = pred_pick_xy + [1.02]
            pred_place_pos = pred_place_xy + [1.02]
        print(f"Predicted pick_pos={pred_pick_pos}, place_pos={pred_place_pos}")

        # Run pick-and-place using predicted positions
        success = pick_and_place(env, pred_pick_pos, pred_place_pos)
        if success:
            print("Pick-and-place SUCCESS")
            success_count += 1
        else:
            print("Pick-and-place FAIL")
        time.sleep(args.sleep)

        if hasattr(env, "viewer") and env.viewer is not None:
            try:
                env.viewer.close()
            except Exception:
                pass
        del env

    print(f"Success: {success_count} / {args.episodes} episodes")

if __name__ == "__main__":
    main()
