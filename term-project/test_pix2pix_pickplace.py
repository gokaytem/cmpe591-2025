import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import argparse
from environment import make_env
from imitate_pick_and_place import PickPlaceCNN
from pix2pix_train import Generator
from env_shared import pick_and_place
from sample_env import save_env_image, randomize_env
from test_pickplace import preprocess_img

def postprocess_pix2pix_img(tensor):
    # Remove batch dimension and move tensor to CPU
    img = tensor.squeeze(0).cpu()
    # Convert from [-1, 1] to [0, 1] and clamp values
    img = (img * 0.5 + 0.5).clamp(0, 1)
    # Convert to numpy array, change from CHW to HWC, and scale to [0, 255]
    img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # Return the processed image
    return img

def main():
    """
    Runs pick-and-place evaluation using a pix2pix image translation model and a pick-and-place regression model.

    For each episode:
    - Randomizes the environment with new pick and place positions.
    - If --compare is set, runs the regression model on the original environment image (env-a) and attempts pick-and-place.
    - Runs the pix2pix generator to translate the env-b image to env-a style, then runs the regression model on the generated image and attempts pick-and-place.
    - Tracks and prints the number of successful pick-and-place attempts for both env-a and env-b (pix2pix).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pix2pix_model_path", default="pix2pix_checkpoints/generator_epoch100.pth")
    parser.add_argument("--pickplace_model_path", default="pickplace_model.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--compare", action="store_true", default=True, help="Compare pix2pix vs original image in env-a and env-b")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pix2pix generator
    pix2pix_G = Generator().to(device)
    pix2pix_G.load_state_dict(torch.load(args.pix2pix_model_path, map_location=device))
    pix2pix_G.eval()

    # Load pick-and-place regression model
    pickplace_model = PickPlaceCNN().to(device)
    pickplace_model.load_state_dict(torch.load(args.pickplace_model_path, map_location=device))
    pickplace_model.eval()

    success_count_a = 0
    success_count_b = 0

    for ep in range(args.episodes):
        pick_pos, place_pos = randomize_env()
        print(f"Episode {ep+1}: true pick_pos={pick_pos}, true place_pos={place_pos}")

        if args.compare:
            # --- Run on env-a (original image) ---
            env_a = make_env(version="A", render_mode="gui", pick_pos=pick_pos, place_pos=place_pos)
            img_a_orig = save_env_image(env_a, filename=None)
            img_a_tensor = preprocess_img(img_a_orig).to(device)
            with torch.no_grad():
                preds_a = pickplace_model(img_a_tensor)
                preds_a = preds_a.squeeze(0).cpu().numpy()
                pred_pick_xy_a = preds_a[:2].tolist()
                pred_place_xy_a = preds_a[2:].tolist()
                pred_pick_pos_a = pred_pick_xy_a + [1.02]
                pred_place_pos_a = pred_place_xy_a + [1.02]
            print(f"[env-a] Predicted pick_pos={pred_pick_pos_a}, place_pos={pred_place_pos_a}")
            success_a = pick_and_place(env_a, pred_pick_pos_a, pred_place_pos_a)
            if success_a:
                print("[env-a] Pick-and-place SUCCESS")
                success_count_a += 1
            else:
                print("[env-a] Pick-and-place FAIL")
            time.sleep(args.sleep)
            if hasattr(env_a, "viewer") and env_a.viewer is not None:
                try:
                    env_a.viewer.close()
                except Exception:
                    pass
            del env_a

        # --- Run on env-b (pix2pix) ---
        env_b = make_env(version="B", render_mode="gui", pick_pos=pick_pos, place_pos=place_pos)
        img_b = save_env_image(env_b, filename=None)
        img_b_tensor = preprocess_img(img_b).to(device)
        with torch.no_grad():
            img_b_tensor_norm = img_b_tensor * 2 - 1  # [0,1] -> [-1,1]
            img_a_tensor_fake = pix2pix_G(img_b_tensor_norm)
            img_a_fake = postprocess_pix2pix_img(img_a_tensor_fake)
        img_a_tensor_for_reg = preprocess_img(img_a_fake).to(device)
        with torch.no_grad():
            preds_b = pickplace_model(img_a_tensor_for_reg)
            preds_b = preds_b.squeeze(0).cpu().numpy()
            pred_pick_xy_b = preds_b[:2].tolist()
            pred_place_xy_b = preds_b[2:].tolist()
            pred_pick_pos_b = pred_pick_xy_b + [1.02]
            pred_place_pos_b = pred_place_xy_b + [1.02]
        print(f"[env-b pix2pix] Predicted pick_pos={pred_pick_pos_b}, place_pos={pred_place_pos_b}")
        success_b = pick_and_place(env_b, pred_pick_pos_b, pred_place_pos_b)
        if success_b:
            print("[env-b pix2pix] Pick-and-place SUCCESS")
            success_count_b += 1
        else:
            print("[env-b pix2pix] Pick-and-place FAIL")
        time.sleep(args.sleep)
        if hasattr(env_b, "viewer") and env_b.viewer is not None:
            try:
                env_b.viewer.close()
            except Exception:
                pass
        del env_b

    if args.compare:
        print(f"[env-a] Success: {success_count_a} / {args.episodes} ({success_count_a/args.episodes:.2%})")
    print(f"[env-b pix2pix] Success: {success_count_b} / {args.episodes} ({success_count_b/args.episodes:.2%})")

if __name__ == "__main__":
    main()
