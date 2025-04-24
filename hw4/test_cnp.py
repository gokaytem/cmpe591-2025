import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from homework4 import CNP

def sample_test_case(demos, n_context_min=1, n_target_min=1):
    traj = demos[np.random.randint(len(demos))]
    idxs = np.arange(traj.shape[0])
    np.random.shuffle(idxs)
    n_context = np.random.randint(n_context_min, max(n_context_min+1, len(idxs)//2))
    n_target = np.random.randint(n_target_min, max(n_target_min+1, len(idxs)-n_context))
    obs_idx = idxs[:n_context]
    tar_idx = idxs[n_context:n_context+n_target]
    obs = traj[obs_idx]
    tar = traj[tar_idx]
    obs_x = obs[:, [0, 5]]  # t, h
    obs_y = obs[:, 1:5]     # ey, ez, oy, oz
    tar_x = tar[:, [0, 5]]  # t, h
    tar_y = tar[:, 1:5]     # ey, ez, oy, oz
    obs = np.concatenate([obs_x, obs_y], axis=-1)
    return obs, tar_x, tar_y

if __name__ == "__main__":
    # Load test data and model
    test_demos = np.load('demos_test.npz')['demos']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNP(in_shape=(2, 4), hidden_size=256, num_hidden_layers=4, min_std=1e-3).to(device)
    model.load_state_dict(torch.load('cnp_model.pth', map_location=device))
    model.eval()

    n_tests = 100 # Number of test cases to sample
    mse_ee = []
    mse_obj = []

    for _ in range(n_tests):
        obs, tar_x, tar_y = sample_test_case(test_demos)
        obs = torch.tensor(obs[None, :, :], dtype=torch.float32, device=device)
        tar_x = torch.tensor(tar_x[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_mean, _ = model(obs, tar_x)
            pred_mean = pred_mean[0].cpu().numpy()
        gt = tar_y
        mse_ee.append(np.mean((pred_mean[:, :2] - gt[:, :2]) ** 2))
        mse_obj.append(np.mean((pred_mean[:, 2:] - gt[:, 2:]) ** 2))

    mse_ee = np.array(mse_ee)
    mse_obj = np.array(mse_obj)

    print('End-effector MSE: mean =', mse_ee.mean(), 'std =', mse_ee.std())
    print('Object MSE: mean =', mse_obj.mean(), 'std =', mse_obj.std())

    # Bar plot for MSEs
    means = [mse_ee.mean(), mse_obj.mean()]
    stds = [mse_ee.std(), mse_obj.std()]
    labels = ['End-Effector', 'Object']

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, means, yerr=stds, capsize=8, color=['#377eb8', '#e41a1c'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
    ax.set_title('CNP Prediction Error on Test Set', fontsize=16, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
