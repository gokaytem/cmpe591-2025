import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from homework4 import CNP
import matplotlib.pyplot as plt  # <-- add this import

def load_demos(path="demos.npz", split_ratio=0.8): # Split ratio for train/test
    data = np.load(path)["demos"]
    num_traj = data.shape[0]
    split = int(num_traj * split_ratio)
    train_data = data[:split]
    test_data = data[split:]
    return train_data, test_data

def pad_and_stack(arrays, pad_value=0.0):
    max_len = max(a.shape[0] for a in arrays)
    batch = []
    mask = []
    for a in arrays:
        pad_width = ((0, max_len - a.shape[0]), (0, 0))
        padded = np.pad(a, pad_width, mode='constant', constant_values=pad_value)
        batch.append(padded)
        mask.append(np.concatenate([np.ones(a.shape[0]), np.zeros(max_len - a.shape[0])]))
    return np.stack(batch), np.stack(mask)

def sample_batch(demos, batch_size=16, n_context_min=3, n_target_min=3):
    obs_list = []
    tar_x_list = []
    tar_y_list = []
    for _ in range(batch_size):
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
        obs_list.append(np.concatenate([obs_x, obs_y], axis=-1))
        tar_x_list.append(tar_x)
        tar_y_list.append(tar_y)
    obs, obs_mask = pad_and_stack(obs_list)
    tar_x, tar_mask = pad_and_stack(tar_x_list)
    tar_y, _ = pad_and_stack(tar_y_list)
    obs = torch.tensor(obs, dtype=torch.float32)
    obs_mask = torch.tensor(obs_mask, dtype=torch.float32)
    tar_x = torch.tensor(tar_x, dtype=torch.float32)
    tar_y = torch.tensor(tar_y, dtype=torch.float32)
    tar_mask = torch.tensor(tar_mask, dtype=torch.float32)
    return obs, tar_x, tar_y, obs_mask, tar_mask

if __name__ == "__main__":
    train_demos, test_demos = load_demos("demos.npz", split_ratio=0.8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Initialize model
    model = CNP(in_shape=(2, 4), hidden_size=256, num_hidden_layers=4, min_std=1e-3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32
    n_epochs = 30000
    train_losses = []
    test_losses = []
    log_epochs = []

    for epoch in range(n_epochs):
        model.train()
        obs, tar_x, tar_y, obs_mask, tar_mask = sample_batch(train_demos, batch_size=batch_size)
        obs = obs.to(device)
        tar_x = tar_x.to(device)
        tar_y = tar_y.to(device)
        obs_mask = obs_mask.to(device)
        tar_mask = tar_mask.to(device)
        loss = model.nll_loss(obs, tar_x, tar_y, observation_mask=obs_mask, target_mask=tar_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 100 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                obs, tar_x, tar_y, obs_mask, tar_mask = sample_batch(test_demos, batch_size=batch_size)
                obs = obs.to(device)
                tar_x = tar_x.to(device)
                tar_y = tar_y.to(device)
                obs_mask = obs_mask.to(device)
                tar_mask = tar_mask.to(device)
                test_loss = model.nll_loss(obs, tar_x, tar_y, observation_mask=obs_mask, target_mask=tar_mask)
            print(f"\rEpoch {epoch+1}/{n_epochs} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}", end="")
            # Record losses for plotting
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            log_epochs.append(epoch+1)
    print()  # for newline after training
    # Save model
    torch.save(model.state_dict(), "cnp_model.pth")

    # Plot train and test loss
    plt.figure()
    plt.plot(log_epochs, train_losses, label="Train Loss")
    plt.plot(log_epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("NLL Loss")
    plt.legend()
    plt.title("Train/Test Loss")
    plt.savefig("loss_plot.png")
    plt.show()
