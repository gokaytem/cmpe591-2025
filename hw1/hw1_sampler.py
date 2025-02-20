import numpy as np
from multiprocessing import Process
import torch
from homework1 import Hw1Env

def collect(idx, N):
    print(f"Process {idx} started.")
    env = Hw1Env(render_mode="offscreen")
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    for i in range(N):
        action_id = np.random.randint(4)
        _, pixels_before = env.state()
        env.step(action_id)
        obj_pos, pixels_after = env.state()
        positions[i] = torch.tensor(obj_pos)
        actions[i] = action_id
        imgs_before[i] = pixels_before
        imgs_after[i] = pixels_after
        env.reset()
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"Process {idx}: {i + 1}/{N} samples collected.")
    # Split data into training and test sets
    test_size = int(0.1 * N)
    train_size = N - test_size

    # Training data
    torch.save(positions[:train_size], f"./hw1_sample/positions_{idx}.pt")
    torch.save(actions[:train_size], f"./hw1_sample/actions_{idx}.pt")
    torch.save(imgs_before[:train_size], f"./hw1_sample/imgsBefore_{idx}.pt")
    torch.save(imgs_after[:train_size], f"./hw1_sample/imgsAfter_{idx}.pt")

    # Test data
    torch.save(positions[train_size:], f"./hw1_test/positions_{idx}.pt")
    torch.save(actions[train_size:], f"./hw1_test/actions_{idx}.pt")
    torch.save(imgs_before[train_size:], f"./hw1_test/imgsBefore_{idx}.pt")
    torch.save(imgs_after[train_size:], f"./hw1_test/imgsAfter_{idx}.pt")
    print(f"Process {idx} finished.")

if __name__ == "__main__":
    processes = []
    for i in range(8):
        p = Process(target=collect, args=(i, 125))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All processes finished.")