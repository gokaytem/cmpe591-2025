import numpy as np
import matplotlib.pyplot as plt
from homework4 import Hw5Env, bezier

if __name__ == "__main__":
    env = Hw5Env(render_mode="offscreen")
    demos = []
    for i in range(1000):
        env.reset()
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        traj = []
        for t_idx, p in enumerate(curve):
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            state = env.high_level_state()
            ey, ez, oy, oz, h = state[0], state[1], state[2], state[3], state[4]
            t = t_idx / (len(curve) - 1)
            traj.append([t, ey, ez, oy, oz, h])
        traj = np.stack(traj)
        demos.append(traj)
        print(f"Collected {i+1} trajectories.", end="\r")

    demos = np.array(demos)
    np.savez("demos.npz", demos=demos)

    fig, ax = plt.subplots(1, 2)
    for traj in demos:
        ax[0].plot(traj[:, 1], traj[:, 2], alpha=0.2, color="b")
        ax[0].set_xlabel("e_y")
        ax[0].set_ylabel("e_z")
        ax[1].plot(traj[:, 3], traj[:, 4], alpha=0.2, color="r")
        ax[1].set_xlabel("o_y")
        ax[1].set_ylabel("o_z")
    plt.show()
