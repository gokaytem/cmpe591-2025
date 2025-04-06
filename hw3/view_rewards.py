import numpy as np
import matplotlib.pyplot as plt

def smooth_data(data, window=1):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def plot_rewards(file_path):
    rewards = np.load(file_path)
    rewards = smooth_data(rewards)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progress')
    plt.show()

if __name__ == "__main__":
    plot_rewards("rews_uhem_10000.npy")