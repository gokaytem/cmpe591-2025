import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

from homework2 import Hw2Env

N_ACTIONS = 8
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_DECAY_ITER = 100
MIN_EPSILON = 0.05
LEARNING_RATE = 0.005
BATCH_SIZE = 32
UPDATE_FREQ = 10
TARGET_NETWORK_UPDATE_FREQ = 100
BUFFER_LENGTH = 10000
NUM_OF_EPISODES = 5000

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.network(x)
    
def select_action(state, epsilon, n_actions, policy_net):
    if state is None:
        return np.random.randint(n_actions)
    if random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    # Convert batch data to tensors and move to the appropriate device
    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0] if s is not None])
    actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[3] if s is not None])
    dones = torch.tensor(batch[4], dtype=torch.float32)

    # Compute Q values for current states and actions
    q_values = policy_net(states).gather(1, actions)
    # Compute Q values for next states using the target network
    next_q_values = target_net(next_states).max(1)[0].detach()
    # Compute expected Q values using the Bellman equation
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Compute loss between current Q values and expected Q values
    loss = nn.functional.mse_loss(q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    global EPSILON  # Declare EPSILON as global to modify it within the function

    # Initialize the environment and the policy and target networks
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    policy_net = DQN(N_ACTIONS).train()  # Ensure training mode
    target_net = DQN(N_ACTIONS).eval()

    # Load previously saved models if they exist
    if os.path.exists('policy_net_2.pth'):
        policy_net.load_state_dict(torch.load('policy_net_2.pth'))
    if os.path.exists('target_net_2.pth'):
        target_net.load_state_dict(torch.load('target_net_2.pth'))

    # Initialize the optimizer and the replay memory
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=BUFFER_LENGTH)

    steps_done = 0

    # Load training progress if it exists
    start_episode = 0
    rewards_list = []
    rps_list = []

    if os.path.exists('training_progress_2.json'):
        with open('training_progress_2.json', 'r') as f:
            progress = json.load(f)
            start_episode = progress['last_episode'] + 1
            rewards_list = progress['rewards_list']
            rps_list = progress['rps_list']

    # Main training loop
    for episode in range(start_episode, NUM_OF_EPISODES):

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY

        state = env.reset()  # Initialize the state
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        old_reward = 0.0  # Initialize old_reward

        while not done:
            # Select an action using the epsilon-greedy policy
            action = select_action(state, EPSILON, N_ACTIONS, policy_net)
            next_state, new_reward, is_terminal, is_truncated = env.step(action)
            reward = new_reward - old_reward  # Calculate reward as new_reward - old_reward
            reward = reward * 2 if reward < 0 else reward  # Scale negative rewards by 2
            old_reward = new_reward  # Update old_reward
            done = is_terminal or is_truncated

            # Store the transition in memory
            if state is not None and next_state is not None:
                memory.append((state, action, reward, next_state, done))
            state = next_state
            cumulative_reward += reward
            episode_steps += 1

            # Optimize the model every UPDATE_FREQ steps
            if steps_done % UPDATE_FREQ == 0:
                optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA)

            # Update the target network every TARGET_NETWORK_UPDATE_FREQ steps
            if steps_done % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1

        # Record the cumulative reward and reward per step (RPS) for the episode
        rewards_list.append(cumulative_reward)
        rps_list.append(cumulative_reward / episode_steps)
        print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")

        # Save the model and training progress every 100 episodes
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), 'policy_net_2.pth')
            torch.save(target_net.state_dict(), 'target_net_2.pth')
            with open('training_progress_2.json', 'w') as f:
                json.dump({
                    'last_episode': episode,
                    'rewards_list': rewards_list,
                    'rps_list': rps_list
                }, f)

    # Save the trained model and training progress at the end of training
    torch.save(policy_net.state_dict(), 'policy_net_2.pth')
    torch.save(target_net.state_dict(), 'target_net_2.pth')
    with open('training_progress_2.json', 'w') as f:
        json.dump({
            'last_episode': NUM_OF_EPISODES - 1,
            'rewards_list': rewards_list,
            'rps_list': rps_list
        }, f)

    # Plotting the rewards and RPS
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')

    plt.subplot(1, 2, 2)
    plt.plot(rps_list)
    plt.xlabel('Episode')
    plt.ylabel('RPS (Reward per Step)')
    plt.title('RPS over Episodes')

    plt.tight_layout()
    plt.savefig('rewards_and_rps.png')  # Save the figure
    plt.show()

if __name__ == "__main__":
    train()