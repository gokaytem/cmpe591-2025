import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from homework2 import Hw2Env

N_ACTIONS = 8
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_DECAY_ITER = 10
MIN_EPSILON = 0.1
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
UPDATE_FREQ = 4
TARGET_NETWORK_UPDATE_FREQ = 100
BUFFER_LENGTH = 10000

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),  # (-1, 3, 128, 128) -> (-1, 32, 64, 64)
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # (-1, 32, 64, 64) -> (-1, 64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),  # (-1, 64, 32, 32) -> (-1, 128, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),  # (-1, 128, 16, 16) -> (-1, 256, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),  # (-1, 256, 8, 8) -> (-1, 512, 4, 4)
            nn.AdaptiveAvgPool2d((1, 1)),  # average pooling over the spatial dimensions  (-1, 512, 4, 4) -> (-1, 512)
            nn.Flatten(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.network(x)

def select_action(state, epsilon, n_actions, policy_net):
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

    states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[0]])
    actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(batch[2], dtype=torch.float32)
    next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch[3]])
    dones = torch.tensor(batch[4], dtype=torch.float32)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = nn.functional.mse_loss(q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
policy_net = DQN(N_ACTIONS)
target_net = DQN(N_ACTIONS)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=BUFFER_LENGTH)

steps_done = 0

for episode in range(5):
    state = env.reset()
    done = False
    cumulative_reward = 0.0
    episode_steps = 0

    while not done:
        action = select_action(state, EPSILON, N_ACTIONS, policy_net)
        next_state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        cumulative_reward += reward
        episode_steps += 1

        if steps_done % UPDATE_FREQ == 0:
            optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA)

        if steps_done % TARGET_NETWORK_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY

    print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")