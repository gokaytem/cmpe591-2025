import torch
from torch import optim
from torch.distributions import Normal

from model import VPG
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamma = 0.99
lr = 1e-4

class Agent():
    def __init__(self):
        self.model = VPG(obs_dim=6, act_dim=2, hl=[256, 256]).to(device)
        self.rewards = []
        self.log_probs = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def decide_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + 5e-2
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.detach()

    def update_model(self):
        discounted_returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)

        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32).to(device)

        mean = discounted_returns.mean()
        std = discounted_returns.std(unbiased=False) + 1e-8
        discounted_returns = (discounted_returns - mean) / std

        if len(discounted_returns) == len(self.log_probs):
            loss = []
            for log_p, R in zip(self.log_probs, discounted_returns):
                loss.append(-log_p.sum() * R)

            self.optimizer.zero_grad()
            torch.stack(loss).sum().backward()
            self.optimizer.step()

        self.rewards = []
        self.log_probs = []

    def add_reward(self, reward):
        self.rewards.append(reward)
