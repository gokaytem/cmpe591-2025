import torch
from torch import optim
from torch.distributions import Normal
import torch.nn.functional as F

from model import VPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
gamma = 0.75
alpha = 0.2

class Agent():
    def __init__(self):
        self.model = VPG(obs_dim=6, act_dim=2, hl=[512, 512]).to(device)
        self.critic1 = VPG(obs_dim=6 + 2, act_dim=1, hl=[512, 512]).to(device)
        self.critic2 = VPG(obs_dim=6 + 2, act_dim=1, hl=[512, 512]).to(device)
        self.optimizer_actor = optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        self.alpha = alpha
        self.gamma = gamma
        self.memory = []
        self.rewards = []

    def decide_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action_mean, act_std = self.model(state_t).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + 5e-2
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob.detach()

    def update_model(self):
        if not self.memory:
            return
        states, actions, rewards, next_states, dones, log_probs = zip(*self.memory)
        states_t = torch.tensor(states, dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
        log_probs_t = torch.stack(log_probs).unsqueeze(-1).to(device)

        # Critic update
        with torch.no_grad():
            a2_mean, a2_std = self.model(next_states_t).chunk(2, dim=-1)
            a2_std = F.softplus(a2_std) + 5e-2
            dist2 = Normal(a2_mean, a2_std)
            next_a = dist2.rsample()
            next_log_p = dist2.log_prob(next_a).sum(dim=-1, keepdim=True)
            q_input_next = torch.cat([next_states_t, next_a], dim=-1)
            q_next1 = self.critic1(q_input_next)
            q_next2 = self.critic2(q_input_next)
            q_next = torch.min(q_next1, q_next2) - self.alpha * next_log_p
            target = rewards_t + self.gamma * (1 - dones_t) * q_next

        q_input = torch.cat([states_t, actions_t], dim=-1)
        q1 = self.critic1(q_input)
        q2 = self.critic2(q_input)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor update
        a_mean, a_std = self.model(states_t).chunk(2, dim=-1)
        a_std = F.softplus(a_std) + 5e-2
        dist_a = Normal(a_mean, a_std)
        act_sample = dist_a.rsample()
        log_p_a = dist_a.log_prob(act_sample).sum(dim=-1, keepdim=True)
        q_input_a = torch.cat([states_t, act_sample], dim=-1)
        q1_a = self.critic1(q_input_a)
        q2_a = self.critic2(q_input_a)
        q_min = torch.min(q1_a, q2_a)
        actor_loss = (self.alpha * log_p_a - q_min).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.memory = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))