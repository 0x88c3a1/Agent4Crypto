import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """Predict continuous agent weights for a single state."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        features = self.backbone(state_tensor)
        weights = self.actor_head(features)
        value = self.critic_head(features)
        return weights.squeeze(0), value.squeeze(0)

    def evaluate(self, state, action):
        """Evaluate a batch of actions under the current policy."""
        features = self.backbone(state)
        weights = self.actor_head(features)
        value = self.critic_head(features)

        dist = torch.distributions.Dirichlet(weights * 10 + 0.1)
        action = torch.clamp(action, min=1e-6)
        action = action / action.sum(dim=-1, keepdim=True)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy


class PPOAgent:
    def __init__(self, state_dim, lr=0.0005, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = []
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            features = self.policy_old.backbone(state_tensor)
            weights = self.policy_old.actor_head(features)
            dist = torch.distributions.Dirichlet(weights.squeeze(0) * 10 + 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.numpy(), log_prob.item()

    def predict_action(self, state):
        """Deterministically return the current policy mean for frozen-test inference."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            features = self.policy_old.backbone(state_tensor)
            weights = self.policy_old.actor_head(features).squeeze(0)
            weights = torch.clamp(weights, min=1e-6)
            weights = weights / weights.sum()
        return weights.numpy()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def update(self):
        if not self.buffer:
            return

        states = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float32)
        log_probs_old = torch.tensor(np.array([t[2] for t in self.buffer]), dtype=torch.float32)
        rewards = torch.tensor(np.array([t[3] for t in self.buffer]), dtype=torch.float32)

        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)

        if returns.numel() > 1:
            returns_std = returns.std(unbiased=False)
            if torch.isfinite(returns_std) and returns_std > 1e-5:
                returns = (returns - returns.mean()) / (returns_std + 1e-5)

        for _ in range(self.K_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            state_values = state_values.squeeze(-1)
            ratio = torch.exp(log_probs - log_probs_old)
            advantages = returns - state_values.detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.loss_fn(state_values, returns)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []
