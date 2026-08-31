"""Actor/critic networks for the equilibrium-seeking router.

Each player (agent/tenant/workflow) runs its own discrete-action SAC
policy over "which backend do I send my next request to". Trained
independently against a shared environment, many simultaneous
best-responses settle into a Nash equilibrium — a routing allocation
where no single player can improve their own latency/cost/success
by unilaterally shifting traffic elsewhere.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    Outputs a probability distribution over backends for one player.

    A pure softmax actor can saturate to a one-hot distribution during
    training: once a probability hits a floating-point extreme, the
    entropy gradient (proportional to p·(1-p) per action) vanishes, so
    the auto-tuned entropy temperature loses all ability to correct the
    policy. To prevent this, the output is mixed with a small uniform
    floor — final_probs = (1-eps)*softmax(logits) + eps/n_actions — so
    no action's probability can ever reach exactly 0 or 1, keeping the
    entropy gradient nonzero for the lifetime of training.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256,
                 epsilon_floor: float = 0.02):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon_floor = epsilon_floor
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs, log_probs), both [batch, n_actions]."""
        logits = self.net(obs)
        raw_probs = F.softmax(logits, dim=-1)
        eps = self.epsilon_floor
        action_probs = (1.0 - eps) * raw_probs + eps / self.n_actions
        log_probs = torch.log(action_probs + 1e-8)
        return action_probs, log_probs

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples an action; returns (action, log_prob_of_action, entropy)."""
        action_probs, log_probs = self.forward(obs)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -torch.sum(action_probs * log_probs, dim=-1)
        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """Q-network: estimates the Q-value of each backend given the player's state."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all backends: [batch, n_actions]."""
        return self.net(obs)
