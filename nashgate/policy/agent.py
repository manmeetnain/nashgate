"""Per-player equilibrium-seeking SAC agent.

One instance per player (agent, tenant, or workflow) competing for
shared backend capacity. Each agent only sees its own local
observation and only controls its own routing decision — no agent is
told what the others are doing. Trained independently and
simultaneously against a shared environment, this is standard
independent-learners multi-agent RL: with well-shaped local rewards,
many simultaneous best-responses settle into a Nash equilibrium
without any central coordinator.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from nashgate.policy.networks import ActorNetwork, CriticNetwork
from nashgate.policy.replay_buffer import ReplayBuffer

_DEFAULT_DEVICE = torch.device("cpu")


class NashSACAgent:
    """
    Discrete-action Soft Actor-Critic agent, one per player.

    Two properties matter for equilibrium-seeking behavior specifically
    (as opposed to single-agent SAC, where they're just standard
    stability knobs):

    - The actor's epsilon floor (see ActorNetwork) keeps every action's
      probability nonzero, so the entropy gradient never vanishes and
      the auto-tuned temperature never loses its ability to correct
      the policy.
    - The entropy target is a moderate, annealable scale — a genuine
      best-response under contention is a confident, near-deterministic
      choice, not a near-random one, so pinning the target near maximum
      entropy forever gives alpha nothing to converge to. Both a max
      and min clamp on alpha keep the temperature from either running
      away (dominating the actor's gradient, degrading behavior) or
      collapsing to exactly zero (removing all regularization).
    """

    def __init__(
        self,
        player_id: int,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
        epsilon_floor: float = 0.02,
        target_entropy_scale: float = 0.3,
        alpha_max: float = 0.5,
        alpha_min: float = 0.01,
    ):
        self.player_id = player_id
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy = auto_entropy
        self.device = device if device is not None else _DEFAULT_DEVICE
        self.epsilon_floor = epsilon_floor

        self.actor = ActorNetwork(obs_dim, n_actions, hidden_dim, epsilon_floor).to(self.device)
        self.critic1 = CriticNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.critic2 = CriticNetwork(obs_dim, n_actions, hidden_dim).to(self.device)

        self.critic1_target = CriticNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.critic2_target = CriticNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        self.target_entropy_scale = target_entropy_scale
        self.target_entropy = -np.log(1.0 / n_actions) * target_entropy_scale
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.log_alpha = torch.tensor(
            np.log(alpha), requires_grad=True, device=self.device, dtype=torch.float32
        )
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = alpha

        self.buffer = ReplayBuffer(buffer_capacity)

        self.total_steps = 0
        self.updates = 0
        self.losses = {"actor": [], "critic1": [], "critic2": [], "alpha": [], "entropy": []}

    def set_epsilon_floor(self, epsilon_floor: float):
        """Update the actor's uniform probability floor mid-training.

        A larger floor early on forces broader sampling of alternative
        backends, giving the critic a chance to actually observe
        transitions for options the policy would otherwise never try —
        anneal it down later to let the policy commit to a confident
        best-response once it's found one.
        """
        self.epsilon_floor = epsilon_floor
        self.actor.epsilon_floor = epsilon_floor

    def set_target_entropy_scale(self, scale: float):
        """Update the entropy target mid-training (for an annealing schedule)."""
        self.target_entropy_scale = scale
        self.target_entropy = -np.log(1.0 / self.n_actions) * scale

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """
        explore=True: sample from the policy distribution (training-time
        environment interaction — always used while training, since a
        deterministic argmax during data collection starves the critic
        of the exploration it needs to learn accurate Q-values).
        explore=False: take the greedy argmax (evaluation / serving only).
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if explore:
                action, _, _ = self.actor.sample(obs_t)
                return action.item()
            action_probs, _ = self.actor.forward(obs_t)
            return action_probs.argmax(dim=-1).item()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def update(self) -> Optional[dict]:
        if len(self.buffer) < self.batch_size:
            return None

        obs, acts, rews, next_obs, dones = self.buffer.sample(self.batch_size, self.device)

        # 1. Critics
        with torch.no_grad():
            next_action_probs, next_log_probs = self.actor.forward(next_obs)
            q1_next = self.critic1_target(next_obs)
            q2_next = self.critic2_target(next_obs)
            q_next = torch.min(q1_next, q2_next)
            v_next = (next_action_probs * (q_next - self.alpha * next_log_probs)).sum(dim=-1)
            q_target = rews + self.gamma * (1.0 - dones) * v_next

        q1_pred = self.critic1(obs).gather(1, acts.unsqueeze(-1)).squeeze(-1)
        q2_pred = self.critic2(obs).gather(1, acts.unsqueeze(-1)).squeeze(-1)

        critic1_loss = F.mse_loss(q1_pred, q_target)
        critic2_loss = F.mse_loss(q2_pred, q_target)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optim.step()

        # 2. Actor
        action_probs, log_probs = self.actor.forward(obs)
        q1 = self.critic1(obs)
        q2 = self.critic2(obs)
        q_min = torch.min(q1, q2)

        actor_loss = (action_probs * (self.alpha * log_probs - q_min)).sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # 3. Entropy temperature alpha
        if self.auto_entropy:
            entropy = -(action_probs * log_probs).sum(dim=-1).mean()
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            # Clamp log_alpha in place (not just the read-out value) so the
            # Adam optimizer's internal momentum doesn't keep pushing it
            # past the bound while the clamp only masks the symptom.
            self.log_alpha.data.clamp_(
                min=float(np.log(self.alpha_min)), max=float(np.log(self.alpha_max))
            )
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)
            entropy = torch.tensor(0.0)

        # 4. Soft-update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        self.updates += 1

        loss_dict = {
            "critic1": critic1_loss.item(),
            "critic2": critic2_loss.item(),
            "actor": actor_loss.item(),
            "alpha": alpha_loss.item() if self.auto_entropy else 0.0,
            "entropy": entropy.item() if hasattr(entropy, "item") else float(entropy),
            "alpha_val": self.alpha,
        }
        for k in ["actor", "critic1", "critic2", "alpha", "entropy"]:
            self.losses[k].append(loss_dict.get(k, 0.0))
        return loss_dict

    def _soft_update(self, source, target):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.tau * src_p.data + (1.0 - self.tau) * tgt_p.data)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "total_steps": self.total_steps,
            "updates": self.updates,
            "epsilon_floor": self.epsilon_floor,
        }, os.path.join(path, f"player_{self.player_id}.pt"))

    def load(self, path: str):
        ckpt = torch.load(
            os.path.join(path, f"player_{self.player_id}.pt"), map_location=self.device
        )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.alpha = self.log_alpha.exp().item()
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates = ckpt.get("updates", 0)
