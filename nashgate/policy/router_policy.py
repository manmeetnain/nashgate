"""Coordinates one NashSACAgent per player into a single routing policy.

This is the piece that actually makes it "Nash-SAC" rather than plain
SAC: N independent agents, each with only its own local observation
and reward, act simultaneously against the shared backend state every
step. No agent sees another agent's policy or reward directly — they
only feel each other's effects through the shared environment (e.g.
a backend getting slower because another player is also routing to
it). Trained this way, independent best-responses settle into a Nash
equilibrium: a stable allocation where no player can do better by
unilaterally routing elsewhere.
"""

from typing import Dict, Optional

import numpy as np

from nashgate.policy.agent import NashSACAgent


class NashEquilibriumRouter:
    def __init__(self, n_players: int, obs_dim: int, n_backends: int, **agent_kwargs):
        self.n_players = n_players
        self.agents: Dict[int, NashSACAgent] = {
            i: NashSACAgent(player_id=i, obs_dim=obs_dim, n_actions=n_backends, **agent_kwargs)
            for i in range(n_players)
        }

    def route(self, obs: Dict[int, np.ndarray], explore: bool = True) -> Dict[int, int]:
        """One observation per player in, one backend choice per player out."""
        return {
            i: self.agents[i].select_action(obs[i], explore=explore)
            for i in range(self.n_players)
        }

    def store(self, obs, actions, rewards, next_obs, done: bool):
        done_f = float(done)
        for i in range(self.n_players):
            self.agents[i].store(obs[i], actions[i], rewards[i], next_obs[i], done_f)

    def update(self) -> Dict[int, Optional[dict]]:
        return {i: self.agents[i].update() for i in range(self.n_players)}

    def save(self, path: str):
        for agent in self.agents.values():
            agent.save(path)

    def load(self, path: str):
        for agent in self.agents.values():
            agent.load(path)
