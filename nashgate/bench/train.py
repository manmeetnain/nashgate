"""Minimal training loop — trains a fresh policy against the routing
game so `nashgate bench` has something real to compare when no
checkpoint is given. Not a substitute for a proper training script;
just enough steps to reach a stable, non-random policy."""

from typing import List

from nashgate.env.backend_state import BackendConfig
from nashgate.env.caller_state import CallerConfig
from nashgate.env.routing_env import MultiAgentRoutingEnv
from nashgate.policy import NashEquilibriumRouter


def train_policy(
    backend_configs: List[BackendConfig],
    caller_configs: List[CallerConfig],
    steps: int = 20_000,
    window_steps: int = 50,
    batch_size: int = 256,
    seed: int = 0,
) -> NashEquilibriumRouter:
    env = MultiAgentRoutingEnv(
        backend_configs, caller_configs, episode_len=steps, window_steps=window_steps, seed=seed
    )
    policy = NashEquilibriumRouter(
        n_players=env.n_callers, obs_dim=env.obs_dim, n_backends=env.n_backends, batch_size=batch_size
    )

    obs = env.reset()
    for _ in range(steps):
        actions = policy.route(obs, explore=True)
        next_obs, rewards, done, _info = env.step(actions)
        policy.store(obs, actions, rewards, next_obs, done)
        policy.update()
        obs = next_obs
        if done:
            obs = env.reset()

    return policy
