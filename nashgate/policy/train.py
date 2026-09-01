"""Trains a NashEquilibriumRouter against the routing game.

The canonical training loop — used directly by `nashgate policy
train`, and as the lightweight pretrain `nashgate bench` runs against
when no `--checkpoint` is given.
"""

from collections.abc import Callable

from nashgate.env.backend_state import BackendConfig
from nashgate.env.caller_state import CallerConfig
from nashgate.env.routing_env import MultiAgentRoutingEnv
from nashgate.policy.router_policy import NashEquilibriumRouter


def train_policy(
    backend_configs: list[BackendConfig],
    caller_configs: list[CallerConfig],
    steps: int = 20_000,
    window_steps: int = 50,
    batch_size: int = 256,
    seed: int = 0,
    on_progress: Callable[[int, int, dict], None] | None = None,
    progress_every: int = 1000,
) -> NashEquilibriumRouter:
    """Runs `steps` training steps against a fresh `MultiAgentRoutingEnv`.

    `on_progress(step, steps, stats)` — stats has `mean_reward` and
    `mean_alpha` since the last call — is invoked every `progress_every`
    steps if given. Kept as a callback rather than a print statement so
    this stays usable as a library function, not just a CLI internal.
    """
    env = MultiAgentRoutingEnv(
        backend_configs, caller_configs, episode_len=steps, window_steps=window_steps, seed=seed
    )
    policy = NashEquilibriumRouter(
        n_players=env.n_callers, obs_dim=env.obs_dim, n_backends=env.n_backends, batch_size=batch_size
    )

    obs = env.reset()
    reward_sum = 0.0
    reward_count = 0
    for step in range(1, steps + 1):
        actions = policy.route(obs, explore=True)
        next_obs, rewards, done, _info = env.step(actions)
        policy.store(obs, actions, rewards, next_obs, done)
        policy.update()
        reward_sum += sum(rewards.values())
        reward_count += len(rewards)
        obs = next_obs
        if done:
            obs = env.reset()

        if on_progress and step % progress_every == 0:
            mean_alpha = sum(a.alpha for a in policy.agents.values()) / len(policy.agents)
            on_progress(step, steps, {
                "mean_reward": reward_sum / max(1, reward_count),
                "mean_alpha": mean_alpha,
            })
            reward_sum = 0.0
            reward_count = 0

    return policy
