"""The routing game.

One "episode" is a window of routing decisions. Each step, every
caller (agent, tenant, or workflow sharing this gateway) picks one
backend for its next request. All callers act simultaneously; a
caller's own choice, and every other caller's choice that step,
together determine how congested each backend is — which is exactly
what makes this a game rather than N independent bandit problems.

OBSERVATION (per caller) — local (5) + shared, per-backend (5 x n_backends):

  local:
    cost_budget_remaining   fraction of this caller's per-window cost budget left, [0, 1]
    own_inflight            this caller's own unfinished requests, normalized, [0, 1]
    last_reward             previous step's reward, clipped to [-1, 1]
    request_size            token size of the request about to be routed, normalized, [0, 1]
    own_success_rate        EMA of this caller's own success rate, [0, 1]

  per backend b (same values visible to every caller, like a load
  balancer's own dashboard — nobody sees what a specific other caller
  is about to do, only the aggregate effect):
    queue_depth[b]          in-flight requests across all callers, normalized, [0, 1]
    latency_ema[b]          recent observed latency / this caller's SLA, clipped [0, 3]
    error_rate[b]           EMA of recent errors/timeouts, [0, 1]
    rate_limit_headroom[b]  fraction of this window's rate limit unused, [0, 1]
    cost_per_1k[b]          backend's cost per 1k tokens, normalized, [0, 1]

  See nashgate/env/features.py — this is the exact function the live
  router builds observations with too, against real state instead of
  simulated state.

ACTION (per caller): Discrete(n_backends) — which backend to send the
next request to.

REWARD (per caller, per request): see nashgate/env/reward.py.
  +1.0                          if the request succeeded within SLA
  -1.5                          if it was rate-limited, errored, or missed SLA
  -0.4 * clip(latency / sla, 0, 3)   always-on latency pressure
  -0.3 * clip(cost / budget, 0, 3)   always-on cost pressure

  Routing everyone to the fastest/cheapest backend drives its queue
  depth up, which drives latency up for the next request sent there —
  that feedback loop is the whole point: it's what makes "always pick
  the best backend" a losing static strategy and gives independent
  learners something to find an equilibrium over.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

from nashgate.env.backend_state import BackendConfig, BackendState
from nashgate.env.caller_state import CallerConfig, CallerRuntime
from nashgate.env.features import build_local_obs, build_shared_obs, obs_dim
from nashgate.env.reward import HARD_TIMEOUT_MULTIPLE, compute_reward

RANDOM_ERROR_RATE = 0.01


class MultiAgentRoutingEnv:
    """Gym-style multi-agent env: reset() -> obs, step(actions) -> (obs, rewards, done, info)."""

    def __init__(
        self,
        backend_configs: List[BackendConfig],
        caller_configs: List[CallerConfig],
        episode_len: int = 200,
        window_steps: int = 50,
        seed: int = 0,
    ):
        self.backend_configs = backend_configs
        self.caller_configs = caller_configs
        self.n_backends = len(backend_configs)
        self.n_callers = len(caller_configs)
        self.episode_len = episode_len
        self.window_steps = window_steps
        self.obs_dim = obs_dim(self.n_backends)

        self._rng = random.Random(seed)
        self.backends: List[BackendState] = []
        self.callers: List[CallerRuntime] = []
        self.step_count = 0
        self.reset()

    def reset(self) -> Dict[int, np.ndarray]:
        self.backends = [BackendState(config=c) for c in self.backend_configs]
        self.callers = [CallerRuntime(config=c) for c in self.caller_configs]
        self.step_count = 0
        for caller in self.callers:
            caller.pending_tokens = self._rng.randint(
                caller.config.min_request_tokens, caller.config.max_request_tokens
            )
        return self._build_all_obs()

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, dict]:
        rewards: Dict[int, float] = {}
        info: dict = {"rate_limited": [], "errored": [], "success": {}, "backend_chosen": dict(actions)}

        for caller_id, backend_idx in actions.items():
            caller = self.callers[caller_id]
            backend = self.backends[backend_idx]

            if backend.is_rate_limited():
                latency_ms = backend.config.base_latency_ms * HARD_TIMEOUT_MULTIPLE
                success = False
                errored = True
                cost = 0.0
                info["rate_limited"].append(caller_id)
            else:
                congestion = backend.config.congestion_latency_ms_per_inflight * backend.in_flight
                jitter = self._rng.uniform(-1.0, 1.0) * backend.config.latency_jitter_ms
                latency_ms = max(1.0, backend.config.base_latency_ms + congestion + jitter)
                random_error = self._rng.random() < RANDOM_ERROR_RATE
                errored = random_error or latency_ms > caller.config.sla_latency_ms * HARD_TIMEOUT_MULTIPLE
                cost = (caller.pending_tokens / 1000.0) * backend.config.cost_per_1k_tokens
                success = (not errored) and latency_ms <= caller.config.sla_latency_ms

                backend.in_flight += 1
                backend.requests_this_window += 1
                if errored:
                    info["errored"].append(caller_id)

            backend.update_ema(latency_ms, errored)
            reward = compute_reward(latency_ms, cost, success, caller.config)
            info["success"][caller_id] = success

            caller.inflight += 1
            caller.budget_remaining = max(0.0, caller.budget_remaining - cost)
            caller.success_ema = 0.9 * caller.success_ema + 0.1 * float(success)
            caller.last_reward = reward
            caller.pending_tokens = self._rng.randint(
                caller.config.min_request_tokens, caller.config.max_request_tokens
            )
            rewards[caller_id] = reward

        for backend in self.backends:
            backend.decay_in_flight()
        for caller in self.callers:
            caller.inflight = max(0, caller.inflight - 1)

        self.step_count += 1
        if self.step_count % self.window_steps == 0:
            for backend in self.backends:
                backend.reset_window()
            for caller in self.callers:
                caller.reset_window()

        done = self.step_count >= self.episode_len
        return self._build_all_obs(), rewards, done, info

    def _build_all_obs(self) -> Dict[int, np.ndarray]:
        shared = build_shared_obs(self.backends, self.caller_configs[0].sla_latency_ms)
        return {i: build_local_obs(self.callers[i], shared) for i in range(self.n_callers)}


if __name__ == "__main__":
    backends = [
        BackendConfig(base_latency_ms=300, latency_jitter_ms=50, cost_per_1k_tokens=0.01, rate_limit_per_window=40),
        BackendConfig(base_latency_ms=600, latency_jitter_ms=80, cost_per_1k_tokens=0.002, rate_limit_per_window=100),
        BackendConfig(base_latency_ms=1200, latency_jitter_ms=150, cost_per_1k_tokens=0.0005, rate_limit_per_window=200),
    ]
    callers = [CallerConfig() for _ in range(4)]
    env = MultiAgentRoutingEnv(backends, callers, episode_len=20, window_steps=10, seed=1)
    obs = env.reset()
    print(f"obs_dim={env.obs_dim}  n_backends={env.n_backends}  n_callers={env.n_callers}")
    for _ in range(5):
        actions = {i: random.randrange(env.n_backends) for i in range(env.n_callers)}
        obs, rewards, done, info = env.step(actions)
        print(f"actions={actions}  rewards={ {k: round(v, 2) for k, v in rewards.items()} }")
    print("OK")
