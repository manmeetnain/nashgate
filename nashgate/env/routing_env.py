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

ACTION (per caller): Discrete(n_backends) — which backend to send the
next request to.

REWARD (per caller, per request):
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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from nashgate.env.backend_state import BackendConfig, BackendState

LOCAL_OBS_DIM = 5
SHARED_OBS_PER_BACKEND = 5

W_SUCCESS = 1.0
W_VIOLATION = 1.5
W_LATENCY = 0.4
W_COST = 0.3

OWN_CONCURRENCY_CAP = 8.0
QUEUE_DEPTH_NORM = 20.0
MAX_REQUEST_TOKENS = 4000.0
MAX_COST_PER_1K_REF = 0.06
HARD_TIMEOUT_MULTIPLE = 3.0
RANDOM_ERROR_RATE = 0.01


@dataclass
class CallerConfig:
    sla_latency_ms: float = 2000.0
    cost_budget_per_window: float = 1.0
    min_request_tokens: int = 200
    max_request_tokens: int = 2000


@dataclass
class _CallerRuntime:
    config: CallerConfig
    inflight: int = 0
    budget_remaining: float = field(init=False)
    last_reward: float = 0.0
    success_ema: float = 1.0
    pending_tokens: int = 0

    def __post_init__(self):
        self.budget_remaining = self.config.cost_budget_per_window


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
        self.obs_dim = LOCAL_OBS_DIM + SHARED_OBS_PER_BACKEND * self.n_backends

        self._rng = random.Random(seed)
        self.backends: List[BackendState] = []
        self.callers: List[_CallerRuntime] = []
        self.step_count = 0
        self.reset()

    def reset(self) -> Dict[int, np.ndarray]:
        self.backends = [BackendState(config=c) for c in self.backend_configs]
        self.callers = [_CallerRuntime(config=c) for c in self.caller_configs]
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
        info: dict = {"rate_limited": [], "errored": []}

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

            reward = (W_SUCCESS if success else -W_VIOLATION)
            reward -= W_LATENCY * min(latency_ms / caller.config.sla_latency_ms, HARD_TIMEOUT_MULTIPLE)
            reward -= W_COST * min(cost / max(caller.config.cost_budget_per_window, 1e-6), HARD_TIMEOUT_MULTIPLE)

            caller.inflight = min(int(OWN_CONCURRENCY_CAP), caller.inflight + 1)
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
                caller.budget_remaining = caller.config.cost_budget_per_window

        done = self.step_count >= self.episode_len
        return self._build_all_obs(), rewards, done, info

    def _build_all_obs(self) -> Dict[int, np.ndarray]:
        shared = self._build_shared_obs()
        return {i: self._build_local_obs(i, shared) for i in range(self.n_callers)}

    def _build_shared_obs(self) -> np.ndarray:
        features = []
        for backend in self.backends:
            caller_sla_ref = self.caller_configs[0].sla_latency_ms
            features.extend([
                min(backend.in_flight / QUEUE_DEPTH_NORM, 1.0),
                min(backend.latency_ema_ms / caller_sla_ref, HARD_TIMEOUT_MULTIPLE),
                backend.error_rate_ema,
                backend.rate_limit_headroom(),
                min(backend.config.cost_per_1k_tokens / MAX_COST_PER_1K_REF, 1.0),
            ])
        return np.array(features, dtype=np.float32)

    def _build_local_obs(self, caller_id: int, shared: np.ndarray) -> np.ndarray:
        caller = self.callers[caller_id]
        local = np.array([
            caller.budget_remaining / max(caller.config.cost_budget_per_window, 1e-6),
            min(caller.inflight / OWN_CONCURRENCY_CAP, 1.0),
            float(np.clip(caller.last_reward, -1.0, 1.0)),
            min(caller.pending_tokens / MAX_REQUEST_TOKENS, 1.0),
            caller.success_ema,
        ], dtype=np.float32)
        return np.concatenate([local, shared])


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
