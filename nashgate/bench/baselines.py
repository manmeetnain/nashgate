"""The static routing strategies real gateways ship today. Each one
implements the same interface as the trained policy — route(obs_dict)
-> actions_dict — reading the identical shared, per-backend features
the policy sees (nashgate.env.features), so the comparison is over
routing logic, not information available.

None of these know or care that other callers are choosing at the
same time — that's the point being tested.
"""

import random
from typing import Dict, List

import numpy as np

from nashgate.env.backend_state import BackendConfig
from nashgate.env.features import LOCAL_OBS_DIM, SHARED_OBS_PER_BACKEND


def shared_features(obs: np.ndarray, n_backends: int) -> np.ndarray:
    """[n_backends, 5] -> columns: queue_depth, latency_ema, error_rate, rate_limit_headroom, cost_per_1k."""
    return obs[LOCAL_OBS_DIM:].reshape(n_backends, SHARED_OBS_PER_BACKEND)


class RoundRobinRouter:
    """Cycles through backends per caller in fixed order. What a gateway
    with no routing logic at all does."""

    def __init__(self, n_callers: int, n_backends: int):
        self.n_backends = n_backends
        self._next = {i: 0 for i in range(n_callers)}

    def route(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
        actions = {}
        for caller_id in obs_dict:
            actions[caller_id] = self._next[caller_id]
            self._next[caller_id] = (self._next[caller_id] + 1) % self.n_backends
        return actions


class WeightedRouter:
    """Fixed probability per backend, set once at startup from configured
    capacity and never adjusted again — LiteLLM's `weighted` strategy."""

    def __init__(self, weights: List[float], seed: int = 0):
        self.n_backends = len(weights)
        self.weights = weights
        self._rng = random.Random(seed)

    @classmethod
    def from_backend_configs(cls, backend_configs: List[BackendConfig], seed: int = 0) -> "WeightedRouter":
        total = sum(c.rate_limit_per_window for c in backend_configs)
        weights = [c.rate_limit_per_window / total for c in backend_configs]
        return cls(weights, seed=seed)

    def route(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
        choices = list(range(self.n_backends))
        return {cid: self._rng.choices(choices, weights=self.weights)[0] for cid in obs_dict}


class LatencyBasedRouter:
    """Greedy: always send to whichever backend currently reports the
    lowest latency. Reacts to load, but every caller reacts identically
    and simultaneously — with no notion that everyone else just made the
    exact same greedy choice, they all pile onto the same "fastest"
    backend together."""

    def __init__(self, n_backends: int):
        self.n_backends = n_backends

    def route(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
        actions = {}
        for caller_id, obs in obs_dict.items():
            latency_col = shared_features(obs, self.n_backends)[:, 1]
            actions[caller_id] = int(np.argmin(latency_col))
        return actions


class CostBasedRouter:
    """Greedy: always send to whichever backend is cheapest per 1k
    tokens. Static in the sense that matters — it never weighs cost
    against the congestion its own greediness is causing."""

    def __init__(self, n_backends: int):
        self.n_backends = n_backends

    def route(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
        actions = {}
        for caller_id, obs in obs_dict.items():
            cost_col = shared_features(obs, self.n_backends)[:, 4]
            actions[caller_id] = int(np.argmin(cost_col))
        return actions


class TrainedRouterAdapter:
    """Wraps NashEquilibriumRouter to the same route(obs_dict) interface,
    evaluating greedily (no exploration, no learning) so the comparison
    is apples-to-apples against the static baselines above."""

    def __init__(self, policy):
        self.policy = policy

    def route(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
        return self.policy.route(obs_dict, explore=False)
