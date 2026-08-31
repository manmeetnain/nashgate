"""Wires live requests to the trained equilibrium-seeking policy.

Usage from the gateway layer, per request:

    routed = router.select_backend(caller_id, request_tokens)
    # ... gateway actually calls routed.backend_id, times it, catches errors ...
    router.report_result(routed, latency_ms, cost, success)

select_backend() builds an observation from the router's own live
backend/caller state (nashgate.env.features — the same functions the
training env uses, so there's no train/serve skew) and asks that
caller's agent which backend to use. report_result() scores the
outcome with the same reward used in training, updates live state so
the *next* request already sees this one's effect on load, and — if
online_learning is on — pushes the transition into that agent's
replay buffer and runs an update step, so the policy keeps adapting
to real traffic instead of freezing at whatever it learned offline.
"""

import time
from dataclasses import dataclass

import numpy as np

from nashgate.env.backend_state import BackendConfig, BackendState
from nashgate.env.caller_state import CallerConfig, CallerRuntime
from nashgate.env.features import OWN_CONCURRENCY_CAP, build_local_obs, build_shared_obs, obs_dim
from nashgate.env.reward import compute_reward
from nashgate.policy import NashEquilibriumRouter


@dataclass
class RoutedRequest:
    """Returned by select_backend(); pass it back into report_result()."""
    caller_id: int
    backend_id: int
    obs: np.ndarray


class LiveRouter:
    def __init__(
        self,
        backend_configs: list[BackendConfig],
        caller_configs: list[CallerConfig],
        policy: NashEquilibriumRouter | None = None,
        explore: bool = False,
        online_learning: bool = True,
        window_seconds: float = 60.0,
    ):
        self.backend_configs = backend_configs
        self.caller_configs = caller_configs
        self.n_backends = len(backend_configs)
        self.n_callers = len(caller_configs)
        self.explore = explore
        self.online_learning = online_learning
        self.window_seconds = window_seconds

        self.backends = [BackendState(config=c) for c in backend_configs]
        self.callers = [CallerRuntime(config=c) for c in caller_configs]
        self._window_start = time.monotonic()

        # Bring your own trained policy (NashEquilibriumRouter.load(path)) —
        # falls back to a fresh, untrained one so the wiring works standalone.
        self.policy = policy or NashEquilibriumRouter(
            n_players=self.n_callers,
            obs_dim=obs_dim(self.n_backends),
            n_backends=self.n_backends,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        backend_configs: list[BackendConfig],
        caller_configs: list[CallerConfig],
        **kwargs,
    ) -> "LiveRouter":
        router = cls(backend_configs, caller_configs, **kwargs)
        router.policy.load(path)
        return router

    def _maybe_reset_window(self):
        if time.monotonic() - self._window_start >= self.window_seconds:
            for backend in self.backends:
                backend.reset_window()
            for caller in self.callers:
                caller.reset_window()
            self._window_start = time.monotonic()

    def select_backend(self, caller_id: int, request_tokens: int) -> RoutedRequest:
        self._maybe_reset_window()
        caller = self.callers[caller_id]
        caller.pending_tokens = request_tokens

        shared = build_shared_obs(self.backends, caller.config.sla_latency_ms)
        obs = build_local_obs(caller, shared)

        backend_id = self.policy.agents[caller_id].select_action(obs, explore=self.explore)

        backend = self.backends[backend_id]
        backend.in_flight += 1
        backend.requests_this_window += 1
        caller.inflight = min(int(OWN_CONCURRENCY_CAP), caller.inflight + 1)

        return RoutedRequest(caller_id=caller_id, backend_id=backend_id, obs=obs)

    def report_result(
        self, routed: RoutedRequest, latency_ms: float, cost: float, success: bool
    ) -> float:
        caller = self.callers[routed.caller_id]
        backend = self.backends[routed.backend_id]

        backend.update_ema(latency_ms, errored=not success)
        backend.in_flight = max(0, backend.in_flight - 1)
        caller.inflight = max(0, caller.inflight - 1)

        reward = compute_reward(latency_ms, cost, success, caller.config)
        caller.budget_remaining = max(0.0, caller.budget_remaining - cost)
        caller.success_ema = 0.9 * caller.success_ema + 0.1 * float(success)
        caller.last_reward = reward

        if self.online_learning:
            next_shared = build_shared_obs(self.backends, caller.config.sla_latency_ms)
            next_obs = build_local_obs(caller, next_shared)
            agent = self.policy.agents[routed.caller_id]
            agent.store(routed.obs, routed.backend_id, reward, next_obs, done=False)
            agent.update()

        return reward


if __name__ == "__main__":
    import random

    backends = [
        BackendConfig(base_latency_ms=300, latency_jitter_ms=50, cost_per_1k_tokens=0.01, rate_limit_per_window=40),
        BackendConfig(base_latency_ms=600, latency_jitter_ms=80, cost_per_1k_tokens=0.002, rate_limit_per_window=100),
        BackendConfig(base_latency_ms=1200, latency_jitter_ms=150, cost_per_1k_tokens=0.0005, rate_limit_per_window=200),
    ]
    callers = [CallerConfig() for _ in range(4)]
    router = LiveRouter(backends, callers, explore=True, window_seconds=5.0)

    for step in range(10):
        outcomes = {}
        for caller_id in range(len(callers)):
            tokens = random.randint(200, 2000)
            routed = router.select_backend(caller_id, tokens)
            backend_cfg = backends[routed.backend_id]
            latency_ms = backend_cfg.base_latency_ms + random.uniform(0, 200)
            cost = (tokens / 1000.0) * backend_cfg.cost_per_1k_tokens
            success = latency_ms <= callers[caller_id].sla_latency_ms
            reward = router.report_result(routed, latency_ms, cost, success)
            outcomes[caller_id] = (routed.backend_id, round(reward, 2))
        print(f"step {step}: caller -> (backend, reward) = {outcomes}")
    print("OK — live router wired to policy, no simulation env involved")
