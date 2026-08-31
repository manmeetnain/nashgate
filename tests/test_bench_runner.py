import pytest

from nashgate.bench.baselines import RoundRobinRouter
from nashgate.bench.runner import jain_fairness_index, run_episode
from nashgate.env.backend_state import BackendConfig
from nashgate.env.caller_state import CallerConfig
from nashgate.env.routing_env import MultiAgentRoutingEnv


def test_jain_fairness_perfectly_even_split_is_one():
    assert jain_fairness_index([10, 10, 10]) == pytest.approx(1.0)


def test_jain_fairness_all_on_one_backend_is_one_over_n():
    assert jain_fairness_index([30, 0, 0]) == pytest.approx(1.0 / 3.0)


def test_jain_fairness_empty_traffic_is_zero():
    assert jain_fairness_index([0, 0, 0]) == 0.0


def test_jain_fairness_uneven_split_is_between_bounds():
    j = jain_fairness_index([20, 5, 5])
    assert 1.0 / 3.0 < j < 1.0


def test_run_episode_round_robin_produces_perfectly_fair_split():
    backends = [
        BackendConfig(base_latency_ms=200, latency_jitter_ms=0, cost_per_1k_tokens=0.01, rate_limit_per_window=10_000),
        BackendConfig(base_latency_ms=200, latency_jitter_ms=0, cost_per_1k_tokens=0.01, rate_limit_per_window=10_000),
    ]
    callers = [CallerConfig()]
    env = MultiAgentRoutingEnv(backends, callers, episode_len=100, window_steps=1000, seed=0)
    router = RoundRobinRouter(n_callers=1, n_backends=2)
    result = run_episode(env, router, n_steps=100)
    assert result.n_requests == 100
    assert sum(result.backend_counts) == 100
    assert result.fairness_jain == pytest.approx(1.0)


def test_run_episode_reports_full_violation_rate_when_backend_always_rate_limited():
    backends = [BackendConfig(base_latency_ms=100, latency_jitter_ms=0, cost_per_1k_tokens=0.01, rate_limit_per_window=0)]
    callers = [CallerConfig()]
    env = MultiAgentRoutingEnv(backends, callers, episode_len=20, window_steps=1000, seed=0)
    router = RoundRobinRouter(n_callers=1, n_backends=1)
    result = run_episode(env, router, n_steps=20)
    assert result.violation_rate == pytest.approx(1.0)
    assert result.success_rate == pytest.approx(0.0)
