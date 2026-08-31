import pytest

from nashgate.env.caller_state import CallerConfig
from nashgate.env.reward import HARD_TIMEOUT_MULTIPLE, W_COST, W_LATENCY, W_SUCCESS, W_VIOLATION, compute_reward


def test_success_within_sla_scores_positively():
    cfg = CallerConfig(sla_latency_ms=1000.0, cost_budget_per_window=1.0)
    reward = compute_reward(latency_ms=200.0, cost=0.01, success=True, caller_config=cfg)
    assert reward > 0


def test_violation_scores_lower_than_a_success_at_the_same_latency_and_cost():
    cfg = CallerConfig(sla_latency_ms=1000.0, cost_budget_per_window=1.0)
    success_reward = compute_reward(latency_ms=200.0, cost=0.01, success=True, caller_config=cfg)
    violation_reward = compute_reward(latency_ms=200.0, cost=0.01, success=False, caller_config=cfg)
    assert violation_reward < success_reward
    assert violation_reward == pytest.approx(success_reward - (W_SUCCESS + W_VIOLATION))


def test_higher_latency_scores_worse_even_when_still_successful():
    cfg = CallerConfig(sla_latency_ms=1000.0, cost_budget_per_window=1.0)
    fast = compute_reward(latency_ms=100.0, cost=0.01, success=True, caller_config=cfg)
    slow = compute_reward(latency_ms=900.0, cost=0.01, success=True, caller_config=cfg)
    assert slow < fast


def test_latency_penalty_clips_at_hard_timeout_multiple():
    cfg = CallerConfig(sla_latency_ms=1000.0, cost_budget_per_window=1.0)
    at_cap = compute_reward(latency_ms=3000.0, cost=0.0, success=False, caller_config=cfg)
    beyond_cap = compute_reward(latency_ms=30_000.0, cost=0.0, success=False, caller_config=cfg)
    assert at_cap == beyond_cap


def test_cost_penalty_scales_with_budget_fraction_used():
    cfg = CallerConfig(sla_latency_ms=1000.0, cost_budget_per_window=1.0)
    cheap = compute_reward(latency_ms=100.0, cost=0.1, success=True, caller_config=cfg)
    expensive = compute_reward(latency_ms=100.0, cost=0.9, success=True, caller_config=cfg)
    assert expensive < cheap


def test_reward_formula_matches_weights_exactly():
    cfg = CallerConfig(sla_latency_ms=1000.0, cost_budget_per_window=2.0)
    reward = compute_reward(latency_ms=500.0, cost=1.0, success=True, caller_config=cfg)
    expected = W_SUCCESS - W_LATENCY * min(500.0 / 1000.0, HARD_TIMEOUT_MULTIPLE) - W_COST * min(1.0 / 2.0, HARD_TIMEOUT_MULTIPLE)
    assert reward == pytest.approx(expected)
