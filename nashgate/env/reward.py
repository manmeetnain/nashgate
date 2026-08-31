"""Reward shaping — shared by the training env and the live router so
a request that succeeds or fails scores identically in both places."""

from nashgate.env.caller_state import CallerConfig

W_SUCCESS = 1.0
W_VIOLATION = 1.5
W_LATENCY = 0.4
W_COST = 0.3
HARD_TIMEOUT_MULTIPLE = 3.0


def compute_reward(latency_ms: float, cost: float, success: bool, caller_config: CallerConfig) -> float:
    reward = W_SUCCESS if success else -W_VIOLATION
    reward -= W_LATENCY * min(latency_ms / caller_config.sla_latency_ms, HARD_TIMEOUT_MULTIPLE)
    reward -= W_COST * min(cost / max(caller_config.cost_budget_per_window, 1e-6), HARD_TIMEOUT_MULTIPLE)
    return reward
