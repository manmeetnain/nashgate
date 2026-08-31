"""A caller (agent, tenant, or workflow) sharing the gateway."""

from dataclasses import dataclass, field


@dataclass
class CallerConfig:
    sla_latency_ms: float = 2000.0
    cost_budget_per_window: float = 1.0
    min_request_tokens: int = 200
    max_request_tokens: int = 2000


@dataclass
class CallerRuntime:
    config: CallerConfig
    inflight: int = 0
    budget_remaining: float = field(init=False)
    last_reward: float = 0.0
    success_ema: float = 1.0
    pending_tokens: int = 0

    def __post_init__(self):
        self.budget_remaining = self.config.cost_budget_per_window

    def reset_window(self):
        self.budget_remaining = self.config.cost_budget_per_window
