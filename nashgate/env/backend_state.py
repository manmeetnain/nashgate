"""Simulated backend (a model endpoint / provider / API key) that
callers contend for. Tracks the state the router actually needs to
observe: how loaded it is, how it's been performing, how much
rate-limit headroom is left in the current window.
"""

from dataclasses import dataclass, field


@dataclass
class BackendConfig:
    base_latency_ms: float
    latency_jitter_ms: float
    cost_per_1k_tokens: float
    rate_limit_per_window: int
    congestion_latency_ms_per_inflight: float = 40.0


@dataclass
class BackendState:
    config: BackendConfig
    in_flight: int = 0
    requests_this_window: int = 0
    latency_ema_ms: float = field(init=False)
    error_rate_ema: float = 0.0

    def __post_init__(self):
        self.latency_ema_ms = self.config.base_latency_ms

    def rate_limit_headroom(self) -> float:
        """Fraction of this window's request budget still unused, in [0, 1]."""
        used = self.requests_this_window / max(1, self.config.rate_limit_per_window)
        return max(0.0, 1.0 - used)

    def is_rate_limited(self) -> bool:
        return self.requests_this_window >= self.config.rate_limit_per_window

    def reset_window(self):
        self.requests_this_window = 0

    def decay_in_flight(self, fraction: float = 0.3):
        """Requests complete and leave the queue between steps."""
        self.in_flight = max(0, int(self.in_flight * (1.0 - fraction)))

    def update_ema(self, observed_latency_ms: float, errored: bool, alpha: float = 0.1):
        self.latency_ema_ms = (1 - alpha) * self.latency_ema_ms + alpha * observed_latency_ms
        self.error_rate_ema = (1 - alpha) * self.error_rate_ema + alpha * float(errored)
