import pytest

from nashgate.env.backend_state import BackendConfig
from nashgate.env.caller_state import CallerConfig


@pytest.fixture
def backend_configs():
    return [
        BackendConfig(base_latency_ms=300, latency_jitter_ms=50, cost_per_1k_tokens=0.01, rate_limit_per_window=40),
        BackendConfig(base_latency_ms=600, latency_jitter_ms=80, cost_per_1k_tokens=0.002, rate_limit_per_window=100),
        BackendConfig(base_latency_ms=1200, latency_jitter_ms=150, cost_per_1k_tokens=0.0005, rate_limit_per_window=200),
    ]


@pytest.fixture
def caller_configs():
    return [CallerConfig() for _ in range(4)]
