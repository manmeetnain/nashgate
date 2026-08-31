
import pytest

from nashgate.env.backend_state import BackendConfig
from nashgate.gateway.backends import GatewayBackend, backends_from_dicts


def test_api_key_reads_from_configured_env_var(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "sk-abc123")
    backend = GatewayBackend(
        name="b1", base_url="https://x.invalid/v1", api_key_env="MY_TEST_KEY", model="m1",
        routing_config=BackendConfig(base_latency_ms=1, latency_jitter_ms=1, cost_per_1k_tokens=0.01, rate_limit_per_window=1),
    )
    assert backend.api_key() == "sk-abc123"


def test_api_key_raises_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    backend = GatewayBackend(
        name="b1", base_url="https://x.invalid/v1", api_key_env="MISSING_KEY", model="m1",
        routing_config=BackendConfig(base_latency_ms=1, latency_jitter_ms=1, cost_per_1k_tokens=0.01, rate_limit_per_window=1),
    )
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        backend.api_key()


def test_backends_from_dicts_parses_required_and_optional_fields():
    raw = [{
        "name": "fast", "base_url": "https://x.invalid/v1", "api_key_env": "K", "model": "m",
        "cost_per_1k_tokens": 0.02, "rate_limit_per_window": 50,
    }]
    backends = backends_from_dicts(raw)
    assert len(backends) == 1
    b = backends[0]
    assert b.name == "fast"
    assert b.routing_config.cost_per_1k_tokens == 0.02
    assert b.routing_config.rate_limit_per_window == 50
    # base_latency_ms / latency_jitter_ms have defaults when omitted
    assert b.routing_config.base_latency_ms == 500.0
    assert b.routing_config.latency_jitter_ms == 100.0


def test_backends_from_dicts_honors_explicit_overrides():
    raw = [{
        "name": "b", "base_url": "u", "api_key_env": "K", "model": "m",
        "cost_per_1k_tokens": 0.01, "rate_limit_per_window": 10,
        "base_latency_ms": 42.0, "congestion_latency_ms_per_inflight": 7.0,
    }]
    backends = backends_from_dicts(raw)
    assert backends[0].routing_config.base_latency_ms == 42.0
    assert backends[0].routing_config.congestion_latency_ms_per_inflight == 7.0
