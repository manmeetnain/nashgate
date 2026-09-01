"""Runs the gateway against a real, live Anthropic API call — not
mocked, not simulated. Skipped unless NASHGATE_ANTHROPIC_TEST_KEY is
set, so this never runs in a normal local `pytest` or in the regular
tests.yml CI job; only the dedicated real-api-check.yml workflow (push
to main, with the key set as a repo secret) actually exercises it.
This is deliberately outside what `make check` runs — it costs real
money on every invocation, so it isn't something to run casually."""

import os

import pytest
from fastapi.testclient import TestClient

from nashgate.env.backend_state import BackendConfig
from nashgate.env.caller_state import CallerConfig
from nashgate.gateway.app import create_app
from nashgate.gateway.backends import GatewayBackend
from nashgate.gateway.callers import NamedCaller

KEY_ENV = "NASHGATE_ANTHROPIC_TEST_KEY"

pytestmark = pytest.mark.skipif(
    not os.environ.get(KEY_ENV), reason=f"requires a real Anthropic key in {KEY_ENV}"
)


def _make_app():
    backend = GatewayBackend(
        name="anthropic-real",
        base_url="https://api.anthropic.com/v1",
        api_key_env=KEY_ENV,
        model="claude-haiku-4-5-20251001",
        routing_config=BackendConfig(
            base_latency_ms=500, latency_jitter_ms=100, cost_per_1k_tokens=0.001, rate_limit_per_window=1000
        ),
    )
    caller = NamedCaller(name="ci", config=CallerConfig(sla_latency_ms=10_000, cost_budget_per_window=10.0))
    return create_app([backend], [caller], explore=False, online_learning=False)


def test_real_non_streaming_request_succeeds():
    with TestClient(_make_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Reply with exactly the word: pong"}], "max_tokens": 10},
            headers={"X-Nashgate-Caller": "ci"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["nashgate"]["backend"] == "anthropic-real"
    assert body["usage"]["prompt_tokens"] > 0


def test_real_streaming_request_succeeds():
    with TestClient(_make_app()) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Reply with exactly the word: pong"}],
                "max_tokens": 10,
                "stream": True,
            },
            headers={"X-Nashgate-Caller": "ci"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" in r.text
