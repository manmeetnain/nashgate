from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nashgate.env.caller_state import CallerConfig
from nashgate.gateway.app import create_app
from nashgate.gateway.backends import GatewayBackend
from nashgate.gateway.callers import NamedCaller
from nashgate.gateway.proxy import ForwardResult


@pytest.fixture
def app(backend_configs, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    backends = [
        GatewayBackend(name=f"backend-{i}", base_url="https://x.invalid/v1", api_key_env="FAKE_KEY", model="m", routing_config=cfg)
        for i, cfg in enumerate(backend_configs)
    ]
    callers = [NamedCaller(name="coding-agent", config=CallerConfig(sla_latency_ms=2000, cost_budget_per_window=5.0))]
    return create_app(backends, callers, explore=True, online_learning=True)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_healthz_lists_configured_backends(client, backend_configs):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert len(r.json()["backends"]) == len(backend_configs)


def test_unknown_caller_is_rejected(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Nashgate-Caller": "ghost"},
    )
    assert r.status_code == 400


def test_missing_caller_header_is_rejected(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 422


OK_RESULT = ForwardResult(
    ok=True, status_code=200, latency_ms=250.0,
    body={"id": "chatcmpl-1", "choices": [{"message": {"role": "assistant", "content": "hi"}}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    completion_tokens=5, prompt_tokens=10,
)
FAIL_RESULT = ForwardResult(ok=False, status_code=503, latency_ms=30000.0, body={"error": {"message": "boom"}})


def test_successful_request_annotates_response_with_routing_info(client):
    with patch("nashgate.gateway.app.forward_chat_completion", new=AsyncMock(return_value=OK_RESULT)):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Nashgate-Caller": "coding-agent"},
        )
    assert r.status_code == 200
    body = r.json()
    assert "nashgate" in body
    assert set(body["nashgate"].keys()) == {"backend", "latency_ms", "cost", "reward"}
    assert r.headers["x-nashgate-backend"] == body["nashgate"]["backend"]


def test_backend_failure_propagates_as_error_and_still_reports_to_policy(app, backend_configs):
    agent = app.state.live_router.policy.agents[0]
    buffer_before = len(agent.buffer)
    with TestClient(app) as client, patch(
        "nashgate.gateway.app.forward_chat_completion", new=AsyncMock(return_value=FAIL_RESULT)
    ):
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Nashgate-Caller": "coding-agent"},
        )
    assert r.status_code == 503
    # even a failed request is scored and fed back to the policy, since
    # that's exactly the outcome the router needs to learn to avoid
    assert len(agent.buffer) == buffer_before + 1


def test_repeated_requests_route_across_multiple_backends(client):
    seen_backends = set()
    with patch("nashgate.gateway.app.forward_chat_completion", new=AsyncMock(return_value=OK_RESULT)):
        for _ in range(20):
            r = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi " * 20}]},
                headers={"X-Nashgate-Caller": "coding-agent"},
            )
            seen_backends.add(r.json()["nashgate"]["backend"])
    # explore=True with a fresh (near-uniform) policy should sample more than one backend
    assert len(seen_backends) > 1
