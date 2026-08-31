import httpx
import pytest

from nashgate.env.backend_state import BackendConfig
from nashgate.gateway.backends import GatewayBackend
from nashgate.gateway.proxy import forward_chat_completion


def make_backend(base_url="https://backend.invalid/v1"):
    return GatewayBackend(
        name="test-backend", base_url=base_url, api_key_env="TEST_KEY", model="model-x",
        routing_config=BackendConfig(base_latency_ms=1, latency_jitter_ms=1, cost_per_1k_tokens=0.01, rate_limit_per_window=1),
    )


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")


async def test_forward_success_substitutes_model_and_returns_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-test"
        body = request.read()
        import json
        payload = json.loads(body)
        assert payload["model"] == "model-x"  # substituted, not the caller's requested model
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await forward_chat_completion(client, make_backend(), {"model": "whatever-caller-asked-for", "messages": []})

    assert result.ok
    assert result.status_code == 200
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 3


async def test_forward_http_error_status_is_not_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await forward_chat_completion(client, make_backend(), {"messages": []})

    assert not result.ok
    assert result.status_code == 503
    assert result.body["error"]["message"] == "overloaded"


async def test_forward_timeout_returns_failed_result_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await forward_chat_completion(client, make_backend(), {"messages": []}, timeout_s=5.0)

    assert not result.ok
    assert result.status_code is None
    assert "timed out" in result.body["error"]["message"]


async def test_forward_connection_error_returns_failed_result_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await forward_chat_completion(client, make_backend(), {"messages": []})

    assert not result.ok
    assert result.status_code is None
