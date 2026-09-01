import json

import httpx
import pytest
from httpx._content import AsyncIteratorByteStream

from nashgate.env.backend_state import BackendConfig
from nashgate.gateway.backends import GatewayBackend
from nashgate.gateway.proxy import forward_chat_completion, iter_stream_chunks, open_chat_completion_stream


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


def _sse_response(status_code: int, lines: list[bytes]) -> httpx.Response:
    # httpx.MockTransport's async path requires an AsyncByteStream, and a
    # plain async generator doesn't satisfy that ABC on its own — this is
    # httpx's own internal wrapper for exactly this case, reached into
    # because there's no public constructor for a streaming mock Response.
    async def gen():
        for line in lines:
            yield line

    return httpx.Response(
        status_code, headers={"content-type": "text/event-stream"}, stream=AsyncIteratorByteStream(gen())
    )


async def test_open_stream_success_reports_status_and_leaves_response_open():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload["model"] == "model-x"
        assert payload["stream"] is True
        return _sse_response(200, [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response, result = await open_chat_completion_stream(
            client, make_backend(), {"model": "whatever-caller-asked-for", "messages": []}
        )
        assert response is not None
        assert result.ok
        assert result.status_code == 200
        await response.aclose()


async def test_iter_stream_chunks_passes_through_bytes_and_extracts_trailing_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(200, [
            b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"}}],'
            b'"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
            b"data: [DONE]\n\n",
        ])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response, result = await open_chat_completion_stream(client, make_backend(), {"messages": []})
        chunks = [chunk async for chunk in iter_stream_chunks(response, result)]

    full = b"".join(chunks)
    assert b'"content":"hel"' in full
    assert b'"content":"lo"' in full
    assert b"[DONE]" in full
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 2
    assert result.ok
    assert result.latency_ms > 0


async def test_open_stream_error_status_returns_none_response_with_error_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response, result = await open_chat_completion_stream(client, make_backend(), {"messages": []})

    assert response is None
    assert not result.ok
    assert result.status_code == 429
    assert result.error_body["error"]["message"] == "rate limited"


async def test_open_stream_timeout_returns_none_response_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response, result = await open_chat_completion_stream(client, make_backend(), {"messages": []}, timeout_s=2.0)

    assert response is None
    assert not result.ok
    assert "timed out" in result.error_body["error"]["message"]
