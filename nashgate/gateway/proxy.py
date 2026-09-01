"""Forwards an OpenAI-compatible chat completion request to a chosen backend."""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from nashgate.gateway.backends import GatewayBackend


@dataclass
class ForwardResult:
    ok: bool
    status_code: int | None
    latency_ms: float
    body: dict
    completion_tokens: int | None = None
    prompt_tokens: int | None = None


async def forward_chat_completion(
    client: httpx.AsyncClient, backend: GatewayBackend, request_body: dict, timeout_s: float = 30.0
) -> ForwardResult:
    payload = dict(request_body)
    payload["model"] = backend.model

    started = time.monotonic()
    try:
        resp = await client.post(
            f"{backend.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {backend.api_key()}"},
            timeout=timeout_s,
        )
        latency_ms = (time.monotonic() - started) * 1000.0
    except httpx.TimeoutException:
        return ForwardResult(
            ok=False, status_code=None, latency_ms=timeout_s * 1000.0,
            body={"error": {"message": f"backend '{backend.name}' timed out"}},
        )
    except httpx.HTTPError as exc:
        return ForwardResult(
            ok=False, status_code=None, latency_ms=(time.monotonic() - started) * 1000.0,
            body={"error": {"message": f"backend '{backend.name}' request failed: {exc}"}},
        )

    try:
        body = resp.json()
    except ValueError:
        body = {"error": {"message": "backend returned a non-JSON response"}}

    ok = resp.status_code < 400
    usage = body.get("usage") or {}
    return ForwardResult(
        ok=ok,
        status_code=resp.status_code,
        latency_ms=latency_ms,
        body=body,
        completion_tokens=usage.get("completion_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
    )


@dataclass
class StreamResult:
    """Filled in as a stream is opened and consumed — read it only after
    the response has either failed to open or been fully iterated."""
    ok: bool = True
    status_code: int | None = None
    latency_ms: float = 0.0
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    error_body: dict | None = None
    started_at: float = field(default_factory=time.monotonic)


async def open_chat_completion_stream(
    client: httpx.AsyncClient, backend: GatewayBackend, request_body: dict, timeout_s: float = 60.0
) -> tuple[httpx.Response | None, StreamResult]:
    """Opens a streaming POST to the backend.

    On success: returns `(response, result)` with `result.ok=True`. The
    caller must fully consume `iter_stream_chunks(response, result)` —
    that's what closes the connection.

    On failure — a network error, or the backend responding with a
    status >= 400 before any content streams — returns `(None, result)`
    with `result.ok=False` and `result.error_body`/`status_code` set.
    No cleanup needed from the caller in this case; the connection (if
    any was opened) is already closed.
    """
    result = StreamResult()
    payload = dict(request_body)
    payload["model"] = backend.model
    payload["stream"] = True

    try:
        request = client.build_request(
            "POST", f"{backend.base_url.rstrip('/')}/chat/completions",
            json=payload, headers={"Authorization": f"Bearer {backend.api_key()}"}, timeout=timeout_s,
        )
        response = await client.send(request, stream=True)
    except httpx.TimeoutException:
        result.ok = False
        result.latency_ms = (time.monotonic() - result.started_at) * 1000.0
        result.error_body = {"error": {"message": f"backend '{backend.name}' timed out"}}
        return None, result
    except httpx.HTTPError as exc:
        result.ok = False
        result.latency_ms = (time.monotonic() - result.started_at) * 1000.0
        result.error_body = {"error": {"message": f"backend '{backend.name}' request failed: {exc}"}}
        return None, result

    result.status_code = response.status_code
    if response.status_code >= 400:
        body = await response.aread()
        await response.aclose()
        result.ok = False
        result.latency_ms = (time.monotonic() - result.started_at) * 1000.0
        try:
            result.error_body = json.loads(body)
        except ValueError:
            result.error_body = {"error": {"message": body.decode(errors="replace") or "backend error"}}
        return None, result

    return response, result


def _maybe_extract_usage(line: bytes, result: StreamResult) -> None:
    if not line.startswith(b"data: "):
        return
    data = line[len(b"data: "):].strip()
    if not data or data == b"[DONE]":
        return
    try:
        chunk = json.loads(data)
    except ValueError:
        return
    usage = chunk.get("usage")
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens")
        result.completion_tokens = usage.get("completion_tokens")


async def iter_stream_chunks(response: httpx.Response, result: StreamResult) -> AsyncIterator[bytes]:
    """Yields raw SSE bytes from an already-open, already-successful
    stream exactly as the backend sent them — passthrough fidelity
    matters more here than reparsing, since the caller's own SSE parser
    has to make sense of it. Extracts usage/token counts best-effort
    along the way (per the OpenAI `stream_options: {include_usage: true}`
    convention — the backend may or may not include it). Always updates
    `result.latency_ms` and closes `response`, whether the stream
    finishes cleanly or drops mid-way.
    """
    buffer = b""
    try:
        async for chunk in response.aiter_bytes():
            buffer += chunk
            *complete_lines, buffer = buffer.split(b"\n")
            for line in complete_lines:
                _maybe_extract_usage(line, result)
            yield chunk
        _maybe_extract_usage(buffer, result)
    except httpx.HTTPError:
        result.ok = False
    finally:
        result.latency_ms = (time.monotonic() - result.started_at) * 1000.0
        await response.aclose()
