"""Forwards an OpenAI-compatible chat completion request to a chosen backend."""

import time
from dataclasses import dataclass

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
