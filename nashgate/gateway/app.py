"""The OpenAI-compatible proxy — what agents actually talk to.

    POST /v1/chat/completions
      -> resolve caller from the X-Nashgate-Caller header
      -> router.select_backend()      (the trained policy picks one)
      -> forward_chat_completion()    (the real HTTP call)
      -> router.report_result()       (score it, feed it back to the policy)
      -> return the backend's response, annotated with which backend served it

For a streaming request (`body["stream"]` truthy), the connection to
the backend is opened and its status checked *before* any response is
returned to the caller — that's what lets a backend error still come
back as a proper HTTP error status instead of a 200 stream with an
error buried in the body. Only once the backend has accepted the
request does this become a `StreamingResponse` passing chunks through
live; the routing outcome is reported to the policy only after the
stream finishes (or drops), since latency/cost/success aren't known
until then.
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from nashgate.gateway.backends import GatewayBackend
from nashgate.gateway.callers import CALLER_HEADER, CallerRegistry, NamedCaller
from nashgate.gateway.proxy import forward_chat_completion, iter_stream_chunks, open_chat_completion_stream
from nashgate.gateway.tokens import estimate_request_tokens, total_tokens_from_usage
from nashgate.router import LiveRouter


def create_app(
    backends: list[GatewayBackend],
    callers: list[NamedCaller],
    policy_checkpoint: str | None = None,
    explore: bool = False,
    online_learning: bool = True,
) -> FastAPI:
    registry = CallerRegistry(callers)
    routing_backend_configs = [b.routing_config for b in backends]

    live_router = (
        LiveRouter.from_checkpoint(
            policy_checkpoint, routing_backend_configs, registry.configs,
            explore=explore, online_learning=online_learning,
        )
        if policy_checkpoint else
        LiveRouter(
            routing_backend_configs, registry.configs,
            explore=explore, online_learning=online_learning,
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = httpx.AsyncClient()
        yield
        await app.state.http_client.aclose()

    app = FastAPI(title="nashgate", lifespan=lifespan)
    app.state.live_router = live_router
    app.state.backends = backends
    app.state.registry = registry

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "backends": [b.name for b in backends]}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: dict, x_nashgate_caller: str = Header(..., alias=CALLER_HEADER)
    ):
        try:
            caller_id = registry.resolve(x_nashgate_caller)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        messages = body.get("messages", [])
        request_tokens = estimate_request_tokens(messages)

        routed = live_router.select_backend(caller_id, request_tokens)
        backend = backends[routed.backend_id]

        if body.get("stream"):
            response, stream_result = await open_chat_completion_stream(app.state.http_client, backend, body)
            if response is None:
                live_router.report_result(routed, latency_ms=stream_result.latency_ms, cost=0.0, success=False)
                raise HTTPException(status_code=stream_result.status_code or 502, detail=stream_result.error_body)

            async def _generate():
                async for chunk in iter_stream_chunks(response, stream_result):
                    yield chunk
                usage = (
                    {"prompt_tokens": stream_result.prompt_tokens, "completion_tokens": stream_result.completion_tokens}
                    if stream_result.prompt_tokens is not None or stream_result.completion_tokens is not None
                    else None
                )
                total_tokens = total_tokens_from_usage(usage or {}, fallback=request_tokens)
                cost = (total_tokens / 1000.0) * backend.routing_config.cost_per_1k_tokens
                live_router.report_result(
                    routed, latency_ms=stream_result.latency_ms, cost=cost, success=stream_result.ok
                )

            return StreamingResponse(
                _generate(), media_type="text/event-stream", headers={"X-Nashgate-Backend": backend.name}
            )

        result = await forward_chat_completion(app.state.http_client, backend, body)

        total_tokens = total_tokens_from_usage(
            result.body.get("usage"), fallback=request_tokens
        )
        cost = (total_tokens / 1000.0) * backend.routing_config.cost_per_1k_tokens
        reward = live_router.report_result(
            routed, latency_ms=result.latency_ms, cost=cost, success=result.ok
        )

        if not result.ok:
            raise HTTPException(
                status_code=result.status_code or 502, detail=result.body
            )

        response_body = dict(result.body)
        response_body["nashgate"] = {
            "backend": backend.name,
            "latency_ms": round(result.latency_ms, 1),
            "cost": round(cost, 6),
            "reward": round(reward, 4),
        }
        return JSONResponse(response_body, headers={"X-Nashgate-Backend": backend.name})

    return app
