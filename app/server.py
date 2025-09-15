from __future__ import annotations
import os
import time
import json
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, Depends, Header, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    CollectorRegistry,
    generate_latest,
)


from app.api.models import (
    ChatRequest,
    ChatResponse,
    Choice,
    Usage,
    ModelsResponse,
)
from app.api.errors import error_response
from app.core.backends.inmemory_backend import InMemoryBackend
from app.core.backends.transformers_backend import TransformersBackend


class Settings:
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    AUTH_TOKEN: Optional[str] = os.getenv("AUTH_TOKEN")
    MODEL_ID: str = os.getenv("MODEL_ID", "distilgpt2")
    BACKEND_KIND: str = os.getenv("BACKEND", "transformers")


def _to_plain_messages(msgs):
    out = []
    for m in msgs:
        if isinstance(m, dict):
            # Preserve as-is so unknown/extra keys survive.
            out.append(m)
        else:
            md = getattr(m, "model_dump", None)  # pydantic v2
            if callable(md):
                out.append(md(exclude_none=True))
            else:
                d = {
                    "role": getattr(m, "role", None),
                    "content": getattr(m, "content", None),
                }
                name = getattr(m, "name", None)
                if name is not None:
                    d["name"] = name
                out.append(d)
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_backend() -> InMemoryBackend:
    if get_settings().BACKEND_KIND == "transformers":
        return TransformersBackend(model_id=get_settings().MODEL_ID)
    return InMemoryBackend(model_id=get_settings().MODEL_ID)


async def auth_dependency(
    authorization: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.AUTH_ENABLED:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"error": "missing_bearer"})
    token = authorization.split(" ", 1)[1]
    if token != (settings.AUTH_TOKEN or ""):
        raise HTTPException(status_code=401, detail={"error": "invalid_bearer"})
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Service", version="0.1.0")

    # Metrics (keep cardinality low)
    registry = CollectorRegistry()
    app.state.registry = registry
    app.state.metrics = {
        "REQ_TOTAL": Counter(
            "llm_requests_total",
            "Total LLM API requests",
            labelnames=("route",),
            registry=registry,
        ),
        "TOKENS_TOTAL": Counter(
            "llm_tokens_generated_total",
            "Total completion tokens",
            labelnames=("model",),
            registry=registry,
        ),
        "TTFB": Histogram(
            "llm_ttfb_seconds",
            "Time to first byte/ token",
            labelnames=("route",),
            registry=registry,
        ),
        "LAT": Histogram(
            "llm_latency_seconds",
            "End-to-end latency",
            labelnames=("route",),
            registry=registry,
        ),
    }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> Dict[str, str]:
        # In a real server you'd check model load, GPU memory, etc.
        return {"status": "ready"}

    @app.get("/v1/models", response_model=ModelsResponse)
    def models(_: None = Depends(auth_dependency), request: Request = None):
        request.app.state.metrics["REQ_TOTAL"].labels("/v1/models").inc()
        settings = get_settings()
        return {
            "object": "list",
            "data": [{"id": settings.MODEL_ID, "object": "model", "owned_by": "owner"}],
        }

    @app.get("/metrics")
    def metrics(request: Request):
        return Response(
            generate_latest(request.app.state.registry), media_type=CONTENT_TYPE_LATEST
        )

    @app.post("/v1/chat/completions")
    async def chat(
        req: ChatRequest, _: None = Depends(auth_dependency), request: Request = None
    ):
        start = time.perf_counter()
        m = request.app.state.metrics
        m["REQ_TOTAL"].labels("/v1/chat/completions").inc()
        backend = get_backend()

        try:
            if req.stream:
                async def sse() -> AsyncGenerator[str, None]:
                    # Prepare ids/timestamps
                    created = int(time.time())
                    chunk_id = backend.new_request_id()

                    # First role frame
                    role_frame = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model or backend.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(role_frame)}\n\n"

                    # Generate content stream
                    ttfb = time.perf_counter()
                    m["TTFB"].labels("/v1/chat/completions").observe(ttfb - start)

                    messages = _to_plain_messages(req.messages)
                    async for piece in backend.stream_generate(
                        messages, req.max_tokens
                    ):
                        data = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model or backend.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                    # Final frame with finish_reason
                    end_frame = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model or backend.model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(end_frame)}\n\n"
                    yield "data: [DONE]\n\n"

                headers = {
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
                resp = StreamingResponse(
                    sse(), media_type="text/event-stream", headers=headers
                )
                return resp

            # Non-streaming path
            ttfb = time.perf_counter()
            m["TTFB"].labels("/v1/chat/completions").observe(ttfb - start)

            messages = _to_plain_messages(req.messages)

            text, usage = await backend.generate(messages, req.max_tokens)
            m["TOKENS_TOTAL"].labels(backend.model_id).inc(usage["completion_tokens"])

            created = int(time.time())
            response = ChatResponse(
                id=backend.new_request_id(),
                created=created,
                model=req.model or backend.model_id,
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message={"role": "assistant", "content": text},
                    )
                ],
                usage=Usage(**usage),
            )
            m["LAT"].labels("/v1/chat/completions").observe(time.perf_counter() - start)
            return JSONResponse(response.model_dump())

        except HTTPException:
            raise
        except Exception as e:
            m["LAT"].labels("/v1/chat/completions").observe(time.perf_counter() - start)
            return error_response(e)

    # Optional classic completions endpoint for parity
    @app.post("/v1/completions")
    async def completions(
        payload: Dict[str, Any],
        _: None = Depends(auth_dependency),
        request: Request = None,
    ):
        m = request.app.state.metrics
        m["REQ_TOTAL"].labels("/v1/completions").inc()
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", 256))
        backend = get_backend()
        # Adapt prompt to chat format
        messages = [{"role": "user", "content": str(prompt)}]
        text, usage = await backend.generate(messages, max_tokens)
        m["TOKENS_TOTAL"].labels(backend.model_id).inc(usage["completion_tokens"])
        created = int(time.time())
        resp = {
            "id": backend.new_request_id(),
            "object": "text_completion",
            "created": created,
            "model": payload.get("model") or backend.model_id,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        return JSONResponse(resp)

    return app


# Uvicorn factory
app = create_app()
