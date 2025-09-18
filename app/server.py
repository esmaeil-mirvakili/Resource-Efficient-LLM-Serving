# app/server.py
from __future__ import annotations

import json
import os
import time
import asyncio
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional
import sys
import uuid
from contextlib import suppress
from loguru import logger
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)

from app.api.errors import error_response
from app.api.models import ChatRequest, ChatResponse, Choice, ModelsResponse, Usage
from app.core.backends.inmemory_backend import InMemoryBackend


class TokenBucket:
    def __init__(self, rate, burst):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.updated = time.monotonic()

    def take(self, n=1):
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= n:
            self.tokens -= n
            return True, int(self.tokens)
        return False, int(self.tokens)


def rate_limit_key(request: Request) -> str:
    auth = request.headers.get("authorization")
    return auth or request.client.host or "anon"


def install_rate_limit(app: FastAPI, rate: float, burst: int):
    buckets: dict[str, TokenBucket] = defaultdict(lambda: TokenBucket(rate, burst))

    @app.middleware("http")
    async def rl_mw(request: Request, call_next):
        ok, remaining = buckets[rate_limit_key(request)].take(1)
        if not ok:
            return _structured_error(
                429, "rate_limited", "Too Many Requests.", _get_req_id(request)
            )
        resp = await call_next(request)
        resp.headers["RateLimit-Remaining"] = str(max(remaining, 0))
        return resp


async def _warmup_backend(app: FastAPI) -> None:
    """
    Ensure backend is instantiated and can generate at least 1 token.
    Sets app.state.ready and app.state.warmup_error accordingly.
    """
    app.state.warmup_started = True
    app.state.warmup_error = None
    try:
        backend = get_backend()
        # Minimal, deterministic single-token generate
        messages = [{"role": "user", "content": "ping"}]
        # Params dict is optional; backends should tolerate None
        if hasattr(backend, "generate"):
            # Avoid using the app semaphore during startup
            import asyncio
            from contextlib import suppress

            with suppress(Exception):
                # Python 3.11 timeout guard
                async with asyncio.timeout(get_settings().WARMUP_TIMEOUT_S):
                    await backend.generate(
                        messages,
                        1,
                        params={
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": 0,
                            "seed": 0,
                            "stop": [],
                        },
                    )
        app.state.ready = True
    except Exception as e:
        app.state.ready = False
        app.state.warmup_error = f"{type(e).__name__}: {e}"


def _normalize_messages(msgs: List[Any]) -> List[Dict[str, Any]]:
    """Convert Pydantic v2 models (e.g., ChatMessage) to dicts so backends can .get(...)."""
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if isinstance(m, dict):
            out.append(m)
            continue
        md = getattr(m, "model_dump", None)
        if callable(md):
            out.append(md(exclude_none=True))
        else:
            out.append(
                {
                    "role": getattr(m, "role", None),
                    "content": getattr(m, "content", None),
                    "name": getattr(m, "name", None),
                }
            )
    return out


def _structured_error(
    status: int, code: str, message: str, req_id: Optional[str] = None
) -> JSONResponse:
    payload = {
        "error": {
            "message": message,
            "type": type(message).__name__ if not isinstance(message, str) else "Error",
            "param": None,
            "code": code,
            "traceback": None,
        },
        "created": int(time.time()),
        "id": req_id or "",
    }
    return JSONResponse(payload, status_code=status)


def _get_req_id(request: Request | None) -> str | None:
    try:
        return getattr(request.state, "request_id", None)
    except Exception:
        return None


def _configure_logging(settings: "Settings") -> None:
    # Make loguru the single sink (JSON optional), quiet by default
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL.upper(),
        enqueue=True,
        backtrace=False,
        diagnose=False,
        serialize=settings.LOG_JSON,
    )


def _configure_tracing_if_enabled(app: FastAPI, settings: "Settings") -> None:
    if not settings.OTEL_ENABLED:
        return
    with suppress(Exception):
        # Lazy import so tests / env without otel deps don’t fail
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT or "http://localhost:4318"
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # Instrument FastAPI app (Starlette middleware under the hood)
        FastAPIInstrumentor.instrument_app(app)
        logger.info(
            "OpenTelemetry tracing enabled",
            endpoint=endpoint,
            service=settings.OTEL_SERVICE_NAME,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Settings & DI
# ─────────────────────────────────────────────────────────────────────────────


class Settings:
    def __init__(self) -> None:
        # Read env at instance time so tests can flip env + cache_clear()
        self.CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
        self.AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"
        self.AUTH_TOKEN: Optional[str] = os.getenv("AUTH_TOKEN")
        self.MODEL_ID: str = os.getenv("MODEL_ID", "dummy-1")
        self.BACKEND: str = os.getenv("BACKEND", "inmemory").lower()
        # New: runtime controls
        self.MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "4"))
        self.GEN_TIMEOUT_S: float = float(os.getenv("GEN_TIMEOUT_S", "30"))
        # Non-blocking acquire timeout; if we can't grab a slot fast, return 429
        self.ACQUIRE_TIMEOUT_S: float = float(os.getenv("ACQUIRE_TIMEOUT_S", "0.005"))
        # Warmup & readiness gating (default OFF to not break tests)
        self.WARMUP_ON_START: bool = (
            os.getenv("WARMUP_ON_START", "false").lower() == "true"
        )
        self.READINESS_REQUIRES_WARMUP: bool = (
            os.getenv("READINESS_REQUIRES_WARMUP", "false").lower() == "true"
        )
        self.WARMUP_TIMEOUT_S: float = float(os.getenv("WARMUP_TIMEOUT_S", "30"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_JSON: bool = os.getenv("LOG_JSON", "false").lower() == "true"
        # OpenTelemetry
        self.OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"
        self.OTEL_EXPORTER_OTLP_ENDPOINT: str | None = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )  # e.g. "http://localhost:4318"
        self.OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "llm-service")
        self.OTEL_SAMPLER: str = os.getenv(
            "OTEL_SAMPLER", "parentbased_always_on"
        )  # keep simple defaults


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_backend():
    """Env-driven backend switch; tests often monkeypatch this symbol directly."""
    settings = get_settings()
    if settings.BACKEND == "transformers":
        from app.core.backends.transformers_backend import TransformersBackend

        return TransformersBackend(model_id=settings.MODEL_ID, device="cpu")
    if settings.BACKEND == "vllm":
        from app.core.backends.vllm_backend import VLLMBackend

        # Let vLLM pick device; override with BACKEND_DEVICE if you want
        device = os.getenv("BACKEND_DEVICE")  # e.g. "cuda", "cpu"
        return VLLMBackend(model_id=settings.MODEL_ID, device=device)
    return InMemoryBackend(model_id=settings.MODEL_ID)


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


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Service (Minimal)", version="0.1.0")

    # Per-app Prometheus registry to avoid duplicate timeseries in tests
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
            "Time to first byte/token",
            labelnames=("route",),
            registry=registry,
        ),
        "LAT": Histogram(
            "llm_latency_seconds",
            "End-to-end latency",
            labelnames=("route",),
            registry=registry,
        ),
        "INFLIGHT": Gauge(
            "llm_requests_inflight",
            "Requests in flight",
            labelnames=("route",),
            registry=registry,
        ),
        "REQS_BY_STATUS": Counter(
            "llm_requests_total_by_status",
            "Total requests by status",
            labelnames=("route", "status"),
            registry=registry,
        ),
        "TOKENS_PROMPT": Counter(
            "llm_prompt_tokens_total",
            "Total prompt tokens",
            labelnames=("model",),
            registry=registry,
        ),
        "TOKENS_COMPLETION": Counter(
            "llm_completion_tokens_total",
            "Total completion tokens",
            labelnames=("model",),
            registry=registry,
        ),
    }

    # Concurrency gate (per app)
    settings = get_settings()
    app.state.sem = asyncio.Semaphore(settings.MAX_CONCURRENCY)
    app.state.acquire_timeout_s = settings.ACQUIRE_TIMEOUT_S

    # Readiness flags
    app.state.ready = (
        not settings.READINESS_REQUIRES_WARMUP
    )  # ready by default when gating is off
    app.state.warmup_started = False
    app.state.warmup_error = None

    _configure_logging(settings)
    _configure_tracing_if_enabled(app, settings)

    @app.middleware("http")
    async def _request_id_and_access_log(request: Request, call_next):
        start = time.perf_counter()
        incoming_rid = request.headers.get("x-request-id")
        req_id = incoming_rid or f"req-{uuid.uuid4().hex[:16]}"
        request.state.request_id = req_id

        # Start line (minimal PII)
        logger.bind(request_id=req_id).info(
            "http.request.start",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query or ""),
            client=getattr(request.client, "host", None),
            ua=request.headers.get("user-agent"),
        )
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception with req_id and re-raise to your existing handlers
            logger.bind(request_id=req_id).exception("http.request.error", error=str(e))
            raise
        finally:
            dur_ms = int((time.perf_counter() - start) * 1000)
            # Note: 'response' may not exist if we crashed above; guard it
            status = getattr(locals().get("response", None), "status_code", 500)
            logger.bind(request_id=req_id).info(
                "http.request.end",
                status=status,
                duration_ms=dur_ms,
                path=request.url.path,
            )

        # Propagate the ID back to clients
        try:
            response.headers["x-request-id"] = req_id
        except Exception:
            pass
        return response

    # Kick warmup at startup if requested
    if settings.WARMUP_ON_START or settings.READINESS_REQUIRES_WARMUP:

        @app.on_event("startup")
        async def _do_warmup() -> None:
            await _warmup_backend(app)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rate = float(os.getenv("RATE_LIMIT_QPS", "0"))
    burst = int(os.getenv("RATE_LIMIT_BURST", "0"))
    if rate > 0 and burst > 0:
        install_rate_limit(app, rate, burst)

    # --- Health ---
    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request) -> Dict[str, str]:
        s = get_settings()
        if not s.READINESS_REQUIRES_WARMUP:
            # Backwards-compatible: always ready when gating is off
            return {"status": "ready"}
        # Gated readiness
        if request.app.state.ready:
            return {"status": "ready"}
        # Not ready yet — expose state for operators; keep 200 or flip to 503 if you prefer
        # To avoid breaking probes unexpectedly, stick with 200 + status message by default.
        # If you want real K8s semantics, return Response(..., status_code=503).
        return {
            "status": "warming" if request.app.state.warmup_error is None else "error"
        }

    # --- Models ---
    @app.get("/v1/models", response_model=ModelsResponse)
    def models(_: None = Depends(auth_dependency), request: Request = None):
        request.app.state.metrics["REQ_TOTAL"].labels("/v1/models").inc()
        s = get_settings()
        return {
            "object": "list",
            "data": [{"id": s.MODEL_ID, "object": "model", "owned_by": "owner"}],
        }

    # --- Metrics ---
    @app.get("/metrics")
    def metrics(request: Request):
        return Response(
            generate_latest(request.app.state.registry), media_type=CONTENT_TYPE_LATEST
        )

    # --- Chat Completions (OpenAI-style) ---
    @app.post("/v1/chat/completions")
    async def chat(
        req: ChatRequest, _: None = Depends(auth_dependency), request: Request = None
    ):
        start = time.perf_counter()
        m = request.app.state.metrics
        m["REQ_TOTAL"].labels("/v1/chat/completions").inc()
        backend = get_backend()
        s = get_settings()

        # Normalize messages & collect generation params
        messages = _normalize_messages(req.messages)
        gen_params = {
            "temperature": 0.7 if req.temperature is None else req.temperature,
            "top_p": 1.0 if req.top_p is None else req.top_p,
            "top_k": 0 if getattr(req, "top_k", None) is None else int(req.top_k),
            "stop": req.stop or [],
            "seed": getattr(req, "seed", None),
        }

        # Concurrency cap: try to acquire fast; else 429
        acquired = False
        try:
            await asyncio.wait_for(
                request.app.state.sem.acquire(),
                timeout=request.app.state.acquire_timeout_s,
            )
            acquired = True
        except asyncio.TimeoutError:
            return _structured_error(
                429,
                "too_many_requests",
                "Server is busy, please retry.",
                _get_req_id(request),
            )

        route = "/v1/chat/completions"
        m["INFLIGHT"].labels(route).inc()
        try:
            if req.stream:

                async def sse() -> AsyncGenerator[str, None]:
                    created = int(time.time())
                    chunk_id = backend.new_request_id()

                    # Initial role frame
                    role_frame = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model
                        or getattr(backend, "model_id", None)
                        or s.MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(role_frame)}\n\n"

                    # TTFB mark
                    m["TTFB"].labels("/v1/chat/completions").observe(
                        time.perf_counter() - start
                    )

                    # Stream content with global timeout
                    try:
                        # Python 3.11 timeout context
                        async with asyncio.timeout(s.GEN_TIMEOUT_S):
                            async for piece in backend.stream_generate(
                                messages, req.max_tokens, params=gen_params
                            ):
                                data = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": req.model
                                    or getattr(backend, "model_id", None)
                                    or s.MODEL_ID,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": piece},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                    except TimeoutError:
                        # End stream with a final frame + [DONE]; client got partial text.
                        end_frame = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req.model
                            or getattr(backend, "model_id", None)
                            or s.MODEL_ID,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "length"}
                            ],
                        }
                        yield f"data: {json.dumps(end_frame)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Normal end
                    end_frame = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model
                        or getattr(backend, "model_id", None)
                        or s.MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(end_frame)}\n\n"
                    yield "data: [DONE]\n\n"

                headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
                return StreamingResponse(
                    sse(), media_type="text/event-stream", headers=headers
                )

            # Non-streaming path with timeout
            m["TTFB"].labels("/v1/chat/completions").observe(
                time.perf_counter() - start
            )

            try:
                text, usage = await asyncio.wait_for(
                    backend.generate(messages, req.max_tokens, params=gen_params),
                    timeout=s.GEN_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                return _structured_error(
                    504,
                    "generation_timeout",
                    "Generation timed out.",
                    _get_req_id(request),
                )

            model_id = getattr(backend, "model_id", None) or s.MODEL_ID
            m["TOKENS_TOTAL"].labels(model_id).inc(usage.get("completion_tokens", 0))

            created = int(time.time())
            response = ChatResponse(
                id=backend.new_request_id(),
                created=created,
                model=req.model or model_id,
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
            m["REQS_BY_STATUS"].labels(route, "200").inc()
            m["TOKENS_PROMPT"].labels(model_id).inc(usage.get("prompt_tokens", 0))
            m["TOKENS_COMPLETION"].labels(model_id).inc(
                usage.get("completion_tokens", 0)
            )
            return JSONResponse(response.model_dump())

        except HTTPException:
            m["REQS_BY_STATUS"].labels(route, str(he.status_code)).inc()
            raise
        except Exception as e:  # noqa: BLE001
            m["LAT"].labels("/v1/chat/completions").observe(time.perf_counter() - start)
            m["REQS_BY_STATUS"].labels(route, "500").inc()
            return error_response(e)
        finally:
            m["INFLIGHT"].labels(route).dec()
            if acquired:
                request.app.state.sem.release()

    # --- Classic Completions (optional) ---
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
        s = get_settings()

        messages = [{"role": "user", "content": str(prompt)}]
        try:
            text, usage = await asyncio.wait_for(
                backend.generate(messages, max_tokens, params=None),
                timeout=s.GEN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _structured_error(
                504, "generation_timeout", "Generation timed out.", _get_req_id(request)
            )

        model_id = getattr(backend, "model_id", None) or s.MODEL_ID
        m["TOKENS_TOTAL"].labels(model_id).inc(usage.get("completion_tokens", 0))

        created = int(time.time())
        resp = {
            "id": backend.new_request_id(),
            "object": "text_completion",
            "created": created,
            "model": payload.get("model") or model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": usage,
        }
        return JSONResponse(resp)

    return app


# Export a module-level app for uvicorn & tests
app = create_app()
