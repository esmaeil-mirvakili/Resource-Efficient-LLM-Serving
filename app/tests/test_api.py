import json as _json
import os
from contextlib import contextmanager
import importlib
from fastapi.testclient import TestClient
import sys, pathlib
import pytest


xf = pytest.importorskip("transformers", reason="transformers not installed")


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# Helpers ---------------------------------------------------------------------


def _make_app(*, auth_enabled: bool = False, token: str = "secret"):
    """Create a fresh FastAPI app instance with desired auth settings.
    This clears lru_caches so settings/backend reflect the env for this app.
    """
    # Set env and build a fresh app
    os.environ["AUTH_ENABLED"] = "true" if auth_enabled else "false"
    os.environ["AUTH_TOKEN"] = token
    os.environ["BACKEND"] = "inmemory"
    os.environ["MODEL_ID"] = "dummy-1"
    from app import server as sm
    sm = importlib.reload(sm)

    # Clear caches so settings/backend pick up new env
    try:
        sm.get_settings.cache_clear()
    except Exception:
        pass
    try:
        sm.get_backend.cache_clear()
    except Exception:
        pass

    return sm.create_app()


@contextmanager
def _client_app(*, auth_enabled: bool = False, token: str = "secret"):
    app = _make_app(auth_enabled=auth_enabled, token=token)
    with TestClient(app) as client:
        yield client


# Tests -----------------------------------------------------------------------


def test_health_and_readiness():
    with _client_app() as client:
        r = client.get("/healthz")
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        r = client.get("/readyz")
        assert r.status_code == 200 and r.json() == {"status": "ready"}


def test_models_endpoint_default_model():
    with _client_app() as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list) and len(data["data"]) == 1
        assert data["data"][0]["id"] == "dummy-1"


def test_chat_completions_non_streaming_shape_and_usage():
    payload = {
        "model": "dummy-1",
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": 16,
        "stream": False,
    }
    with _client_app() as client:
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        text = body["choices"][0]["message"]["content"]
        assert "hello there" in text  # echo-y deterministic reply
        assert body["choices"][0]["finish_reason"] == "stop"
        usage = body["usage"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert isinstance(usage[k], int) and usage[k] >= 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )


def test_chat_completions_streaming_sse_contract():
    payload = {
        "model": "dummy-1",
        "messages": [{"role": "user", "content": "stream please"}],
        "max_tokens": 16,
        "stream": True,
    }
    with _client_app() as client:
        chunks = []
        saw_role = False
        saw_done = False
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            # content-type should include text/event-stream
            assert "text/event-stream" in resp.headers.get("content-type", "")
            for raw in resp.iter_lines():
                if not raw:
                    continue
                assert raw.startswith("data:")
                datum = raw[len("data:") :].strip()
                if datum == "[DONE]":
                    saw_done = True
                    break
                obj = _json.loads(datum)
                choice = obj["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("role") == "assistant":
                    saw_role = True
                piece = delta.get("content")
                if piece:
                    chunks.append(piece)
        text = "".join(chunks).strip()
        assert saw_role and saw_done
        assert "stream please" in text


def test_completions_endpoint_basic():
    payload = {"model": "dummy-1", "prompt": "classic completion", "max_tokens": 12}
    with _client_app() as client:
        r = client.post("/v1/completions", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert "classic completion" in body["choices"][0]["text"]
        usage = body["usage"]
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )


def test_metrics_endpoint_exposes_counters():
    with _client_app() as client:
        # touch an endpoint to increment counters
        client.get("/v1/models")
        r = client.get("/metrics")
        assert r.status_code == 200
        content = r.text
        assert "llm_requests_total" in content


def test_auth_enforced_when_enabled():
    # Build an app with auth ON
    with _client_app(auth_enabled=True, token="sekrit") as client:
        # Without header → 401
        r = client.get("/v1/models")
        assert r.status_code == 401
        # With correct bearer → 200
        r = client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
        assert r.status_code == 200


def test_error_shape_when_backend_throws(monkeypatch):
    # NOTE: import path—adjust to your tree
    from app import server as sm  # or: from serving import server as sm

    # Ensure auth is OFF for this test and cached settings are rebuilt
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    try:
        sm.get_settings.cache_clear()
        sm.get_backend.cache_clear()
    except Exception:
        pass

    class BoomBackend(sm.InMemoryBackend):
        async def generate(self, messages, max_tokens):  # type: ignore[override]
            raise RuntimeError("boom")

        async def stream_generate(self, messages, max_tokens):  # pragma: no cover
            raise RuntimeError("boom stream")

    # Patch backend factory to return the failing backend
    monkeypatch.setattr(sm, "get_backend", lambda: BoomBackend())

    app = sm.create_app()
    app.dependency_overrides[sm.auth_dependency] = lambda: None
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "dummy-1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
            },
        )
        assert r.status_code == 500
        body = r.json()
        assert body.get("error", {}).get("code") == "internal_error"
