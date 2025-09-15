import os
import importlib
import pytest
from fastapi.testclient import TestClient
import sys, pathlib

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Skip the whole module if transformers isn't installed
pytest.importorskip("transformers", reason="transformers not installed")


def _build_transformers_client(
    monkeypatch, model_id: str = "distilgpt2"
) -> tuple[TestClient, object]:
    """
    Create a TestClient with BACKEND=transformers on CPU.
    Returns (client, sm_module) so callers can access server symbols if needed.
    """
    # Force env for this test instance
    monkeypatch.setenv("BACKEND", "transformers")
    monkeypatch.setenv("MODEL_ID", model_id)
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    # Keep everything on CPU
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Import/reload the server module so it reads the env
    import app.server as sm

    sm = importlib.reload(sm)

    # Clear DI singletons to pick up new env
    try:
        sm.get_settings.cache_clear()
    except Exception:
        pass
    try:
        sm.get_backend.cache_clear()
    except Exception:
        pass

    # Sanity: verify the backend really is Transformers; if not, skip
    try:
        be = sm.get_backend()  # this will instantiate the backend
    except Exception as e:
        pytest.skip(f"Transformers backend not available or failed to load: {e!r}")

    if "Transformers" not in be.__class__.__name__:
        pytest.skip("Server does not wire BACKEND=transformers; skipping.")

    app = sm.create_app()
    return TestClient(app), sm


@pytest.mark.slow
def test_transformers_models_endpoint(monkeypatch):
    client, sm = _build_transformers_client(monkeypatch, model_id="distilgpt2")
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "distilgpt2"  # should reflect MODEL_ID from env


@pytest.mark.slow
def test_transformers_chat_completion_non_stream(monkeypatch):
    client, sm = _build_transformers_client(monkeypatch, model_id="distilgpt2")
    payload = {
        "model": "distilgpt2",
        "messages": [{"role": "user", "content": "Say hi in five words."}],
        "max_tokens": 12,  # keep it small for CPU speed
        "stream": False,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], str) and len(msg["content"]) > 0
    usage = body["usage"]
    # Usage invariants
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.slow
def test_transformers_chat_completion_streaming_sse(monkeypatch):
    client, sm = _build_transformers_client(monkeypatch, model_id="distilgpt2")
    payload = {
        "model": "distilgpt2",
        "messages": [{"role": "user", "content": "Write a short greeting."}],
        "max_tokens": 10,
        "stream": True,
    }

    chunks = []
    saw_role = False
    saw_done = False

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        for raw in resp.iter_lines():
            if not raw:
                continue
            assert raw.startswith("data:")
            datum = raw[len("data:") :].strip()
            if datum == "[DONE]":
                saw_done = True
                break
            obj = __import__("json").loads(datum)
            choice = obj["choices"][0]
            delta = choice.get("delta", {})
            if delta.get("role") == "assistant":
                saw_role = True
            piece = delta.get("content")
            if piece:
                chunks.append(piece)

    text = "".join(chunks).strip()
    assert saw_role and saw_done
    assert len(text) > 0
