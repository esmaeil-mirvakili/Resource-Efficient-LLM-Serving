# app/tests/test_api_vllm.py
import os
import importlib
import pytest
from fastapi.testclient import TestClient
import sys, pathlib

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Skip the whole module if vLLM isn't installed
pytest.importorskip("vllm", reason="vLLM not installed")

# Pick a tiny model vLLM supports; override with env if you want
DEFAULT_MODEL = os.environ.get("VLLM_TEST_MODEL", "EleutherAI/pythia-70m-deduped")


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _build_vllm_client(
    monkeypatch, model_id: str = DEFAULT_MODEL
) -> tuple[TestClient, object]:
    """
    Create a TestClient with BACKEND=vllm.
    We pre-instantiate VLLMBackend to fail fast (and skip) on unsupported envs.
    """
    # Env for this test instance
    monkeypatch.setenv("BACKEND", "vllm")
    monkeypatch.setenv("MODEL_ID", model_id)
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    # Prefer CPU unless CUDA is clearly available or explicitly forced
    if not os.environ.get("BACKEND_DEVICE"):
        device = (
            "cuda" if (_has_cuda() or os.getenv("VLLM_FORCE_CUDA") == "1") else "cpu"
        )
        monkeypatch.setenv("BACKEND_DEVICE", device)

    # Be generous with timeouts to avoid flakes on slow CPU envs
    monkeypatch.setenv("GEN_TIMEOUT_S", "120")
    monkeypatch.setenv("ACQUIRE_TIMEOUT_S", "1.0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")

    # Import/reload server so it reads env
    import app.server as sm

    sm = importlib.reload(sm)

    # Clear DI caches
    try:
        sm.get_settings.cache_clear()
    except Exception:
        pass
    try:
        sm.get_backend.cache_clear()
    except Exception:
        pass

    # Try to stand up the vLLM backend now; skip if this env can't run it
    try:
        from app.core.backends.vllm_backend import VLLMBackend

        be = VLLMBackend(model_id=model_id, device=os.environ.get("BACKEND_DEVICE"))
    except Exception as e:
        pytest.skip(f"vLLM backend failed to initialize for '{model_id}': {e!r}")

    # Freeze this backend instance so routes reuse it (no double init)
    monkeypatch.setattr(sm, "get_backend", lambda: be, raising=True)

    app = sm.create_app()
    return TestClient(app), sm


@pytest.mark.slow
def test_vllm_models_endpoint(monkeypatch):
    client, _ = _build_vllm_client(monkeypatch)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    # Should reflect MODEL_ID from env
    assert data["data"][0]["id"] == os.environ.get("MODEL_ID", DEFAULT_MODEL)


@pytest.mark.slow
def test_vllm_chat_completion_non_stream(monkeypatch):
    client, _ = _build_vllm_client(monkeypatch)
    payload = {
        "model": os.environ.get("MODEL_ID", DEFAULT_MODEL),
        "messages": [{"role": "user", "content": "Say hi in five words."}],
        "max_tokens": 12,
        "stream": False,
        "temperature": 0.7,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], str) and len(msg["content"]) > 0
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.slow
def test_vllm_chat_completion_streaming_sse(monkeypatch):
    client, _ = _build_vllm_client(monkeypatch)
    payload = {
        "model": os.environ.get("MODEL_ID", DEFAULT_MODEL),
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
