# app/tests/test_generation_params.py
import os
import importlib
import pytest
from fastapi.testclient import TestClient
import sys, pathlib

# Ensure project root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytest.importorskip("transformers", reason="transformers not installed")

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"  # ships safetensors


def _client_with_transformers(monkeypatch, model_id: str = MODEL_ID):
    # Keep everything deterministic/safe on CPU
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    import app.server as sm

    sm = importlib.reload(sm)
    try:
        sm.get_settings.cache_clear()
        sm.get_backend.cache_clear()
    except Exception:
        pass

    from app.core.backends.transformers_backend import TransformersBackend

    try:
        backend = TransformersBackend(model_id=model_id, device="cpu")
    except Exception as e:
        pytest.skip(f"Could not load model '{model_id}': {e!r}")

    # Force this backend instance
    monkeypatch.setattr(sm, "get_backend", lambda: backend, raising=True)
    app = sm.create_app()
    return TestClient(app), sm


@pytest.mark.slow
def test_seed_determinism(monkeypatch):
    client, sm = _client_with_transformers(monkeypatch)

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say hi in five words."}],
        "max_tokens": 16,
        "stream": False,
        "temperature": 0.9,
        "top_p": 0.95,
        "seed": 1234,
    }

    r1 = client.post("/v1/chat/completions", json=payload)
    r2 = client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200, (r1.text, r2.text)
    t1 = r1.json()["choices"][0]["message"]["content"]
    t2 = r2.json()["choices"][0]["message"]["content"]
    assert t1 == t2  # same seed → identical text

    # Different seed → likely different text (not guaranteed but extremely likely on random sampling)
    payload2 = dict(payload, seed=9876)
    r3 = client.post("/v1/chat/completions", json=payload2)
    assert r3.status_code == 200, r3.text
    t3 = r3.json()["choices"][0]["message"]["content"]
    # If equal (rare), at least we proved same-seed determinism above.
    if t3 == t1:
        pytest.xfail("Different seed produced same text (rare with tiny model).")


@pytest.mark.slow
def test_stop_sequences(monkeypatch):
    client, sm = _client_with_transformers(monkeypatch)

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "Write two sentences. Then say: NEXT USER"}
        ],
        "max_tokens": 32,
        "stream": False,
        "temperature": 0.7,
        "top_p": 1.0,
        "stop": ["\nUser:"],  # simulate chat turn boundary
    }

    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    text = r.json()["choices"][0]["message"]["content"]
    assert "\nUser:" not in text  # must truncate before emitting the stop sequence
