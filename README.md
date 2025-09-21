# Resource-Efficient-LLM-Serving

## Overview
This project provides a **production-grade LLM serving stack** with:
- **FastAPI backend** exposing **OpenAI-compatible APIs** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/healthz`, `/readyz`, `/metrics`).
- Flexible **inference backends**: HuggingFace Transformers and [vLLM](https://vllm.ai) (with continuous batching, paged KV cache, quantization, speculative decoding, FlashAttention).
- Support for **LoRA/QLoRA adapters** (hot-swappable).
- **Hydra configuration system** for clean overrides per environment.
- **Observability**: Prometheus, OpenTelemetry, Grafana, Tempo.
- **UI**: Next.js/React chat interface for interacting with the model.

---

## Backend: FastAPI LLM Service

### Features
- OpenAI-compatible APIs (`/v1/chat/completions` etc).
- **Transformers backend** for dev/local use.
- **vLLM backend** for high-throughput serving.
- Token accounting + deterministic `seed`.
- Middleware for **auth, rate limiting, logging**.
- Configurable via `.env` or Hydra YAML.

### Running Locally
1. Install deps:
   ```bash
   conda create -n llm_serving python=3.11
   conda activate llm_serving
   pip install -r requirements.txt
   ```
2. Create `.env`:
   ```env
   MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
   BACKEND=vllm
   MAX_CONCURRENCY=4
   ```
3. Start:
   ```bash
   uvicorn app.server:create_app --factory --reload --port 8000
   ```

---

## Frontend: Next.js UI
The `ui/` folder provides a chat interface.

### Running
```bash
cd ui
npm install
npm run dev  # runs at http://localhost:3000
```

The UI calls the backend (`http://localhost:8000/v1/chat/completions`) for streaming responses.

---

## Observability
- **Prometheus** → metrics from `/metrics`
- **OpenTelemetry + Tempo** → traces
- **Grafana** → dashboards for both

### Start Infra
```bash
cd grafana
docker-compose up -d
```

- Prometheus → http://localhost:9090
- Tempo → http://localhost:3200
- Grafana → http://localhost:3100 (default user/pass: admin / admin)

---

## Example Usage
```bash
curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "stream": true
  }'
```

---

## Testing
```bash
pytest tests/
```

Includes:
- **unit tests** for prompts, policies, loader
- **e2e tests** for API
- **perf tests** (throughput)
