# app/core/backends/vllm_backend.py
from __future__ import annotations
import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from vllm import LLM, SamplingParams
except Exception as e:  # pragma: no cover
    raise ImportError(
        "vLLM is not installed or failed to import. "
        "Install with `pip install vllm` (CUDA) or `pip install vllm-cpu` (experimental)."
    ) from e


class VLLMBackend:
    """
    Minimal vLLM backend:
      - same interface as your other backends
      - honors temperature/top_p/top_k/stop/seed/max_tokens
      - streaming implemented by chunking final text (simple, reliable)
    NOTE: vLLM works best on Linux + CUDA. CPU/Mac may be unsupported/very slow.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,  # "cuda"|"cpu"|None (vLLM auto-detect by default)
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: Optional[str] = None,  # "auto"|"float16"|...
        download_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        # Let vLLM pick device by default; allow override if caller insists
        llm_kwargs: Dict[str, Any] = {
            "model": model_id,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if dtype is not None:
            llm_kwargs["dtype"] = dtype
        if download_dir is not None:
            llm_kwargs["download_dir"] = download_dir
        if device is not None:
            llm_kwargs["device"] = device  # vLLM supports "cuda"|"cpu" when available

        # Respect HF cache if present
        cache_dir = os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HOME")
        if cache_dir and "download_dir" not in llm_kwargs:
            llm_kwargs["download_dir"] = cache_dir

        self.llm = LLM(**llm_kwargs)
        # vLLM exposes the tokenizer via get_tokenizer()
        self.tokenizer = self.llm.get_tokenizer()

    # ---------- public API ----------
    def new_request_id(self) -> str:
        import uuid

        return f"cmpl-{uuid.uuid4().hex[:24]}"

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int | None,
        params: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, int]]:
        prompt = self._encode_chat(messages)
        sp = self._sampling_params(params, max_tokens)

        # vLLM generate() is sync; run off the event loop
        outputs = await asyncio.to_thread(self.llm.generate, [prompt], sp)
        out = outputs[0]

        # vLLM gives token ids for prompt and completion
        prompt_tokens = len(getattr(out, "prompt_token_ids", []) or [])
        # out.outputs is a list (n=1 for single sample)
        completion = out.outputs[0]
        completion_tokens = len(getattr(completion, "token_ids", []) or [])
        text = completion.text

        usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        }
        return text, usage

    async def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int | None,
        params: Dict[str, Any] | None = None,
    ):
        # Simple + reliable: generate fully, then drip content
        text, _ = await self.generate(messages, max_tokens, params)
        # Yield in small chunks to satisfy SSE contract
        stride = 16
        i = 0
        while i < len(text):
            piece = text[i : i + stride]
            if piece:
                yield piece
                await asyncio.sleep(0)
            i += stride

    # ---------- internals ----------
    def _field(self, msg: Any, key: str):
        if isinstance(msg, dict):
            return msg.get(key)
        md = getattr(msg, "model_dump", None)
        if callable(md):
            return md(exclude_none=True).get(key)
        return getattr(msg, key, None)

    def _encode_chat(self, messages: List[Dict[str, Any]] | List[Any]) -> str:
        # Keep identical to your Transformers prompt template for parity
        text = ""
        for m in messages:
            role = (self._field(m, "role") or "user").strip()
            content = (self._field(m, "content") or "").strip()
            if not content:
                continue
            if role == "user":
                text += f"User: {content}\nAssistant:"
            elif role == "assistant":
                text += f" {content}\n"
            else:
                text += f"{role}: {content}\n"
        return text

    def _sampling_params(
        self, params: Dict[str, Any] | None, max_tokens: int | None
    ) -> "SamplingParams":
        p = (params or {}).copy()
        # Coerce None → defaults
        temperature = 0.7 if p.get("temperature") is None else float(p["temperature"])
        top_p = 1.0 if p.get("top_p") is None else float(p["top_p"])
        top_k = 0 if p.get("top_k") is None else int(p["top_k"])
        stop = list((p.get("stop") or []))
        seed = p.get("seed", None)

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=int(max_tokens or 128),
            stop=stop or None,
            seed=None if seed is None else int(seed),
        )
