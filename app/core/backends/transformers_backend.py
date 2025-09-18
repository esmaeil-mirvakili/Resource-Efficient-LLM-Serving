from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenParams:
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop: List[str] | None = None
    seed: Optional[int] = None


class TransformersBackend:
    """
    Minimal, CPU-first Transformers backend with:
      - safetensors-only model loading (avoids torch.load CVE on .bin)
      - seedable sampling
      - token-aware stop sequences
      - streaming (token-by-token) for learning purposes

    NOTE: keep models tiny for tests/local CPU:
      - "hf-internal-testing/tiny-random-gpt2" (ships safetensors)
      - "sshleifer/tiny-gpt2" may be .bin → upgrade torch>=2.6 if you insist
    """

    def __init__(
        self,
        model_id: str = "hf-internal-testing/tiny-random-gpt2",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        # Keep CPU math single-threaded to avoid flakiness under threaded test runners
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            os.environ.setdefault(k, "1")
        try:
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self.model_id = model_id
        self.device = device or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=trust_remote_code
        )

        # Prefer safetensors for security (CVE-2025-32434). If this fails, tell the caller to
        # pick a safetensors model or upgrade PyTorch >= 2.6 to allow .bin safely.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                use_safetensors=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load '{model_id}' with safetensors only. "
                f"Use a safetensors-enabled model (e.g., 'hf-internal-testing/tiny-random-gpt2') "
                f"or upgrade torch to >= 2.6 if the repo only has .bin. "
                f"Underlying error: {type(e).__name__}: {e}"
            ) from e

        self.model.to(self.device)
        self.model.eval()

        # Some GPT2 tokenizers have no pad token; map pad to eos to avoid generate() warnings
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---- public API ----------------------------------------------------------

    def new_request_id(self) -> str:
        import uuid

        return f"cmpl-{uuid.uuid4().hex[:24]}"

    async def generate(
        self,
        messages: List[Dict[str, Any]] | List[Any],
        max_tokens: int | None,
        params: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, int]]:
        gp = self._parse_params(params)
        prompt_ids = self._encode_chat(messages)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        self._apply_seed(gp.seed)

        # We enforce custom stop sequences ourselves → do not set eos_token_id here
        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=int(max_tokens or 128),
                do_sample=(gp.temperature is not None and gp.temperature > 0),
                temperature=float(max(gp.temperature, 1e-5)),
                top_p=float(gp.top_p),
                top_k=int(gp.top_k),
                pad_token_id=self._pad_id(),
                eos_token_id=None,
            )[0].tolist()

        completion_ids = out_ids[len(prompt_ids) :]
        if gp.stop:
            completion_ids = self._truncate_on_stops(
                prompt_ids, completion_ids, gp.stop
            )

        text = self._decode_new(prompt_ids + completion_ids, len(prompt_ids))
        usage = self._usage(prompt_ids, completion_ids)
        await asyncio.sleep(0)  # cooperative yield
        return text, usage

    async def stream_generate(
        self,
        messages: List[Dict[str, Any]] | List[Any],
        max_tokens: int | None,
        params: Dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Simple token-by-token loop (CPU-friendly, pedagogical).
        - Applies temperature/top_p/top_k manually on the last-step logits.
        - Enforces stop sequences by buffering and truncating before yield.
        """
        gp = self._parse_params(params)
        prompt_ids = self._encode_chat(messages)
        max_new = int(max_tokens or 128)
        self._apply_seed(gp.seed)

        generated: List[int] = prompt_ids[:]  # running full sequence
        stop_ids = self._compile_stops(gp.stop)
        max_stop = max((len(s) for s in stop_ids), default=0)

        # Text buffering so we don't leak a stop suffix mid-stream
        flushed_upto = 0  # chars flushed from `accum_text`
        accum_text = ""  # decoded text for completion produced so far

        with torch.no_grad():
            for _ in range(max_new):
                last_ctx = generated[-1024:]  # context window trim (toy)
                logits = self.model(
                    input_ids=torch.tensor([last_ctx], device=self.device)
                ).logits[0, -1]
                next_token = self._sample_from_logits(logits, gp)

                generated.append(int(next_token))

                # Check for custom stops (on token ids)
                if stop_ids and self._has_stop_suffix(generated, stop_ids):
                    # Trim away the stop pattern from completion
                    trimmed = self._trim_stop_suffix(prompt_ids, generated, stop_ids)
                    # Decode only the *new* non-stop part
                    accum_text += self._decode_new(
                        trimmed,
                        len(prompt_ids)
                        + len(self._completion_ids(prompt_ids, trimmed))
                        - 1,
                    )[
                        -0:
                    ]  # no-op decode safeguard
                    break

                # Decode just this one token to text and append
                piece = self.tokenizer.decode(
                    [int(next_token)], skip_special_tokens=True
                )
                if piece:
                    accum_text += piece

                # Flush safely: keep at least `max_stop` chars unflushed to avoid leaking a stop suffix
                if max_stop == 0:
                    # No custom stops → flush immediately
                    yield piece
                else:
                    safe_flush_upto = max(0, len(accum_text) - max_stop)
                    if safe_flush_upto > flushed_upto:
                        yield accum_text[flushed_upto:safe_flush_upto]
                        flushed_upto = safe_flush_upto

                await asyncio.sleep(0)  # cooperative yield

            # Final flush (whatever remains that wasn't part of a stop)
            tail = accum_text[flushed_upto:]
            if tail:
                yield tail

    # ---- internals ----------------------------------------------------------

    def _pad_id(self) -> Optional[int]:
        return (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

    def _parse_params(self, params: Dict[str, Any] | None) -> GenParams:
        if not params:
            return GenParams()

        # Coerce None → defaults
        raw_temp = params.get("temperature", 0.7)
        if raw_temp is None:
            raw_temp = 0.7

        raw_top_p = params.get("top_p", 1.0)
        if raw_top_p is None:
            raw_top_p = 1.0

        raw_top_k = params.get("top_k", 0)
        if raw_top_k is None:
            raw_top_k = 0

        raw_stop = params.get("stop") or []
        raw_seed = params.get("seed")

        return GenParams(
            temperature=float(raw_temp),
            top_p=float(raw_top_p),
            top_k=int(raw_top_k),
            stop=list(raw_stop),
            seed=raw_seed,
        )
        
    def _truncate_on_stops(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        stops: list[str] | None,
    ) -> list[int]:
        """
        Find the earliest occurrence of any stop pattern in the *completion*
        and truncate before it. Patterns are matched on token ids to be robust
        to multi-token stops.
        """
        stop_ids = self._compile_stops(stops)
        if not stop_ids or not completion_ids:
            return completion_ids

        full = prompt_ids + completion_ids
        plen = len(prompt_ids)

        # Scan forward once; when a stop suffix matches at position i, cut there.
        for i in range(plen, len(full) + 1):
            for pat in stop_ids:
                n = len(pat)
                if n and i >= n and full[i - n : i] == pat:
                    # Cut completion *before* the stop pattern start
                    cut_len = (i - n) - plen
                    return completion_ids[: max(cut_len, 0)]

        return completion_ids

    # Support both dict messages and Pydantic objects
    def _field(self, msg: Any, key: str):
        if isinstance(msg, dict):
            return msg.get(key)
        # pydantic v2 BaseModel
        md = getattr(msg, "model_dump", None)
        if callable(md):
            return md(exclude_none=True).get(key)
        return getattr(msg, key, None)

    def _encode_chat(self, messages: List[Dict[str, Any]] | List[Any]) -> List[int]:
        # Ultra-minimal prompt template; swap for ChatML or your house style later.
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
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode_new(self, ids: List[int], start_idx: int) -> str:
        if start_idx >= len(ids):
            return ""
        return self.tokenizer.decode(ids[start_idx:], skip_special_tokens=True)

    def _usage(
        self, prompt_ids: List[int], completion_ids: List[int]
    ) -> Dict[str, int]:
        return {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": len(prompt_ids) + len(completion_ids),
        }

    def _apply_seed(self, seed: Optional[int]):
        if seed is None:
            return
        torch.manual_seed(int(seed))
        # (GPU later) torch.cuda.manual_seed_all(seed)

    # --- stop sequences (token-aware) ---

    def _compile_stops(self, stops: List[str] | None) -> List[List[int]]:
        toks: List[List[int]] = []
        for s in stops or []:
            s = (s or "").strip()
            if not s:
                continue
            toks.append(self.tokenizer.encode(s, add_special_tokens=False))
        return [t for t in toks if t]

    def _has_stop_suffix(self, full_ids: List[int], stop_ids: List[List[int]]) -> bool:
        for pat in stop_ids:
            n = len(pat)
            if n and full_ids[-n:] == pat:
                return True
        return False

    def _trim_stop_suffix(
        self, prompt_ids: List[int], full_ids: List[int], stop_ids: List[List[int]]
    ) -> List[int]:
        # Remove the trailing stop pattern from `full_ids` and return the trimmed list
        for pat in stop_ids:
            n = len(pat)
            if n and full_ids[-n:] == pat:
                return full_ids[:-n]
        return full_ids

    def _completion_ids(self, prompt_ids: List[int], full_ids: List[int]) -> List[int]:
        return full_ids[len(prompt_ids) :]

    # --- sampling ---

    def _sample_from_logits(self, logits: torch.Tensor, gp: GenParams) -> int:
        """
        Manual temperature/top-k/top-p sampling for streaming.
        """
        # Greedy if temperature is effectively zero
        if gp.temperature is not None and gp.temperature <= 1e-5:
            return int(torch.argmax(logits).item())

        # Temperature
        logits = logits / float(max(gp.temperature, 1e-5))

        # Convert to probabilities (stable)
        probs = torch.softmax(logits, dim=-1)

        # Top-k
        if gp.top_k and gp.top_k > 0:
            topk = min(int(gp.top_k), probs.numel())
            vals, idx = torch.topk(probs, topk)
            mask = torch.zeros_like(probs)
            mask.scatter_(0, idx, vals)
            probs = mask
            probs /= probs.sum()

        # Top-p (nucleus)
        if gp.top_p and gp.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep = cumsum <= gp.top_p
            # Always keep at least one token
            if not bool(keep.any()):
                keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered.scatter_(0, sorted_idx[keep], sorted_probs[keep])
            probs = filtered
            probs_sum = probs.sum()
            if probs_sum.item() == 0.0:
                # fallback: greedy
                return int(torch.argmax(logits).item())
            probs /= probs_sum

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())
