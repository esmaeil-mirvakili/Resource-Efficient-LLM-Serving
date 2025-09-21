from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


@dataclass
class GenParams:
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop: List[str] | None = None
    seed: Optional[int] = None


class TransformersBackend:
    def __init__(
        self,
        model_id: str = "hf-internal-testing/tiny-random-gpt2",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        # Keep CPU math single-threaded to avoid flakiness under threaded test runners
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        try:
            torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self.model_id = model_id
        self.device = device or "cpu"

        self.quant_mode = os.getenv(
            "TRANSFORMERS_QUANT", "none"
        ).lower()  # none|8bit|4bit
        self.attn_impl = os.getenv("ATTN_IMPL", "sdpa")  # sdpa | flash2 | eager
        self.adapters = [s for s in os.getenv("ADAPTERS", "").split(",") if s.strip()]
        self.draft_model_id = os.getenv("DRAFT_MODEL_ID", "").strip() or None

        self.DEFAULT_STOPS = [
            "\nUser:",  # ideal match
            "\nUser",  # safety: colon sometimes comes a token later
            "\nuser:",
            "\nuser",
            "\nHuman:",
            "\nHuman",
        ]
        self.rep_penalty = float(os.getenv("REPETITION_PENALTY", "1.05"))  # >1.0 reduces repeats
        self.no_repeat_ngram = int(os.getenv("NO_REPEAT_NGRAM", "3"))      # 0 disables

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=trust_remote_code
        )

        model_kwargs = {"attn_implementation": self.attn_impl} if self.attn_impl else {}
        quant_cfg = None
        if self.quant_mode in ("8bit", "4bit") and BitsAndBytesConfig is not None:
            if self.quant_mode == "8bit":
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            else:
                # 4-bit QLoRA-style config
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            model_kwargs["quantization_config"] = quant_cfg
            model_kwargs["device_map"] = "auto"

        used_device_map = "device_map" in model_kwargs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_safetensors=True,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        if not used_device_map:
            self.model.to(self.device)

        if self.adapters and PeftModel is not None:
            for adapter in self.adapters:
                try:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        adapter,
                        trust_remote_code=trust_remote_code,
                        is_trainable=False,
                    )
                except:
                    pass
        self.model.eval()

        # draft model for speculative decoding
        self.draft_model = None
        if self.draft_model_id:
            try:
                self.draft_tok = AutoTokenizer.from_pretrained(
                    self.draft_model_id, use_fast=True
                )
                dm_kwargs = ({"device_map": "auto"} if quant_cfg else {})
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.draft_model_id,
                    **dm_kwargs,
                ).eval()
                if "device_map" not in dm_kwargs:
                    self.draft_model.to(self.device)
            except Exception:
                pass

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
            gen_kwargs = dict(
                input_ids,
                max_new_tokens=int(max_tokens or 128),
                do_sample=(gp.temperature is not None and gp.temperature > 0),
                temperature=float(max(gp.temperature, 1e-5)),
                top_p=float(gp.top_p),
                top_k=int(gp.top_k),
                pad_token_id=self._pad_id(),
                eos_token_id=None,
            )
            if self.draft_model is not None:
                gen_kwargs["assistant_model"] = self.draft_model
            if self.rep_penalty and self.rep_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = float(self.rep_penalty)
            if self.no_repeat_ngram and self.no_repeat_ngram > 0:
                gen_kwargs["no_repeat_ngram_size"] = int(self.no_repeat_ngram)
            out_ids = self.model.generate(
                input_ids,
                **gen_kwargs,
            )[0].tolist()

        completion_ids = out_ids[len(prompt_ids) :]
        if gp.stop:
            completion_ids = self._truncate_on_stops(
                prompt_ids, completion_ids, gp.stop
            )

        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        usage = self._usage(prompt_ids, completion_ids)
        await asyncio.sleep(0)  # cooperative yield
        return text, usage

    async def stream_generate(
        self,
        messages: List[Dict[str, Any]] | List[Any],
        max_tokens: int | None,
        params: Dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        gp = self._parse_params(params)
        prompt_ids = self._encode_chat(messages)
        start_idx = len(prompt_ids)
        max_new = int(max_tokens or 128)
        self._apply_seed(gp.seed)

        generated: List[int] = prompt_ids[:]  # running full sequence

        # Compile stops (token patterns) and compute a conservative char buffer
        stop_ids = self._compile_stops(gp.stop)
        def _pat_text(pat: List[int]) -> str:
            return self.tokenizer.decode(
                pat, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        max_stop_chars = 0
        for pat in stop_ids:
            t = _pat_text(pat)
            if len(t) > max_stop_chars:
                max_stop_chars = len(t)

        # Text buffering so we don't leak a stop suffix mid-stream
        flushed_upto = 0      # chars already yielded
        full_text_cached = "" # cumulative decoded text for completion
        stop_trim_chars = 0   # how many chars to trim at the very end (matched stop)

        with torch.no_grad():
            for _ in range(max_new):
                last_ctx = generated[-1024:]  # naive context window
                logits = self.model(
                    input_ids=torch.tensor([last_ctx], device=self.device)
                ).logits[0, -1]

                next_token = int(self._sample_from_logits(logits, gp, generated))
                generated.append(next_token)

                # Cumulative decode of the *completion*; emit only the new tail
                full_text = self.tokenizer.decode(
                    generated[start_idx:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # Append only what changed since last step
                if len(full_text) > len(full_text_cached):
                    new_piece = full_text[len(full_text_cached):]
                else:
                    new_piece = ""

                # If no custom stops → flush new text right away
                if not stop_ids:
                    if new_piece:
                        yield new_piece
                    full_text_cached = full_text
                    # Early break on EOS if present
                    if (
                        self.tokenizer.eos_token_id is not None
                        and next_token == self.tokenizer.eos_token_id
                    ):
                        break
                    await asyncio.sleep(0)
                    continue

                # With custom stops: keep a safety buffer to avoid leaking suffixes
                if new_piece:
                    # Compute safe flush boundary keeping max_stop_chars buffered
                    safe_flush_upto = max(0, len(full_text) - max_stop_chars)
                    if safe_flush_upto > flushed_upto:
                        yield full_text[flushed_upto:safe_flush_upto]
                        flushed_upto = safe_flush_upto
                    full_text_cached = full_text

                # Detect a stop match based on token pattern suffix
                if stop_ids and self._has_stop_suffix(generated, stop_ids):
                    # Work out which pattern matched to remove its textual form
                    matched_len_chars = 0
                    for pat in stop_ids:
                        n = len(pat)
                        if n and generated[-n:] == pat:
                            matched_len_chars = len(_pat_text(pat))
                            break
                    stop_trim_chars = matched_len_chars
                    break

                # Also stop on EOS if provided by tokenizer
                if (
                    self.tokenizer.eos_token_id is not None
                    and next_token == self.tokenizer.eos_token_id
                ):
                    break

                await asyncio.sleep(0)  # cooperative yield

            # Final flush (whatever remains that wasn't part of a stop)
            # Remove the textual stop suffix if we matched one.
            end_len = max(0, len(full_text_cached) - stop_trim_chars)
            tail = full_text_cached[flushed_upto:end_len]
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
            # If caller didn’t set params, use defaults AND default stops
            return GenParams(stop=list(self.DEFAULT_STOPS))

        raw_temp = params.get("temperature", 0.7) or 0.7
        raw_top_p = params.get("top_p", 1.0) or 1.0
        raw_top_k = params.get("top_k", 0) or 0
        raw_stop = params.get("stop") or []   # may be []
        raw_seed = params.get("seed")

        stops = list(raw_stop) if raw_stop else list(self.DEFAULT_STOPS)

        return GenParams(
            temperature=float(raw_temp),
            top_p=float(raw_top_p),
            top_k=int(raw_top_k),
            stop=stops,
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
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if tmpl:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer.encode(text, add_special_tokens=False)
        text = ""
        for m in messages:
            role = (self._field(m, "role") or "user").strip().lower()
            content = (self._field(m, "content") or "").strip()
            if not content:
                continue

            if role == "system":
                # Skip for non-instruction-tuned models; otherwise they echo it.
                continue
            elif role == "user":
                # Space after Assistant: reduces chance of immediate newline
                text += f"User: {content}\nAssistant: "
            elif role == "assistant":
                text += f"{content}\n"
            else:
                # Unknown roles treated as user context
                text += f"{role.capitalize()}: {content}\n"
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

    def _sample_from_logits(
        self, logits: torch.Tensor, gp: GenParams, generated: List[int]
    ) -> int:
        """
        Manual temperature/top-k/top-p sampling for streaming.
        """
        if self.rep_penalty and self.rep_penalty > 1.0 and generated:
            recent = list(set(generated[-128:]))  # small window
            idx = torch.tensor(recent, device=logits.device)
            logits.index_put_((idx,), logits.index_select(0, idx) / float(self.rep_penalty))
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
