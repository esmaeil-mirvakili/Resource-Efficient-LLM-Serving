from __future__ import annotations
import asyncio
import hashlib
import uuid
from typing import Any, AsyncGenerator, Dict, List, Tuple, Optional
from app.core.backends.base_backend import LLMBackend


class InMemoryBackend(LLMBackend):
    """
    A tiny, deterministic, dependency-free "backend" to make the API usable
    without GPUs. It synthesizes answers based on the last user message and
    simulates token streaming by splitting into words.
    """

    def __init__(self, model_id: str = "dummy-1"):
        self.model_id = model_id

    def _field(self, msg: Any, key: str):
        if isinstance(msg, dict):
            return msg.get(key)
        return getattr(msg, key, None)

    def new_request_id(self) -> str:
        return f"cmpl-{uuid.uuid4().hex[:24]}"

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.strip().split() if t]

    def _count_tokens(
        self, messages: List[Dict[str, Any]], completion_text: str
    ) -> Dict[str, int]:
        prompt_tokens = sum(
            len(self._tokenize(self._field(m, "content") or "")) for m in messages
        )
        completion_tokens = len(self._tokenize(completion_text))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _deterministic_reply(
        self, messages: List[Dict[str, Any]], max_tokens: int
    ) -> str:
        # Use the last user message as base; echo with a tiny transformation
        last_user = next(
            (self._field(m, "content") for m in reversed(messages) if self._field(m, "role") == "user"),
            "",
        )
        if not last_user:
            last_user = "(no user content)"
        # Hash to make output stable per input, then choose a canned style.
        h = int(hashlib.sha256(last_user.encode()).hexdigest(), 16)
        styles = [
            "Here's a concise reply:",
            "Quick take:",
            "TL;DR:",
            "Short answer:",
        ]
        prefix = styles[h % len(styles)]
        body = last_user.strip()
        reply = f"{prefix} {body}"
        words = self._tokenize(reply)[: max_tokens or 256]
        return " ".join(words)

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int | None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        text = self._deterministic_reply(messages, max_tokens or 256)
        usage = self._count_tokens(messages, text)
        # Pretend there's some latency
        await asyncio.sleep(0.01)
        return text, usage

    async def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int | None,
        params: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        text = self._deterministic_reply(messages, max_tokens or 256)
        # Stream word by word
        for w in text.split():
            await asyncio.sleep(0.005)
            yield w + " "
