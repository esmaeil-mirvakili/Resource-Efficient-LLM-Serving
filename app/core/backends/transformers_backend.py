from __future__ import annotations
import asyncio, math
from typing import Any, AsyncGenerator, Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersBackend:
    def __init__(self, model_id: str = "distilgpt2", device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def new_request_id(self) -> str:
        import uuid

        return f"cmpl-{uuid.uuid4().hex[:24]}"

    # ---- token utilities ----
    def _encode_chat(self, messages: List[Dict[str, Any]]) -> List[int]:
        # ultra-minimal: just concatenate roles; replace with a template later
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
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
        torch.manual_seed(seed)
        # (GPU later) torch.cuda.manual_seed_all(seed)

    # ---- generation ----
    async def generate(
        self, messages: List[Dict[str, Any]], max_tokens: int | None
    ) -> Tuple[str, Dict[str, int]]:
        prompt_ids = self._encode_chat(messages)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        max_new = int(max_tokens or 128)
        self._apply_seed(None)  # hook up req.seed when you plumb it through

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                eos_token_id=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0].tolist()

        completion_ids = out_ids[len(prompt_ids) :]
        text = self._decode_new(out_ids, len(prompt_ids))
        usage = self._usage(prompt_ids, completion_ids)
        await asyncio.sleep(0)  # yield control
        return text, usage

    async def stream_generate(
        self, messages: List[Dict[str, Any]], max_tokens: int | None
    ) -> AsyncGenerator[str, None]:
        # super-simple token-by-token loop (greedy) for learning; replace with faster search later
        prompt_ids = self._encode_chat(messages)
        max_new = int(max_tokens or 128)
        self._apply_seed(None)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated: List[int] = prompt_ids[:]
        with torch.no_grad():
            for _ in range(max_new):
                outputs = self.model(
                    input_ids=torch.tensor([generated[-1024:]], device=self.device)
                )  # context trim
                next_token = int(outputs.logits[0, -1].argmax())
                generated.append(next_token)
                piece = self.tokenizer.decode([next_token], skip_special_tokens=True)
                if piece:
                    yield piece
                    await asyncio.sleep(0)  # cooperative
                # naive stop on EOS if defined
                if (
                    self.tokenizer.eos_token_id is not None
                    and next_token == self.tokenizer.eos_token_id
                ):
                    break
