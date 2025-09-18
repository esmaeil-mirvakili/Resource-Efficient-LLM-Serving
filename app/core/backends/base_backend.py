from typing import Any, AsyncGenerator, Dict, List, Tuple


class LLMBackend:
    model_id: str

    def new_request_id(self) -> str:
        raise NotImplementedError()
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int | None,
        params: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, int]]:
        raise NotImplementedError()
    
    async def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int | None,
        params: Dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError()
