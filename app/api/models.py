from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal
from typing import Any, Dict, List, Optional


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None  # JSONSchema (not used in minimal)


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ChatMessage(BaseModel):
    role: str  # "system"|"user"|"assistant"|"tool"
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    max_tokens: Optional[int] = 256
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    seed: Optional[int] = None
    tools: Optional[List[ToolSpec]] = None
    tool_choice: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    finish_reason: Optional[str] = None
    message: Dict[str, Any]


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
