"""Abstract LLM provider interface – every model backend implements this.

Supports streaming responses and tool/function calling.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class LLMMessage:
    """A single message in the conversation."""

    role: str  # system | user | assistant | tool
    content: str | list[dict[str, Any]] = ""
    tool_call_id: str | None = None
    name: str | None = None  # tool name for role=tool


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """A complete (non-streaming) LLM response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StreamDelta:
    """An incremental streaming chunk."""

    text: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None


class LLMProvider(abc.ABC):
    """Abstract base for an LLM provider (Anthropic, OpenAI, etc.)."""

    @property
    @abc.abstractmethod
    def provider_id(self) -> str:
        """e.g. 'anthropic', 'openai'."""

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a complete response."""

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream response deltas. Default impl wraps complete() as a single delta."""
        response = await self.complete(
            messages, model=model, tools=tools, max_tokens=max_tokens,
            temperature=temperature, system=system,
        )
        yield StreamDelta(text=response.text, finish_reason=response.stop_reason)
