"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from openclaw.models.base import LLMMessage, LLMProvider, LLMResponse, StreamDelta, ToolCall

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider."""

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120.0,
        )

    @property
    def provider_id(self) -> str:
        return "anthropic"

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert internal messages to Anthropic Messages API format."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue  # system prompt handled separately
            if msg.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                    }],
                })
            elif msg.role == "assistant" and isinstance(msg.content, list):
                out.append({"role": "assistant", "content": msg.content})
            else:
                out.append({"role": msg.role, "content": msg.content})
        return out

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert tool definitions to Anthropic format."""
        if not tools:
            return None
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", t.get("input_schema", {"type": "object"})),
            }
            for t in tools
        ]

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
        # extract system prompt from messages or param
        sys_prompt = system
        if not sys_prompt:
            for msg in messages:
                if msg.role == "system":
                    sys_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

        body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if sys_prompt:
            body["system"] = sys_prompt

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            body["tools"] = converted_tools

        resp = await self._client.post("/messages", json=body)
        resp.raise_for_status()
        data = resp.json()

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=data.get("stop_reason", "end_turn"),
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
        )

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
        sys_prompt = system
        if not sys_prompt:
            for msg in messages:
                if msg.role == "system":
                    sys_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

        body: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if sys_prompt:
            body["system"] = sys_prompt
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            body["tools"] = converted_tools

        async with self._client.stream("POST", "/messages", json=body) as resp:
            resp.raise_for_status()
            current_tool: dict[str, Any] | None = None
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                event = json.loads(payload)
                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {"id": block["id"], "name": block["name"], "input_json": ""}

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamDelta(text=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta" and current_tool:
                        current_tool["input_json"] += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    if current_tool:
                        try:
                            args = json.loads(current_tool["input_json"]) if current_tool["input_json"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield StreamDelta(tool_call=ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=args,
                        ))
                        current_tool = None

                elif event_type == "message_delta":
                    stop = event.get("delta", {}).get("stop_reason")
                    if stop:
                        yield StreamDelta(finish_reason=stop)
