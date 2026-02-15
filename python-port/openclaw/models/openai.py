"""OpenAI (GPT) LLM provider."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from openclaw.models.base import LLMMessage, LLMProvider, LLMResponse, StreamDelta, ToolCall

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions API provider (also works for compatible APIs)."""

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    @property
    def provider_id(self) -> str:
        return "openai"

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                })
            else:
                out.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        return out

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", t.get("input_schema", {"type": "object"})),
                },
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
        api_messages = self._convert_messages(messages)

        # inject system prompt as first message if not already present
        if system and (not api_messages or api_messages[0].get("role") != "system"):
            api_messages.insert(0, {"role": "system", "content": system})

        body: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            body["tools"] = converted_tools

        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"]) if fn["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=args))

        return LLMResponse(
            text=msg.get("content") or "",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason", "stop"),
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
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
        api_messages = self._convert_messages(messages)
        if system and (not api_messages or api_messages[0].get("role") != "system"):
            api_messages.insert(0, {"role": "system", "content": system})

        body: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            body["tools"] = converted_tools

        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            # track partial tool calls across deltas
            partial_tools: dict[int, dict[str, Any]] = {}

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                choice = event["choices"][0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                # text content
                if delta.get("content"):
                    yield StreamDelta(text=delta["content"])

                # tool call deltas
                for tc in delta.get("tool_calls", []):
                    idx = tc["index"]
                    if idx not in partial_tools:
                        partial_tools[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                    if tc.get("id"):
                        partial_tools[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        partial_tools[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        partial_tools[idx]["arguments"] += fn["arguments"]

                if finish:
                    # emit any accumulated tool calls
                    for pt in partial_tools.values():
                        try:
                            args = json.loads(pt["arguments"]) if pt["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        yield StreamDelta(tool_call=ToolCall(
                            id=pt["id"], name=pt["name"], arguments=args,
                        ))
                    partial_tools.clear()
                    yield StreamDelta(finish_reason=finish)
