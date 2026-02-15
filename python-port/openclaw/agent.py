"""Agent runner – orchestrates LLM calls, tool execution, and session management.

This is the heart of OpenClaw: the agentic loop that receives a user message,
builds context from the session transcript, calls the LLM (possibly multiple
times if tools are invoked), and returns the final response.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Awaitable

from openclaw.config import AgentConfig, OpenClawConfig, resolve_agent
from openclaw.models.base import LLMMessage, LLMProvider, LLMResponse, ToolCall
from openclaw.session import SessionStore, Turn
from openclaw.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 25  # safety cap on agentic loops


@dataclass
class AgentRunResult:
    """Final output of an agent run."""

    text: str = ""
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    rounds: int = 0


@dataclass
class StreamEvent:
    """Real-time event emitted during an agent run."""

    type: str  # text_delta | tool_call | tool_result | done
    text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""


class AgentRunner:
    """Runs the agentic LLM loop for a single conversation turn.

    Flow:
        1. Load session context from JSONL
        2. Build system prompt from workspace bootstrap files
        3. Call LLM with messages + tool definitions
        4. If LLM requests tools → execute → feed results → repeat
        5. When LLM produces final text → return it and append turn to JSONL
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        session_store: SessionStore,
        tools: list[Tool],
        config: OpenClawConfig,
        agent_config: AgentConfig,
    ) -> None:
        self.provider = provider
        self.session_store = session_store
        self.tools = {t.name: t for t in tools}
        self.config = config
        self.agent_config = agent_config

    # -- public API ----------------------------------------------------------

    async def run(
        self,
        user_text: str,
        *,
        session_id: str,
        on_event: Callable[[StreamEvent], Awaitable[None]] | None = None,
    ) -> AgentRunResult:
        """Execute one full agent turn (possibly multiple LLM rounds)."""
        model_ref = self.agent_config.model.primary
        provider_id, model_id = _parse_model_ref(model_ref)

        # load session history
        history = self.session_store.build_context_messages(session_id, limit=50)
        system_prompt = self._build_system_prompt()
        tool_schemas = [t.to_schema() for t in self.tools.values()]

        # build initial messages
        messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]
        for msg in history:
            messages.append(LLMMessage(role=msg["role"], content=msg.get("content", "")))
        messages.append(LLMMessage(role="user", content=user_text))

        total_input = 0
        total_output = 0
        tool_results_log: list[dict[str, Any]] = []
        final_text = ""
        rounds = 0

        while rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            response = await self.provider.complete(
                messages,
                model=model_id,
                tools=tool_schemas if self.tools else None,
                system=system_prompt,
            )
            total_input += response.input_tokens
            total_output += response.output_tokens

            # accumulate text
            if response.text:
                final_text += response.text
                if on_event:
                    await on_event(StreamEvent(type="text_delta", text=response.text))

            # if no tool calls, we're done
            if not response.tool_calls:
                break

            # append the assistant response (with tool calls) to messages
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            messages.append(LLMMessage(role="assistant", content=assistant_content))

            # execute each tool call
            for tc in response.tool_calls:
                if on_event:
                    await on_event(StreamEvent(
                        type="tool_call", tool_name=tc.name, tool_args=tc.arguments,
                    ))

                result = await self._execute_tool(tc)
                tool_results_log.append({
                    "tool": tc.name,
                    "args": tc.arguments,
                    "output": result.output,
                    "error": result.error,
                })

                if on_event:
                    await on_event(StreamEvent(
                        type="tool_result", tool_name=tc.name, tool_output=result.output,
                    ))

                # feed result back as a tool message
                content = result.output
                if result.is_error and result.error:
                    content = f"Error: {result.error}\n{result.output}" if result.output else f"Error: {result.error}"
                messages.append(LLMMessage(
                    role="tool",
                    content=content,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

            # clear text for next round (tool results may produce new text)
            # the final_text already captured the partial; next round appends more
            final_text = ""

        # the last round's text is the final answer
        # (if the loop ended after a tool round, the LLM's final response is in final_text)

        # append turn to session transcript
        turn_messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": final_text},
        ]
        turn = Turn(
            id=f"turn-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            messages=turn_messages,
        )
        self.session_store.append_turn(session_id, turn)
        self.session_store.update_stats(
            session_id,
            input_tokens=total_input,
            output_tokens=total_output,
        )

        if on_event:
            await on_event(StreamEvent(type="done", text=final_text))

        return AgentRunResult(
            text=final_text,
            tool_results=tool_results_log,
            input_tokens=total_input,
            output_tokens=total_output,
            rounds=rounds,
        )

    # -- internals -----------------------------------------------------------

    async def _execute_tool(self, tc: ToolCall) -> ToolResult:
        tool = self.tools.get(tc.name)
        if not tool:
            return ToolResult(output="", error=f"Unknown tool: {tc.name}", is_error=True)
        try:
            return await tool.execute(tc.arguments)
        except Exception as exc:
            log.exception("Tool %s failed", tc.name)
            return ToolResult(output="", error=str(exc), is_error=True)

    def _build_system_prompt(self) -> str:
        """Build the system prompt from workspace bootstrap files + identity."""
        parts: list[str] = []
        identity = self.agent_config.identity
        parts.append(f"You are {identity.name}, a helpful AI assistant.")

        # load workspace bootstrap files if they exist
        workspace = self.agent_config.workspace
        if workspace:
            ws_path = Path(workspace).expanduser()
            for filename in ("SOUL.md", "AGENTS.md", "IDENTITY.md"):
                boot_file = ws_path / filename
                if boot_file.exists():
                    parts.append(boot_file.read_text().strip())

        # tool descriptions
        if self.tools:
            tool_lines = ["", "# Available Tools", ""]
            for tool in self.tools.values():
                tool_lines.append(f"- **{tool.name}**: {tool.description}")
            parts.append("\n".join(tool_lines))

        return "\n\n".join(parts)


def _parse_model_ref(ref: str) -> tuple[str, str]:
    """Parse 'provider/model-id' into (provider, model_id)."""
    if "/" in ref:
        provider, model_id = ref.split("/", 1)
        return provider, model_id
    return "anthropic", ref


def create_provider(config: OpenClawConfig, provider_id: str) -> LLMProvider:
    """Factory: create an LLM provider from config."""
    provider_cfg = config.models.providers.get(provider_id)
    api_key = provider_cfg.api_key if provider_cfg else None
    base_url = provider_cfg.base_url if provider_cfg else None

    if provider_id == "anthropic":
        from openclaw.models.anthropic import AnthropicProvider
        return AnthropicProvider(api_key=api_key or "", base_url=base_url)
    elif provider_id == "openai":
        from openclaw.models.openai import OpenAIProvider
        return OpenAIProvider(api_key=api_key or "", base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_id}")
