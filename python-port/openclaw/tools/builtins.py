"""Built-in tools – read, write, exec (shell), send_message."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from openclaw.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)


class ReadTool(Tool):
    """Read a file from the workspace."""

    def __init__(self, workspace: str) -> None:
        self._workspace = Path(workspace)

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Returns the file text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file (relative to workspace)"},
            },
            "required": ["path"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        rel = arguments.get("path", "")
        target = (self._workspace / rel).resolve()
        # simple sandbox: must stay under workspace
        if not str(target).startswith(str(self._workspace.resolve())):
            return ToolResult(output="", error="Path escapes workspace", is_error=True)
        if not target.exists():
            return ToolResult(output="", error=f"File not found: {rel}", is_error=True)
        try:
            text = target.read_text(errors="replace")
            return ToolResult(output=text)
        except Exception as exc:
            return ToolResult(output="", error=str(exc), is_error=True)


class WriteTool(Tool):
    """Write content to a file in the workspace."""

    def __init__(self, workspace: str) -> None:
        self._workspace = Path(workspace)

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "Write text to a file, creating directories as needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file (relative to workspace)"},
                "content": {"type": "string", "description": "File content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        rel = arguments.get("path", "")
        content = arguments.get("content", "")
        target = (self._workspace / rel).resolve()
        if not str(target).startswith(str(self._workspace.resolve())):
            return ToolResult(output="", error="Path escapes workspace", is_error=True)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return ToolResult(output=f"Wrote {len(content)} bytes to {rel}")
        except Exception as exc:
            return ToolResult(output="", error=str(exc), is_error=True)


class ExecTool(Tool):
    """Execute a shell command in the workspace."""

    def __init__(self, workspace: str, *, timeout: int = 30) -> None:
        self._workspace = Path(workspace)
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command in the workspace directory. Returns stdout + stderr."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
            },
            "required": ["command"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cmd = arguments.get("command", "")
        timeout = arguments.get("timeout", self._timeout)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace")
            if stderr:
                output += "\n" + stderr.decode(errors="replace")
            if proc.returncode != 0:
                return ToolResult(output=output, error=f"Exit code {proc.returncode}", is_error=True)
            return ToolResult(output=output)
        except asyncio.TimeoutError:
            return ToolResult(output="", error=f"Command timed out after {timeout}s", is_error=True)
        except Exception as exc:
            return ToolResult(output="", error=str(exc), is_error=True)


class SendMessageTool(Tool):
    """Send a message via a channel – injected at runtime with a delivery callback."""

    def __init__(self, deliver_fn: Any) -> None:
        self._deliver = deliver_fn  # async (channel, to, text) -> OutboundResult

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        return "Send a message to a user or group via a messaging channel."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel id (telegram, discord, etc.)"},
                "to": {"type": "string", "description": "Recipient identifier"},
                "text": {"type": "string", "description": "Message text to send"},
            },
            "required": ["channel", "to", "text"],
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        channel = arguments.get("channel", "")
        to = arguments.get("to", "")
        text = arguments.get("text", "")
        try:
            result = await self._deliver(channel, to, text)
            return ToolResult(output=f"Sent to {channel}:{to} (id={result.message_id})")
        except Exception as exc:
            return ToolResult(output="", error=str(exc), is_error=True)


def create_default_tools(workspace: str) -> list[Tool]:
    """Create the standard set of workspace tools (no send_message – that's injected later)."""
    return [
        ReadTool(workspace),
        WriteTool(workspace),
        ExecTool(workspace),
    ]
