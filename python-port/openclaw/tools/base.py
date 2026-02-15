"""Tool system – JSON-schema-defined functions the LLM can invoke."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """The output of a tool execution."""

    output: str
    error: str | None = None
    is_error: bool = False


class Tool(abc.ABC):
    """Abstract base for a tool the agent can use."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Tool name as seen by the LLM."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Brief description shown in the LLM's system prompt."""

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters."""

    @abc.abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool and return its output."""

    def to_schema(self) -> dict[str, Any]:
        """Convert to the schema dict sent to the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
