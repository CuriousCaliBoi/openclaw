"""Session management – JSONL-backed conversation transcripts.

Each session is an append-only JSONL file at:
    ~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl

A session index (sessions.json) tracks active sessions + metadata.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    """A single conversation turn (user message + assistant reply + tool calls)."""

    id: str
    timestamp: float
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "turn",
            "id": self.id,
            "timestamp": self.timestamp,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            messages=data["messages"],
        )


@dataclass
class SessionStats:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.input_tokens + self.output_tokens,
            "costUsd": self.cost_usd,
        }


@dataclass
class SessionMeta:
    session_id: str
    session_key: str
    channel: str = ""
    display_name: str = ""
    updated_at: float = 0.0
    stats: SessionStats = field(default_factory=SessionStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "channel": self.channel,
            "displayName": self.display_name,
            "updatedAt": self.updated_at,
            "stats": self.stats.to_dict(),
        }


class SessionStore:
    """Manages session transcripts and the session index for a single agent."""

    def __init__(self, agent_dir: Path) -> None:
        self.agent_dir = agent_dir
        self.sessions_dir = agent_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.sessions_dir / "sessions.json"
        self._index: dict[str, SessionMeta] = {}
        self._load_index()

    # -- index ---------------------------------------------------------------

    def _load_index(self) -> None:
        if self._index_path.exists():
            raw = json.loads(self._index_path.read_text())
            for key, entry in raw.items():
                self._index[key] = SessionMeta(
                    session_id=entry["sessionId"],
                    session_key=key,
                    channel=entry.get("channel", ""),
                    display_name=entry.get("displayName", ""),
                    updated_at=entry.get("updatedAt", 0),
                    stats=SessionStats(
                        turns=entry.get("stats", {}).get("turns", 0),
                        input_tokens=entry.get("stats", {}).get("inputTokens", 0),
                        output_tokens=entry.get("stats", {}).get("outputTokens", 0),
                        cost_usd=entry.get("stats", {}).get("costUsd", 0.0),
                    ),
                )

    def _save_index(self) -> None:
        data = {key: meta.to_dict() for key, meta in self._index.items()}
        self._index_path.write_text(json.dumps(data, indent=2))

    # -- sessions ------------------------------------------------------------

    def get_or_create(self, session_key: str, *, channel: str = "") -> SessionMeta:
        """Return existing session metadata or create a new session."""
        if session_key in self._index:
            return self._index[session_key]
        meta = SessionMeta(
            session_id=str(uuid.uuid4()),
            session_key=session_key,
            channel=channel,
            updated_at=time.time(),
        )
        self._index[session_key] = meta
        self._save_index()
        return meta

    def list_sessions(self) -> list[SessionMeta]:
        return list(self._index.values())

    # -- transcript ----------------------------------------------------------

    def _transcript_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.jsonl"

    def append_turn(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session's JSONL transcript."""
        path = self._transcript_path(session_id)
        with path.open("a") as f:
            f.write(json.dumps(turn.to_dict()) + "\n")
        # update index stats
        for meta in self._index.values():
            if meta.session_id == session_id:
                meta.stats.turns += 1
                meta.updated_at = time.time()
                break
        self._save_index()

    def load_turns(self, session_id: str, *, limit: int = 50) -> list[Turn]:
        """Load the most recent N turns from a session transcript."""
        path = self._transcript_path(session_id)
        if not path.exists():
            return []
        turns: list[Turn] = []
        for line in path.read_text().strip().splitlines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("type") == "turn":
                turns.append(Turn.from_dict(data))
        return turns[-limit:]

    def build_context_messages(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """Build a flat list of {role, content} messages from recent turns."""
        turns = self.load_turns(session_id, limit=limit)
        messages: list[dict[str, Any]] = []
        for turn in turns:
            messages.extend(turn.messages)
        return messages

    def update_stats(
        self,
        session_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        for meta in self._index.values():
            if meta.session_id == session_id:
                meta.stats.input_tokens += input_tokens
                meta.stats.output_tokens += output_tokens
                meta.stats.cost_usd += cost_usd
                meta.updated_at = time.time()
                break
        self._save_index()
