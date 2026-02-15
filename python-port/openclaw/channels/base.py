"""Abstract channel interface – every messaging platform implements this.

Mirrors the TypeScript ChannelPlugin / ChannelDock contract.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class ChannelCapabilities:
    """Feature flags for a channel."""

    chat_types: list[str] = field(default_factory=lambda: ["direct"])
    polls: bool = False
    reactions: bool = False
    threads: bool = False
    media: bool = False
    native_commands: bool = False


@dataclass
class MsgContext:
    """Normalized inbound message – the common currency between channels.

    Every channel adapter converts its native message into this struct before
    handing it to the router.
    """

    # content
    body: str = ""
    body_for_agent: str = ""

    # routing & identity
    sender_id: str = ""
    sender_name: str = ""
    sender_username: str = ""
    recipient_id: str = ""

    # session & threading
    message_id: str = ""
    reply_to_id: str | None = None
    reply_to_body: str | None = None
    thread_id: str | None = None

    # group context
    chat_type: str = "direct"  # direct | group | channel | thread
    group_name: str | None = None

    # media
    media_path: str | None = None
    media_url: str | None = None
    media_type: str | None = None

    # channel metadata
    channel: str = ""
    account_id: str = "default"
    timestamp: float = 0.0
    was_mentioned: bool = False

    # raw platform payload (for channel-specific logic)
    raw: Any = None


@dataclass
class OutboundResult:
    """Result of delivering a message back through a channel."""

    channel: str
    message_id: str = ""
    chat_id: str = ""


# Type alias for the inbound callback the gateway registers
InboundHandler = Callable[[MsgContext], Awaitable[None]]


class Channel(abc.ABC):
    """Abstract base for a channel adapter (Telegram, Discord, etc.)."""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Unique channel identifier (e.g. 'telegram', 'discord')."""

    @property
    @abc.abstractmethod
    def capabilities(self) -> ChannelCapabilities:
        """Declare what this channel supports."""

    @property
    def text_chunk_limit(self) -> int:
        """Max characters per outbound message. Subclasses override."""
        return 4096

    # -- lifecycle -----------------------------------------------------------

    @abc.abstractmethod
    async def start(self, on_message: InboundHandler) -> None:
        """Begin listening for inbound messages and call *on_message* for each."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel listener."""

    # -- outbound ------------------------------------------------------------

    @abc.abstractmethod
    async def send_text(self, to: str, text: str, *, reply_to: str | None = None) -> OutboundResult:
        """Send a text message. Implementations should handle chunking."""

    async def send_media(
        self, to: str, url: str, *, caption: str = "", media_type: str = ""
    ) -> OutboundResult:
        """Send media (image/video/audio). Default falls back to text with URL."""
        return await self.send_text(to, f"{caption}\n{url}".strip())

    # -- helpers -------------------------------------------------------------

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks respecting the channel's limit."""
        limit = self.text_chunk_limit
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            # try to split at paragraph boundary
            cut = text.rfind("\n\n", 0, limit)
            if cut < limit // 4:
                cut = text.rfind("\n", 0, limit)
            if cut < limit // 4:
                cut = text.rfind(" ", 0, limit)
            if cut < limit // 4:
                cut = limit
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks
