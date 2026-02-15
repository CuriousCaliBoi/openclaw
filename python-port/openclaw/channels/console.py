"""Console (stdin/stdout) channel – useful for local testing without any API keys."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid

from openclaw.channels.base import (
    Channel,
    ChannelCapabilities,
    InboundHandler,
    MsgContext,
    OutboundResult,
)

log = logging.getLogger(__name__)


class ConsoleChannel(Channel):
    """Interactive console channel – reads from stdin, prints to stdout."""

    def __init__(self, *, user_id: str = "console-user") -> None:
        self._user_id = user_id
        self._on_message: InboundHandler | None = None
        self._running = False
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    @property
    def id(self) -> str:
        return "console"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(chat_types=["direct"])

    @property
    def text_chunk_limit(self) -> int:
        return 100_000  # no real limit for terminal

    async def start(self, on_message: InboundHandler) -> None:
        self._on_message = on_message
        self._running = True
        self._task = asyncio.create_task(self._read_loop())
        log.info("console: started (type messages, Ctrl-C to quit)")

    async def _read_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            ctx = MsgContext(
                body=line,
                body_for_agent=line,
                sender_id=self._user_id,
                sender_name="User",
                recipient_id="console",
                message_id=str(uuid.uuid4())[:8],
                chat_type="direct",
                channel="console",
                timestamp=time.time(),
            )
            if self._on_message:
                await self._on_message(ctx)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("console: stopped")

    async def send_text(self, to: str, text: str, *, reply_to: str | None = None) -> OutboundResult:
        print(f"\n{text}\n")
        return OutboundResult(channel="console", message_id="", chat_id=to)
