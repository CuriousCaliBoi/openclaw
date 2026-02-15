"""Telegram channel adapter – uses python-telegram-bot (grammy equivalent)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openclaw.channels.base import (
    Channel,
    ChannelCapabilities,
    InboundHandler,
    MsgContext,
    OutboundResult,
)

log = logging.getLogger(__name__)


class TelegramChannel(Channel):
    """Telegram bot channel via python-telegram-bot library."""

    def __init__(self, token: str, *, account_id: str = "default", allow_from: list[str] | None = None) -> None:
        self._token = token
        self._account_id = account_id
        self._allow_from = set(allow_from) if allow_from else None
        self._app: Any = None  # telegram.ext.Application
        self._on_message: InboundHandler | None = None

    @property
    def id(self) -> str:
        return "telegram"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            chat_types=["direct", "group", "channel"],
            polls=True,
            reactions=True,
            threads=True,
            media=True,
            native_commands=True,
        )

    @property
    def text_chunk_limit(self) -> int:
        return 4000

    # -- lifecycle -----------------------------------------------------------

    async def start(self, on_message: InboundHandler) -> None:
        try:
            from telegram.ext import Application, MessageHandler, filters
        except ImportError as exc:
            raise ImportError("Install python-telegram-bot: pip install 'openclaw[all]'") from exc

        self._on_message = on_message
        self._app = Application.builder().token(self._token).build()

        async def _handle(update: Any, context: Any) -> None:
            msg = update.effective_message
            if msg is None or msg.text is None:
                return
            sender = update.effective_user
            chat = update.effective_chat

            sender_id = str(sender.id) if sender else ""

            # allowlist enforcement
            if self._allow_from and sender_id not in self._allow_from:
                log.debug("telegram: ignoring message from non-allowed sender %s", sender_id)
                return

            chat_type = "direct"
            if chat and chat.type in ("group", "supergroup"):
                chat_type = "group"
            elif chat and chat.type == "channel":
                chat_type = "channel"

            ctx = MsgContext(
                body=msg.text,
                body_for_agent=msg.text,
                sender_id=sender_id,
                sender_name=sender.full_name if sender else "",
                sender_username=sender.username or "" if sender else "",
                recipient_id=str(chat.id) if chat else "",
                message_id=str(msg.message_id),
                reply_to_id=str(msg.reply_to_message.message_id) if msg.reply_to_message else None,
                thread_id=str(msg.message_thread_id) if msg.message_thread_id else None,
                chat_type=chat_type,
                group_name=chat.title if chat and chat_type != "direct" else None,
                channel="telegram",
                account_id=self._account_id,
                timestamp=time.time(),
                was_mentioned=False,  # simplified; real impl checks bot username in text
                raw=update,
            )

            if self._on_message:
                await self._on_message(ctx)

        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle))

        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling()
        log.info("telegram: started polling (account=%s)", self._account_id)

    async def stop(self) -> None:
        if self._app:
            if self._app.updater:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            log.info("telegram: stopped")

    # -- outbound ------------------------------------------------------------

    async def send_text(self, to: str, text: str, *, reply_to: str | None = None) -> OutboundResult:
        if not self._app or not self._app.bot:
            raise RuntimeError("Telegram channel not started")

        chunks = self.chunk_text(text)
        last_msg_id = ""
        for chunk in chunks:
            kwargs: dict[str, Any] = {"chat_id": to, "text": chunk}
            if reply_to:
                kwargs["reply_to_message_id"] = int(reply_to)
                reply_to = None  # only reply to the first chunk
            sent = await self._app.bot.send_message(**kwargs)
            last_msg_id = str(sent.message_id)

        return OutboundResult(channel="telegram", message_id=last_msg_id, chat_id=to)

    async def send_media(
        self, to: str, url: str, *, caption: str = "", media_type: str = ""
    ) -> OutboundResult:
        if not self._app or not self._app.bot:
            raise RuntimeError("Telegram channel not started")

        if media_type.startswith("image"):
            sent = await self._app.bot.send_photo(chat_id=to, photo=url, caption=caption or None)
        elif media_type.startswith("video"):
            sent = await self._app.bot.send_video(chat_id=to, video=url, caption=caption or None)
        elif media_type.startswith("audio"):
            sent = await self._app.bot.send_audio(chat_id=to, audio=url, caption=caption or None)
        else:
            sent = await self._app.bot.send_document(chat_id=to, document=url, caption=caption or None)

        return OutboundResult(channel="telegram", message_id=str(sent.message_id), chat_id=to)
