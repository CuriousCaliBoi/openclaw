"""Gateway server – the central orchestrator.

Runs a WebSocket + HTTP server that:
  1. Manages channel adapters (start/stop listeners)
  2. Routes inbound messages to the correct agent + session
  3. Runs the agent loop and dispatches responses back through channels
  4. Broadcasts events to connected WebSocket clients (dashboards, WebChat)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from openclaw.agent import AgentRunner, StreamEvent, create_provider
from openclaw.channels.base import Channel, InboundHandler, MsgContext, OutboundResult
from openclaw.config import OpenClawConfig, default_data_dir, resolve_agent
from openclaw.router import ResolvedRoute, resolve_route
from openclaw.session import SessionStore
from openclaw.tools.base import Tool
from openclaw.tools.builtins import SendMessageTool, create_default_tools

log = logging.getLogger(__name__)


@dataclass
class GatewayState:
    """Mutable runtime state for the gateway."""

    channels: dict[str, Channel] = field(default_factory=dict)
    session_stores: dict[str, SessionStore] = field(default_factory=dict)  # agent_id → store
    ws_clients: list[WebSocket] = field(default_factory=list)
    # per-session lock to ensure one agent run at a time
    session_locks: dict[str, asyncio.Lock] = field(default_factory=dict)


class Gateway:
    """The OpenClaw gateway – ties everything together."""

    def __init__(self, config: OpenClawConfig) -> None:
        self.config = config
        self.data_dir = default_data_dir()
        self.state = GatewayState()
        self.app = self._build_app()

    # -- channel management --------------------------------------------------

    def register_channel(self, channel: Channel) -> None:
        """Register a channel adapter."""
        self.state.channels[channel.id] = channel

    async def start_channels(self) -> None:
        """Start all registered channels, providing them our inbound handler."""
        for ch in self.state.channels.values():
            try:
                await ch.start(self._on_inbound_message)
                log.info("gateway: started channel %s", ch.id)
            except Exception:
                log.exception("gateway: failed to start channel %s", ch.id)

    async def stop_channels(self) -> None:
        for ch in self.state.channels.values():
            try:
                await ch.stop()
            except Exception:
                log.exception("gateway: failed to stop channel %s", ch.id)

    # -- session store management --------------------------------------------

    def _get_session_store(self, agent_id: str) -> SessionStore:
        if agent_id not in self.state.session_stores:
            agent_dir = self.data_dir / "agents" / agent_id
            self.state.session_stores[agent_id] = SessionStore(agent_dir)
        return self.state.session_stores[agent_id]

    # -- inbound message handling (the main pipeline) ------------------------

    async def _on_inbound_message(self, ctx: MsgContext) -> None:
        """Central inbound handler – called by every channel adapter."""
        log.info(
            "gateway: inbound [%s] from=%s body=%s",
            ctx.channel, ctx.sender_id, ctx.body[:80],
        )

        # 1. route to agent + session
        route = resolve_route(self.config, ctx)
        log.info(
            "gateway: routed → agent=%s session=%s (matched_by=%s)",
            route.agent_id, route.session_key, route.matched_by,
        )

        # 2. ensure session exists
        store = self._get_session_store(route.agent_id)
        session_meta = store.get_or_create(route.session_key, channel=ctx.channel)

        # 3. acquire per-session lock (one agent run at a time per session)
        lock = self.state.session_locks.setdefault(route.session_key, asyncio.Lock())
        async with lock:
            # 4. run the agent
            result = await self._run_agent(route, session_meta.session_id, ctx)

        # 5. send response back through the originating channel
        if result.text:
            await self._dispatch_outbound(ctx, result.text)

        # 6. broadcast event to WebSocket clients
        await self._broadcast_event({
            "type": "agent",
            "agentId": route.agent_id,
            "sessionKey": route.session_key,
            "channel": ctx.channel,
            "text": result.text,
            "rounds": result.rounds,
            "inputTokens": result.input_tokens,
            "outputTokens": result.output_tokens,
        })

    async def _run_agent(
        self, route: ResolvedRoute, session_id: str, ctx: MsgContext,
    ) -> Any:
        """Create an AgentRunner and execute one turn."""
        agent_cfg = resolve_agent(self.config, route.agent_id)
        provider_id, _ = _parse_model_ref(agent_cfg.model.primary)
        provider = create_provider(self.config, provider_id)

        workspace = agent_cfg.workspace or str(self.data_dir / "workspace")
        Path(workspace).expanduser().mkdir(parents=True, exist_ok=True)

        store = self._get_session_store(route.agent_id)

        # build tools (including send_message wired to our dispatch)
        tools: list[Tool] = create_default_tools(workspace)
        tools.append(SendMessageTool(self._deliver_message))

        runner = AgentRunner(
            provider=provider,
            session_store=store,
            tools=tools,
            config=self.config,
            agent_config=agent_cfg,
        )

        async def _on_event(event: StreamEvent) -> None:
            # broadcast streaming events to WS clients
            await self._broadcast_event({
                "type": f"agent.{event.type}",
                "agentId": route.agent_id,
                "sessionKey": route.session_key,
                "text": event.text,
                "toolName": event.tool_name,
            })

        return await runner.run(ctx.body, session_id=session_id, on_event=_on_event)

    # -- outbound dispatch ---------------------------------------------------

    async def _dispatch_outbound(self, ctx: MsgContext, text: str) -> None:
        """Send the agent's response back through the originating channel."""
        channel = self.state.channels.get(ctx.channel)
        if not channel:
            log.warning("gateway: no channel adapter for %s, dropping response", ctx.channel)
            return
        try:
            await channel.send_text(
                ctx.recipient_id or ctx.sender_id,
                text,
                reply_to=ctx.message_id,
            )
        except Exception:
            log.exception("gateway: failed to send outbound on %s", ctx.channel)

    async def _deliver_message(self, channel_id: str, to: str, text: str) -> OutboundResult:
        """Callback used by SendMessageTool – sends via any registered channel."""
        channel = self.state.channels.get(channel_id)
        if not channel:
            raise ValueError(f"Unknown channel: {channel_id}")
        return await channel.send_text(to, text)

    # -- WebSocket broadcast -------------------------------------------------

    async def _broadcast_event(self, event: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        data = json.dumps(event)
        for ws in self.state.ws_clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.state.ws_clients.remove(ws)

    # -- FastAPI app ---------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw Gateway")

        @app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({
                "status": "ok",
                "channels": list(self.state.channels.keys()),
                "uptime": time.time(),
            })

        @app.get("/api/sessions")
        async def list_sessions() -> JSONResponse:
            all_sessions: list[dict[str, Any]] = []
            for agent_id, store in self.state.session_stores.items():
                for meta in store.list_sessions():
                    entry = meta.to_dict()
                    entry["agentId"] = agent_id
                    all_sessions.append(entry)
            return JSONResponse(all_sessions)

        @app.get("/api/channels")
        async def list_channels() -> JSONResponse:
            return JSONResponse([
                {
                    "id": ch.id,
                    "capabilities": {
                        "chatTypes": ch.capabilities.chat_types,
                        "polls": ch.capabilities.polls,
                        "reactions": ch.capabilities.reactions,
                        "threads": ch.capabilities.threads,
                        "media": ch.capabilities.media,
                    },
                }
                for ch in self.state.channels.values()
            ])

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            self.state.ws_clients.append(ws)
            log.info("gateway: ws client connected (%d total)", len(self.state.ws_clients))
            try:
                while True:
                    data = await ws.receive_text()
                    # handle client → gateway messages (e.g. WebChat sends)
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "message":
                            # WebChat sends a message as if it came from the "web" channel
                            ctx = MsgContext(
                                body=msg.get("text", ""),
                                body_for_agent=msg.get("text", ""),
                                sender_id=msg.get("senderId", "web-user"),
                                sender_name=msg.get("senderName", "Web User"),
                                recipient_id="web",
                                message_id=msg.get("messageId", ""),
                                chat_type="direct",
                                channel="web",
                                account_id="default",
                                timestamp=time.time(),
                            )
                            asyncio.create_task(self._on_inbound_message(ctx))
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                if ws in self.state.ws_clients:
                    self.state.ws_clients.remove(ws)
                log.info("gateway: ws client disconnected (%d total)", len(self.state.ws_clients))

        return app


def _parse_model_ref(ref: str) -> tuple[str, str]:
    if "/" in ref:
        return ref.split("/", 1)
    return "anthropic", ref
