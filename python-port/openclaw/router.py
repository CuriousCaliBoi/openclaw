"""Message routing – resolves inbound messages to agents and session keys.

Implements the binding-precedence algorithm:
  exact-peer > guild > team > account > channel > default
"""

from __future__ import annotations

from dataclasses import dataclass

from openclaw.channels.base import MsgContext
from openclaw.config import AgentBinding, OpenClawConfig


@dataclass
class ResolvedRoute:
    """The result of routing an inbound message."""

    agent_id: str
    channel: str
    account_id: str
    session_key: str
    main_session_key: str
    matched_by: str  # binding.peer | binding.guild | binding.team | binding.account | binding.channel | default


def resolve_route(cfg: OpenClawConfig, ctx: MsgContext) -> ResolvedRoute:
    """Route an inbound message to an agent and session key.

    Binding precedence (highest → lowest):
        1. Exact peer match
        2. Guild match  (e.g. Discord server)
        3. Team match   (e.g. Slack workspace)
        4. Account match
        5. Channel match
        6. Default agent
    """
    # extract routing fields from message context
    channel = ctx.channel
    account_id = ctx.account_id or "default"
    peer_id = ctx.sender_id
    # guild_id / team_id would come from ctx.raw in real impl
    guild_id: str | None = None
    team_id: str | None = None
    if ctx.raw and isinstance(ctx.raw, dict):
        guild_id = ctx.raw.get("guild_id")
        team_id = ctx.raw.get("team_id")

    # walk bindings in precedence order
    best_agent_id: str | None = None
    best_match_type = "default"

    for binding in cfg.bindings:
        m = binding.match
        # channel must match (if specified)
        if m.channel and m.channel != channel:
            continue

        # peer binding (highest precedence)
        if m.peer_id and m.peer_id == peer_id:
            best_agent_id = binding.agent_id
            best_match_type = "binding.peer"
            break

        # guild binding
        if m.guild_id and guild_id and m.guild_id == guild_id:
            if best_match_type not in ("binding.peer",):
                best_agent_id = binding.agent_id
                best_match_type = "binding.guild"

        # team binding
        if m.team_id and team_id and m.team_id == team_id:
            if best_match_type not in ("binding.peer", "binding.guild"):
                best_agent_id = binding.agent_id
                best_match_type = "binding.team"

        # account binding
        if m.account_id and m.account_id == account_id and not m.peer_id and not m.guild_id and not m.team_id:
            if best_match_type == "default":
                best_agent_id = binding.agent_id
                best_match_type = "binding.account"

        # channel-only binding
        if m.channel and not m.peer_id and not m.guild_id and not m.team_id and not m.account_id:
            if best_match_type == "default":
                best_agent_id = binding.agent_id
                best_match_type = "binding.channel"

    # fall back to default agent
    if not best_agent_id:
        for agent in cfg.agents.agents:
            if agent.default:
                best_agent_id = agent.id
                break
        if not best_agent_id:
            best_agent_id = cfg.agents.agents[0].id if cfg.agents.agents else "main"

    # compute session key based on dm_scope
    session_key = _compute_session_key(
        agent_id=best_agent_id,
        channel=channel,
        account_id=account_id,
        peer_id=peer_id,
        chat_type=ctx.chat_type,
        dm_scope=cfg.session.dm_scope,
    )

    main_session_key = f"agent:{best_agent_id}:main"

    return ResolvedRoute(
        agent_id=best_agent_id,
        channel=channel,
        account_id=account_id,
        session_key=session_key,
        main_session_key=main_session_key,
        matched_by=best_match_type,
    )


def _compute_session_key(
    *,
    agent_id: str,
    channel: str,
    account_id: str,
    peer_id: str,
    chat_type: str,
    dm_scope: str,
) -> str:
    """Build a deterministic session key from routing fields.

    dm_scope modes:
        main             – all DMs share one session per agent
        per-peer         – each peer gets their own session
        per-channel-peer – each (channel, peer) combo is unique
    """
    base = f"agent:{agent_id}"

    if chat_type != "direct":
        # groups always get per-channel-peer isolation
        return f"{base}:{channel}:{account_id}:{peer_id}"

    if dm_scope == "per-peer":
        return f"{base}:peer:{peer_id}"
    if dm_scope == "per-channel-peer":
        return f"{base}:{channel}:{peer_id}"
    # default: "main"
    return f"{base}:main"
