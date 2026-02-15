"""CLI entry point – mirrors `openclaw gateway run`, `openclaw agent --message`, etc."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from openclaw.config import load_config, default_data_dir, resolve_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openclaw")


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="Config file path")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """OpenClaw – personal AI assistant gateway."""
    ctx.ensure_object(dict)
    path = Path(config_path) if config_path else None
    ctx.obj["config"] = load_config(path)


# ---------------------------------------------------------------------------
# openclaw gateway run
# ---------------------------------------------------------------------------

@cli.group()
def gateway() -> None:
    """Gateway server commands."""


@gateway.command("run")
@click.option("--host", default=None, help="Bind address")
@click.option("--port", default=None, type=int, help="Port")
@click.pass_context
def gateway_run(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Start the OpenClaw gateway server."""
    cfg = ctx.obj["config"]

    from openclaw.gateway import Gateway

    gw = Gateway(cfg)
    bind_host = host or cfg.gateway.host
    bind_port = port or cfg.gateway.port

    # auto-register channels from config
    _auto_register_channels(gw, cfg)

    async def _run() -> None:
        import uvicorn

        await gw.start_channels()
        config = uvicorn.Config(
            gw.app,
            host=bind_host,
            port=bind_port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            await gw.stop_channels()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# openclaw agent --message "..."
# ---------------------------------------------------------------------------

@cli.command("agent")
@click.option("--message", "-m", required=True, help="Message to send to the agent")
@click.option("--agent-id", default=None, help="Agent ID (default: default agent)")
@click.pass_context
def agent_cmd(ctx: click.Context, message: str, agent_id: str | None) -> None:
    """Send a one-shot message to the agent (CLI mode)."""
    cfg = ctx.obj["config"]
    asyncio.run(_run_agent_cli(cfg, message, agent_id))


async def _run_agent_cli(cfg: object, message: str, agent_id: str | None) -> None:
    from openclaw.agent import AgentRunner, StreamEvent, create_provider
    from openclaw.config import OpenClawConfig
    from openclaw.session import SessionStore
    from openclaw.tools.builtins import create_default_tools

    assert isinstance(cfg, OpenClawConfig)
    agent_cfg = resolve_agent(cfg, agent_id)
    provider_id = agent_cfg.model.primary.split("/")[0] if "/" in agent_cfg.model.primary else "anthropic"
    provider = create_provider(cfg, provider_id)

    data_dir = default_data_dir()
    store = SessionStore(data_dir / "agents" / agent_cfg.id)
    session_meta = store.get_or_create("cli:main", channel="console")

    workspace = agent_cfg.workspace or str(data_dir / "workspace")
    Path(workspace).expanduser().mkdir(parents=True, exist_ok=True)

    runner = AgentRunner(
        provider=provider,
        session_store=store,
        tools=create_default_tools(workspace),
        config=cfg,
        agent_config=agent_cfg,
    )

    async def _on_event(event: StreamEvent) -> None:
        if event.type == "text_delta" and event.text:
            print(event.text, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[tool] {event.tool_name}({event.tool_args})", flush=True)
        elif event.type == "tool_result":
            preview = event.tool_output[:200]
            print(f"[result] {preview}", flush=True)

    result = await runner.run(message, session_id=session_meta.session_id, on_event=_on_event)
    print()  # newline after streaming


# ---------------------------------------------------------------------------
# openclaw console
# ---------------------------------------------------------------------------

@cli.command("console")
@click.option("--agent-id", default=None, help="Agent ID")
@click.pass_context
def console_cmd(ctx: click.Context, agent_id: str | None) -> None:
    """Interactive console chat (no external channels needed)."""
    cfg = ctx.obj["config"]
    asyncio.run(_run_console(cfg, agent_id))


async def _run_console(cfg: object, agent_id: str | None) -> None:
    from openclaw.config import OpenClawConfig
    from openclaw.gateway import Gateway
    from openclaw.channels.console import ConsoleChannel

    assert isinstance(cfg, OpenClawConfig)
    gw = Gateway(cfg)
    gw.register_channel(ConsoleChannel())
    await gw.start_channels()

    print("OpenClaw console – type your messages (Ctrl-C to quit)")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await gw.stop_channels()


# ---------------------------------------------------------------------------
# openclaw status
# ---------------------------------------------------------------------------

@cli.command("status")
@click.pass_context
def status_cmd(ctx: click.Context) -> None:
    """Show gateway and session status."""
    cfg = ctx.obj["config"]
    data_dir = default_data_dir()
    agents_dir = data_dir / "agents"

    print(f"Data dir: {data_dir}")
    print(f"Gateway:  {cfg.gateway.host}:{cfg.gateway.port} (mode={cfg.gateway.mode})")
    print(f"Agents:   {len(cfg.agents.agents)}")
    print()

    if agents_dir.exists():
        for agent_dir in sorted(agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            from openclaw.session import SessionStore
            store = SessionStore(agent_dir)
            sessions = store.list_sessions()
            print(f"  Agent: {agent_dir.name} ({len(sessions)} sessions)")
            for s in sessions[:5]:
                print(f"    {s.session_key}: {s.stats.turns} turns, ${s.stats.cost_usd:.4f}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _auto_register_channels(gw: "Gateway", cfg: object) -> None:  # type: ignore[name-defined]
    """Auto-register channels based on config (Telegram, etc.)."""
    from openclaw.config import OpenClawConfig

    assert isinstance(cfg, OpenClawConfig)

    # Telegram
    for account_id, account in cfg.channels.telegram.accounts.items():
        if account.enabled and account.token:
            from openclaw.channels.telegram import TelegramChannel
            gw.register_channel(TelegramChannel(
                account.token,
                account_id=account_id,
                allow_from=account.allow_from or cfg.channels.telegram.allow_from or None,
            ))

    # Always register console as a fallback channel
    from openclaw.channels.console import ConsoleChannel
    gw.register_channel(ConsoleChannel())


if __name__ == "__main__":
    cli()
