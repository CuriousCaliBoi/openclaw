"""Configuration system – single JSON file with Pydantic validation.

Maps to ~/.openclaw/openclaw.json (or OPENCLAW_CONFIG env override).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ModelProviderConfig(BaseModel):
    """Credentials + endpoint for a single LLM provider."""

    api: str = "anthropic-messages"  # anthropic-messages | openai-completions | google-generative-ai
    base_url: str | None = None
    api_key: str | None = None  # supports ${ENV_VAR} substitution


class ModelsConfig(BaseModel):
    providers: dict[str, ModelProviderConfig] = Field(default_factory=dict)


class AgentModelConfig(BaseModel):
    primary: str = "anthropic/claude-sonnet-4-5-20250929"
    fallback: str | None = None
    thinking: str = "off"  # off | low | high


class IdentityConfig(BaseModel):
    name: str = "Clawd"
    emoji: str = ""


class AgentConfig(BaseModel):
    id: str = "main"
    default: bool = True
    workspace: str | None = None
    agent_dir: str | None = None
    model: AgentModelConfig = Field(default_factory=AgentModelConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)


class AgentsConfig(BaseModel):
    defaults: AgentModelConfig = Field(default_factory=AgentModelConfig)
    agents: list[AgentConfig] = Field(default_factory=lambda: [AgentConfig()])


class ChannelAccountConfig(BaseModel):
    """Generic per-account config – channels extend via extra fields."""

    token: str | None = None
    name: str | None = None
    enabled: bool = True
    allow_from: list[str] = Field(default_factory=list)


class ChannelConfig(BaseModel):
    accounts: dict[str, ChannelAccountConfig] = Field(default_factory=dict)
    allow_from: list[str] = Field(default_factory=list)


class ChannelsConfig(BaseModel):
    telegram: ChannelConfig = Field(default_factory=ChannelConfig)
    discord: ChannelConfig = Field(default_factory=ChannelConfig)
    whatsapp: ChannelConfig = Field(default_factory=ChannelConfig)
    signal: ChannelConfig = Field(default_factory=ChannelConfig)
    slack: ChannelConfig = Field(default_factory=ChannelConfig)


class BindingMatch(BaseModel):
    channel: str | None = None
    account_id: str | None = None
    peer_id: str | None = None
    guild_id: str | None = None
    team_id: str | None = None


class AgentBinding(BaseModel):
    agent_id: str
    match: BindingMatch = Field(default_factory=BindingMatch)


class SessionResetConfig(BaseModel):
    mode: str = "daily"  # daily | idle | manual
    at_hour: int = 4


class SessionConfig(BaseModel):
    dm_scope: str = "main"  # main | per-peer | per-channel-peer
    reset: SessionResetConfig = Field(default_factory=SessionResetConfig)


class GatewayConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18789
    mode: str = "local"  # local | remote


class ToolPolicyConfig(BaseModel):
    policy: str = "allowlist"  # allowlist | open | sandbox


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class OpenClawConfig(BaseModel):
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    bindings: list[AgentBinding] = Field(default_factory=list)
    session: SessionConfig = Field(default_factory=SessionConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _substitute_env(value: Any) -> Any:
    """Recursively substitute ${VAR} / $VAR in string values."""
    if isinstance(value, str):
        def _replacer(m: re.Match) -> str:
            var = m.group(1) or m.group(2)
            return os.environ.get(var, m.group(0))
        return _ENV_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def default_config_path() -> Path:
    return Path(os.environ.get("OPENCLAW_CONFIG", "~/.openclaw/openclaw.json")).expanduser()


def default_data_dir() -> Path:
    return Path(os.environ.get("OPENCLAW_DATA", "~/.openclaw")).expanduser()


def load_config(path: Path | None = None) -> OpenClawConfig:
    """Load and validate config from JSON file (with env-var substitution)."""
    path = path or default_config_path()
    if not path.exists():
        return OpenClawConfig()
    raw = json.loads(path.read_text())
    substituted = _substitute_env(raw)
    return OpenClawConfig.model_validate(substituted)


def resolve_agent(cfg: OpenClawConfig, agent_id: str | None = None) -> AgentConfig:
    """Resolve an agent by ID, falling back to the default agent."""
    for agent in cfg.agents.agents:
        if agent_id and agent.id == agent_id:
            return agent
        if not agent_id and agent.default:
            return agent
    return cfg.agents.agents[0] if cfg.agents.agents else AgentConfig()
