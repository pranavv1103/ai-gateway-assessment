"""Centralized, environment-driven configuration for all components."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

VALID_ROLES = ("admin", "viewer")


def parse_token_roles(raw: str) -> dict[str, str]:
    """Parse ``{"<token>": "<role>"}`` JSON and reject unknown roles."""
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("token map must be valid JSON") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("token map must be a non-empty JSON object")
    for token, role in mapping.items():
        if not isinstance(token, str) or not token:
            raise ValueError("tokens must be non-empty strings")
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role for a configured token; expected one of {VALID_ROLES}")
    return mapping


@dataclass(frozen=True)
class GatewaySettings:
    downstream_url: str
    token_roles: dict[str, str] = field(repr=False)
    downstream_timeout_seconds: float = 10.0
    max_body_bytes: int = 1_048_576
    max_batch_size: int = 50

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        raw_tokens = os.environ.get("AIGW_GATEWAY_TOKENS")
        if not raw_tokens:
            raise RuntimeError(
                "AIGW_GATEWAY_TOKENS is required: a JSON object mapping each token to "
                '"admin" or "viewer"'
            )
        return cls(
            downstream_url=os.environ.get("AIGW_DOWNSTREAM_URL", "http://127.0.0.1:9001/mcp"),
            token_roles=parse_token_roles(raw_tokens),
            downstream_timeout_seconds=float(os.environ.get("AIGW_DOWNSTREAM_TIMEOUT", "10")),
        )


@dataclass(frozen=True)
class LLMGatewaySettings:
    provider_url: str = "http://127.0.0.1:9002/v1/chat/completions"
    provider_timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 3.0
    max_pending_chars: int = 256

    @classmethod
    def from_env(cls) -> "LLMGatewaySettings":
        return cls(
            provider_url=os.environ.get("AIGW_LLM_PROVIDER_URL", cls.provider_url),
            provider_timeout_seconds=float(os.environ.get("AIGW_LLM_TIMEOUT", "30")),
        )


@dataclass(frozen=True)
class RouterSettings:
    primary_url: str = "http://127.0.0.1:9003/v1/completions"
    secondary_url: str = "http://127.0.0.1:9004/v1/completions"
    db_path: Path = Path("rate_limits.db")
    tokens_per_window: int = 50_000
    window_seconds: float = 60.0
    primary_deadline_seconds: float = 3.0
    secondary_deadline_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "RouterSettings":
        defaults = cls()
        return cls(
            primary_url=os.environ.get("AIGW_PRIMARY_URL", defaults.primary_url),
            secondary_url=os.environ.get("AIGW_SECONDARY_URL", defaults.secondary_url),
            db_path=Path(os.environ.get("AIGW_RATE_LIMIT_DB", str(defaults.db_path))),
        )
