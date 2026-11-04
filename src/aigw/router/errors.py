"""Standardized, sanitized gateway errors for the model router."""

import uuid
from dataclasses import dataclass, field


@dataclass
class GatewayError(Exception):
    status: int
    code: str
    message: str
    headers: dict[str, str] = field(default_factory=dict)

    def body(self, request_id: str) -> dict:
        return {
            "error": {
                "type": "gateway_error",
                "code": self.code,
                "message": self.message,
                "request_id": request_id,
            }
        }


def new_request_id() -> str:
    return uuid.uuid4().hex


def unauthenticated() -> GatewayError:
    return GatewayError(
        401, "invalid_api_key", "A valid API key is required", {"WWW-Authenticate": "Bearer"}
    )


def rate_limited(retry_after_seconds: int | None, message: str) -> GatewayError:
    headers = {"Retry-After": str(retry_after_seconds)} if retry_after_seconds else {}
    return GatewayError(429, "rate_limit_exceeded", message, headers)


def limiter_unavailable() -> GatewayError:
    return GatewayError(503, "rate_limiter_unavailable", "Rate limiting is temporarily unavailable")


def providers_rate_limited() -> GatewayError:
    return GatewayError(
        429, "upstream_rate_limited", "All model providers are rate limited; retry later"
    )


def providers_timeout() -> GatewayError:
    return GatewayError(504, "upstream_timeout", "Model providers did not respond in time")


def providers_unavailable() -> GatewayError:
    return GatewayError(502, "upstream_unavailable", "Model providers are unavailable")


def internal_error() -> GatewayError:
    return GatewayError(500, "internal_error", "Internal gateway error")
