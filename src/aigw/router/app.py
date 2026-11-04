"""FastAPI endpoint: tenant auth -> token reservation -> primary/secondary routing."""

import hmac
import logging
import math

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aigw.config import RouterSettings
from aigw.router import errors
from aigw.router.fallback import ModelRouter
from aigw.router.limiter import LimiterUnavailable, RateLimitExceeded, SQLiteRateLimiter

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1024 * 1024


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt: str = Field(min_length=1, max_length=200_000)
    max_tokens: int = Field(default=256, ge=1, le=100_000)
    mock: dict | None = None  # passed through so local mock providers can simulate faults


def estimate_tokens(prompt: str, max_tokens: int) -> int:
    """Input estimate (~4 chars/token) plus the worst-case output budget."""
    return math.ceil(len(prompt) / 4) + max_tokens


def _error_response(error: errors.GatewayError, request_id: str) -> JSONResponse:
    headers = {**error.headers, "X-Request-ID": request_id}
    return JSONResponse(error.body(request_id), status_code=error.status, headers=headers)


def _extract_api_key(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    value = value.strip()
    if scheme.lower() != "bearer" or not value or len(value) > 256:
        return None
    return value


def create_app(
    settings: RouterSettings,
    client: httpx.AsyncClient | None = None,
    limiter: SQLiteRateLimiter | None = None,
    valid_keys: set[str] | None = None,
) -> FastAPI:
    """``valid_keys`` restricts tenants; None accepts any non-empty key (demo mode)."""
    app = FastAPI(title="LLM routing gateway")
    http_client = client or httpx.AsyncClient(timeout=settings.secondary_deadline_seconds + 1)
    app.state.limiter = limiter or SQLiteRateLimiter(
        settings.db_path, settings.tokens_per_window, settings.window_seconds
    )
    router = ModelRouter(
        http_client,
        settings.primary_url,
        settings.secondary_url,
        settings.primary_deadline_seconds,
        settings.secondary_deadline_seconds,
    )

    def key_is_valid(key: str) -> bool:
        if valid_keys is None:
            return True
        return any(hmac.compare_digest(key.encode(), v.encode()) for v in valid_keys)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request):
        request_id = errors.new_request_id()
        try:
            api_key = _extract_api_key(request)
            if api_key is None or not key_is_valid(api_key):
                return _error_response(errors.unauthenticated(), request_id)

            raw = await request.body()
            if len(raw) > MAX_BODY_BYTES:
                return _error_response(
                    errors.GatewayError(413, "request_too_large", "Request body too large"),
                    request_id,
                )
            try:
                parsed = CompletionRequest.model_validate_json(raw)
            except ValidationError:
                return _error_response(
                    errors.GatewayError(400, "invalid_request", "Invalid request body"),
                    request_id,
                )

            estimate = estimate_tokens(parsed.prompt, parsed.max_tokens)
            try:
                reservation = await app.state.limiter.reserve(api_key, estimate)
            except RateLimitExceeded as exc:
                message = (
                    "Request exceeds the per-window token limit"
                    if exc.retry_after_seconds is None
                    else "Token rate limit exceeded"
                )
                return _error_response(
                    errors.rate_limited(exc.retry_after_seconds, message), request_id
                )
            except LimiterUnavailable:
                return _error_response(errors.limiter_unavailable(), request_id)

            payload = {"prompt": parsed.prompt, "max_tokens": parsed.max_tokens}
            if parsed.mock is not None:
                payload["mock"] = parsed.mock
            try:
                result = await router.complete(payload)
            except BaseException:
                # Failure or client cancellation: nothing was consumed, give the tokens back.
                await _safe_release(app.state.limiter, reservation)
                raise
            await _safe_settle(
                app.state.limiter, reservation, result.prompt_tokens + result.completion_tokens
            )
            return JSONResponse(
                {
                    "text": result.body.get("text", ""),
                    "provider": result.provider,
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                    },
                },
                headers={"X-Request-ID": request_id},
            )
        except errors.GatewayError as exc:
            return _error_response(exc, request_id)
        except Exception:
            logger.exception("unhandled error in request %s", request_id)
            return _error_response(errors.internal_error(), request_id)

    return app


async def _safe_release(limiter: SQLiteRateLimiter, reservation) -> None:
    import anyio

    with anyio.CancelScope(shield=True):
        try:
            await limiter.release(reservation)
        except LimiterUnavailable:
            logger.warning("could not release reservation; it will expire with the window")


async def _safe_settle(limiter: SQLiteRateLimiter, reservation, actual: int) -> None:
    try:
        await limiter.settle(reservation, actual)
    except LimiterUnavailable:
        logger.warning("could not settle reservation; estimate stays counted")


def build_app_from_env() -> FastAPI:
    return create_app(RouterSettings.from_env())
