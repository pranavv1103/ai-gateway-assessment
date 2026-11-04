"""Primary/secondary provider routing with a hard primary deadline."""

import asyncio
import logging
from dataclasses import dataclass

import httpx

from aigw.router import errors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    body: dict
    prompt_tokens: int
    completion_tokens: int


class ProviderRateLimited(Exception):
    pass


class ProviderFailed(Exception):
    pass


class ModelRouter:
    """Calls the primary once; on 429 or deadline expiry calls the secondary once."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        primary_url: str,
        secondary_url: str,
        primary_deadline_seconds: float = 3.0,
        secondary_deadline_seconds: float = 10.0,
        primary_api_key: str | None = None,
        secondary_api_key: str | None = None,
    ):
        self._client = client
        self._targets = {
            "primary": (primary_url, primary_deadline_seconds, primary_api_key),
            "secondary": (secondary_url, secondary_deadline_seconds, secondary_api_key),
        }

    async def _call(self, name: str, payload: dict) -> ProviderResult:
        url, deadline, api_key = self._targets[name]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # asyncio.timeout bounds the whole exchange and cancels the in-flight request on expiry.
        async with asyncio.timeout(deadline):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ProviderFailed(type(exc).__name__) from None
        if response.status_code == 429:
            raise ProviderRateLimited
        if response.status_code != 200:
            raise ProviderFailed(f"status {response.status_code}")
        try:
            body = response.json()
            usage = body["usage"]
            return ProviderResult(
                name, body, int(usage["prompt_tokens"]), int(usage["completion_tokens"])
            )
        except (ValueError, KeyError, TypeError):
            raise ProviderFailed("malformed response") from None

    async def complete(self, payload: dict) -> ProviderResult:
        """Return a provider result or raise errors.GatewayError (sanitized)."""
        try:
            return await self._call("primary", payload)
        except (ProviderRateLimited, TimeoutError, ProviderFailed) as exc:
            logger.warning("primary failed (%s); falling back", type(exc).__name__)
            primary_failure = exc
        # Only 429 and timeouts trigger fallback; other primary errors are not retried elsewhere.
        if isinstance(primary_failure, ProviderFailed):
            raise errors.providers_unavailable()
        try:
            return await self._call("secondary", payload)
        except ProviderRateLimited:
            raise errors.providers_rate_limited() from None
        except TimeoutError:
            raise errors.providers_timeout() from None
        except ProviderFailed:
            raise errors.providers_unavailable() from None
