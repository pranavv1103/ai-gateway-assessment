import asyncio

import httpx
import pytest

from aigw.config import RouterSettings
from aigw.router import errors
from aigw.router.app import create_app
from aigw.router.fallback import ModelRouter
from aigw.router.mock_provider import create_mock_provider

PRIMARY = "http://primary.internal.example/v1/completions"
SECONDARY = "http://secondary.internal.example/v1/completions"


class Providers:
    def __init__(self):
        self.primary = create_mock_provider("primary")
        self.secondary = create_mock_provider("secondary")
        self.calls: list[str] = []

    def transport(self) -> httpx.MockTransport:
        primary_t = httpx.ASGITransport(app=self.primary)
        secondary_t = httpx.ASGITransport(app=self.secondary)

        class Dispatch(httpx.AsyncBaseTransport):
            async def handle_async_request(inner, request):  # noqa: N805
                host = request.url.host
                request.url = request.url.copy_with(scheme="http", host="testserver")
                return await (
                    primary_t if host.startswith("primary") else secondary_t
                ).handle_async_request(request)

        return Dispatch()


@pytest.fixture
def providers():
    return Providers()


@pytest.fixture
def settings(tmp_path):
    return RouterSettings(
        primary_url=PRIMARY,
        secondary_url=SECONDARY,
        db_path=tmp_path / "limits.db",
        primary_deadline_seconds=0.2,
        secondary_deadline_seconds=1.0,
    )


@pytest.fixture
async def client(settings, providers):
    upstream = httpx.AsyncClient(transport=providers.transport())
    app = create_app(settings, client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        c.app = app
        yield c
    await upstream.aclose()


def both(options):
    return {"primary": options, "secondary": options}


def auth(key="tenant-key"):
    return {"Authorization": f"Bearer {key}"}


async def post(client, body, key="tenant-key"):
    return await client.post("/v1/completions", json=body, headers=auth(key))


async def test_primary_success(client, providers):
    response = await post(client, {"prompt": "hello world", "max_tokens": 10})
    assert response.status_code == 200
    assert response.json()["provider"] == "primary"
    assert providers.primary.state.request_count == 1
    assert providers.secondary.state.request_count == 0


async def test_primary_429_falls_back_once(client, providers):
    response = await post(client, {"prompt": "hi", "mock": {"primary": {"status": 429}}})
    assert response.status_code == 200
    assert response.json()["provider"] == "secondary"
    assert providers.primary.state.request_count == 1
    assert providers.secondary.state.request_count == 1


async def test_primary_timeout_falls_back(client, providers):
    response = await post(client, {"prompt": "hi", "mock": {"primary": {"delay_ms": 1000}}})
    assert response.status_code == 200
    assert response.json()["provider"] == "secondary"
    assert providers.primary.state.request_count == 1
    assert providers.secondary.state.request_count == 1


async def test_primary_5xx_does_not_fall_back(client, providers):
    response = await post(client, {"prompt": "hi", "mock": {"primary": {"status": 500}}})
    assert response.status_code == 502
    assert providers.secondary.state.request_count == 0


async def test_both_rate_limited_is_429_sanitized(client):
    response = await post(client, {"prompt": "hi", "mock": both({"status": 429})})
    assert response.status_code == 429
    body = response.json()["error"]
    assert body["code"] == "upstream_rate_limited"
    assert "request_id" in body


async def test_both_time_out_is_504(client):
    response = await post(client, {"prompt": "hi", "mock": both({"delay_ms": 5000})})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"


async def test_errors_never_leak_internals(client):
    for mock in (both({"status": 500}), both({"status": 429}), both({"delay_ms": 5000})):
        text = (await post(client, {"prompt": "hi", "mock": mock})).text
        assert "internal.example" not in text
        assert "Traceback" not in text
        assert "tenant-key" not in text


async def test_unreachable_provider_is_sanitized(settings):
    def refuse(request):
        raise httpx.ConnectError("secret-host:1234 refused")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    app = create_app(settings, client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        response = await post(c, {"prompt": "hi"})
    assert response.status_code == 502
    assert "secret-host" not in response.text


async def test_malformed_provider_response_is_502(settings):
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"nope": 1}))
    )
    app = create_app(settings, client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        response = await post(c, {"prompt": "hi"})
    assert response.status_code == 502


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": 5},
        {"prompt": "x", "max_tokens": 0},
        {"prompt": "x", "max_tokens": "10"},
        {"prompt": "x", "max_tokens": True},
        {"prompt": "x", "max_tokens": -5},
    ],
)
async def test_invalid_bodies_rejected_before_reservation(client, body):
    response = await post(client, body)
    assert response.status_code == 400
    assert await client.app.state.limiter.usage("tenant-key") == 0


async def test_not_json_rejected(client):
    response = await client.post("/v1/completions", content=b"{oops", headers=auth())
    assert response.status_code == 400


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer ", "Basic abc", "abc"])
async def test_missing_or_bad_auth(client, providers, header):
    headers = {"Authorization": header} if header is not None else {}
    response = await client.post("/v1/completions", json={"prompt": "hi"}, headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert providers.primary.state.request_count == 0


async def test_unknown_key_rejected_when_allowlist_set(settings, providers):
    upstream = httpx.AsyncClient(transport=providers.transport())
    app = create_app(settings, client=upstream, valid_keys={"good"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as c:
        assert (await post(c, {"prompt": "hi"}, key="bad")).status_code == 401
        assert (await post(c, {"prompt": "hi"}, key="good")).status_code == 200


async def test_rate_limit_returns_429_with_retry_after_and_skips_upstream(client, providers):
    big = {"prompt": "a" * 400, "max_tokens": 49_000}  # estimate 49,100
    assert (await post(client, big)).status_code == 200
    # Actual usage settled to the small real figure, so exhaust via a fresh tenant instead.
    limiter = client.app.state.limiter
    await limiter.reserve("tenant-2", 50_000)
    before = providers.primary.state.request_count
    response = await post(client, {"prompt": "hi", "max_tokens": 10}, key="tenant-2")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(response.headers["Retry-After"]) >= 1
    assert providers.primary.state.request_count == before


async def test_oversized_request_never_fits_window(client):
    response = await post(client, {"prompt": "hi", "max_tokens": 60_000})
    assert response.status_code == 429
    assert "Retry-After" not in response.headers


async def test_tokens_settled_to_actual_usage(client):
    await post(client, {"prompt": "hello world", "max_tokens": 4_000})
    used = await client.app.state.limiter.usage("tenant-key")
    assert 0 < used < 100


async def test_failed_request_releases_tokens(client):
    await post(client, {"prompt": "hi", "max_tokens": 4_000, "mock": both({"status": 500})})
    assert await client.app.state.limiter.usage("tenant-key") == 0


async def test_concurrent_requests_respect_quota(client):
    limiter = client.app.state.limiter
    await limiter.reserve("tenant-key", 30_000)
    body = {
        "prompt": "x" * 40,
        "max_tokens": 5_000,  # estimate 5,010
        "mock": {"primary": {"delay_ms": 100}},  # keeps all requests in flight together
    }
    responses = await asyncio.gather(*(post(client, body) for _ in range(10)))
    ok = [r for r in responses if r.status_code == 200]
    limited = [r for r in responses if r.status_code == 429]
    assert len(ok) + len(limited) == 10
    assert len(ok) == 3  # 20,000 headroom // 5,010 estimate
    assert await limiter.usage("tenant-key") <= 50_000


async def test_limiter_unavailable_is_503(client, monkeypatch):
    from aigw.router.limiter import LimiterUnavailable

    async def broken(*args, **kwargs):
        raise LimiterUnavailable("db locked at /secret/path.db")

    monkeypatch.setattr(client.app.state.limiter, "reserve", broken)
    response = await post(client, {"prompt": "hi"})
    assert response.status_code == 503
    assert "/secret" not in response.text


async def test_unexpected_exception_is_sanitized_500(client, monkeypatch):
    async def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(client.app.state.limiter, "reserve", explode)
    response = await post(client, {"prompt": "hi"})
    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["error"]["code"] == "internal_error"


async def test_cancelled_primary_is_cleaned_up_before_fallback():
    cancelled = asyncio.Event()

    class SlowThenOk(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if "primary" in request.url.host:
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return httpx.Response(
                200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "text": "ok"}
            )

    async with httpx.AsyncClient(transport=SlowThenOk()) as upstream:
        router = ModelRouter(upstream, PRIMARY, SECONDARY, 0.1, 1.0)
        result = await router.complete({"prompt": "hi"})
    assert result.provider == "secondary"
    assert cancelled.is_set()


async def test_primary_deadline_is_enforced_quickly():
    class Hang(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await asyncio.sleep(30)

    async with httpx.AsyncClient(transport=Hang()) as upstream:
        router = ModelRouter(upstream, PRIMARY, SECONDARY, 0.1, 0.1)
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(errors.GatewayError) as info:
            await router.complete({"prompt": "hi"})
    assert info.value.status == 504
    assert loop.time() - start < 1.0


async def test_client_cancellation_releases_reservation(settings):
    started = asyncio.Event()

    class Hang(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            started.set()
            await asyncio.sleep(30)

    upstream = httpx.AsyncClient(transport=Hang())
    s = RouterSettings(**{**settings.__dict__, "primary_deadline_seconds": 30})
    app = create_app(s, client=upstream)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        task = asyncio.create_task(post(c, {"prompt": "hi", "max_tokens": 1000}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
        assert await app.state.limiter.usage("tenant-key") == 0
    await upstream.aclose()
