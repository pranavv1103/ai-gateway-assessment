import asyncio
import sqlite3

import pytest

from aigw.router.limiter import (
    LimiterUnavailable,
    RateLimitExceeded,
    SQLiteRateLimiter,
    tenant_key,
)


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(tmp_path, clock):
    return SQLiteRateLimiter(tmp_path / "limits.db", 50_000, 60.0, clock=clock)


async def test_exact_limit_allowed_and_one_more_denied(limiter):
    await limiter.reserve("k", 50_000)
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("k", 1)


async def test_over_limit_single_request_denied_without_retry_hint(limiter):
    with pytest.raises(RateLimitExceeded) as info:
        await limiter.reserve("k", 50_001)
    assert info.value.retry_after_seconds is None


async def test_zero_token_reservation_allowed(limiter):
    reservation = await limiter.reserve("k", 0)
    assert reservation.remaining == 50_000


async def test_negative_tokens_rejected(limiter):
    with pytest.raises(ValueError):
        await limiter.reserve("k", -1)


async def test_window_slides_precisely(limiter, clock):
    await limiter.reserve("k", 30_000)
    clock.now += 30
    await limiter.reserve("k", 20_000)
    with pytest.raises(RateLimitExceeded) as info:
        await limiter.reserve("k", 1)
    assert info.value.retry_after_seconds == 30
    clock.now += 29.9  # first reservation still 59.9s old
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("k", 30_000)
    clock.now += 0.1  # exactly 60s: expired
    await limiter.reserve("k", 30_000)
    assert await limiter.usage("k") == 50_000


async def test_tenants_are_isolated(limiter):
    await limiter.reserve("a", 50_000)
    await limiter.reserve("b", 50_000)
    with pytest.raises(RateLimitExceeded):
        await limiter.reserve("a", 1)


async def test_settle_adjusts_to_actual_usage(limiter):
    reservation = await limiter.reserve("k", 40_000)
    await limiter.settle(reservation, 10_000)
    assert await limiter.usage("k") == 10_000
    await limiter.reserve("k", 40_000)


async def test_release_returns_tokens(limiter):
    reservation = await limiter.reserve("k", 50_000)
    await limiter.release(reservation)
    assert await limiter.usage("k") == 0


async def test_concurrent_reservations_never_exceed_quota(tmp_path):
    limiter = SQLiteRateLimiter(tmp_path / "c.db", 50_000, 60.0, busy_timeout_seconds=10)
    results = await asyncio.gather(
        *(limiter.reserve("k", 1_000) for _ in range(80)), return_exceptions=True
    )
    granted = [r for r in results if not isinstance(r, Exception)]
    denied = [r for r in results if isinstance(r, RateLimitExceeded)]
    assert len(granted) == 50
    assert len(denied) == 30
    assert await limiter.usage("k") == 50_000


async def test_concurrent_across_separate_limiter_instances(tmp_path):
    path = tmp_path / "multi.db"
    instances = [SQLiteRateLimiter(path, 10_000, 60.0, busy_timeout_seconds=10) for _ in range(4)]
    results = await asyncio.gather(
        *(instances[i % 4].reserve("k", 500) for i in range(40)), return_exceptions=True
    )
    assert sum(not isinstance(r, Exception) for r in results) == 20


async def test_state_persists_across_restart(tmp_path, clock):
    path = tmp_path / "p.db"
    first = SQLiteRateLimiter(path, 50_000, 60.0, clock=clock)
    await first.reserve("k", 45_000)
    del first
    second = SQLiteRateLimiter(path, 50_000, 60.0, clock=clock)
    assert await second.usage("k") == 45_000
    with pytest.raises(RateLimitExceeded):
        await second.reserve("k", 6_000)


async def test_database_is_on_disk_with_indexes_and_hashed_keys(tmp_path):
    path = tmp_path / "disk.db"
    limiter = SQLiteRateLimiter(path)
    await limiter.reserve("super-secret-key", 10)
    assert path.exists() and path.stat().st_size > 0
    with sqlite3.connect(path) as conn:
        indexes = {r[1] for r in conn.execute("PRAGMA index_list(token_reservations)")}
        assert "idx_reservations_tenant_time" in indexes
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        (tenant,) = conn.execute("SELECT tenant FROM token_reservations").fetchone()
    assert tenant == tenant_key("super-secret-key")
    assert "super-secret-key" not in tenant


async def test_purge_expired_removes_old_rows(limiter, clock):
    await limiter.reserve("a", 10)
    await limiter.reserve("b", 10)
    clock.now += 61
    assert await limiter.purge_expired() == 2


async def test_expired_rows_cleaned_on_reserve(tmp_path, clock):
    path = tmp_path / "clean.db"
    limiter = SQLiteRateLimiter(path, clock=clock)
    await limiter.reserve("k", 10)
    clock.now += 61
    await limiter.reserve("k", 10)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM token_reservations").fetchone()[0] == 1


async def test_locked_database_raises_limiter_unavailable(tmp_path):
    path = tmp_path / "lock.db"
    limiter = SQLiteRateLimiter(path, busy_timeout_seconds=0.05, max_attempts=2)
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(LimiterUnavailable):
            await limiter.reserve("k", 1)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    await limiter.reserve("k", 1)  # recovers once the lock is gone


async def test_corrupt_database_raises_limiter_unavailable(tmp_path):
    path = tmp_path / "bad.db"
    path.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(LimiterUnavailable):
        SQLiteRateLimiter(path)


async def test_failed_transaction_rolls_back(limiter, monkeypatch):
    original = limiter._retry_after

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(limiter, "_retry_after", boom)
    await limiter.reserve("k", 50_000)
    with pytest.raises(RuntimeError):
        await limiter.reserve("k", 1)
    monkeypatch.setattr(limiter, "_retry_after", original)
    assert await limiter.usage("k") == 50_000
