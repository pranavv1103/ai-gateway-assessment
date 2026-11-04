"""Token-aware sliding-window rate limiter backed by an on-disk SQLite database.

Every reservation is a row ``(tenant, tokens, reserved_at)``. A tenant's usage is the sum of
its rows younger than the window, so the window slides continuously and expiry is exact.
Reservations run inside ``BEGIN IMMEDIATE`` transactions: SQLite's single-writer lock makes the
check-then-insert atomic across threads *and* processes, so concurrent requests can never
overshoot the quota. State is committed to disk (WAL mode), so it survives restarts and crashes;
a crash between reserve and settle leaves the reservation counted until it expires, which errs
on the side of enforcing the limit.
"""

import asyncio
import hashlib
import logging
import math
import random
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: int | None, requested: int, limit: int):
        super().__init__("rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds  # None: can never fit in one window
        self.requested = requested
        self.limit = limit


class LimiterUnavailable(Exception):
    """The backing store could not be used (locked too long, corrupt, unwritable)."""


@dataclass(frozen=True)
class Reservation:
    id: int
    tenant: str
    tokens: int
    remaining: int


def tenant_key(api_key: str) -> str:
    """Tenants are stored as a hash so raw API keys never touch the database."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class SQLiteRateLimiter:
    def __init__(
        self,
        db_path: str | Path,
        limit_tokens: int = 50_000,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
        busy_timeout_seconds: float = 5.0,
        max_attempts: int = 3,
    ):
        self._path = str(db_path)
        self._limit = limit_tokens
        self._window = window_seconds
        self._clock = clock
        self._busy_timeout = busy_timeout_seconds
        self._max_attempts = max_attempts
        self._init_schema()

    @property
    def limit(self) -> int:
        return self._limit

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=self._busy_timeout, isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout * 1000)}")
        return conn

    def _init_schema(self) -> None:
        try:
            with closing(self._connect()) as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS token_reservations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant TEXT NOT NULL,
                        tokens INTEGER NOT NULL CHECK (tokens >= 0),
                        reserved_at REAL NOT NULL
                    )"""
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reservations_tenant_time "
                    "ON token_reservations (tenant, reserved_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_reservations_time "
                    "ON token_reservations (reserved_at)"
                )
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
        except sqlite3.Error as exc:
            raise LimiterUnavailable("cannot initialize rate limiter database") from exc

    def _run(self, operation: Callable[[sqlite3.Connection], object]) -> object:
        """Run one transaction, retrying briefly on lock contention."""
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                with closing(self._connect()) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = operation(conn)
                    except BaseException:
                        conn.execute("ROLLBACK")
                        raise
                    conn.execute("COMMIT")
                    return result
            except sqlite3.OperationalError as exc:
                last_error = exc
                logger.warning("SQLite operational error (attempt %d): %s", attempt + 1, exc)
                time.sleep(random.uniform(0.005, 0.03) * (attempt + 1))  # noqa: S311
            except sqlite3.DatabaseError as exc:
                raise LimiterUnavailable("rate limiter database error") from exc
        raise LimiterUnavailable("rate limiter database is busy") from last_error

    def _reserve_sync(self, api_key: str, tokens: int) -> Reservation:
        tenant = tenant_key(api_key)

        def operation(conn: sqlite3.Connection) -> Reservation:
            now = self._clock()
            cutoff = now - self._window
            conn.execute(
                "DELETE FROM token_reservations WHERE tenant = ? AND reserved_at <= ?",
                (tenant, cutoff),
            )
            (used,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_reservations WHERE tenant = ?",
                (tenant,),
            ).fetchone()
            if used + tokens > self._limit:
                raise RateLimitExceeded(
                    self._retry_after(conn, tenant, used, tokens, now), tokens, self._limit
                )
            cursor = conn.execute(
                "INSERT INTO token_reservations (tenant, tokens, reserved_at) VALUES (?, ?, ?)",
                (tenant, tokens, now),
            )
            return Reservation(cursor.lastrowid, tenant, tokens, self._limit - used - tokens)

        return self._run(operation)  # type: ignore[return-value]

    def _retry_after(
        self, conn: sqlite3.Connection, tenant: str, used: int, requested: int, now: float
    ) -> int | None:
        if requested > self._limit:
            return None
        needed = used + requested - self._limit
        freed = 0
        rows = conn.execute(
            "SELECT tokens, reserved_at FROM token_reservations WHERE tenant = ? "
            "ORDER BY reserved_at, id",
            (tenant,),
        )
        for tokens, reserved_at in rows:
            freed += tokens
            if freed >= needed:
                return max(1, math.ceil(reserved_at + self._window - now))
        return math.ceil(self._window)

    def _settle_sync(self, reservation: Reservation, actual_tokens: int | None) -> None:
        def operation(conn: sqlite3.Connection) -> None:
            if actual_tokens is None:
                conn.execute("DELETE FROM token_reservations WHERE id = ?", (reservation.id,))
            else:
                conn.execute(
                    "UPDATE token_reservations SET tokens = ? WHERE id = ?",
                    (max(0, actual_tokens), reservation.id),
                )

        self._run(operation)

    def _usage_sync(self, api_key: str) -> int:
        tenant = tenant_key(api_key)

        def operation(conn: sqlite3.Connection) -> int:
            cutoff = self._clock() - self._window
            (used,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_reservations "
                "WHERE tenant = ? AND reserved_at > ?",
                (tenant, cutoff),
            ).fetchone()
            return used

        return self._run(operation)  # type: ignore[return-value]

    def _purge_sync(self) -> int:
        def operation(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "DELETE FROM token_reservations WHERE reserved_at <= ?",
                (self._clock() - self._window,),
            )
            return cursor.rowcount

        return self._run(operation)  # type: ignore[return-value]

    async def reserve(self, api_key: str, tokens: int) -> Reservation:
        """Atomically reserve tokens or raise RateLimitExceeded / LimiterUnavailable."""
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        return await asyncio.to_thread(self._reserve_sync, api_key, tokens)

    async def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate with actual consumption."""
        await asyncio.to_thread(self._settle_sync, reservation, actual_tokens)

    async def release(self, reservation: Reservation) -> None:
        """Return a reservation in full (the request consumed nothing)."""
        await asyncio.to_thread(self._settle_sync, reservation, None)

    async def usage(self, api_key: str) -> int:
        return await asyncio.to_thread(self._usage_sync, api_key)

    async def purge_expired(self) -> int:
        """Delete expired rows for every tenant; returns the number removed."""
        return await asyncio.to_thread(self._purge_sync)
