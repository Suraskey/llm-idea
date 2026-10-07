"""Single entry point for Postgres connections (sync and async).

Wraps psycopg.connect to always register the pgvector adapter so numpy arrays
serialize as VECTOR values. Prevents RESEARCH Pitfall 6 (no adapter found for
numpy.ndarray on INSERT into a VECTOR column).

Usage (sync):
    from ascension.common.db import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")

Usage (async — Phase 4.0+):
    from ascension.common.db import get_async_pool

    pool = await get_async_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")

Convention set here: never call psycopg.connect(...) or
psycopg.AsyncConnection.connect(...) directly in application code — always
get_connection() / get_async_pool(). A grep for either raw call in any Phase
2.0+ PR is a review red flag; the only authorized async connection site is
the `configure` callback of `_ASYNC_POOL` below.

Phase 4.0 adds the async sibling `get_async_pool()` (and `close_async_pool()`)
below for the live LLM-call dual-write path — the pool registers pgvector
per-connection via `_configure_async_conn` (Pitfall 4: pgvector's adapter is
per-connection, not global, so every pooled conn must pass through the
callback before its first VECTOR column INSERT).
"""

from __future__ import annotations

import asyncio

import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg_pool import AsyncConnectionPool

from ascension.common.config import settings

# Module-level async pool singleton. Lazily initialized by the first
# `await get_async_pool()` call; closed explicitly by `close_async_pool()`
# (or via LLMClient.aclose() in the smoke / Phase 7+ lifecycle).
_ASYNC_POOL: AsyncConnectionPool | None = None
# WR-03 fix (2026-04-22): async lock guarding the lazy init.
# Council peak = 5 Council + 3 Justice + 1 Argus = 9 concurrent agents,
# all of which may race the first `await get_async_pool()` call. Without
# this lock the loser of the race constructs a pool that is silently
# orphaned (its connections leak until process exit). The lock is bound
# to the running event loop on first acquire (Python 3.10+ behavior); a
# module that gets re-imported under a fresh loop pays a one-time
# binding cost on first access.
_ASYNC_POOL_LOCK: asyncio.Lock = asyncio.Lock()


def get_connection() -> psycopg.Connection:
    """Open a new Postgres connection with pgvector registered.

    Callers get a vanilla psycopg Connection — same context-manager semantics
    as psycopg.connect() — but numpy ndarrays serialize to VECTOR automatically.
    """
    conn = psycopg.connect(str(settings.DATABASE_URL))
    register_vector(conn)
    return conn


async def _configure_async_conn(conn: psycopg.AsyncConnection) -> None:
    """Pool-level configure callback — register pgvector on every connection.

    Pitfall 4: pgvector's numpy-array adapter is registered per-connection,
    not globally. Every pooled connection MUST pass through this callback
    before the first VECTOR column INSERT.
    """
    await register_vector_async(conn)


async def get_async_pool() -> AsyncConnectionPool:
    """Module-level async pool singleton. Lazily initialized on first call.

    Why pooled: TCP + TLS + auth + register_vector_async is ~20-50ms per
    new connection. Phase 10.0+ peak (~5 agents + 4 metabolism + 1
    orchestrator + 1 transient + 1 buffer = 12 concurrent) is covered
    by max_size=12 with zero handshake on the hot path (RESEARCH Phase
    10.0 Pitfall 5 + Correction 4 — raised from the pre-orchestrator
    value of 8). Per-call connect-and-close would add latency_ms noise
    to the cost_usd attribution (CONTEXT §specifics D-10).

    Concurrency (WR-03 fix): double-checked-locking pattern. The unlocked
    fast-path returns the cached pool without contention; only the very
    first call (per process lifetime, or after `close_async_pool()`)
    pays the lock cost.

    L-004 note: reads `settings.DATABASE_URL` at first construction. Tests
    that monkeypatch the singleton AFTER the pool opens will NOT redirect
    the pool — test fixtures MUST patch `settings.DATABASE_URL` BEFORE any
    `get_async_pool()` call, or explicitly `close_async_pool()` first to
    force re-construct on the patched DSN.
    """
    global _ASYNC_POOL
    # Fast path — already initialized; no lock needed.
    if _ASYNC_POOL is not None:
        return _ASYNC_POOL
    # Slow path — first call (or post-close re-init). Take the lock and
    # re-check inside it so concurrent first-callers don't double-construct.
    async with _ASYNC_POOL_LOCK:
        if _ASYNC_POOL is not None:
            return _ASYNC_POOL
        # RESEARCH Phase 10.0 Pitfall 5 + Correction 4: 5 placeholder agents +
        # 4 metabolism services + 1 orchestrator heartbeat writer + 1 transient
        # runs writer + 1 buffer = 12. Previous value (8) was pre-orchestrator;
        # under peak load it sporadically TimeoutErrors on checkout. Cost: zero
        # — psycopg_pool allocates lazily; idle connections don't burn memory.
        pool = AsyncConnectionPool(
            conninfo=str(settings.DATABASE_URL),
            min_size=1,
            max_size=12,
            configure=_configure_async_conn,
            open=False,
        )
        await pool.open()
        _ASYNC_POOL = pool
    return _ASYNC_POOL


async def close_async_pool() -> None:
    """Clean shutdown — called from LLMClient.aclose() and test teardown."""
    global _ASYNC_POOL
    async with _ASYNC_POOL_LOCK:
        if _ASYNC_POOL is not None:
            await _ASYNC_POOL.close()
            _ASYNC_POOL = None
