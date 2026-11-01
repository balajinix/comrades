"""Async SQLAlchemy engine + session factory.

Engine is process-global and lazy: the first call to `get_session()` triggers
both engine creation and table creation. This keeps cold-start cheap when the
agent isn't yet exercised, and avoids requiring callers to know about an
init step.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import KG_AGENT_DB_URL
from .models import Base


logger = logging.getLogger("kg_agent.db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_initialised = False


async def _ensure_initialised() -> None:
    global _engine, _session_factory, _initialised
    if _initialised:
        return
    _engine = create_async_engine(KG_AGENT_DB_URL, future=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_runs_table(conn)
    logger.info("kg-agent persistence ready (%s)", KG_AGENT_DB_URL.split("://")[0])
    _initialised = True


async def _migrate_runs_table(conn) -> None:
    """Apply forward-only column additions for the cache layer.

    `Base.metadata.create_all` only creates tables that don't exist; it
    won't ALTER an existing `runs` table to add new columns. The UI's
    deployed kg_agent.db has weeks of run history we don't want to drop,
    so we add the cache columns idempotently here. Both SQLite and
    Postgres support `ALTER TABLE ADD COLUMN`; on a duplicate-column
    error we log and continue (the column is already there).
    """
    statements = [
        # Column adds. Safe to run repeatedly — duplicate-column errors
        # are caught individually below.
        "ALTER TABLE runs ADD COLUMN question_hash VARCHAR(64)",
        "ALTER TABLE runs ADD COLUMN output_json TEXT",
        # cache_hit defaults to false / 0. SQLite's BOOLEAN is INTEGER under
        # the hood; both 0 and FALSE work as the default literal. Existing
        # rows are backfilled to 0 (= miss / pre-cache run).
        "ALTER TABLE runs ADD COLUMN cache_hit BOOLEAN DEFAULT 0 NOT NULL",
        # Per-step wall-clock latency (LLM + tool + bookkeeping). NULL on
        # rows persisted before this column existed.
        "ALTER TABLE run_steps ADD COLUMN elapsed_ms INTEGER",
        # Index for the cache-key lookup. CREATE INDEX IF NOT EXISTS works
        # in both SQLite and Postgres without dialect-specific syntax.
        "CREATE INDEX IF NOT EXISTS ix_runs_question_hash ON runs (question_hash)",
    ]
    for sql in statements:
        try:
            await conn.exec_driver_sql(sql)
        except Exception as exc:
            # Duplicate column / "column already exists" — fine.
            logger.debug("migration step skipped (%s): %s", sql, exc)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async context manager that yields a session and commits/rolls back."""
    await _ensure_initialised()
    assert _session_factory is not None
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
