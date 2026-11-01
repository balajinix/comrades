"""SQLAlchemy ORM for run history.

Two tables:
- runs:       one row per agent invocation
- run_steps:  one row per ReAct iteration

Schema is portable across SQLite (local dev) and Postgres (production) — no
dialect-specific types. Add Postgres-specific fields (jsonb, range types for
temporal queries) here only when we actually need them.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    steps_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_invoked: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    client: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Cache key + payload. See CACHE_DESIGN.md.
    # `question_hash` is SHA-256 of the normalised question; the index makes
    # cache lookups sub-ms. `output_json` is KgOutput.model_dump_json() —
    # storing the full envelope means a cache hit returns identical
    # renderings/raw_result/source as a fresh run. NULL on legacy rows or
    # uncacheable outputs (errors, blanks, **Error: prefixed answers).
    question_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    output_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # True iff this row records a cache hit (no LLM/MCP work done — just a
    # `runs` table lookup). False for normal ReAct runs. Lets analytics
    # filter "real work" from "cached responses" with a single predicate
    # and lets the dashboard show hit/miss alongside each other.
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class RunStep(Base):
    __tablename__ = "run_steps"

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True,
    )
    step: Mapped[int] = mapped_column(Integer, primary_key=True)
    thought: Mapped[str | None] = mapped_column(Text, nullable=True)
    action: Mapped[str | None] = mapped_column(String(128), nullable=True)
    action_input: Mapped[str | None] = mapped_column(Text, nullable=True)
    observation: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Wall-clock time for this iteration (LLM + tool + bookkeeping). NULL
    # on rows that pre-date the timing instrumentation.
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
