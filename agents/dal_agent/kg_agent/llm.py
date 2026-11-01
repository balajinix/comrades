"""LLM construction. Provider-aware so a single kg-agent process can route
each request to either provider or the litellm proxy (which fronts Claude
for us) without restarting.

ChatLiteLLM exposes `api_key` / `api_base` as top-level fields and a
`model_kwargs` dict that is forwarded as kwargs to `litellm.completion`. Older
versions of this module passed a `litellm_params={...}` kwarg, which has no
special handling on ChatLiteLLM — those settings were silently dropped and the
calls only succeeded because env-var fallbacks (PROVIDER_API_KEY etc.) happened
to be set in the same process. We now pass the documented fields explicitly.
"""
from __future__ import annotations

import re
import threading
import time

from langchain_litellm import ChatLiteLLM

from .config import (
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    PROVIDER_API_KEY,
    PROVIDER_MODEL_NAME,
    PROVIDER_PROJECT_ID,
    PROVIDER_URL,
)


# Circuit breaker for transient LLM-provider outages (provider disconnects,
# connection resets, etc.). Without this, every request waits through 3
# retries + backoff before failing — when the provider is genuinely down,
# that compounds to multiple minutes per user before they see an error.
# Once THRESHOLD consecutive transient failures hit, the breaker opens for
# COOLDOWN_S; subsequent calls return a fast "provider unavailable" error
# without invoking the LLM. After the cooldown, the next call is a
# half-open probe — success closes the breaker, failure re-opens it.
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 30

_TRANSIENT_HINTS = (
    "server disconnected",
    "internal server error",
    "service unavailable",
    "connection error",
    "connection reset",
    "timeout",
    "providerexception",
)


class _Breaker:
    """Tiny circuit breaker; thread-safe with a plain Lock (no awaits while held)."""

    def __init__(self) -> None:
        self._consec_failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            return (time.monotonic() - self._opened_at) < _BREAKER_COOLDOWN_S

    def cooldown_remaining(self) -> int:
        with self._lock:
            if self._opened_at is None:
                return 0
            return max(0, _BREAKER_COOLDOWN_S - int(time.monotonic() - self._opened_at))

    def record_success(self) -> None:
        with self._lock:
            self._consec_failures = 0
            self._opened_at = None

    def record_failure(self, exc: BaseException) -> None:
        """Bump the failure counter iff the exception looks transient."""
        msg = str(exc).lower()
        if not any(hint in msg for hint in _TRANSIENT_HINTS):
            return
        with self._lock:
            self._consec_failures += 1
            if self._consec_failures >= _BREAKER_THRESHOLD:
                self._opened_at = time.monotonic()

    @property
    def consec_failures(self) -> int:
        with self._lock:
            return self._consec_failures


# Module-global breaker shared across all in-process requests. kg-agent
# currently only dispatches to provider in prod; if we add a second
# provider later, key this by `resolved.split("/")[0]` instead.
breaker = _Breaker()


def resolve_model(model: str | None) -> str:
    """Pick the model id to use, falling back to env. Adds the `provider/`
    prefix when the id has no provider prefix — anthropic/, gemini/, openai/,
    provider/ all pass through.

    Splitting this out from `build_llm` lets callers persist or surface the
    *resolved* name (the actual model used), which differs from the raw arg
    when the arg is None or unprefixed.
    """
    name = model or PROVIDER_MODEL_NAME
    if not any(name.startswith(p) for p in (
        "anthropic/", "gemini/", "openai/", "provider/",
    )):
        name = f"provider/{name}"
    return name


# Claude 4.x family on AWS Bedrock (Opus 4.7, Sonnet 4.6, Haiku 4.5, etc.)
# deprecated the `temperature` request parameter — Bedrock now rejects calls
# that send it with "temperature is deprecated for this model". Match on the
# name fragment so future 4.x point releases work without code changes.
_CLAUDE_4_FAMILY = re.compile(r"claude-(opus|sonnet|haiku)-4-\d+", re.I)


def _omits_temperature(model_name: str) -> bool:
    return bool(_CLAUDE_4_FAMILY.search(model_name))


def build_llm(
    temperature: float = 0.0,
    max_tokens: int = 1500,
    model: str | None = None,
) -> ChatLiteLLM:
    """Construct a ChatLiteLLM ready for `ainvoke([SystemMessage, ...])`.

    Routes credentials by model prefix:
      - `provider/...`  → PROVIDER_API_KEY/URL/PROJECT_ID
      - `openai/...` / `anthropic/...` → OPENAI_API_KEY/BASE_URL (the
                          litellm proxy, which exposes Claude and other
                          providers behind an OpenAI-compatible interface).
                          If you wire direct vendor keys later, branch here.

    The `temperature` kwarg is silently dropped for Claude 4.x models since
    Bedrock rejects it for that family. Other models receive it as passed.
    """
    resolved = resolve_model(model)
    # ChatLiteLLM has its own retry layer; the top-level `max_retries` kwarg
    # is what it actually honours (default = 1). Putting `num_retries` inside
    # `model_kwargs` is silently ignored — found this the hard way after a
    # bump that didn't take. LiteLLM retries on RateLimitError (HTTP 429) and
    # transient disconnects with exponential backoff + jitter, honouring the
    # server's Retry-After when present. A budget of 3 (~7s) was enough for a
    # one-off "Server disconnected" hiccup but NOT for sustained provider 429s
    # under concurrent experiment load, which exhausted it and failed the call.
    # 8 stretches the backoff to a ~1–2 min window so a sweep rides out a
    # throttle instead of failing; the extra wall-clock is paid only when
    # actually rate-limited (one logical call either way — no double-counting).
    common_kwargs = {"timeout": 60}

    base_kwargs = {
        "model": resolved,
        "max_tokens": max_tokens,
        "max_retries": 8,
    }
    if not _omits_temperature(resolved):
        base_kwargs["temperature"] = temperature

    if resolved.startswith("provider/"):
        return ChatLiteLLM(
            **base_kwargs,
            api_key=PROVIDER_API_KEY,
            api_base=PROVIDER_URL,
            model_kwargs={**common_kwargs, "project_id": PROVIDER_PROJECT_ID},
        )

    return ChatLiteLLM(
        **base_kwargs,
        api_key=OPENAI_API_KEY,
        api_base=OPENAI_API_BASE,
        model_kwargs=common_kwargs,
    )
