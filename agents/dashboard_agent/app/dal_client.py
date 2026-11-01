"""Fetch data from the DAL agent over A2A.

This makes the dashboard agent a true *consumption* agent: instead of reading a
local CSV, it asks the central DAL agent (the data-access layer) for the data it
needs, over the A2A protocol. That call is the inter-agent communication the
COMRADES study measures.

The ``retries`` knob is the consumption agent's failure policy toward the DAL
agent: ``0`` is fail-fast (give up if the DAL agent fails), ``>0`` retries the
DAL agent (recovers transient failures, but amplifies inter-agent communication).
It defaults from ``DASHBOARD_DAL_RETRIES``. Every attempt is counted so the
caller can report inter-agent message volume.
"""
from __future__ import annotations

import json
import logging
import os
import uuid

import httpx
import pandas as pd
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart,
)

logger = logging.getLogger(__name__)

DAL_AGENT_URL = os.getenv("DAL_AGENT_URL", "http://localhost:8005")
# Phrases the DAL agent uses when a tool/backend failed and it gave up. Used by
# the consumption agent to decide whether to retry the DAL agent.
_GIVEUP_MARKERS = (
    "could not", "couldn't", "can not", "cannot", "unable to", "failed to",
    "not reachable", "unreachable", "not be reached", "connection error",
    "nameresolution", "unavailable", "no incidents", "not be listed",
    "no data", "not retrieve", "**error", "timed out", "timeout",
)


def _looks_like_giveup(text: str) -> bool:
    t = (text or "").lower()
    return not t.strip() or any(m in t for m in _GIVEUP_MARKERS)


def _to_dataframe(artifacts: list) -> pd.DataFrame | None:
    """Build a DataFrame from a DAL `table` rendering, if present."""
    for art in artifacts or []:
        for part in art.get("parts", []):
            text = part.get("text")
            if not text:
                continue
            try:
                renderings = json.loads(text)
            except (ValueError, TypeError):
                continue
            for r in renderings if isinstance(renderings, list) else []:
                if r.get("type") == "table" and r.get("rows"):
                    return pd.DataFrame(r["rows"])
    return None


async def _one_call(client: A2AClient, prompt: str) -> tuple[bool, str, list]:
    msg = Message(role=Role.user, messageId=uuid.uuid4().hex,
                  parts=[Part(root=TextPart(text=prompt))])
    resp = await client.send_message(
        SendMessageRequest(id=uuid.uuid4().hex,
                           params=MessageSendParams(message=msg)))
    d = resp.model_dump().get("result", {}) or {}
    status = d.get("status", {}) or {}
    state = str(status.get("state", "")).lower()
    parts = (status.get("message") or {}).get("parts", []) or []
    text = " ".join(p.get("text", "") for p in parts if p.get("text"))
    artifacts = d.get("artifacts") or []
    ok = ("completed" in state) and not _looks_like_giveup(text)
    return ok, text, artifacts


async def fetch_from_dal(nl_query: str, retries: int | None = None,
                         timeout: float = 180.0) -> tuple[pd.DataFrame | None, dict]:
    """Ask the DAL agent for data. Returns (dataframe_or_None, meta).

    meta = {a2a_calls, success, answer}. a2a_calls is the inter-agent message
    count this fetch produced (1 + retries actually used).
    """
    if retries is None:
        retries = int(os.getenv("DASHBOARD_DAL_RETRIES", "0"))

    a2a_calls = 0
    last_text = ""
    async with httpx.AsyncClient(timeout=timeout) as hc:
        # Card path differs across a2a-sdk versions (agent-card.json vs
        # agent.json). Try the resolver, then fall back to the legacy path.
        try:
            card = await A2ACardResolver(hc, DAL_AGENT_URL).get_agent_card()
        except Exception:
            from a2a.types import AgentCard
            r = await hc.get(DAL_AGENT_URL.rstrip("/") + "/.well-known/agent.json")
            r.raise_for_status()
            card = AgentCard.model_validate(r.json())
        client = A2AClient(hc, card, url=DAL_AGENT_URL)
        for attempt in range(retries + 1):
            a2a_calls += 1
            try:
                ok, text, artifacts = await _one_call(client, nl_query)
            except Exception as e:  # transport failure counts as a failed attempt
                last_text = f"{type(e).__name__}: {e}"
                logger.warning("DAL A2A call failed (attempt %d): %s", attempt + 1, last_text)
                continue
            last_text = text
            if ok:
                df = _to_dataframe(artifacts)
                return df, {"a2a_calls": a2a_calls, "success": True, "answer": text}
            logger.info("DAL returned a give-up (attempt %d/%d)", attempt + 1, retries + 1)

    return None, {"a2a_calls": a2a_calls, "success": False, "answer": last_text}
