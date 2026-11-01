#!/usr/bin/env python3
"""Fire a natural-language query at the DAL agent over A2A and print the answer.

Usage:
    python scripts/ask.py "list up to 3 ITSM incidents"
    DAL_AGENT_URL=http://localhost:8005 python scripts/ask.py "top cloud costs this month"

Run it with the dal_agent venv so the a2a-sdk is available:
    agents/dal_agent/.venv/bin/python scripts/ask.py "..."
(comrades_up.sh prints this; the dal_agent service must be running.)
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Make the a2a-sdk importable by reusing dal_agent's venv site-packages if this
# script is run with a different interpreter.
_DAL_VENV = Path(__file__).resolve().parents[1] / "agents" / "dal_agent" / ".venv"
for sp in _DAL_VENV.glob("lib/python*/site-packages"):
    if str(sp) not in sys.path:
        sys.path.insert(0, str(sp))

try:
    import httpx
    from a2a.client import A2ACardResolver, A2AClient
    from a2a.types import (
        Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart,
    )
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing a2a-sdk ({e}). Run with: agents/dal_agent/.venv/bin/python scripts/ask.py \"...\"")

AGENT_URL = os.getenv("DAL_AGENT_URL", "http://localhost:8005")


async def ask(prompt: str) -> None:
    async with httpx.AsyncClient(timeout=180) as hc:
        card = await A2ACardResolver(hc, AGENT_URL).get_agent_card()
        client = A2AClient(hc, card, url=AGENT_URL)
        msg = Message(role=Role.user, messageId=uuid.uuid4().hex,
                      parts=[Part(root=TextPart(text=prompt))])
        req = SendMessageRequest(id=uuid.uuid4().hex,
                                 params=MessageSendParams(message=msg))
        resp = await client.send_message(req)

    result = resp.model_dump().get("result", {}) or {}
    status = result.get("status", {}) or {}
    state = status.get("state", "unknown")
    parts = (status.get("message") or {}).get("parts", []) or []
    text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    print(f"[{state}]\n{text or '(no text returned)'}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: ask.py "your natural-language question"')
    prompt = " ".join(sys.argv[1:])
    try:
        asyncio.run(ask(prompt))
    except Exception as e:  # pragma: no cover
        sys.exit(f"query failed: {type(e).__name__}: {e} "
                 f"(is dal_agent running at {AGENT_URL}? try scripts/comrades_up.sh)")


if __name__ == "__main__":
    main()
