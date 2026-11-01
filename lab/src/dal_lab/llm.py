"""LLM gateway: one place where every model call is made and metered.

Records, per turn, the provider-reported prompt and completion tokens, the
cumulative input tokens of the run (the quantity the cost model predicts grows
roughly quadratically with turns), and latency. Credentials come from the
environment (litellm conventions, e.g. provider_URL / provider_APIKEY /
provider_PROJECT_ID); nothing is read from files here.

Models named ``gateway/<id>`` go to a LiteLLM gateway (OpenAI-compatible,
``LITELLM_BASE_URL`` and ``LITELLM_API_KEY``). The gateway is billed, so those calls
are logged with tokens and dollar cost (``LITELLM_TOKEN_LOG``, a JSONL file), priced
from the gateway's own price card, and stopped at ``LITELLM_COST_CAP_USD`` (default
25), checked over the whole log before every call.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

GATEWAY = "gateway/"
RATE_LIMIT_RETRIES = 8      # provider 429s are infrastructure, not model behaviour
RATE_LIMIT_BACKOFF_S = 2.0  # doubled per attempt, capped at 60 s, with jitter


class CostCapReached(RuntimeError):
    """The gateway token log has reached the configured dollar cap."""


def _gateway_base() -> str:
    base = os.environ["LITELLM_BASE_URL"].rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


_PRICES: dict[str, tuple[float, float]] = {}


def gateway_prices() -> dict[str, tuple[float, float]]:
    """(input, output) dollars per token for each gateway model, from /v1/model/info."""
    if not _PRICES:
        req = urllib.request.Request(
            _gateway_base() + "/model/info",
            headers={"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - known host
            data = json.load(resp).get("data", [])
        for m in data:
            info = m.get("model_info") or {}
            _PRICES[m.get("model_name")] = (float(info.get("input_cost_per_token") or 0),
                                            float(info.get("output_cost_per_token") or 0))
    return _PRICES


def _as_text(message) -> str:
    """The reply as text in the loop's JSON-action format.

    Some models (gpt-oss on provider) answer with a native tool call and empty
    content. The first call is rendered as ``{"tool": name, "args": {...}}`` so
    every model is driven through the same text protocol.
    """
    content = message.content or ""
    calls = getattr(message, "tool_calls", None) or []
    if content.strip() or not calls:
        return content
    fn = calls[0].function
    try:
        args = json.loads(fn.arguments or "{}")
    except json.JSONDecodeError:
        args = {"_raw": fn.arguments}
    return json.dumps({"tool": fn.name, "args": args})


def logged_cost(log: Path) -> float:
    """Total dollars recorded in a gateway token log."""
    if not log.exists():
        return 0.0
    return sum(json.loads(line).get("cost_usd", 0.0) for line in log.open() if line.strip())


@dataclass
class TurnRecord:
    """Metering for one model call."""

    turn: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class LLMGateway:
    """Metered chat-completion client.

    Attributes:
        model: litellm model id, e.g. "provider/meta-llama/llama-3-3-70b-instruct".
        max_tokens: Completion cap per turn.
        temperature: Sampling temperature (0 for reproducibility).
        turns: Per-turn records for the current run.
    """

    model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout_s: float = 120.0
    seed: int | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    _no_temperature: bool = False

    def reset(self) -> None:
        """Start metering a new run."""
        self.turns = []

    def chat(self, messages: list[dict]) -> str:
        """Send ``messages`` and return the assistant text, recording usage.

        Args:
            messages: OpenAI-style chat messages.

        Returns:
            The completion text.
        """
        import litellm  # imported lazily so the offline lab does not need it

        litellm.suppress_debug_info = True
        t0 = time.time()
        rate_limited = getattr(litellm, "RateLimitError", ())
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                resp = self._call(litellm, messages)
                break
            except rate_limited:
                # A 429 would otherwise end the run as llm_error and be scored as a
                # model failure; wait it out instead.
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                delay = min(60.0, RATE_LIMIT_BACKOFF_S * 2 ** attempt)
                time.sleep(delay * (0.5 + random.random()))
        usage = resp.usage
        self.turns.append(TurnRecord(len(self.turns) + 1, int(usage.prompt_tokens or 0),
                                     int(usage.completion_tokens or 0), time.time() - t0))
        return _as_text(resp.choices[0].message)

    def _call(self, litellm, messages: list[dict]):
        """One provider call, through the gateway or directly."""
        if self.model.startswith(GATEWAY):
            return self._gateway_call(litellm, messages)
        extra = {} if self.seed is None else {"seed": self.seed}
        return litellm.completion(model=self.model, messages=messages,
                                  max_tokens=self.max_tokens,
                                  temperature=self.temperature,
                                  timeout=self.timeout_s, num_retries=2, **extra)

    def _gateway_call(self, litellm, messages: list[dict]):
        """One billed call through the LiteLLM gateway, logged and capped."""
        log = Path(os.environ.get("LITELLM_TOKEN_LOG", "findings/gateway_tokens.jsonl"))
        cap = float(os.environ.get("LITELLM_COST_CAP_USD", "25"))
        spent = logged_cost(log)
        if spent >= cap:
            raise CostCapReached(f"gateway spend {spent:.2f} USD reached the cap {cap}")
        model_id = self.model[len(GATEWAY):]
        kwargs = dict(model="openai/" + model_id, messages=messages,
                      api_base=_gateway_base(), api_key=os.environ["LITELLM_API_KEY"],
                      max_completion_tokens=self.max_tokens, timeout=self.timeout_s,
                      num_retries=2)
        if not self._no_temperature:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        try:
            resp = litellm.completion(**kwargs)
        except Exception as exc:  # some models accept only their default temperature
            if "temperature" not in str(exc).lower() or self._no_temperature:
                raise
            self._no_temperature = True
            kwargs.pop("temperature")
            resp = litellm.completion(**kwargs)
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        pin, pout = gateway_prices().get(model_id, (0.0, 0.0))
        rec = {"ts": time.time(), "model": model_id,
               "prompt_tokens": int(usage.prompt_tokens or 0),
               "completion_tokens": int(usage.completion_tokens or 0),
               "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
               "finish_reason": resp.choices[0].finish_reason,
               "temperature_sent": not self._no_temperature,
               "cost_usd": (usage.prompt_tokens or 0) * pin
                           + (usage.completion_tokens or 0) * pout,
               "run_tag": os.environ.get("LITELLM_RUN_TAG", "")}
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return resp

    def totals(self) -> dict:
        """Totals for the current run."""
        return {
            "turns": len(self.turns),
            "prompt_tokens": sum(t.prompt_tokens for t in self.turns),
            "completion_tokens": sum(t.completion_tokens for t in self.turns),
            "llm_latency_s": round(sum(t.latency_s for t in self.turns), 2),
            "max_prompt_tokens": max((t.prompt_tokens for t in self.turns), default=0),
        }
