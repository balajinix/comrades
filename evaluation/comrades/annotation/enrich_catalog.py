"""Enrich the mcp_dal tool catalog with the context a human annotator needs.

Combines two sources:
  (1) the MCP catalog (catalog_mcp_dal.json) — full description + per-parameter
      {name, type, required, description, enum, default} (these carry the
      Annotated Field descriptions, the richest reliable signal); and
  (2) the mcp_dal source tree — the BACKEND each tool belongs to and best-effort
      API endpoint hints (the REST paths / base URLs the tool's module calls,
      e.g. FinOps /v3/..., ITSM /api/now/table/..., Resilience
      /core/api/v1/...).

Output: catalog_mcp_dal.enriched.json — consumed by gen_candidates.py so every
annotation row shows the tool's real API surface, not just its name.

    python enrich_catalog.py            # uses ./catalog_mcp_dal.json + $MCP_DAL_DIR
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MCP_DAL_DIR = Path(os.getenv("MCP_DAL_DIR", os.path.expanduser("~/src/mcp_dal")))
CATALOG = HERE / "catalog_mcp_dal.json"

# backend label by source-path fragment (first match wins)
BACKEND_BY_PATH = [
    ("finops", "finops (multi-cloud cost / FinOps)", "cost"),
    ("resilience", "Resilience (resilience / compliance / security)", "resilience"),
    ("infra", "Infra (cloud infrastructure graph)", "infra"),
    ("apm", "APM (APM / observability)", "apm"),
    ("itsm", "ITSM (ITSM)", "itsm"),
]
# endpoint-ish strings worth surfacing as API hints
ENDPOINT_RE = re.compile(
    r"(https?://[\w.\-]+(?:/[\w/{}.\-]*)?"            # base URLs
    r"|/api/now/table/[\w]+"                          # itsm tables
    r"|/(?:core/)?api/v?\d?[\w/{}.\-]*"               # /core/api/v1/..., /api/...
    r"|/v\d/[\w/{}.\-]*)"                             # /v3/...
)


def _params(input_schema: dict) -> list[dict]:
    props = (input_schema or {}).get("properties", {}) or {}
    required = set((input_schema or {}).get("required", []) or [])
    out = []
    for name, spec in props.items():
        if name == "ctx":
            continue
        # resolve type across anyOf/null unions
        t = spec.get("type")
        if not t and "anyOf" in spec:
            t = "|".join(sorted({a.get("type", "?") for a in spec["anyOf"] if a.get("type") != "null"})) or "any"
        out.append({
            "name": name, "type": t or "any", "required": name in required,
            "description": (spec.get("description") or "").strip(),
            **({"enum": spec["enum"]} if "enum" in spec else {}),
            **({"default": spec["default"]} if "default" in spec else {}),
        })
    return out


def _source_files() -> list[Path]:
    skip = ("/.venv/", "/venv/", "/__pycache__/", "/tests/", "/examples/", "/scripts/", "/.git/")
    return [p for p in MCP_DAL_DIR.rglob("*.py")
            if not any(s in str(p) for s in skip)]


def _locate(name: str, files: list[Path]) -> Path | None:
    """Find the file that defines a tool: `def name(`, `async def name(`, or a
    `name="name"` / `"name"` registration string."""
    pats = [re.compile(rf"\b(?:async\s+)?def\s+{re.escape(name)}\s*\("),
            re.compile(rf"""name\s*=\s*["']{re.escape(name)}["']"""),
            re.compile(rf"""["']{re.escape(name)}["']""")]
    for pat in pats:                      # prefer a real def over a string mention
        for f in files:
            try:
                if pat.search(f.read_text(errors="ignore")):
                    return f
            except OSError:
                continue
    return None


def _backend(path: Path) -> tuple[str, str]:
    s = str(path).lower()
    for frag, label, fam in BACKEND_BY_PATH:
        if frag in s:
            return label, fam
    return "mcp_dal (other)", "other"


def _api_hints(path: Path, limit: int = 8) -> list[str]:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    seen, hints = set(), []
    for m in ENDPOINT_RE.findall(text):
        h = m.strip().rstrip('",\'')
        if 4 < len(h) < 120 and h not in seen and not h.endswith((".py", ".json")):
            seen.add(h); hints.append(h)
        if len(hints) >= limit:
            break
    return hints


def main() -> None:
    catalog = json.loads(CATALOG.read_text())
    files = _source_files()
    enriched = []
    located = 0
    for t in catalog:
        name = t["name"]
        src = _locate(name, files)
        backend, family = _backend(src) if src else ("mcp_dal (other)", "other")
        if src:
            located += 1
        enriched.append({
            "name": name,
            "description": (t.get("description") or "").strip(),
            "backend": backend,
            "family_guess": family,
            "params": _params(t.get("inputSchema") or {}),
            "api_hints": _api_hints(src) if src else [],
            "source_file": str(src.relative_to(MCP_DAL_DIR)) if src else None,
        })
    out = HERE / "catalog_mcp_dal.enriched.json"
    out.write_text(json.dumps(enriched, indent=1))
    fams = {}
    for e in enriched:
        fams[e["family_guess"]] = fams.get(e["family_guess"], 0) + 1
    print(f"enriched {len(enriched)} tools -> {out}")
    print(f"  source-located: {located}/{len(enriched)} | by family: {dict(sorted(fams.items()))}")
    with_hints = sum(1 for e in enriched if e["api_hints"])
    print(f"  with API endpoint hints: {with_hints}/{len(enriched)}")


if __name__ == "__main__":
    main()
