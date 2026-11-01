"""ITBench-Lite SRE snapshots as an observability store.

Source: ``ibm-research/ITBench-Lite`` on Hugging Face (Apache 2.0), folder
``snapshots/sre/<version>/Scenario-<n>``. Each scenario is a Kubernetes cluster
running the OpenTelemetry demo shop with one injected fault, exported as raw
Kubernetes objects and events, OpenTelemetry spans and logs, Prometheus alert
snapshots and a ground-truth file. Topology is derived from the records, not read
from a provided graph.

The store is relational with uid foreign keys, so every lab interface (SQL, tools,
GraphQL, skills, code) works unchanged, and the uid joins give the graph view:
Pod RUNS_ON Node, Pod OWNED_BY ReplicaSet OWNED_BY Deployment, Span IN Pod,
Event ABOUT object, Alert ON Pod. The scale knob is the number of scenarios loaded.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
import os
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

STORE_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR = Path(os.environ.get("ITBENCH_DIR", STORE_DIR / "itbench-lite"))
SCALES = {"s1": 1, "s5": 5, "s35": 35}
HF_REPO = "ibm-research/ITBench-Lite"
HF_REVISION = "d0916b08ba421ce5e672e9ad68aa947d938dfef0"  # 2026-04-21, apache-2.0

csv.field_size_limit(1 << 30)
# Snapshot file names carry the time as 2025-12-15T17-55-46.713 or 2025-12-15T175546.713.
_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-?(\d{2})-?(\d{2})(\.\d+)?")
_ATTR = {k: re.compile(r"'" + re.escape(k) + r"': '([^']*)'") for k in
         ("k8s.pod.uid", "k8s.pod.name", "k8s.deployment.name", "k8s.namespace.name",
          "k8s.node.name")}

SCHEMA = """
CREATE TABLE scenario (scenario_id INTEGER PRIMARY KEY, fault_entity VARCHAR,
    fault_kind VARCHAR, fault_category VARCHAR, fault_mechanism VARCHAR,
    root_cause_namespace VARCHAR, ground_truth VARCHAR);
CREATE TABLE k8s_object (uid VARCHAR PRIMARY KEY, scenario_id INTEGER, kind VARCHAR,
    name VARCHAR, namespace VARCHAR, node_uid VARCHAR, node_name VARCHAR,
    phase VARCHAR, owner_uid VARCHAR, owner_kind VARCHAR, owner_name VARCHAR,
    created_at TIMESTAMP, first_seen TIMESTAMP, last_seen TIMESTAMP,
    restarts INTEGER, deleted BOOLEAN, labels VARCHAR);
CREATE TABLE k8s_event (event_id VARCHAR, scenario_id INTEGER, ts TIMESTAMP,
    type VARCHAR, reason VARCHAR, object_uid VARCHAR, object_kind VARCHAR,
    object_name VARCHAR, namespace VARCHAR, message VARCHAR, count INTEGER);
CREATE TABLE span (span_id VARCHAR, trace_id VARCHAR, parent_span_id VARCHAR,
    scenario_id INTEGER, ts TIMESTAMP, service VARCHAR, operation VARCHAR,
    kind VARCHAR, duration_ms DOUBLE, status VARCHAR, pod_uid VARCHAR,
    deployment VARCHAR, namespace VARCHAR);
CREATE TABLE alert (alert_id VARCHAR, scenario_id INTEGER, alertname VARCHAR,
    severity VARCHAR, namespace VARCHAR, pod_uid VARCHAR, pod VARCHAR,
    service VARCHAR, deployment VARCHAR, active_at TIMESTAMP, first_seen TIMESTAMP,
    last_seen TIMESTAMP, summary VARCHAR);
CREATE TABLE log (scenario_id INTEGER, ts TIMESTAMP, service VARCHAR,
    severity VARCHAR, pod_uid VARCHAR, body VARCHAR);
"""

# Data dictionary, stored as column comments: value sets and meaning, as an
# observability API's schema endpoint would document them.
NOTES = {
    ("span", "status"): "'Ok', 'Error' or 'Unset' (exact case)",
    ("span", "kind"): "'Server', 'Client', 'Internal', 'Producer' or 'Consumer' (exact case)",
    ("span", "parent_span_id"): "span_id of the parent span in the same trace; a call "
                                "from service A to service B is a span of B whose parent "
                                "span belongs to A",
    ("span", "duration_ms"): "span duration in milliseconds",
    ("span", "ts"): "span start time, UTC",
    ("span", "pod_uid"): "uid of the pod (k8s_object) that emitted the span",
    ("k8s_object", "kind"): "Kubernetes kind, e.g. 'Pod', 'Node', 'Deployment', "
                            "'ReplicaSet', 'Service' (exact case)",
    ("k8s_object", "owner_uid"): "uid of the owning object (Pod -> ReplicaSet -> Deployment)",
    ("k8s_object", "node_uid"): "uid of the Node a Pod runs on",
    ("k8s_object", "node_name"): "name of the Node a Pod runs on",
    ("k8s_object", "deleted"): "true if the object was missing from the last 5-minute "
                               "listing, i.e. deleted during the window",
    ("k8s_event", "type"): "'Normal' or 'Warning'",
    ("k8s_event", "object_uid"): "uid of the object (k8s_object) the event is about",
    ("k8s_event", "ts"): "time the event was last observed, UTC",
    ("alert", "pod_uid"): "uid of the pod the alert names, if any",
    ("alert", "first_seen"): "first alert snapshot (one per minute) the alert appeared in, UTC",
    ("log", "severity"): "as emitted, mixed case: 'INFO', 'info', 'Information', 'WARN' or empty",
    ("scenario", "fault_category"): "category of the injected fault, e.g. 'Change'",
}

# Foreign keys for the GraphQL facade and skills: (child, column) -> (parent, column).
FOREIGN_KEYS = {
    ("k8s_object", "scenario_id"): ("scenario", "scenario_id"),
    ("k8s_object", "node_uid"): ("k8s_object", "uid"),
    ("k8s_object", "owner_uid"): ("k8s_object", "uid"),
    ("k8s_event", "object_uid"): ("k8s_object", "uid"),
    ("k8s_event", "scenario_id"): ("scenario", "scenario_id"),
    ("span", "pod_uid"): ("k8s_object", "uid"),
    ("span", "scenario_id"): ("scenario", "scenario_id"),
    ("alert", "pod_uid"): ("k8s_object", "uid"),
    ("alert", "scenario_id"): ("scenario", "scenario_id"),
    ("log", "pod_uid"): ("k8s_object", "uid"),
    ("log", "scenario_id"): ("scenario", "scenario_id"),
}


def download(data_dir: Path = DATA_DIR) -> Path:
    """Fetch the SRE snapshots (without per-pod metrics) at the pinned revision.

    About 9 GB. Needs the ``data`` extra (``huggingface_hub``).
    """
    from huggingface_hub import snapshot_download

    pattern = "snapshots/sre/*/Scenario-*/"
    snapshot_download(HF_REPO, repo_type="dataset", revision=HF_REVISION,
                      local_dir=str(data_dir), max_workers=8,
                      allow_patterns=["README.md", pattern + "ground_truth.yaml",
                                      pattern + "k8s_*_raw.tsv",
                                      pattern + "otel_*_raw.tsv", pattern + "alerts/*"])
    return data_dir


def scenario_dirs(data_dir: Path = DATA_DIR) -> dict[int, Path]:
    """Map scenario id to its directory, for every downloaded SRE scenario."""
    out = {}
    for p in sorted(data_dir.glob("snapshots/sre/*/Scenario-*")):
        out[int(p.name.split("-")[1])] = p
    return out


def _rows(path: Path):
    with path.open(newline="") as fh:
        yield from csv.DictReader(fh, delimiter="\t")


def _attr(attrs: str, key: str) -> str | None:
    m = _ATTR[key].search(attrs or "")
    return m.group(1) if m else None


def _ts(s: str | None) -> str | None:
    # ClickHouse nanosecond timestamps and RFC 3339 strings; DuckDB keeps microseconds.
    if not s:
        return None
    s = s.replace("T", " ").rstrip("Z")
    head, _, frac = s.partition(".")
    return f"{head}.{frac[:6]}" if frac else head


def _load_objects(sid: int, path: Path) -> list[tuple]:
    latest: dict[str, dict] = {}
    seen: dict[str, list[str]] = {}
    for row in _rows(path):
        body = json.loads(row["Body"])
        obj = body.get("object", body)
        meta = obj.get("metadata", {})
        uid = meta.get("uid")
        if not uid:
            continue
        ts = _ts(row["Timestamp"])
        span = seen.setdefault(uid, [ts, ts])
        span[0], span[1] = min(span[0], ts), max(span[1], ts)
        if uid not in latest or ts >= latest[uid]["_ts"]:
            obj["_ts"] = ts
            latest[uid] = obj
    node_uid = {o["metadata"]["name"]: u for u, o in latest.items() if o.get("kind") == "Node"}
    # Objects are re-listed every 5 minutes; one missing from the last listing was
    # deleted (the only way a snapshot shows a delete, as in a snapshot merge).
    last = max(v[1] for v in seen.values())
    cutoff = (datetime.fromisoformat(last) - timedelta(seconds=60)).isoformat(" ")
    out = []
    for uid, o in latest.items():
        meta, spec, status = o.get("metadata", {}), o.get("spec") or {}, o.get("status") or {}
        owners = meta.get("ownerReferences") or [{}]
        node = spec.get("nodeName") if o.get("kind") == "Pod" else None
        restarts = sum(c.get("restartCount", 0) for c in status.get("containerStatuses") or [])
        out.append((uid, sid, o.get("kind"), meta.get("name"), meta.get("namespace"),
                    node_uid.get(node), node, status.get("phase"), owners[0].get("uid"),
                    owners[0].get("kind"), owners[0].get("name"),
                    _ts(meta.get("creationTimestamp")), seen[uid][0], seen[uid][1],
                    restarts if o.get("kind") == "Pod" else None,
                    seen[uid][1] < cutoff,
                    json.dumps(meta.get("labels") or {}, sort_keys=True)))
    return out


def _load_events(sid: int, path: Path) -> list[tuple]:
    # Each event is exported twice: as core/v1 (involvedObject, message, count) and as
    # events.k8s.io/v1 (regarding, note, deprecatedCount). Read both; keep core/v1.
    out: dict[str, tuple] = {}
    for row in _rows(path):
        e = json.loads(row["Body"]).get("object", {})
        core = e.get("apiVersion") == "v1"
        eid = e.get("metadata", {}).get("uid")
        if eid in out and not core:
            continue
        inv = e.get("involvedObject") or e.get("regarding") or {}
        ts = _ts(e.get("lastTimestamp") or e.get("deprecatedLastTimestamp")
                 or e.get("eventTime") or row["Timestamp"])
        out[eid] = (eid, sid, ts, e.get("type"), e.get("reason"), inv.get("uid"),
                    inv.get("kind"), inv.get("name"), inv.get("namespace"),
                    e.get("message") or e.get("note"),
                    e.get("count") or e.get("deprecatedCount") or 1)
    return list(out.values())


def _load_spans(sid: int, path: Path) -> list[tuple]:
    out = []
    for r in _rows(path):
        ra = r["ResourceAttributes"]
        out.append((r["SpanId"], r["TraceId"], r["ParentSpanId"] or None, sid,
                    _ts(r["Timestamp"]), r["ServiceName"], r["SpanName"], r["SpanKind"],
                    int(r["Duration"]) / 1e6, r["StatusCode"], _attr(ra, "k8s.pod.uid"),
                    _attr(ra, "k8s.deployment.name"), _attr(ra, "k8s.namespace.name")))
    return out


def _load_logs(sid: int, path: Path) -> list[tuple]:
    return [(sid, _ts(r["Timestamp"]), r["ServiceName"], r["SeverityText"],
             _attr(r["ResourceAttributes"], "k8s.pod.uid"), r["Body"][:2000])
            for r in _rows(path)]


def _load_alerts(sid: int, alert_dir: Path, pod_uid: dict) -> list[tuple]:
    alerts: dict[str, list] = {}
    for f in sorted(alert_dir.glob("*.json")):
        m = _STAMP.search(f.stem)
        if not m:
            continue
        snap_ts = f"{m[1]} {m[2]}:{m[3]}:{m[4]}{(m[5] or '')[:7]}"
        doc = json.loads(f.read_text()) or []
        if isinstance(doc, dict):  # the Prometheus API envelope, in some snapshots
            doc = (doc.get("data") or {}).get("alerts") or []
        for a in doc:
            if not isinstance(a, dict):
                continue
            lab = a.get("labels", {})
            key = json.dumps(lab, sort_keys=True)
            rec = alerts.get(key)
            if rec is None:
                pod = lab.get("pod")
                alerts[key] = [f"{sid}:{len(alerts)}", sid, lab.get("alertname"),
                               lab.get("severity"), lab.get("namespace"),
                               pod_uid.get((lab.get("namespace"), pod)), pod,
                               lab.get("service"), lab.get("deployment"),
                               _ts(a.get("activeAt")), snap_ts, snap_ts,
                               (a.get("annotations") or {}).get("summary")]
            else:
                rec[10], rec[11] = min(rec[10], snap_ts), max(rec[11], snap_ts)
    return [tuple(v) for v in alerts.values()]


def _insert(con: duckdb.DuckDBPyConnection, table: str, rows: list[tuple]) -> None:
    """Bulk insert through a DataFrame (executemany is row-at-a-time in DuckDB)."""
    if not rows:
        return
    cols = [c[0] for c in con.execute(f"DESCRIBE {table}").fetchall()]
    df = pd.DataFrame(rows, columns=cols)  # noqa: F841 (read by DuckDB by name)
    verb = "INSERT OR IGNORE" if table in ("scenario", "k8s_object") else "INSERT"
    con.execute(f"{verb} INTO {table} SELECT * FROM df")


def build(db: Path, scenarios: list[int], data_dir: Path = DATA_DIR) -> Path:
    """Load ``scenarios`` into a fresh DuckDB store at ``db``.

    Args:
        db: Output database path (replaced if present).
        scenarios: Scenario ids to load.
        data_dir: Root of the ITBench-Lite download.

    Returns:
        ``db``.
    """
    dirs = scenario_dirs(data_dir)
    missing = [s for s in scenarios if s not in dirs]
    if missing:
        raise FileNotFoundError(f"scenarios not downloaded: {missing}")
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    con.execute(SCHEMA)
    for sid in scenarios:
        d = dirs[sid]
        gt_path = d / "ground_truth.yaml"
        gt = (yaml.safe_load(gt_path.read_text()) if gt_path.exists() else None) or {}
        fault = (gt.get("fault") or [{}])[0]
        entity = fault.get("entity") or {}
        root = next((g for g in gt.get("groups") or [] if g.get("root_cause")), {})
        con.execute("INSERT INTO scenario VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [sid, entity.get("name"), entity.get("kind"), fault.get("category"),
                     fault.get("fault_mechanism"), root.get("namespace"),
                     json.dumps(gt, default=str)])
        objs = _load_objects(sid, d / "k8s_objects_raw.tsv")
        _insert(con, "k8s_object", objs)
        pod_uid = {(o[4], o[3]): o[0] for o in objs if o[2] == "Pod"}
        for table, rows in (("k8s_event", _load_events(sid, d / "k8s_events_raw.tsv")),
                            ("span", _load_spans(sid, d / "otel_traces_raw.tsv")),
                            ("log", _load_logs(sid, d / "otel_logs_raw.tsv")),
                            ("alert", _load_alerts(sid, d / "alerts", pod_uid))):
            _insert(con, table, rows)
    for (t, c), note in NOTES.items():
        con.execute(f"COMMENT ON COLUMN {t}.{c} IS '{note.replace(chr(39), chr(39) * 2)}'")
    con.execute("CHECKPOINT")
    con.close()
    return db


def db_path(scale: str) -> Path:
    """Store path for a scale (``s1``, ``s5`` or ``s35`` scenarios)."""
    return STORE_DIR / f"itbench_{scale}.duckdb"


def ensure_itbench(scale: str = "s1", data_dir: Path = DATA_DIR) -> Path:
    """Build the store for ``scale`` if absent: the first N scenarios by id."""
    path = db_path(scale)
    if not path.exists():
        ids = sorted(scenario_dirs(data_dir))[:SCALES[scale]]
        build(path, ids, data_dir)
    return path
