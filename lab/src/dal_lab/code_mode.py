"""Code execution as a data-access modality ("code mode").

Instead of calling tools one at a time and reading every result, the model writes a
short Python script. Inside the script, ``sql(query)`` returns rows and
``save_answer(rows)`` stores a result of any size and returns a handle. Only what the
script prints comes back to the model, so intermediate rows never enter its context.
This generalises result handles: data stays where it is computed.

Model-written code runs in a separate interpreter under an OS sandbox: ``sandbox-exec``
on macOS, bubblewrap (``bwrap``) on Linux. Either way: no network, no file writes
outside a per-call temp directory, the store opened read-only, a wall-clock timeout,
and capped output. Without a sandbox the tool refuses to run code (fail closed).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import duckdb

from .facades import Catalog, Facade, HandleFacade

STDOUT_LIMIT = 4000
TIMEOUT_S = 60

_RUNNER = r'''
import decimal, json, sys
import duckdb
_con = duckdb.connect(sys.argv[1], read_only=True)
_out = sys.argv[2]

def sql(query):
    """Run a read-only SQL query; return a list of dicts."""
    cur = _con.execute(query)
    cols = [d[0] for d in cur.description]
    return _record([dict(zip(cols, r)) for r in cur.fetchall()])

def _plain(v):
    # Keep numbers numeric: DuckDB returns DECIMAL as Decimal, which default=str
    # would turn into text and make a correct answer compare unequal.
    if isinstance(v, decimal.Decimal):
        return float(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalars from pandas
        return v.item()
    return str(v)

_PROV = sys.argv[3] == "1"
_SEEN = set()      # every text value any sql() call returned
_CALLS = [0]

def _record(rows):
    _CALLS[0] += 1
    if _PROV:
        for r in rows:
            for v in r.values():
                if v is not None and not isinstance(v, (int, float, decimal.Decimal)):
                    _SEEN.add(str(_plain(v)))
    return rows

def save_answer(rows):
    """Store the final answer rows (list of dicts or lists); return a handle."""
    if _PROV:
        if _GLOBALS.get("sql") is not _sql_tool:
            raise RuntimeError("provenance: sql() was redefined; use the provided sql()")
        if not _CALLS[0]:
            raise RuntimeError("provenance: no sql() call has run; answers must come "
                               "from the database")
        for r in rows:
            for v in (r.values() if isinstance(r, dict) else r):
                if v is None or isinstance(v, (int, float, decimal.Decimal, bool)):
                    continue
                if str(_plain(v)) not in _SEEN:
                    raise RuntimeError(f"provenance: value {str(v)[:60]!r} was not "
                                       "returned by any sql() call")
    with open(_out + "/answer.json", "w") as fh:
        json.dump([list(r.values()) if isinstance(r, dict) else list(r) for r in rows],
                  fh, default=_plain)
    return "saved"

_sql_tool = sql
_GLOBALS = {"sql": sql, "save_answer": save_answer}
exec(compile(sys.stdin.read(), "<model>", "exec"), _GLOBALS)
'''


def _sandbox_profile(tmp: str) -> str:
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* (subpath "{tmp}") (subpath "/private/var/folders") '
            '(literal "/dev/null"))')


def _sandboxed(cmd: list[str], tmp: str) -> list[str] | None:
    """Wrap ``cmd`` in the platform sandbox, or return None if there is none."""
    if shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", _sandbox_profile(tmp)] + cmd
    if shutil.which("bwrap"):
        # Read-only root, private /dev, /proc and /tmp, writable temp dir only, and a
        # fresh network namespace with no interfaces.
        return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "--tmpfs", "/tmp", "--bind", tmp, tmp, "--unshare-net", "--unshare-pid",
                "--unshare-ipc", "--die-with-parent", "--new-session", "--chdir", tmp
                ] + cmd
    return None


class CodeFacade(Facade):
    """One ``run_python`` tool; answers can be saved as handles of any size."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 handles: HandleFacade, provenance: bool = False) -> None:
        super().__init__("code+prov" if provenance else "code")
        self.provenance = provenance
        self.con, self.handles = con, handles
        self.store = con.execute("SELECT path FROM duckdb_databases() "
                                 "WHERE database_name = 'store'").fetchone()[0]
        self._add("run_python", self.run_python,
                  "Run a Python script. In it, sql(query) runs read-only DuckDB SQL and "
                  "returns a list of dicts; save_answer(rows) stores the final answer "
                  "rows (any size) and the tool returns a handle you can answer with. "
                  "pandas and numpy are installed. "
                  f"Only printed output is returned (first {STDOUT_LIMIT} chars). "
                  + ("save_answer checks provenance: every text value it saves must "
                     "have been returned by sql(), and sql() must not be redefined. "
                     if provenance else "") +
                  f"Schema:\n{catalog.describe()}",
                  {"code": {"type": "string"}}, ["code"])

    def run_python(self, code: str) -> dict:
        """Execute model-written code in the sandbox.

        Args:
            code: Python source.

        Returns:
            ``{stdout, stderr?, handle?, saved_rows?}``.
        """
        tmp = tempfile.mkdtemp(prefix="codemode-")
        try:
            cmd = [sys.executable, "-I", "-c", _RUNNER, self.store, tmp,
                   "1" if self.provenance else "0"]
            cmd = _sandboxed(cmd, tmp)
            if cmd is None:
                return {"error": "no sandbox available (sandbox-exec or bwrap); "
                                 "refusing to run model-written code"}
            try:
                proc = subprocess.run(cmd, input=code, capture_output=True, text=True,
                                      timeout=TIMEOUT_S, cwd=tmp)
            except subprocess.TimeoutExpired:
                return {"error": f"timeout after {TIMEOUT_S}s"}
            out: dict = {"stdout": proc.stdout[:STDOUT_LIMIT]}
            if len(proc.stdout) > STDOUT_LIMIT:
                out["note"] = f"output truncated to {STDOUT_LIMIT} chars"
            if proc.returncode != 0:
                out["stderr"] = proc.stderr[-1500:]
            answer = Path(tmp) / "answer.json"
            if answer.exists():
                out.update(self._register(json.loads(answer.read_text())))
            elif not proc.stdout and proc.returncode == 0:
                # An empty result is otherwise indistinguishable from "no rows".
                out["note"] = ("the script printed nothing and saved no answer; "
                               "print() what you need or call save_answer(rows)")
            return out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _register(self, rows: list[list]) -> dict:
        """Load saved answer rows into a temp table and expose it as a handle.

        Column types are inferred from the JSON values (integer, number, else text),
        so the stored answer compares like the store's own types.
        """
        handle = f"c{uuid.uuid4().hex[:8]}"
        width = max((len(r) for r in rows), default=1)
        rows = [list(r) + [None] * (width - len(r)) for r in rows]
        types = []
        for i in range(width):
            vals = [r[i] for r in rows if r[i] is not None]
            if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                types.append("BIGINT")
            elif vals and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                              for v in vals):
                types.append("DOUBLE")
            else:
                types.append("VARCHAR")
        cols = ", ".join(f"c{i} {ty}" for i, ty in enumerate(types))
        self.con.execute(f"CREATE TEMP TABLE {handle} ({cols})")
        if rows:
            casted = [[v if v is None or ty != "VARCHAR" else str(v)
                       for v, ty in zip(r, types)] for r in rows]
            self.con.executemany(
                f"INSERT INTO {handle} VALUES ({', '.join('?' * width)})", casted)
        self.handles.handles[handle] = f"temp.main.{handle}"
        return {"handle": handle, "saved_rows": len(rows)}
