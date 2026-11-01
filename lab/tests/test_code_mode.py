"""Tests for the code-execution modality."""
from __future__ import annotations

from pathlib import Path

from dal_lab.code_mode import CodeFacade
from dal_lab.facades import HandleFacade


def test_code_mode_returns_only_printed_output(con, catalog):
    fac = CodeFacade(con, catalog, HandleFacade(con, catalog))
    out = fac.run_python("rows = sql('SELECT * FROM customer')\nprint(len(rows))")
    assert out["stdout"].strip() == "1500" and "handle" not in out


def test_code_mode_saves_large_answer_as_handle(con, catalog):
    handles = HandleFacade(con, catalog)
    fac = CodeFacade(con, catalog, handles)
    out = fac.run_python("save_answer(sql('SELECT c_custkey, c_acctbal FROM customer'))")
    table = handles.table_of(out["handle"])
    assert out["saved_rows"] == 1500
    assert con.execute(f"SELECT count(*), typeof(any_value(c0)) FROM {table}").fetchone() \
        == (1500, "BIGINT")


def test_code_mode_reports_errors_and_blocks_network(con, catalog):
    fac = CodeFacade(con, catalog, HandleFacade(con, catalog))
    out = fac.run_python("import urllib.request\n"
                         "urllib.request.urlopen('https://example.com', timeout=3)")
    assert "stderr" in out


def test_code_mode_blocks_writes_outside_temp_dir(con, catalog):
    # Not tmp_path: the macOS profile allows the system temp area by design.
    target = Path(__file__).resolve().parent / ".sandbox_escape_test.txt"
    target.unlink(missing_ok=True)
    try:
        out = CodeFacade(con, catalog, HandleFacade(con, catalog)).run_python(
            f"open({str(target)!r}, 'w').write('x')")
        assert "stderr" in out
        assert not target.exists()
    finally:
        target.unlink(missing_ok=True)


def test_code_mode_refuses_without_sandbox(con, catalog, monkeypatch):
    monkeypatch.setattr("dal_lab.code_mode.shutil.which", lambda name: None)
    out = CodeFacade(con, catalog, HandleFacade(con, catalog)).run_python("print(1)")
    assert "no sandbox" in out["error"]


def test_code_mode_keeps_decimal_answers_numeric(con, catalog):
    handles = HandleFacade(con, catalog)
    out = CodeFacade(con, catalog, handles).run_python(
        "save_answer(sql('SELECT sum(o_totalprice) AS t FROM orders'))")
    table = handles.table_of(out["handle"])
    got = con.execute(f"SELECT * FROM {table}").fetchall()
    gold = con.execute("SELECT sum(o_totalprice) FROM orders").fetchone()[0]
    assert isinstance(got[0][0], float)
    assert abs(got[0][0] - float(gold)) < 0.01


def test_code_mode_flags_silent_script(con, catalog):
    out = CodeFacade(con, catalog, HandleFacade(con, catalog)).run_python(
        "sql('SELECT 1')")
    assert out["stdout"] == "" and "printed nothing" in out["note"]


def test_code_mode_can_use_pandas(con, catalog):
    out = CodeFacade(con, catalog, HandleFacade(con, catalog)).run_python(
        "import pandas as pd\nprint(len(pd.DataFrame(sql('SELECT * FROM region'))))")
    assert out["stdout"].strip() == "5"
