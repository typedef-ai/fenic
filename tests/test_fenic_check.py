"""Tests for `fenic check` (`fenic.scripts.fenic_check`).

Each known agent-failure mode (from the gap catalog) maps to a lint or validate
diagnostic; good scripts validate clean and report a result schema — with no API
key and no materialization. These guard against regressions such as the
`__name__` false-pass bug (a `def main(): ... if __name__ == "__main__"` script
that silently did nothing and reported ok).
"""
from __future__ import annotations

import textwrap

from fenic.scripts.fenic_check import check_source, lint


def _finding(result, **kw):
    """Return the first finding whose named fields all contain the given
    (case-insensitive) substrings, else None."""
    for f in result["findings"]:
        if all(str(v).lower() in str(f.get(k, "")).lower() for k, v in kw.items()):
            return f
    return None


# --- lint: static symbol/namespace resolution (no session, no key, no exec) ---

def test_lint_flags_fenic_functions_import():
    findings = lint("from fenic import functions as F\n", "t.py")
    assert any(f["error_type"] == "BadImport" and "fenic.functions" in f["symbol"]
               for f in findings)


def test_lint_flags_fc_array_vs_arr():
    findings = lint("import fenic as fc\nfc.array.size(fc.col('x'))\n", "t.py")
    f = next((f for f in findings if f["symbol"] == "fc.array.size"), None)
    assert f is not None
    assert f["error_type"] == "BadNamespace"
    assert "arr" in f["suggestion"]


def test_lint_flags_fc_explode_not_a_function():
    findings = lint("import fenic as fc\nfc.explode(fc.col('x'))\n", "t.py")
    f = next((f for f in findings if f["symbol"] == "fc.explode"), None)
    assert f is not None
    assert f["error_type"] == "UnknownSymbol"


def test_lint_flags_unknown_namespaced_symbol():
    findings = lint("import fenic as fc\nfc.text.no_such_fn('x')\n", "t.py")
    assert any(f["symbol"] == "fc.text.no_such_fn" for f in findings)


def test_lint_flags_internal_import():
    findings = lint("from fenic.api.session.config import SessionConfig\n", "t.py")
    assert any(f["error_type"] == "InternalImport" for f in findings)


def test_lint_clean_on_valid_namespaces():
    src = "import fenic as fc\nfc.arr.size(fc.col('t'))\nfc.json.jq(fc.col('j'), '.a')\n"
    assert lint(src, "t.py") == []


# --- validate: dry-run plan construction (keyless; no materialization) --------

def _good_src(tmp_path) -> str:
    return textwrap.dedent(f"""
        import fenic as fc
        s = fc.Session.get_or_create(fc.SessionConfig(app_name="chk_good", db_path={str(tmp_path)!r}))
        df = s.create_dataframe({{"tags": [["a", "b"], ["c"]]}})
        df = df.select(fc.arr.size(fc.col("tags")).alias("n"), fc.col("tags"))
        df.show()
    """)


def test_validate_good_script_ok_with_schema(tmp_path):
    r = check_source(_good_src(tmp_path), "good.py")
    assert r["ok"] is True
    assert r["findings"] == []
    assert any("ColumnField(name='n'" in s["schema"] for s in r["schemas"])


def test_validate_catches_unknown_column(tmp_path):
    src = textwrap.dedent(f"""
        import fenic as fc
        s = fc.Session.get_or_create(fc.SessionConfig(app_name="chk_badcol", db_path={str(tmp_path)!r}))
        df = s.create_dataframe({{"a": [1, 2]}})
        df.select(fc.col("nope")).show()
    """)
    r = check_source(src, "badcol.py")
    assert r["ok"] is False
    assert _finding(r, stage="validate", message="not found") is not None


def test_validate_runs_main_guard(tmp_path):
    """Regression: `__name__` must be "__main__" so the conventional main() guard
    fires. Previously this script reported ok:true without building anything."""
    src = textwrap.dedent(f"""
        import fenic as fc
        def main():
            s = fc.Session.get_or_create(fc.SessionConfig(app_name="chk_main", db_path={str(tmp_path)!r}))
            df = s.create_dataframe({{"a": [1]}})
            df.select(fc.col("nope")).show()
        if __name__ == "__main__":
            main()
    """)
    r = check_source(src, "main.py")
    assert r["ok"] is False
    assert _finding(r, message="not found") is not None


def test_check_source_reports_import_error_in_validate():
    r = check_source("from fenic import functions as F\nF.explode('x')\n", "bad.py")
    assert r["ok"] is False
    # caught both by lint (BadImport) and validate (ImportError)
    assert _finding(r, error_type="BadImport") is not None
