"""Tests for `fenic check` (`fenic.scripts.fenic_check`) — a static, non-executing lint.

Each known agent-failure mode (from the gap catalog) maps to a lint finding; clean
scripts produce none. `fenic check` never executes the script — a property the
last test pins down explicitly.
"""
from __future__ import annotations

import pathlib
import tempfile

from fenic.scripts.fenic_check import check_source, lint


def _finding(result, **kw):
    """Return the first finding whose named fields all contain the given substrings."""
    for f in result["findings"]:
        if all(str(v).lower() in str(f.get(k, "")).lower() for k, v in kw.items()):
            return f
    return None


# --- lint: static symbol/namespace resolution (no session/key/exec) ----------

def test_lint_flags_fenic_functions_import():
    findings = lint("from fenic import functions as F\n", "t.py")
    assert any(f["error_type"] == "BadImport" and "fenic.functions" in f["symbol"] for f in findings)


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


# --- check_source: ok/not-ok, and it must NOT execute the script --------------

def test_check_source_ok_on_clean_script():
    r = check_source("import fenic as fc\nfc.arr.size(fc.col('tags'))\n", "good.py")
    assert r["ok"] is True
    assert r["findings"] == []


def test_check_source_not_ok_on_bad_namespace():
    r = check_source("import fenic as fc\nfc.array.size(fc.col('tags'))\n", "bad.py")
    assert r["ok"] is False
    assert _finding(r, error_type="BadNamespace") is not None


def test_check_source_does_not_execute_the_script():
    # `fenic check` is static-only: top-level side effects must never run.
    sentinel = pathlib.Path(tempfile.gettempdir()) / "fenic_check_should_not_exist.txt"
    sentinel.unlink(missing_ok=True)
    src = f"import fenic as fc\nopen({str(sentinel)!r}, 'w').write('x')\n"
    check_source(src, "side_effect.py")
    assert not sentinel.exists()
