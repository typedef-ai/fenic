"""`fenic check` — validate a fenic script WITHOUT executing it.

Two passes, both designed to spend no tokens and (for non-semantic scripts) need
no API key:

  1. lint   — static symbol/namespace resolution against the installed fenic
              (catches the dominant gaps: `fenic.functions`, `fc.explode`,
              `fc.array.size` vs `fc.arr.size`, internal imports). No code is run.
  2. validate — build the logical plan(s) by exec'ing the script with all
              *action* methods (show/collect/count/to_*/write.*/lineage) stubbed
              to capture `.schema` instead of materializing. fenic validates
              eagerly at plan construction (LogicalPlan.__init__), so type/column
              errors surface with no execution and no data.

A script that *configures semantic models* still triggers fenic's live
session-creation key check, so that path needs a provider key in the env;
purely structural / non-semantic scripts validate with no secrets.

Emits JSON: {ok, findings: [...], schemas: [...]}.
"""
from __future__ import annotations

import ast
import contextlib
import difflib
import inspect
import io
import json
import types
from pathlib import Path
from typing import Any, Optional

# Namespaces & action methods (kept in sync with the skill's reference).
_ACTIONS = ["show", "collect", "count", "to_polars", "to_pandas", "to_arrow",
            "to_pydict", "to_pylist", "lineage"]
_WRITER_ACTIONS = ["save_as_table", "parquet", "csv", "json"]

_SUGGEST = {
    "explode": "`explode`/`unnest` are DataFrame methods — `df.explode('col')`, not `fc.explode`.",
    "unnest": "`unnest` is a DataFrame method — `df.unnest('col')`, not `fc.unnest`.",
}


def _finding(stage: str, severity: str, etype: str, message: str,
             symbol: str = "", suggestion: str = "", line: Optional[int] = None) -> dict:
    return {"stage": stage, "severity": severity, "error_type": etype,
            "message": message, "symbol": symbol, "suggestion": suggestion, "line": line}


def _attr_chain(node: ast.Attribute) -> Optional[list[str]]:
    """Return the dotted parts of a pure Name.attr[.attr] chain, else None."""
    parts: list[str] = []
    cur: Any = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return None


def _closest(name: str, options) -> str:
    m = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f" Did you mean `{m[0]}`?" if m else ""


def lint(src: str, path: str) -> list[dict]:
    """Statically resolve fenic symbol/namespace usage; return diagnostic findings."""
    import fenic as fc

    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return [_finding("lint", "error", "SyntaxError", str(e), line=e.lineno)]

    findings: list[dict] = []
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name == "fenic":
                    aliases.add(n.asname or "fenic")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "fenic" and any(n.name == "functions" for n in node.names):
                findings.append(_finding(
                    "lint", "error", "BadImport",
                    "`from fenic import functions` — there is no `fenic.functions` submodule.",
                    "fenic.functions",
                    "Use `import fenic as fc`; functions live on `fc.text/json/markdown/semantic/arr/dt/embedding`.",
                    node.lineno))
            elif mod.startswith("fenic.api") or mod.startswith("fenic.core"):
                findings.append(_finding(
                    "lint", "warning", "InternalImport",
                    f"Importing from internal module `{mod}` couples to private layout.",
                    mod, "Prefer the public surface: `import fenic as fc`; `fc.<Symbol>`.",
                    node.lineno))

    public = [n for n in dir(fc) if not n.startswith("_")]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attr_chain(node)
        if not chain or len(chain) < 2 or chain[0] not in aliases:
            continue
        a = chain[1]
        if not hasattr(fc, a):
            sug = _SUGGEST.get(a, "") or (f"`fc.{a}` does not exist." + _closest(a, public))
            findings.append(_finding("lint", "error", "UnknownSymbol",
                                     f"`fc.{a}` is not a member of fenic.", f"fc.{a}", sug, node.lineno))
            continue
        if len(chain) >= 3:
            obj = getattr(fc, a)
            b = chain[2]
            if isinstance(obj, types.ModuleType):
                if not hasattr(obj, b):
                    opts = [n for n in dir(obj) if not n.startswith("_")]
                    findings.append(_finding("lint", "error", "UnknownSymbol",
                                             f"`fc.{a}.{b}` does not exist.", f"fc.{a}.{b}",
                                             _closest(b, opts).strip(), node.lineno))
            elif (inspect.isfunction(obj) or inspect.isbuiltin(obj)) and not hasattr(obj, b):
                sug = f"`fc.arr.{b}`?" if a == "array" and hasattr(getattr(fc, "arr", None), b) else ""
                findings.append(_finding("lint", "error", "BadNamespace",
                                         f"`fc.{a}` is a function, not a namespace — `.{b}` won't resolve.",
                                         f"fc.{a}.{b}", sug, node.lineno))

    # dedupe
    seen, out = set(), []
    for f in findings:
        k = (f["symbol"], f["message"])
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def validate(src: str, path: str) -> dict:
    """Exec the script with actions stubbed; let eager plan construction validate."""
    import fenic as fc
    from fenic.api.dataframe.dataframe import DataFrame
    DataFrameWriter = fc.DataFrameWriter

    schemas: list[dict] = []
    originals: list[tuple] = []

    def _capture(name):
        def f(self, *a, **k):
            try:
                schemas.append({"at": name, "schema": str(self.schema)})
            except Exception:
                pass
            return None
        return f

    def _noop(self, *a, **k):
        return None

    for m in _ACTIONS:
        if hasattr(DataFrame, m):
            originals.append((DataFrame, m, getattr(DataFrame, m)))
            setattr(DataFrame, m, _capture(m))
    for m in _WRITER_ACTIONS:
        if hasattr(DataFrameWriter, m):
            originals.append((DataFrameWriter, m, getattr(DataFrameWriter, m)))
            setattr(DataFrameWriter, m, _noop)

    finding = None
    # __name__ must be "__main__" so the conventional `if __name__ == "__main__": main()`
    # guard fires and the pipeline actually builds (fenic examples use this pattern).
    ns = {"__name__": "__main__", "__file__": path}
    _sink = io.StringIO()  # swallow the user script's own stdout/stderr so our JSON is clean
    try:
        compiled = compile(src, path, "exec")
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            exec(compiled, ns)
    except BaseException as e:  # noqa: BLE001 — we classify and report
        emod = type(e).__module__ or ""
        es = str(e)
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            finding = _finding("validate", "error", type(e).__name__, es, suggestion="Use the public `fc.` surface.")
        elif "NoneType" in es and schemas:
            # plan(s) built fine; this is the script using a stubbed action's return.
            finding = None
        elif emod.startswith("fenic") or emod.startswith("pydantic"):
            finding = _finding("validate", "error", type(e).__name__, es)
        elif isinstance(e, AttributeError):
            finding = _finding("validate", "error", "AttributeError", es,
                               suggestion="Check the symbol/namespace against `fenic check`'s lint or the reference.")
        else:
            finding = _finding("validate", "error", type(e).__name__, es)
    finally:
        # Also capture the schema of any top-level DataFrame variable (covers
        # scripts that build a frame without calling a captured action).
        for vname, val in list(ns.items()):
            if isinstance(val, DataFrame):
                try:
                    schemas.append({"at": f"var:{vname}", "schema": str(val.schema)})
                except Exception:
                    pass
        for cls, name, fn in originals:
            setattr(cls, name, fn)

    # dedupe by schema text (action-capture and var-scan can overlap)
    seen, deduped = set(), []
    for s in schemas:
        if s["schema"] not in seen:
            seen.add(s["schema"])
            deduped.append(s)
    return {"finding": finding, "schemas": deduped}


def check_source(src: str, path: str = "<stdin>") -> dict:
    """Lint and dry-run-validate a fenic script; return {ok, findings, schemas}."""
    findings = lint(src, path)
    v = validate(src, path)
    if v["finding"]:
        findings.append(v["finding"])
    ok = not any(f["severity"] == "error" for f in findings)
    return {"ok": ok, "path": path, "findings": findings, "schemas": v["schemas"]}


def run(path: Optional[str]) -> int:
    """Read a script (file path or stdin), check it, print JSON, and return an exit code."""
    import sys
    src = sys.stdin.read() if not path or path == "-" else Path(path).read_text()
    result = check_source(src, path or "<stdin>")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1
