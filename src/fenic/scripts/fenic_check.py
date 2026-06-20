"""`fenic check` — statically lint a fenic script's symbol/namespace usage.

It resolves every `fc.<...>` reference against the installed fenic API and flags
the common mistakes: no `fenic.functions`; `fc.array` (a constructor) vs the
`fc.arr` ops namespace; `fc.explode` (a DataFrame method, not a function);
imports from internal modules; and unknown symbols. The script is **not
executed** — purely static analysis over the AST. Emits JSON: {ok, findings}.

Type/column errors that only surface at plan construction are intentionally out
of scope: catching those safely needs first-class dry-run support in fenic (an
execute/session-boundary mode that builds plans without materializing or
mutating the catalog), which doesn't exist yet. Until then `fenic check` stays a
non-executing lint.
"""
from __future__ import annotations

import ast
import difflib
import inspect
import json
import types
from pathlib import Path
from typing import Any, Optional

_SUGGEST = {
    "explode": "`explode`/`unnest` are DataFrame methods — `df.explode('col')`, not `fc.explode`.",
    "unnest": "`unnest` is a DataFrame method — `df.unnest('col')`, not `fc.unnest`.",
}


def _finding(severity: str, etype: str, message: str,
             symbol: str = "", suggestion: str = "", line: Optional[int] = None) -> dict:
    return {"severity": severity, "error_type": etype, "message": message,
            "symbol": symbol, "suggestion": suggestion, "line": line}


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
        return [_finding("error", "SyntaxError", str(e), line=e.lineno)]

    findings: list[dict] = []
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name == "fenic":
                    aliases.add(n.asname or "fenic")
                elif n.name == "fenic.functions" or n.name.startswith("fenic.functions."):
                    findings.append(_finding(
                        "error", "BadImport",
                        "`import fenic.functions` — there is no `fenic.functions` submodule.",
                        n.name,
                        "Use `import fenic as fc`; functions live on `fc.text/json/markdown/semantic/arr/dt/embedding`.",
                        node.lineno))
                elif n.name.startswith("fenic.api") or n.name.startswith("fenic.core"):
                    findings.append(_finding(
                        "warning", "InternalImport",
                        f"Importing internal module `{n.name}` couples to private layout.",
                        n.name, "Prefer the public surface: `import fenic as fc`.", node.lineno))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "fenic" and any(n.name == "functions" for n in node.names):
                findings.append(_finding(
                    "error", "BadImport",
                    "`from fenic import functions` — there is no `fenic.functions` submodule.",
                    "fenic.functions",
                    "Use `import fenic as fc`; functions live on `fc.text/json/markdown/semantic/arr/dt/embedding`.",
                    node.lineno))
            elif mod.startswith("fenic.api") or mod.startswith("fenic.core"):
                findings.append(_finding(
                    "warning", "InternalImport",
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
            findings.append(_finding("error", "UnknownSymbol",
                                     f"`fc.{a}` is not a member of fenic.", f"fc.{a}", sug, node.lineno))
            continue
        if len(chain) >= 3:
            obj = getattr(fc, a)
            b = chain[2]
            if isinstance(obj, types.ModuleType):
                if not hasattr(obj, b):
                    opts = [n for n in dir(obj) if not n.startswith("_")]
                    findings.append(_finding("error", "UnknownSymbol",
                                             f"`fc.{a}.{b}` does not exist.", f"fc.{a}.{b}",
                                             _closest(b, opts).strip(), node.lineno))
            elif (inspect.isfunction(obj) or inspect.isbuiltin(obj)) and not hasattr(obj, b):
                sug = f"`fc.arr.{b}`?" if a == "array" and hasattr(getattr(fc, "arr", None), b) else ""
                findings.append(_finding("error", "BadNamespace",
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


def check_source(src: str, path: str = "<stdin>") -> dict:
    """Lint a fenic script (no execution); return {ok, findings}."""
    findings = lint(src, path)
    ok = not any(f["severity"] == "error" for f in findings)
    return {"ok": ok, "path": path, "findings": findings}


def run(path: Optional[str]) -> int:
    """Read a script (file path or stdin), lint it, print JSON, and return an exit code."""
    import sys
    src = sys.stdin.read() if not path or path == "-" else Path(path).read_text()
    result = check_source(src, path or "<stdin>")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1
