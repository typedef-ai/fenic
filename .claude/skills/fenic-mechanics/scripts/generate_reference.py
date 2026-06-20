#!/usr/bin/env python
"""Generate the version-stamped API reference for the `fenic-mechanics` skill.

This is the GENERATED half of the hybrid skill: it introspects the *installed*
fenic package (signatures + first-line docstrings) so the reference can never
drift from / hallucinate the real API. The hand-authored judgment (namespace
law, silent-gap warnings, "did you mean" corrections) lives in SKILL.md and
gotchas.md and is layered on top.

Re-run on every new fenic version (see the `update-fenic-skill` maintenance
skill). Usage:
    uv run --project . python .claude/skills/fenic-mechanics/scripts/generate_reference.py
"""
from __future__ import annotations

import inspect
import re
from importlib import metadata
from pathlib import Path

import fenic as fc


def _clean(s: str) -> str:
    """Sanitize a signature/annotation string.

    Deterministic (no memory addresses) and readable (short type names).
    """
    s = re.sub(r" at 0x[0-9a-fA-F]+", "", s)              # drop memory addresses (determinism!)
    s = re.sub(r"Annotated\[([^\[\],]+(?:\[[^\]]*\])?),[^\]]*\]", r"\1", s)  # Annotated[T, ...] -> T
    s = re.sub(r"fenic\.[\w.]+\.(\w+)", r"\1", s)          # fenic.api.column.Column -> Column
    s = re.sub(r"<class '([\w.]+)'>", lambda m: m.group(1).split(".")[-1], s)  # <class 'int'> -> int
    s = s.replace("typing.", "").replace("NoneType", "None")
    return s

OUT_DIR = Path(__file__).resolve().parent.parent / "reference"

# Function namespaces exposed on the fenic root (fc.<ns>.<fn>).
NAMESPACES = ["text", "json", "markdown", "semantic", "embedding", "dt", "arr"]

# User-facing config / model / example classes worth documenting with their fields.
CONFIG_CLASSES = [
    "Session", "SessionConfig", "SemanticConfig", "CloudConfig",
    "OpenAILanguageModel", "OpenAIEmbeddingModel",
    "AnthropicLanguageModel",
    "GoogleDeveloperLanguageModel", "GoogleDeveloperEmbeddingModel",
    "GoogleVertexLanguageModel", "GoogleVertexEmbeddingModel",
    "CohereEmbeddingModel", "OpenRouterLanguageModel",
    "MapExample", "MapExampleCollection",
    "ClassifyExample", "ClassifyExampleCollection", "ClassDefinition",
    "JoinExample", "JoinExampleCollection",
    "PredicateExample", "PredicateExampleCollection",
    "ToolParam",
]

# Type singletons / constructors.
TYPE_NAMES = [
    "StringType", "IntegerType", "FloatType", "DoubleType", "BooleanType",
    "DateType", "TimestampType", "JsonType", "MarkdownType", "HtmlType",
    "TranscriptType", "DocumentPathType", "EmbeddingType",
    "ArrayType", "StructType", "StructField", "Schema", "ColumnField",
]


def _fenic_defined(obj) -> bool:
    return str(getattr(obj, "__module__", "")).startswith("fenic")


def _first_doc_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _sig(obj) -> str:
    try:
        return _clean(str(inspect.signature(obj)))
    except (ValueError, TypeError):
        return "(...)"


def _render_callable(name: str, obj, prefix: str) -> str:
    return f"- `{prefix}{name}{_sig(obj)}` — {_first_doc_line(obj)}".rstrip(" —")


def render_namespaces() -> str:
    """Render the fc.* function namespaces and top-level free functions."""
    out = ["# fenic functions reference (generated)", ""]
    # Namespaced functions: fc.text.*, fc.json.*, ...
    for ns in NAMESPACES:
        mod = getattr(fc, ns, None)
        if mod is None:
            continue
        fns = []
        for n in sorted(dir(mod)):
            if n.startswith("_"):
                continue
            o = getattr(mod, n)
            if (inspect.isfunction(o) or inspect.isbuiltin(o)) and _fenic_defined(o):
                fns.append(_render_callable(n, o, f"fc.{ns}."))
        if fns:
            out += [f"## fc.{ns}.*", "", *fns, ""]
    # Top-level free functions: fc.col, fc.lit, fc.when, ...
    top = []
    for n in sorted(dir(fc)):
        if n.startswith("_"):
            continue
        o = getattr(fc, n)
        if (inspect.isfunction(o) or inspect.isbuiltin(o)) and _fenic_defined(o):
            top.append(_render_callable(n, o, "fc."))
    if top:
        out += ["## fc.* (top-level functions)", "", *top, ""]
    return "\n".join(out)


def render_dataframe() -> str:
    """Render the public DataFrame and Column methods."""
    from fenic.api.dataframe.dataframe import DataFrame
    out = ["# DataFrame / Column methods reference (generated)", ""]
    for cls_name, cls in [("DataFrame", DataFrame), ("Column", fc.Column)]:
        out += [f"## {cls_name}", ""]
        for n in sorted(dir(cls)):
            if n.startswith("_"):
                continue
            o = inspect.getattr_static(cls, n)
            if callable(o):
                try:
                    sig = _sig(getattr(cls, n))
                except Exception:
                    sig = "(...)"
                out.append(f"- `{cls_name}.{n}{sig}` — {_first_doc_line(getattr(cls, n))}".rstrip(" —"))
        out.append("")
    return "\n".join(out)


def render_config_types() -> str:
    """Render config/model/example classes (with fields) and the data types."""
    out = ["# Config, models, examples & types reference (generated)", "", "## Classes", ""]
    for name in CONFIG_CLASSES:
        cls = getattr(fc, name, None)
        if cls is None:
            continue
        fields = getattr(cls, "model_fields", None)
        if fields:  # pydantic model — list fields + types
            parts = []
            for fname, finfo in fields.items():
                ann = _clean(str(getattr(finfo, "annotation", "")))
                if len(ann) > 160:  # cap huge Literal[...] model-name lists
                    ann = ann[:157] + "...]"
                req = "" if finfo.is_required() else " (optional)"
                parts.append(f"{fname}: {ann}{req}")
            out.append(f"- `fc.{name}` — fields: {', '.join(parts)}")
        else:
            out.append(f"- `fc.{name}{_sig(cls)}` — {_first_doc_line(cls)}".rstrip(" —"))
    out += ["", "## Data types", ""]
    for name in TYPE_NAMES:
        obj = getattr(fc, name, None)
        if obj is None:
            continue
        kind = "singleton instance" if not inspect.isclass(obj) else "constructor"
        out.append(f"- `fc.{name}` ({kind})")
    return "\n".join(out)


def main() -> None:
    """Generate the reference markdown files and the VERSION stamp."""
    version = metadata.version("fenic")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        f"<!-- GENERATED by generate_reference.py against fenic=={version}. "
        f"Do not edit by hand; re-run the generator. -->\n"
        f"<!-- fenic version: {version} -->\n\n"
    )
    files = {
        "functions.md": render_namespaces(),
        "dataframe.md": render_dataframe(),
        "config-and-types.md": render_config_types(),
    }
    for fname, body in files.items():
        (OUT_DIR / fname).write_text(header + body + "\n")
        print(f"wrote {OUT_DIR / fname}  ({len(body.splitlines())} lines)")
    # Version stamp for the maintenance skill to diff against.
    (OUT_DIR / "VERSION").write_text(version + "\n")
    print(f"fenic version: {version}")


if __name__ == "__main__":
    main()
