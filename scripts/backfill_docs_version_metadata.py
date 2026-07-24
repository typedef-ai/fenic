#!/usr/bin/env python3
"""Canonicalize legacy Mike documentation pages to the latest version."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SITE_ROOT = "https://docs.fenic.ai/latest/"
_CANONICAL_RE = re.compile(
    r"""<link\b(?=[^>]*\brel=["']canonical["'])[^>]*>""",
    flags=re.IGNORECASE,
)
_NOINDEX_RE = re.compile(
    r"""<meta\b(?=[^>]*\bname=["']robots["'])"""
    r"""(?=[^>]*\bcontent=["'][^"']*\bnoindex\b[^"']*["'])[^>]*>""",
    flags=re.IGNORECASE,
)
_BACKFILL_NOINDEX_RE = re.compile(
    r"""\s*<meta name="robots" content="noindex, follow" """
    r"""data-fenic-version-backfill>\s*""",
    flags=re.IGNORECASE,
)
_HEAD_END_RE = re.compile(r"</head>", flags=re.IGNORECASE)


def _insert_before_head_end(document: str, tag: str) -> str:
    match = _HEAD_END_RE.search(document)
    if not match:
        raise ValueError("HTML document has no closing head tag")
    return f"{document[: match.start()]}  {tag}\n{document[match.start() :]}"


def _canonical_url(relative_path: Path) -> str:
    parent = relative_path.parent.as_posix()
    if parent == ".":
        return _SITE_ROOT
    return f"{_SITE_ROOT}{parent}/"


def _set_canonical(document: str, canonical_url: str) -> str:
    canonical_tag = f'<link rel="canonical" href="{canonical_url}">'
    document = _BACKFILL_NOINDEX_RE.sub("\n", document)
    if _CANONICAL_RE.search(document):
        return _CANONICAL_RE.sub(canonical_tag, document, count=1)
    return _insert_before_head_end(document, canonical_tag)


def _set_noindex(document: str) -> str:
    noindex_tag = (
        '<meta name="robots" content="noindex, follow" data-fenic-version-backfill>'
    )
    document = _CANONICAL_RE.sub("", document)
    if _NOINDEX_RE.search(document):
        return document
    return _insert_before_head_end(document, noindex_tag)


def _version_directories(root: Path) -> list[Path]:
    versions_path = root / "versions.json"
    versions = json.loads(versions_path.read_text(encoding="utf-8"))
    return [
        root / entry["version"]
        for entry in versions
        if isinstance(entry, dict) and isinstance(entry.get("version"), str)
    ]


def backfill(root: Path) -> int:
    """Update legacy page metadata under a checked-out Mike deployment."""
    latest = root / "latest"
    if not latest.is_dir():
        raise ValueError(f"Latest documentation alias is missing: {latest}")

    changed = 0
    for version_dir in _version_directories(root):
        if not version_dir.is_dir():
            continue

        for page_path in version_dir.rglob("index.html"):
            relative_path = page_path.relative_to(version_dir)
            latest_page = latest / relative_path
            document = page_path.read_text(encoding="utf-8")

            if latest_page.is_file():
                latest_document = latest_page.read_text(encoding="utf-8")
                if _NOINDEX_RE.search(latest_document):
                    updated = _set_noindex(document)
                else:
                    updated = _set_canonical(
                        document,
                        _canonical_url(relative_path),
                    )
            else:
                updated = _set_noindex(document)

            if updated != document:
                page_path.write_text(updated, encoding="utf-8")
                changed += 1

    return changed


def main() -> None:
    """Backfill a deployment directory supplied on the command line."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: backfill_docs_version_metadata.py <gh-pages-worktree>")

    root = Path(sys.argv[1]).resolve()
    if not (root / ".git").exists():
        raise SystemExit(f"not a Git worktree: {root}")

    changed = backfill(root)
    print(f"Updated metadata for {changed} legacy documentation pages.")


if __name__ == "__main__":
    main()
