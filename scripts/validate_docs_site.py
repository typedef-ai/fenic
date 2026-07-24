#!/usr/bin/env python3
"""Validate crawler and LLM artifacts emitted by the MkDocs build."""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

_CANONICAL_ROOT = "https://docs.fenic.ai/latest/"
_MARKDOWN_LINK_RE = re.compile(
    r"^- \[[^\]]+]\((https://docs\.fenic\.ai/latest/[^)]+)\)"
)


class _HeadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonical: str | None = None
        self.description: str | None = None
        self.markdown_alternate: str | None = None
        self.robots: str | None = None
        self.json_ld: list[dict[str, object]] = []
        self._in_json_ld = False
        self._json_ld_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonical = attributes.get("href")
        if (
            tag == "link"
            and attributes.get("rel") == "alternate"
            and attributes.get("type") == "text/markdown"
        ):
            self.markdown_alternate = attributes.get("href")
        if tag == "meta" and attributes.get("name") == "description":
            self.description = attributes.get("content")
        if tag == "meta" and attributes.get("name") == "robots":
            self.robots = attributes.get("content")
        if tag == "script" and attributes.get("type") == "application/ld+json":
            self._in_json_ld = True
            self._json_ld_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_ld_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_json_ld:
            self._in_json_ld = False
            self.json_ld.append(json.loads("".join(self._json_ld_parts)))


def _fail(message: str) -> None:
    raise SystemExit(f"docs validation failed: {message}")


def _parse_head(path: Path) -> _HeadParser:
    parser = _HeadParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def _site_path_for_url(site_dir: Path, url: str) -> Path:
    relative = urlparse(url).path.removeprefix("/latest/").strip("/")
    return site_dir / relative / "index.html" if relative else site_dir / "index.html"


def _markdown_path_for_url(site_dir: Path, url: str) -> Path:
    relative = urlparse(url).path.removeprefix("/latest/")
    return site_dir / relative


def main() -> None:
    """Validate the generated site directory passed on the command line."""
    site_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "site").resolve()
    if not site_dir.is_dir():
        _fail(f"site directory does not exist: {site_dir}")

    sitemap_path = site_dir / "sitemap.xml"
    root = ET.parse(sitemap_path).getroot()
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [node.text for node in root.findall("s:url/s:loc", namespace) if node.text]
    if not urls:
        _fail("sitemap.xml contains no URLs")
    if any(not url.startswith(_CANONICAL_ROOT) for url in urls):
        _fail("sitemap.xml contains a URL outside the canonical latest path")
    if any("/plans/" in url or "/td-flow/" in url for url in urls):
        _fail("sitemap.xml contains internal planning documents")

    expected_urls = {
        _CANONICAL_ROOT,
        f"{_CANONICAL_ROOT}topics/fenic-mcp/",
        f"{_CANONICAL_ROOT}reference/fenic/api/functions/semantic/",
    }
    missing_urls = expected_urls.difference(urls)
    if missing_urls:
        _fail(f"sitemap.xml is missing: {', '.join(sorted(missing_urls))}")

    descriptions: Counter[str] = Counter()
    for expected_canonical in urls:
        page_path = _site_path_for_url(site_dir, expected_canonical)
        if not page_path.is_file():
            _fail(f"sitemap URL has no generated page: {expected_canonical}")

        parsed = _parse_head(page_path)
        if parsed.canonical != expected_canonical:
            _fail(f"{page_path} has canonical {parsed.canonical!r}")
        if not parsed.description or len(parsed.description) < 30:
            _fail(f"{page_path} has no useful meta description")
        descriptions[parsed.description] += 1
        if not parsed.markdown_alternate:
            _fail(f"{page_path} has no Markdown alternate")
        if not _markdown_path_for_url(
            site_dir,
            parsed.markdown_alternate,
        ).is_file():
            _fail(f"{page_path} has a broken Markdown alternate")
        if not parsed.json_ld:
            _fail(f"{page_path} has no JSON-LD structured data")

    duplicate_descriptions = [
        description for description, count in descriptions.items() if count > 1
    ]
    if duplicate_descriptions:
        _fail("multiple sitemap pages share the same meta description")

    for internal_page in (
        site_dir / "plans/mcp-search-literal-mode-structure/index.html",
        site_dir / "td-flow/create-dataframe-schema/design/index.html",
        site_dir / "td-flow/create-dataframe-schema/plan/index.html",
        site_dir / "td-flow/create-dataframe-schema/research/index.html",
        site_dir / "td-flow/create-dataframe-schema/structure/index.html",
    ):
        parsed = _parse_head(internal_page)
        if parsed.robots != "noindex, follow":
            _fail(f"{internal_page} is not marked noindex, follow")
        if parsed.markdown_alternate:
            _fail(f"{internal_page} advertises a Markdown alternate")

    required_files = {
        "llms.txt": 500,
        "llms-full.txt": 10_000,
        "agents/index.md": 1_000,
        "topics/fenic-mcp/index.md": 1_000,
        "reference/fenic/api/functions/semantic/index.md": 1_000,
    }
    for relative_path, minimum_size in required_files.items():
        path = site_dir / relative_path
        if not path.is_file() or path.stat().st_size < minimum_size:
            _fail(f"{relative_path} is missing or unexpectedly small")

    llms_index = (site_dir / "llms.txt").read_text(encoding="utf-8")
    if _CANONICAL_ROOT not in llms_index:
        _fail("llms.txt does not link to the canonical latest docs")
    for line in llms_index.splitlines():
        match = _MARKDOWN_LINK_RE.match(line)
        if match and not _markdown_path_for_url(site_dir, match.group(1)).is_file():
            _fail(f"llms.txt contains a broken link: {match.group(1)}")

    robots = (site_dir / "robots.txt").read_text(encoding="utf-8")
    expected_sitemap = f"Sitemap: {_CANONICAL_ROOT}sitemap.xml"
    if expected_sitemap not in robots:
        _fail("robots.txt does not reference the canonical sitemap")

    print(
        f"Validated {len(urls)} sitemap URLs, canonical metadata, "
        "Markdown variants, and LLM assets."
    )


if __name__ == "__main__":
    main()
