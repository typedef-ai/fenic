"""MkDocs hook that supplies concise, page-specific meta descriptions."""

from __future__ import annotations

import html
import re
from typing import Any

_MAX_DESCRIPTION_LENGTH = 160
_MIN_DESCRIPTION_LENGTH = 30
_REFERENCE_PREFIX = "reference/"

_HTML_BLOCK_RE = re.compile(
    r"<(?P<tag>div|picture|p)\b[^>]*>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)]\([^)]*\)")
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_ORDERED_LIST_RE = re.compile(r"\d+[.)]\s")


def _plain_text(value: str) -> str:
    value = _IMAGE_RE.sub("", value)
    value = _LINK_RE.sub(r"\1", value)
    value = _TAG_RE.sub(" ", value)
    value = value.replace("`", "").replace("*", "").replace("_", "")
    return _WHITESPACE_RE.sub(" ", html.unescape(value)).strip()


def _truncate(value: str) -> str:
    if len(value) <= _MAX_DESCRIPTION_LENGTH:
        return value

    shortened = value[: _MAX_DESCRIPTION_LENGTH - 1].rsplit(" ", 1)[0]
    return f"{shortened}…"


def description_from_markdown(markdown: str) -> str | None:
    """Return the first useful prose paragraph from a Markdown document."""
    markdown = _HTML_COMMENT_RE.sub("", markdown)
    markdown = _HTML_BLOCK_RE.sub("", markdown)

    if markdown.startswith("---"):
        _, separator, remainder = markdown[3:].partition("\n---")
        if separator:
            markdown = remainder.lstrip("\n")

    paragraph: list[str] = []
    in_fence = False

    for raw_line in markdown.splitlines():
        line = raw_line.strip()

        if line.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if not line:
            if paragraph:
                candidate = _plain_text(" ".join(paragraph))
                if len(candidate) >= _MIN_DESCRIPTION_LENGTH:
                    return _truncate(candidate)
                paragraph = []
            continue

        if line.startswith(("#", "[![", "![", "<", "|", ":::")):
            continue
        if line in {"---", "***", "___"}:
            continue
        if line.startswith(("- ", "* ", "+ ", "> ")) or _ORDERED_LIST_RE.match(line):
            continue

        paragraph.append(line)

    if paragraph:
        candidate = _plain_text(" ".join(paragraph))
        if len(candidate) >= _MIN_DESCRIPTION_LENGTH:
            return _truncate(candidate)
    return None


def on_page_markdown(markdown: str, *, page: Any, **_: Any) -> str:
    """Populate ``page.meta.description`` while leaving Markdown unchanged."""
    if page.meta.get("description"):
        return markdown

    src_uri = page.file.src_uri
    if src_uri.startswith(_REFERENCE_PREFIX):
        symbol = src_uri.removeprefix(_REFERENCE_PREFIX).removesuffix(".md")
        symbol = symbol.removesuffix("/index").replace("/", ".")
        page.meta["description"] = _truncate(
            f"API reference for {symbol} in fenic, including its public "
            "classes, functions, parameters, and return types."
        )
        return markdown

    description = description_from_markdown(markdown)
    if description:
        page.meta["description"] = description
    return markdown
