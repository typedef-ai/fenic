"""Generate LLM-oriented Markdown artifacts from rendered MkDocs pages."""

from __future__ import annotations

import fnmatch
import html
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import ATX, MarkdownConverter

_LOGGER = logging.getLogger("mkdocs.plugins.docs_llms")


@dataclass(frozen=True)
class _MarkdownPage:
    title: str
    output_path: Path
    url: str
    content: str


_base_url = ""
_description = ""
_full_output = "llms-full.txt"
_sections: dict[str, dict[str, str]] = {}
_selected_uris: set[str] = set()
_pages: dict[str, _MarkdownPage] = {}


def _language_callback(tag: Tag) -> str:
    parents = (tag.parent,) if isinstance(tag.parent, Tag) else ()
    for element in chain((tag,), parents):
        for css_class in element.get("class") or ():
            if isinstance(css_class, str) and css_class.startswith("language-"):
                return css_class.removeprefix("language-")
    return ""


_CONVERTER = MarkdownConverter(
    bullets="-",
    code_language_callback=_language_callback,
    escape_underscores=False,
    heading_style=ATX,
)


def _expand_inputs(
    inputs: Sequence[str | Mapping[str, str]],
    page_uris: Sequence[str],
) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for item in inputs:
        if isinstance(item, Mapping):
            if len(item) != 1:
                raise ValueError("Each llms page mapping must contain one path")
            source_uri, description = next(iter(item.items()))
        else:
            source_uri, description = item, ""

        if "*" in source_uri:
            for match in fnmatch.filter(page_uris, source_uri):
                expanded[match] = description
        else:
            expanded[source_uri] = description
    return expanded


def _remove_noise(soup: BeautifulSoup) -> None:
    for element in soup.find_all(("img", "svg")):
        parent = element.parent
        if (
            isinstance(parent, Tag)
            and parent.name == "a"
            and not parent.get_text(strip=True)
        ):
            parent.decompose()
        else:
            element.decompose()

    for element in soup.find_all("a", class_="headerlink"):
        element.decompose()
    for element in soup.select(".twemoji, .tabbed-labels, .doc-labels"):
        element.decompose()
    for element in soup.find_all("autoref"):
        element.replace_with(NavigableString(element.get_text()))

    for table in soup.find_all("table", class_="highlighttable"):
        code = table.find("code")
        if code:
            replacement = BeautifulSoup(
                f"<pre><code>{html.escape(code.get_text())}</code></pre>",
                "html.parser",
            )
            table.replace_with(replacement)


def _absolute_link(href: str, *, current_dir: str) -> str:
    if not href or href.startswith(("/", "#")):
        return href
    try:
        if urlparse(href).scheme:
            return href
    except ValueError:
        return href

    relative_base = urljoin(_base_url, f"{current_dir}/") if current_dir else _base_url
    absolute = urljoin(relative_base, href)
    if absolute.endswith("/"):
        absolute = f"{absolute}index.md"
    return absolute


def _render_markdown(
    page_html: str,
    *,
    page_uri: str,
    canonical_url: str,
) -> str:
    soup = BeautifulSoup(page_html, "html.parser")
    _remove_noise(soup)

    current_dir = Path(page_uri).parent.as_posix()
    if current_dir == ".":
        current_dir = ""
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        if isinstance(href, str):
            link["href"] = _absolute_link(href, current_dir=current_dir)

    markdown = _CONVERTER.convert_soup(soup)
    markdown = re.sub(r"\n[ \t]+\n", "\n\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = markdown.strip()
    first_line, separator, remainder = markdown.partition("\n")
    source = f"Canonical HTML: {canonical_url}"
    if separator and first_line.startswith("#"):
        return f"{first_line}\n\n{source}\n\n{remainder.lstrip()}\n"
    return f"{source}\n\n{markdown}\n"


def _page_title(page: Any) -> str:
    source_uri = page.file.src_uri
    if source_uri.startswith("reference/"):
        symbol = source_uri.removeprefix("reference/").removesuffix(".md")
        return symbol.removesuffix("/index").replace("/", ".")
    return BeautifulSoup(str(page.title or source_uri), "html.parser").get_text(
        " ",
        strip=True,
    )


def on_config(config: Any) -> Any:
    """Load LLM artifact settings after Mike has canonicalized ``site_url``."""
    global _base_url, _description, _full_output

    if not config.site_url:
        raise ValueError("site_url must be set to generate LLM documentation")

    llms_config = config.extra.get("llms", {})
    _base_url = str(config.site_url)
    if not _base_url.endswith("/"):
        _base_url = f"{_base_url}/"
    _description = str(llms_config.get("markdown_description", "")).strip()
    _full_output = str(llms_config.get("full_output", "llms-full.txt"))
    return config


def on_files(files: Any, *, config: Any) -> Any:
    """Resolve configured globs after API Autonav adds its virtual pages."""
    global _pages, _sections, _selected_uris

    llms_config = config.extra.get("llms", {})
    configured_sections = llms_config.get("sections", {})
    page_uris = list(files.src_uris)
    _sections = {
        str(section): _expand_inputs(inputs, page_uris)
        for section, inputs in configured_sections.items()
    }
    _selected_uris = set(chain.from_iterable(_sections.values()))
    _pages = {}
    return files


def on_page_content(page_html: str, *, page: Any, **_: Any) -> str:
    """Capture clean Markdown for selected authored and generated pages."""
    source_uri = page.file.src_uri
    if source_uri not in _selected_uris:
        return page_html

    output_path = Path(page.file.abs_dest_path).with_suffix(".md")
    markdown_uri = Path(page.file.dest_uri).with_suffix(".md").as_posix()
    markdown_url = urljoin(_base_url, markdown_uri)
    _pages[source_uri] = _MarkdownPage(
        title=_page_title(page),
        output_path=output_path,
        url=markdown_url,
        content=_render_markdown(
            page_html,
            page_uri=page.file.dest_uri,
            canonical_url=page.canonical_url,
        ),
    )
    return page_html


def on_post_build(*, config: Any, **_: Any) -> None:
    """Write per-page Markdown plus the concise and full LLM indexes."""
    header = f"# {config.site_name}\n\n"
    if config.site_description:
        header += f"> {config.site_description}\n\n"
    if _description:
        header += f"{_description}\n\n"

    index_parts = [header]
    full_parts = [header]

    for section, configured_pages in _sections.items():
        index_parts.append(f"## {section}\n\n")
        full_parts.append(f"## {section}\n\n")
        section_content: list[str] = []

        for source_uri, description in configured_pages.items():
            page = _pages.get(source_uri)
            if page is None:
                _LOGGER.warning(
                    "Configured LLM page '%s' was not generated; skipping",
                    source_uri,
                )
                continue

            page.output_path.parent.mkdir(parents=True, exist_ok=True)
            page.output_path.write_text(page.content, encoding="utf-8")
            suffix = f": {description}" if description else ""
            index_parts.append(f"- [{page.title}]({page.url}){suffix}\n")
            section_content.append(page.content)

        index_parts.append("\n")
        full_parts.append("\n---\n\n".join(section_content))
        full_parts.append("\n\n")

    site_dir = Path(config.site_dir)
    (site_dir / "llms.txt").write_text("".join(index_parts), encoding="utf-8")
    (site_dir / _full_output).write_text(
        "".join(full_parts),
        encoding="utf-8",
    )
