---
workflow_id: mcp-search-literal-mode
phase: structure
track: engineering
size_class: lightweight
status: approved
portability_level: 2
source_inputs:
  - docs/brainstorms/mcp-search-literal-mode-research.md
last_updated: 2026-06-26
---

# Structure: MCP Search Literal Mode

Generated MCP Search Summary and Search Content tools should support literal substring search as a first-class mode while preserving the current regex behavior. The implementation can stay inside the MCP tool generator because fenic already exposes both literal matching through `Column.contains(...)` and regex matching through `Column.rlike(...)`.

## Desired End State

- Generated Search Content accepts an explicit search mode parameter and can return rows for literal patterns containing regex metacharacters without requiring callers to escape them.
- Generated Search Summary uses the same mode semantics as Search Content and reports per-dataset match counts for literal or regex searches.
- Regex remains the default mode for backward compatibility with existing generated tool contracts and docs that currently describe regex search.
- Search mode metadata is visible through generated callable signatures and FastMCP tool schemas because `FenicMCPServer` preserves generated callable signatures with `@wraps(...)`.
- API docstrings and MCP topic docs describe the mode choices consistently instead of saying only regex or ambiguous substring/regex.

## Implementation Overview

- [x] Phase 1: Add literal mode to Search Content end-to-end
- [x] Phase 2: Add literal mode to Search Summary and align exposed docs/schema

---

## ✅ Phase 1: Add Literal Mode to Search Content End-to-End

This phase makes the smallest useful path work: one generated tool over one table can search literal text and still page/order results. It is independently verifiable by directly executing the generated Search Content callable and collecting its logical plan.

### Phase 1 File Changes

- **`src/fenic/api/mcp/_tool_generation_utils.py`**:
  - Add a local search-mode type near the generator helpers, for example `SearchMode = Literal["regex", "literal"]`, using the existing imported `Literal`.
  - Import `Column` from `fenic.api.column` if `_search_predicate(...)` is annotated with a `Column` return type; otherwise omit that return annotation.
  - Add a private helper that owns predicate construction for one column:
    - Signature: `_search_predicate(column_name: str, pattern: str, mode: SearchMode) -> Column`.
    - `mode == "regex"` returns `col(column_name).rlike(pattern)`.
    - `mode == "literal"` returns `col(column_name).contains(pattern)`.
    - Any other value raises `ValidationError("search_mode must be one of: regex, literal")`.
  - Update `search_rows(...)` inside `_auto_generate_search_content_tool` with a keyword parameter:
    - `search_mode: Annotated[SearchMode, "Search mode: 'regex' treats pattern as a regular expression; 'literal' treats pattern as a plain substring."] = "regex"`.
  - Replace the inline `col(c_name).rlike(pattern)` predicate construction in Search Content with `_search_predicate(c_name, pattern, search_mode)`.
  - Keep existing normalization and validation for `df_name`, `pattern`, `limit`, `offset`, `order_by`, `sort_ascending`, and `search_columns`.
  - Do not add wrapper-level handling in `src/fenic/core/mcp/_server.py`; the server already preserves generated callable signatures and exposes generated parameters.

- **`tests/api/mcp/utils.py`**:
  - Add a helper for mixed typed rows so search behavior tests can create string, integer, and multiple text columns without duplicating save/description setup:
    - Signature: `create_table_from_dict(session: Session, name: str, data: dict[str, list], description: str | None = None) -> None`.
    - Use `session.create_dataframe(data)`, `df.write.save_as_table(name, mode="overwrite")`, and `session.catalog.set_table_description(...)` when a description is provided.
  - Leave `create_table_with_rows(...)` in place for existing tests.

- **`tests/api/mcp/test_tool_generation.py`**:
  - Add tests that retrieve the generated tool whose name ends with `"Search Content"` and execute its `func(...)` directly.
  - Required cases:
    - Literal mode treats regex metacharacters literally, e.g. pattern `"a.b"` matches only a row containing the literal substring `a.b`, while regex mode with the same pattern can match `axb`.
    - `search_columns` still restricts literal matching to selected columns.
    - Paging still works with literal mode: `order_by="id"`, `limit="1"`, `offset="1"` returns the expected second ordered match.
    - Unknown `search_mode` raises `ValidationError` with the mode-specific message.
  - Collect plans through `local_session._session_state.execution.collect(plan)` as the Schema test already does.

### Phase 1 Validation

#### Phase 1 Automated Verification

- [x] `uv run pytest tests/api/mcp/test_tool_generation.py`
- [x] `uv run ruff check src/fenic/api/mcp/_tool_generation_utils.py tests/api/mcp/test_tool_generation.py tests/api/mcp/utils.py`

#### Phase 1 Manual Verification

None.

---

## ✅ Phase 2: Add Literal Mode to Search Summary and Align Exposed Docs/Schema

This phase completes the generated search surface by applying the same mode semantics across all datasets and making the public descriptions match the callable behavior.

### Phase 2 File Changes

- **`src/fenic/api/mcp/_tool_generation_utils.py`**:
  - Update `search_summary(...)` inside `_auto_generate_search_summary_tool` with the same keyword parameter:
    - `search_mode: Annotated[SearchMode, "Search mode: 'regex' treats pattern as a regular expression; 'literal' treats pattern as a plain substring."] = "regex"`.
  - Replace Search Summary's inline `col(c_name).rlike(pattern)` construction with `_search_predicate(c_name, pattern, search_mode)`.
  - Update generated Search Summary and Search Content tool descriptions to name the modes explicitly, for example:
    - Search Summary: "Search across all datasets and return the number of matches per dataset. Use `search_mode='literal'` for plain substring search or `search_mode='regex'` for regular expressions."
    - Search Content: "Return matching rows from a single dataset. Use `search_mode='literal'` for plain substring search or `search_mode='regex'` for regular expressions."
  - Update `pattern` parameter annotations for both generated callables so they no longer imply regex-only behavior.

- **`tests/api/mcp/test_tool_generation.py`**:
  - Add Search Summary execution tests:
    - Literal mode counts only rows containing the literal pattern across all string columns and all generated datasets.
    - Regex mode keeps existing regex behavior and still returns zero-count rows for datasets with no string columns.
  - Extend the existing generation/signature assertions to verify generated Search Summary and Search Content callables include `search_mode` and default it to `"regex"`.

- **`tests/api/mcp/test_server.py`**:
  - Extend `_validate_server_tools(...)` or add a focused assertion that FastMCP schemas expose `search_mode` for the snake-cased generated Search Summary and Search Content tools.
  - Assert the schema default is `"regex"` and that the description mentions both `"literal"` and `"regex"`.
  - Keep existing assertions that wrapper-added `table_format` is present and Search Content does not receive an extra wrapper-level `limit`.

- **`src/fenic/api/mcp/tools.py`**:
  - Update `SystemToolConfig` docstring bullets for Search Summary and Search Content to describe literal and regex modes consistently.
  - Keep `max_result_rows` wording unchanged unless tests expose current behavior mismatch outside the search-mode scope.

- **`docs/topics/fenic-mcp.md`**:
  - Update the auto-generated system tools section so Search Summary and Search Content mention `search_mode="literal"` for plain substring search and `search_mode="regex"` for regular expressions.
  - Keep examples minimal; do not introduce a new MCP server setup path.

### Phase 2 Validation

#### Phase 2 Automated Verification

- [x] `uv run pytest tests/api/mcp/test_tool_generation.py tests/api/mcp/test_server.py`
- [x] `uv run ruff check src/fenic/api/mcp/_tool_generation_utils.py src/fenic/api/mcp/tools.py tests/api/mcp/test_tool_generation.py tests/api/mcp/test_server.py tests/api/mcp/utils.py`

#### Phase 2 Manual Verification

None.

---

## Open Questions

None.

## Decision Log

- **[applied]** Default `search_mode` remains `"regex"` to preserve existing generated callable behavior and user expectations from current docs.
- **[applied]** Literal matching should use existing `Column.contains(...)`, whose local backend transpiles to Polars substring matching with `literal=True`; no new logical expression or backend code is needed.
- **[applied]** Search Content is implemented before Search Summary because it verifies literal mode on a single dataset while keeping paging, column restriction, and existing validation behavior in the same slice.
- **[applied]** FastMCP schema work belongs in the second slice because the server already reflects generated signatures; the remaining risk is schema visibility and docs consistency after both generated callables own the same parameter.
- **[applied]** Repo validation recipes were derived from `CLAUDE.md`, `justfile`, and `pyproject.toml`. This shell resolved `just` to `just 1.54.0`. The direct `uv run ...` commands remain valid targeted equivalents for the per-file checks.
- **[applied]** Structure review ran with inline `plan-eng-review-lite` and `ce-doc-review-lite` checks. Optional subagent review was attempted but blocked by harness model-resolution errors, so `portability_level` remains `2`.
- **[deferred]** Broader cleanup of current Search Content parameter normalization, such as `bool("false") is True`, is outside this literal-mode slice; file a focused MCP generated-tool validation follow-up if this becomes user-visible during implementation.
- **[deferred]** Changing the default from regex to literal would be a breaking behavior change and should be handled as a separately approved API decision.

## Handoff

**Next step (paste into a fresh tab):**

> Use the `td-implement` skill. Structure is ready at
> `docs/plans/mcp-search-literal-mode-structure.md` (track: engineering, size: lightweight).
> Implement the phases in order. Use the repo `just` recipes normally. After editing any fenic pipeline examples, run `fenic check <file>`; this plan does not currently require editing fenic pipeline examples.

**Approved decisions:** add `search_mode` to generated Search Content and Search Summary; accepted values are `"regex"` and `"literal"`; default is `"regex"`; regex mode keeps `Column.rlike(...)`; literal mode uses `Column.contains(...)`; server wrapper changes are not expected because generated signatures already flow through FastMCP.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** changing generated Analyze text-search guidance, changing default search behavior to literal, adding new DataFrame/string expression primitives, fixing unrelated Search Content normalization issues.
**Evidence summary:** Search tools currently build predicates with `col(...).rlike(pattern)` in `src/fenic/api/mcp/_tool_generation_utils.py`; `Column.contains(...)` already exists and transpiles to literal string matching; `FenicMCPServer` exposes generated callable signatures via `@wraps(...)`; existing MCP tests cover generation and server schema but not search execution.
**Known weak assumptions:** FastMCP may encode `typing.Literal["regex", "literal"]` as an enum in the schema; if the exact schema shape differs, assert the stable user-facing properties instead of brittle internals.
**Rollback if:** implementation reveals FastMCP or Pydantic cannot expose `Literal` defaults cleanly for generated function parameters; then switch the generated parameter type to `str`, keep the same runtime validation, and update tests to assert description/default rather than enum metadata.

## Implementation Notes

**Implemented head:** `ff6ad44`
**Spec source:** `docs/plans/mcp-search-literal-mode-structure.md`
**Verification summary:** `uv run --env-file .env pytest tests/api/mcp/test_tool_generation.py tests/api/mcp/test_server.py` passed with 14 tests; `uv run --env-file .env ruff check src/fenic/api/mcp/_tool_generation_utils.py src/fenic/api/mcp/tools.py tests/api/mcp/test_tool_generation.py tests/api/mcp/test_server.py tests/api/mcp/utils.py` passed. Live generated-callable checks confirmed literal vs regex Search Summary counts and FastMCP `search_mode` schema defaults.
**Deliberate tradeoffs / rejected approaches:** Kept regex as the default for backward compatibility; reused existing `Column.contains(...)` and `Column.rlike(...)` rather than adding a new expression; left unrelated generated-tool normalization quirks out of scope.
**Deviations from spec:** None.
**Reviewer context:** `search_mode` is implemented in generated callable signatures, not the MCP server wrapper. Post-implementation subagent review found and fixed no-string invalid-mode validation and Search Content positional compatibility. The server tests assert the reflected FastMCP schema default, enum values, optionality, and description.
