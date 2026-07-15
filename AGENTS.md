# AGENTS.md — writing fenic

fenic is a PySpark-style **semantic DataFrame** library (`import fenic as fc`).
You likely know its DataFrame surface; below are the mechanics that DON'T match
PySpark/pandas intuition. _(Developing fenic itself? See `CLAUDE.md`.)_

> **After writing or editing any fenic pipeline, run `fenic check <file>`** — a
> static lint (no execution) that resolves your `fc.*` symbols against the
> installed fenic and flags namespace/import mistakes.

## Must-knows

- **`import fenic as fc`; everything is flat on `fc`.** There is **no
  `fenic.functions`**, no `fenic.api.types`, and no unified `OpenAIModelConfig`.
- **Function namespaces:** `fc.text` / `fc.json` / `fc.markdown` / `fc.semantic` /
  `fc.embedding` / `fc.dt`, and **`fc.arr`** for array ops (⚠️ `fc.array` is the
  array-literal _constructor_, not the ops namespace).
- **`explode` / `unnest` are DataFrame methods** — `df.explode("c")`,
  `df.unnest("c")` — never `fc.explode`.
- **Language vs embedding models are separate classes** (`fc.OpenAILanguageModel`
  vs `fc.OpenAIEmbeddingModel`) in separate config keys (`language_models` /
  `embedding_models`); `default_language_model` / `default_embedding_model` are
  required when more than one is registered. Anthropic uses split
  `input_tpm` / `output_tpm`, not a single `tpm`.
- **Semantic templates use Jinja2 `{{ var }}` + matching column kwargs:**
  `fc.semantic.predicate("... {{ x }} ...", x=fc.col("x"))`. `parse_pdf` is
  **`fc.semantic.parse_pdf`** (under `semantic`, not `markdown`).
- **Local extras for heavier operators:** `fc.semantic.parse_pdf` and
  `session.read.pdf_metadata` need `fenic[pdf]`;
  `df.semantic.with_cluster_labels` needs `fenic[cluster]`;
  `df.semantic.sim_join` needs `fenic[sim-join]`.

## Traps `fenic check` can't catch — get these right by hand

- `fc.json.jq(col, q)` returns an **array** → `.get_item(0)` before a scalar `.cast`.
- A semantic template with **single braces** `{x}` is not interpolated (silent).
- `fc.dt.datediff(end, start)` returns `end - start` (argument order matters).
- `fc.dt.to_timestamp(col, fmt)` takes **Spark/Java** patterns (`yyyy-MM-dd HH:mm:ss`),
  **not** Python `%`-tokens.

## Full detail

`.claude/skills/fenic-mechanics/` — `SKILL.md` (rules), `gotchas.md` (the
"wrote X, meant Y" table), `reference/` (full signatures, generated per version).
Regenerate after a fenic upgrade with the `update-fenic-skill` skill.
