---
name: fenic-mechanics
description: >-
  Use when writing, editing, or debugging code that uses fenic (the `fenic`
  Python library, imported as `import fenic as fc`) — the semantic-DataFrame
  engine with LLM operators (semantic.map/extract/classify/predicate/reduce,
  embeddings), jq/markdown/transcript processing, chunking, and fuzzy matching.
  Covers fenic's import & namespace structure, semantic-operator calling
  conventions, model/session config, and known silent-failure traps. NOT for
  plain PySpark, pandas, or Polars code.
---

# fenic mechanics

fenic looks like PySpark and you already write its DataFrame surface well
(`select`/`filter`/`join`/`group_by`/`agg`, `semantic.extract`/`classify`).
This skill covers the **mechanics that don't transfer** — where fenic differs
from PySpark/pandas intuition in ways that fail (often loudly, sometimes
silently). For full signatures see `reference/*.md` (generated from the
installed version); for the correction table and traps see `gotchas.md`.

> **Golden rule:** after writing or editing a fenic pipeline, run
> **`fenic check <file>`** — a static lint (no execution) that resolves your
> `fc.*` symbols against the installed fenic and flags namespace/import mistakes
> (`fenic.functions`, `fc.array` vs `fc.arr`, `fc.explode`, …). Fix what it reports.

## 1. Import & namespace law (the #1 source of errors)

- **Always `import fenic as fc`. Everything hangs off `fc.`.** There is **no**
  `fenic.functions` (don't write `from fenic import functions as F`), **no**
  `fenic.api.types`, and **no** unified `OpenAIModelConfig`.
- **Function namespaces:** `fc.text.*`, `fc.json.*`, `fc.markdown.*`,
  `fc.semantic.*`, `fc.embedding.*`, `fc.dt.*`, and **`fc.arr.*`** for array ops.
  ⚠️ `fc.array(...)` is a _constructor_ for array literals; the array-operations
  namespace is **`fc.arr`** (`fc.arr.size`, `fc.arr.contains`, `fc.arr.sort`, …).
- **Flat on `fc`:** free functions (`fc.col`, `fc.lit`, `fc.when`, `fc.coalesce`,
  `fc.count`, `fc.sum`, `fc.avg`, `fc.collect_list`, `fc.struct`, `fc.udf`,
  `fc.async_udf`, …), all types, and all model-config classes.
- **`explode` / `unnest` are DataFrame _methods_, not functions:**
  `df.explode("col")`, `df.unnest("col")` — never `fc.explode(...)`.
- PySpark camelCase aliases (`withColumn`, `groupBy`, `orderBy`,
  `dropDuplicates`) **do** exist and work, but prefer snake_case.

## 2. Session & models

```python
import fenic as fc
session = fc.Session.get_or_create(fc.SessionConfig(
    app_name="my_app",
    semantic=fc.SemanticConfig(
        language_models={"mini": fc.OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)},
        default_language_model="mini",
        # embeddings are a SEPARATE class + SEPARATE dict:
        embedding_models={"emb": fc.OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=500, tpm=200_000)},
        default_embedding_model="emb",
    ),
))
```

- **Language vs embedding models are different classes** (`fc.OpenAILanguageModel`
  vs `fc.OpenAIEmbeddingModel`) and live in **different config keys**
  (`language_models` vs `embedding_models`). No single unified model class.
- `default_language_model` / `default_embedding_model` are **required when more
  than one** of that kind is registered.
- **Anthropic splits rate limits:** `fc.AnthropicLanguageModel(model_name, rpm,
input_tpm, output_tpm)` — **no single `tpm`**. OpenAI/Google/Cohere use `tpm`.
- Session creation performs a **live API-key check** — a valid provider key must
  be in the environment even just to build a semantic session.
- **`session.create_dataframe(data, schema=...)` accepts a complete top-level
  `fc.Schema`.** Use it to stamp logical `fc.JsonType`, `fc.MarkdownType`, and
  `fc.EmbeddingType` onto in-memory data; schema field names are authoritative
  and output columns are reordered to schema order.

## 3. Semantic operators — calling convention

- **Column-level** (use inside `select`/`with_column`): `fc.semantic.map`,
  `extract`, `classify`, `predicate`, `reduce`, `summarize`, `analyze_sentiment`,
  `embed`, `parse_pdf`.
- **DataFrame-level** (`df.semantic.*`): `join` (LLM predicate), `sim_join`
  (embedding similarity), `with_cluster_labels` (clustering). Only these three.
- **Templates use Jinja2 double braces and REQUIRE matching column kwargs:**
  ```python
  fc.semantic.predicate("Is this a complaint? {{ msg }}", msg=fc.col("msg"))
  fc.semantic.map("Summarize {{ body }}", body=fc.col("body"))
  ```
  Same for `fc.text.jinja(template, **columns)` and `df.semantic.join`'s
  predicate (which uses the literal placeholders `{{ left_on }}` / `{{ right_on }}`).
- `fc.semantic.extract(col, MyPydanticModel)` — schema is positional (or
  `response_format=`). `fc.semantic.classify(col, [..>=2 classes..])`.
- **`parse_pdf` is `fc.semantic.parse_pdf`** (under `semantic`, NOT `markdown` —
  it calls the model). Input is a column of PDF _path_ strings (no cast needed):
  `fc.semantic.parse_pdf(fc.col("path"), page_separator="--- PAGE {page} ---")` —
  pass `page_separator` (the `{page}` placeholder is filled per page) when you
  want page breaks; omit it and pages run together.

## 4. ⚠️ The 4 traps `fenic check` can't catch

`fenic check` is a static lint (symbols & namespaces) — it doesn't see these.
The first three run clean and produce **wrong output** (truly silent); the
fourth errors only at execution. Get them right by hand:

1. **`fc.json.jq(col, query)` returns an ARRAY** (`ArrayType(JsonType)`), never a
   scalar. Take one match before casting: `fc.json.jq(c, ".x").get_item(0).cast(fc.IntegerType)`.
2. **Single braces in a semantic template.** `"... {msg} ..."` (one brace) is
   _not_ interpolated — the model receives the literal `{msg}`. Always `{{ msg }}`.
3. **`fc.dt.datediff(end, start)` returns `end - start`.** Reversed args →
   silently negative/wrong. Order matters.
4. **`fc.dt.to_timestamp` / `to_date` / `date_format` take Spark/Java patterns**
   (`yyyy-MM-dd HH:mm:ss`, `MM-dd-yyyy`), **NOT** Python/chrono `%`-tokens — fenic
   converts the Spark pattern to chrono internally. A `%`-style string raises an
   `ExecutionError` at materialization (so `fenic check` won't flag it). With no
   `format`, `to_timestamp` expects ISO-8601-with-ms; `datediff`/`date_trunc`
   take the resulting timestamp/date columns directly.

## 5. Stay in fenic — don't bypass it

Use fenic's native operators (`fc.json.jq`, `fc.markdown.*`,
`fc.text.parse_transcript`, `fc.text.extract` templates, `fc.text.compute_fuzzy_*`)
rather than dropping to `json.loads`, `re`, or manual string parsing. The point
of fenic is a typed, inspectable, rerunnable pipeline — raw-Python escape hatches
throw that away and don't run in the engine.

## More detail

- `reference/functions.md`, `reference/dataframe.md`, `reference/config-and-types.md`
  — full signatures, generated from the installed fenic version.
- `gotchas.md` — the "wrote X, meant Y" correction table (every real failure mode
  observed) and the silent-trap deep dive.
