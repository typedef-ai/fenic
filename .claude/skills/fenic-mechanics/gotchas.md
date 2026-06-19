# fenic gotchas — "wrote X, meant Y"

Every entry below is a real failure mode observed when an agent wrote fenic from
intuition, with the verified correct form. Loud = fails immediately (ImportError /
AttributeError / validation / plan error → `fenic check` catches it). Silent =
runs but produces wrong output (only knowledge prevents it).

## Namespace & imports (Loud)

| Wrote                                                      | Meant                                       | Why                                                                   |
| ---------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------- |
| `from fenic import functions as F` / `F.col`               | `import fenic as fc`; `fc.col`              | No `fenic.functions` submodule; namespaces are flat on `fc`.          |
| `from fenic.api.types import ...`                          | `fc.StringType`, `fc.StructType`, …         | Types are flat on `fc`; no `fenic.api.types`.                         |
| `from fenic.api.session.config import OpenAILanguageModel` | `fc.OpenAILanguageModel`                    | Config classes are re-exported flat on `fc`.                          |
| `fc.explode(col)` / `F.explode(...)` inside `select`       | `df.explode("col")` then `df.unnest("col")` | `explode`/`unnest` are DataFrame **methods**, not functions.          |
| `fc.array.size(col)`                                       | `fc.arr.size(col)`                          | `fc.array` is the array-literal constructor; ops live under `fc.arr`. |
| `fc.markdown.parse_pdf(col)`                               | `fc.semantic.parse_pdf(col)`                | `parse_pdf` calls the LLM → lives under `semantic`, not `markdown`.   |

## Model / session config (Loud)

| Wrote                                   | Meant                                                        | Why                                                                                        |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `OpenAIModelConfig(...)` for embeddings | `fc.OpenAIEmbeddingModel(...)` in `embedding_models`         | No unified config class; language & embedding models are separate types in separate dicts. |
| `AnthropicLanguageModel(..., tpm=...)`  | `AnthropicLanguageModel(..., input_tpm=..., output_tpm=...)` | Anthropic splits token limits; there is no single `tpm`.                                   |
| one model, then a semantic op fails     | set `default_language_model=` / `default_embedding_model=`   | Required when >1 model of a kind is registered (single model auto-defaults).               |

## Function contracts (Loud unless noted)

| Wrote                                                  | Meant                                                                   | Why                                                                                                                                            |
| ------------------------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `fc.json.get_type(col, "$.x")` / JSONPath              | `fc.json.get_type(col)` (whole value); nested → `fc.json.jq(col, ".x")` | `get_type` takes one arg; there is no JSONPath — navigation is jq only.                                                                        |
| `fc.json.contains(col, "urgent")`                      | `fc.json.contains(col, '"urgent"')`                                     | `value` must be valid JSON-encoded (a string needs its quotes).                                                                                |
| `recursive_token_chunk(col, chunk_overlap=...)`        | `...(col, chunk_size, chunk_overlap_percentage=...)`                    | Param is `chunk_overlap_percentage`.                                                                                                           |
| `compute_fuzzy_ratio(a, b, method="token_sort_ratio")` | `compute_fuzzy_token_sort_ratio(a, b)`                                  | `method` ∈ `{indel, levenshtein, damerau_levenshtein, jaro_winkler, jaro, hamming}`; token-sort/set are separate functions, not method values. |
| `dt.truncate(col, "month")`                            | `fc.dt.date_trunc(col, "month")`                                        | Real name is `date_trunc`.                                                                                                                     |
| `dt.diff_days(a, b)`                                   | `fc.dt.datediff(end, start)`                                            | Real name is `datediff`. **(arg order is also a silent trap — see below)**                                                                     |
| `catalog.get_table_description(name)`                  | `catalog.describe_table(name).description`                              | `get_table_description` is backend-only; public reader is `describe_table` → `DatasetMetadata`.                                                |
| `catalog.create_view(...)`                             | (no public Catalog method)                                              | Views are created via DataFrame write paths; Catalog only exposes `list/describe/set_description/does_exist/drop` for views.                   |
| `@fc.udf(return_type=fc.StringType())`                 | `@fc.udf(return_type=fc.StringType)`                                    | Scalar types are singletons — pass `fc.StringType`, no parens.                                                                                 |

## The 4 traps `fenic check` can't catch

Three run clean and produce **wrong output** (truly silent); the fourth (#4)
errors only at execution. None are caught at plan construction — only correct
knowledge prevents them.

1. **`fc.json.jq` returns an array.** `fc.json.jq(c, ".value")` is
   `ArrayType(JsonType)` (every match). Casting it straight to a scalar either
   errors (if caught) or, when shapes happen to line up, mis-shapes data. Take
   one: `fc.json.jq(c, ".value").get_item(0).cast(fc.IntegerType)`. Keep
   aggregations inside the jq query (`".items | length"`).
2. **Single-brace templates.** A semantic/`text.jinja` template with `{var}`
   (one brace) is left literally un-interpolated; the model runs on the raw
   string `{var}`. Plausible-looking, wrong output, no error. Always `{{ var }}`,
   and pass the matching `var=fc.col("var")` kwarg.
3. **`fc.dt.datediff(end, start)` = `end - start`.** Reversed arguments produce
   silently negative/incorrect day counts.
4. **`fc.dt.to_timestamp` / `to_date` / `date_format` take Spark/Java
   SimpleDateFormat patterns** (`yyyy-MM-dd HH:mm:ss`, `MM-dd-yyyy`), **NOT**
   Python/chrono `%`-tokens — fenic converts the Spark pattern to chrono
   internally (`_java_like_to_chrono`, `expressions/dt.py:142`). A `%`-token
   string is mangled and raises an `ExecutionError` at materialization (verified:
   `yyyy-MM-dd HH:mm:ss` parses `"2024-03-15 09:30:00"`; `%Y-%m-%d %H:%M:%S`
   errors). `fenic check` won't flag it (execution-time failure). Default
   `to_timestamp` (no format) is ISO-8601-with-ms; `datediff`/`date_trunc`
   consume the resulting timestamp/date columns directly.

## What agents reliably get RIGHT (don't over-warn)

PySpark-shaped DataFrame ops, `semantic.extract`/`classify`/`join`,
`semantic.reduce`, `with_cluster_labels`, `sim_join`, the full `regexp_*` family,
`text.jinja`, struct casting + `unnest`, UDFs, IO round-trips, and the
`text.extract` `${field:none}` template DSL — all observed correct. The skill's
job is the lists above, not re-teaching these.
