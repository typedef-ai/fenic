# Creating DataFrames with an Explicit Schema

This page describes how to make a fenic `Schema` authoritative when you build a
DataFrame with `Session.create_dataframe`, and the capabilities that unlocks:
logical string-backed types, fixed-size embedding preservation, and typed empty
frames.

## Overview

By default, `create_dataframe(data)` infers a schema from the input. Inference is
convenient, but it cannot know that a string column is really JSON or Markdown,
and it normalizes fixed-size arrays to variable-length lists.

Passing an explicit `schema` makes a complete, top-level `Schema` the source of
truth for the resulting DataFrame:

- **Field names are authoritative** — the data's columns must match the schema.
- **Columns are reordered** to the schema's field order.
- **Values are physically coerced** to the schema's representation.
- **The logical schema is preserved exactly**, including string-backed types.

```python
import fenic as fc

session = fc.Session.get_or_create(fc.SessionConfig(app_name="schema_intro"))

schema = fc.Schema([
    fc.ColumnField("age", fc.IntegerType),
    fc.ColumnField("name", fc.StringType),
])

# Input columns can be in any order; values are coerced to the schema.
df = session.create_dataframe({"name": ["Alice"], "age": ["42"]}, schema=schema)

df.schema
# Schema(
#   ColumnField(name='age', data_type=IntegerType)
#   ColumnField(name='name', data_type=StringType)
# )
```

Reach for an explicit schema when you need to seed logical string-backed types,
preserve embeddings, or guarantee column order and dtypes; rely on inference for
quick, ad-hoc frames.

## The Schema Contract

When you pass `schema`, `create_dataframe` enforces a strict top-level contract:

- **Exact column match.** Every schema field must be present in the data, and the
  data may not contain columns outside the schema. Column _order_ in the input
  does not matter — the result is ordered by the schema.
- **Physical coercion.** Each column is cast to the schema's Polars
  representation (for example `IntegerType` → `Int64`, `EmbeddingType` →
  `Array(Float32, dimensions)`).
- **Typed errors.** Shape problems raise `ValidationError`; coercion or
  schema-validity problems raise `PlanError`.

```python
from fenic.core.error import ValidationError, PlanError

# Extra/missing columns -> ValidationError
try:
    session.create_dataframe({"name": ["A"], "age": ["1"], "extra": [9]}, schema=schema)
except ValidationError as e:
    print(e)
    # Data columns must match the provided schema exactly;
    # missing columns: [...], extra columns: ['extra']

# Duplicate schema field names -> PlanError
dup = fc.Schema([
    fc.ColumnField("x", fc.IntegerType),
    fc.ColumnField("x", fc.StringType),
])
try:
    session.create_dataframe({"x": [1]}, schema=dup)
except PlanError as e:
    print(e)  # Duplicate column names found: 'x'. Column names must be unique. ...
```

### Typed empty frames

With a schema, empty inputs (`[]`, `{}`, `pl.DataFrame()`, `pd.DataFrame()`,
`pa.table({})`) produce a zero-row DataFrame that still carries the requested
schema — useful for seeding a typed, empty starting point.

```python
empty = session.create_dataframe([], schema=schema)
empty.schema.column_names()          # ['age', 'name']
empty.to_polars().height             # 0
```

Without a schema, an empty list is rejected — the no-schema path cannot infer
types from nothing.

## Logical String-Backed Types (JSON / Markdown)

fenic stores `JsonType`, `MarkdownType`, and `HtmlType` as physical strings while
tracking their logical type so type-specific functions (JQ, markdown parsing,
etc.) are available. Declaring these types in the schema tags the columns at
ingestion, so you don't need a follow-up `.cast()`:

```python
schema = fc.Schema([
    fc.ColumnField("payload", fc.JsonType),
    fc.ColumnField("notes", fc.MarkdownType),
])

df = session.create_dataframe(
    {"payload": ['{"a": 1}'], "notes": ["# Title"]},
    schema=schema,
)

df.schema
# ColumnField(name='payload', data_type=JsonType)
# ColumnField(name='notes',   data_type=MarkdownType)

df.to_polars()["payload"].dtype      # String  (logical type, physical string)
```

The logical type stays visible in `df.schema` and drives downstream validation,
while the value is stored as a plain string.

**No content validation.** An explicit schema tags a column's _logical type_; it
does not validate the _content_. A `JsonType` column is not parsed or checked for
well-formed JSON at ingestion — malformed values surface when a JSON operation
runs on them.

For columns produced mid-pipeline (where no ingestion schema exists), continue to
use `.cast(fc.JsonType)` / `.cast(fc.MarkdownType)` as shown in the
[JSON Processing](../examples/json_processing.md) and
[Markdown Processing](../examples/markdown_processing.md) examples.

## Embedding Preservation

fenic represents `EmbeddingType` as a fixed-size Polars array
(`Array(Float32, dimensions)`). The schema-free ingestion path normalizes all
fixed-size arrays to variable-length lists (`pl.List`), which loses the
fixed-width guarantee. An explicit schema preserves the fixed-size array through
local execution all the way to `to_polars()`:

```python
emb_schema = fc.Schema([
    fc.ColumnField("id", fc.IntegerType),
    fc.ColumnField("embedding", fc.EmbeddingType(dimensions=3, embedding_model="example/model")),
])

edf = session.create_dataframe(
    {"id": [1, 2], "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
    schema=emb_schema,
)

edf.to_polars()["embedding"].dtype   # Array(Float32, shape=(3,))
```

This lets you bring precomputed embeddings into fenic without losing their
fixed-size representation. (Constructing the frame does not call an embedding
provider — the `embedding_model` on the type is metadata.)

## Relationship to Schema-Free Ingestion

The no-schema path is unchanged and fully behavior-compatible:

- Empty lists are still rejected.
- Types are still inferred from the data.
- Fixed-size arrays are still normalized to `pl.List`.

Only supplying a `schema` opts you into authoritative names, ordering, coercion,
and logical-type/embedding preservation. For related ingestion-normalization
behavior on temporal columns, see
[Timestamps and Dates](timestamps_and_dates.md).
