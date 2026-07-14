---
workflow_id: create-dataframe-schema
phase: plan
track: engineering
size_class: high-risk
status: approved
portability_level: 3
source_inputs:
  - docs/td-flow/create-dataframe-schema/structure.md
  - docs/td-flow/create-dataframe-schema/research.md
  - docs/td-flow/create-dataframe-schema/design.md
last_updated: 2026-06-26
---

# create_dataframe Schema Handling Implementation Plan

## Overview

Add `schema: Schema | None = None` to `Session.create_dataframe` so callers can make a complete top-level `Schema` authoritative at ingestion time. The implementation keeps the no-schema path behavior-compatible, reuses `InMemorySource.from_schema`, and narrows local physical schema awareness to the embedding paths that current ingestion coercion erases.

## Current State Analysis

### Key Discoveries

- `Session.create_dataframe` currently accepts only `data: DataLike`, normalizes supported inputs through Polars, rejects empty row lists, wraps construction failures in `PlanError`, and always builds `InMemorySource.from_session_state` (`src/fenic/api/session/session.py:132`, `src/fenic/api/session/session.py:186`, `src/fenic/api/session/session.py:217`).
- `Schema.column_names()` preserves ordered field names, while `convert_custom_schema_to_polars_schema` maps fenic logical schemas to physical Polars dtypes including `EmbeddingType -> pl.Array(pl.Float32, dimensions)` and tagged string types to `pl.String` (`src/fenic/core/types/schema.py:117`, `src/fenic/core/_utils/schema.py:56`, `src/fenic/core/_utils/schema.py:185`).
- `InMemorySource.from_schema` and wrapper-level plan serde already preserve a supplied logical schema; `LogicalPlan._validate_schema` already rejects duplicate column names (`src/fenic/core/_logical_plan/plans/source.py:31`, `src/fenic/core/_logical_plan/plans/base.py:35`, `src/fenic/core/_serde/proto/plan_serde.py:48`).
- Local `PlanConverter` currently lowers `InMemorySource` without passing the logical schema, and `InMemorySourceExec` applies `apply_ingestion_coercions(self.df, coerce_array=True)` which converts all fixed-size arrays to lists (`src/fenic/_backends/local/transpiler/plan_converter.py:162`, `src/fenic/_backends/local/physical_plan/source.py:27`, `src/fenic/_backends/local/physical_plan/utils.py:39`).
- Existing tests already cover schema-free `create_dataframe` forms, source serde/equality, local ingestion coercion, local function validation for JSON/Markdown after explicit casts, cloud execution mocks, and source transpilation (`tests/api/test_session.py:48`, `tests/_logical_plan/serde/test_plan_serde.py:247`, `tests/_backends/test_ingestion_coercion.py:10`, `tests/_backends/local/functions/test_jq.py:28`, `tests/_backends/cloud/test_cloud_execution.py:301`, `tests/_backends/local/test_transpiler.py:83`).

## Desired End State

- `Session.create_dataframe(data, schema=...)` and `Session.createDataFrame(data, schema=...)` accept a complete top-level public `Schema`.
- Schema-free calls keep current normalization and error behavior, including rejecting `[]`.
- Schema-backed calls validate top-level shape, reorder columns by `schema.column_fields`, physically coerce through `convert_custom_schema_to_polars_schema`, and construct `InMemorySource.from_schema`.
- Empty `[]`, `{}`, `pl.DataFrame()`, `pd.DataFrame()`, and `pa.table({})` produce zero-row DataFrames with the requested schema only when a schema is provided.
- Logical tagged string types remain visible in `df.schema` and downstream logical validation while storing physical strings.
- Explicit `EmbeddingType` columns materialize locally as fixed-size `pl.Array(pl.Float32, dimensions)` after `to_polars()`, while no-schema arrays still normalize to `pl.List`.

## What We're NOT Doing

- Partial schemas or schema inference mixed with a partial explicit schema.
- Pydantic model shortcuts for `create_dataframe(schema=...)`.
- New source plan nodes or cloud protobuf/request fields.
- Broad recasting of every local source column from the logical schema during execution.
- Tagged-string content validation for `JsonType`, `MarkdownType`, `HtmlType`, transcripts, or document paths.
- User-facing documentation updates in this implementation pass.

## Implementation Overview

- [x] Phase 1: Basic schema-backed creation end to end
- [x] Phase 2: Complete input-shape and error contract
- [x] Phase 3: Logical schema propagation through planning, serde, and cloud request boundaries
- [x] Phase 4: Local embedding physical preservation

---

## ✅ Phase 1: Basic schema-backed creation end to end

### Phase 1 Overview

Make the smallest useful API path work: a primitive explicit schema can create a DataFrame from common inputs, preserve the logical schema, order output columns by the schema, and leave the current no-schema path intact.

### Phase 1 Changes Required

#### 1.1 Public Signature and Schema Imports

**File**: `src/fenic/api/session/session.py`
**Changes**: Around lines 27-29, import `convert_custom_schema_to_polars_schema` and `Schema`; around line 132, add the optional schema parameter and docstring contract.

```diff
@@
 from fenic.core.error import CatalogError, PlanError, ValidationError
+from fenic.core._utils.schema import convert_custom_schema_to_polars_schema
 from fenic.core.types.query_result import DataLike
+from fenic.core.types.schema import Schema
@@
     def create_dataframe(
         self,
         data: DataLike,
+        schema: Schema | None = None,
     ) -> DataFrame:
@@
-            data: Input data. Must be one of:
+            data: Input data. Must be one of:
                 - Polars DataFrame
@@
                 - pyarrow Table
+            schema: Optional complete top-level fenic schema. When provided,
+                field names are authoritative, values are physically coerced to
+                the schema's Polars representation, and the logical DataFrame
+                schema is preserved exactly.
```

#### 1.2 Split Normalization From Source Construction

**File**: `src/fenic/api/session/session.py`
**Changes**: Replace the body at lines 186-220 with a no-schema branch that keeps current behavior and a schema branch that delegates to helpers added below `create_dataframe`.

```diff
@@
-        try:
-            if isinstance(data, pl.DataFrame):
-                pl_df = data
-            elif isinstance(data, pd.DataFrame):
-                pl_df = pl.from_pandas(data)
-            elif isinstance(data, dict):
-                pl_df = pl.DataFrame(data)
-            elif isinstance(data, list):
-                if not data:
-                    raise ValidationError(
-                        "Cannot create DataFrame from empty list. Provide a non-empty list of dictionaries, lists, or other supported data types."
-                    )
-
-                if not isinstance(data[0], dict):
-                    raise ValidationError(
-                        "Cannot create DataFrame from list of non-dict values. Provide a list of dictionaries."
-                    )
-                pl_df = pl.DataFrame(data)
-            elif isinstance(data, pa.Table):
-                pl_df = pl.from_arrow(data)
-
-            else:
-                raise ValidationError(
-                    f"Unsupported data type: {type(data)}. Supported types are: Polars DataFrame, Pandas DataFrame, dict, or list."
-                )
-
-        except ValidationError:
-            raise
-        except Exception as e:
-            raise PlanError(f"Failed to create DataFrame from {data}") from e
+        pl_df, row_oriented = _normalize_data_like_to_polars(
+            data,
+            allow_empty_list=schema is not None,
+        )
+        if schema is None:
+            return DataFrame._from_logical_plan(
+                InMemorySource.from_session_state(pl_df, self._session_state),
+                self._session_state,
+            )
+
+        coerced_pl_df = _coerce_to_schema(pl_df, schema, row_oriented=row_oriented)

         return DataFrame._from_logical_plan(
-            InMemorySource.from_session_state(pl_df, self._session_state),
+            InMemorySource.from_schema(coerced_pl_df, schema),
             self._session_state,
         )
```

#### 1.3 Add Initial Helper Implementations

**File**: `src/fenic/api/session/session.py`
**Changes**: Add module-level helpers below the class before the alias block; keep them private to avoid broadening public API. In this phase, support `dict`, `list[dict]`, and `pl.DataFrame` for schema-backed creation.

```diff
@@
+def _normalize_data_like_to_polars(
+    data: DataLike,
+    *,
+    allow_empty_list: bool,
+) -> tuple[pl.DataFrame, bool]:
+    try:
+        if isinstance(data, pl.DataFrame):
+            return data, False
+        if isinstance(data, pd.DataFrame):
+            return pl.from_pandas(data), False
+        if isinstance(data, dict):
+            return pl.DataFrame(data), False
+        if isinstance(data, list):
+            if not data:
+                if allow_empty_list:
+                    return pl.DataFrame(), True
+                raise ValidationError(
+                    "Cannot create DataFrame from empty list. Provide a non-empty list of dictionaries, lists, or other supported data types."
+                )
+            if not isinstance(data[0], dict):
+                raise ValidationError(
+                    "Cannot create DataFrame from list of non-dict values. Provide a list of dictionaries."
+                )
+            return pl.DataFrame(data), True
+        if isinstance(data, pa.Table):
+            return pl.from_arrow(data), False
+        raise ValidationError(
+            f"Unsupported data type: {type(data)}. Supported types are: Polars DataFrame, Pandas DataFrame, dict, list, or PyArrow Table."
+        )
+    except ValidationError:
+        raise
+    except Exception as e:
+        raise PlanError(f"Failed to create DataFrame from {data}") from e
+
+
+def _coerce_to_schema(
+    pl_df: pl.DataFrame,
+    schema: Schema,
+    *,
+    row_oriented: bool,
+) -> pl.DataFrame:
+    _validate_explicit_schema(schema)
+    target_schema = convert_custom_schema_to_polars_schema(schema)
+    ordered_names = schema.column_names()
+    try:
+        if pl_df.width == 0 and pl_df.height == 0:
+            return _schema_only_empty_frame(schema)
+        return pl_df.select(ordered_names).cast(target_schema)
+    except Exception as e:
+        raise PlanError(f"Failed to create DataFrame with schema {schema}") from e
+
+
+def _schema_only_empty_frame(schema: Schema) -> pl.DataFrame:
+    return pl.DataFrame(schema=convert_custom_schema_to_polars_schema(schema))
+
+
+def _validate_explicit_schema(schema: Schema) -> None:
+    InMemorySource.from_schema(pl.DataFrame(), schema)
```

#### 1.4 Add Basic API Tests

**File**: `tests/api/test_session.py`
**Changes**: Around the existing `create_dataframe` tests at lines 48-145, import `Schema`, `FloatType`, and `PlanError`, then add basic schema-backed tests.

```diff
@@
     ColumnField,
+    FloatType,
@@
     IntegerType,
+    Schema,
@@
 from fenic.core.error import ConfigurationError
+from fenic.core.error import PlanError
 from fenic.core.error import ValidationError as FenicValidationError
@@
+def test_create_dataframe_with_schema_from_dict_coerces_and_orders_columns(local_session):
+    schema = Schema([
+        ColumnField("age", IntegerType),
+        ColumnField("name", StringType),
+        ColumnField("score", FloatType),
+    ])
+    df = local_session.create_dataframe(
+        {"name": ["Alice"], "score": [1], "age": ["42"]},
+        schema=schema,
+    )
+
+    assert df.schema == schema
+    result = df.to_polars()
+    assert result.columns == ["age", "name", "score"]
+    assert result.schema["age"] == pl.Int64
+    assert result.schema["score"] == pl.Float32
+    assert result.to_dict(as_series=False) == {
+        "age": [42],
+        "name": ["Alice"],
+        "score": [1.0],
+    }
+
+
+def test_create_dataframe_with_schema_from_polars_uses_provided_schema(local_session):
+    schema = Schema([ColumnField("name", StringType)])
+    df = local_session.create_dataframe(pl.DataFrame({"name": ["Alice"]}), schema=schema)
+    assert df.schema == schema
+    assert df.to_polars().schema["name"] == pl.String
+
+
+def test_create_dataframe_with_schema_from_list_of_dicts_allows_missing_row_keys(local_session):
+    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
+    df = local_session.create_dataframe([{"name": "Alice"}, {"age": 30}], schema=schema)
+    assert df.schema == schema
+    assert df.to_polars().to_dict(as_series=False) == {
+        "name": ["Alice", None],
+        "age": [None, 30],
+    }
+
+
+def test_create_dataframe_with_schema_allows_empty_list(local_session):
+    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
+    df = local_session.create_dataframe([], schema=schema)
+    assert df.schema == schema
+    assert df.to_polars().schema == {"name": pl.String, "age": pl.Int64}
+    assert df.to_polars().height == 0
```

### Phase 1 Success Criteria

#### Phase 1 Automated Verification

- [x] `OPENAI_API_KEY=dummy-key uv run pytest tests/api/test_session.py -k "create_dataframe_with_schema or create_dataframe_empty_list" -q` (5 passed, 34 deselected)
- [x] `OPENAI_API_KEY=dummy-key uv run pytest --collect-only tests/api/test_session.py -q`
- [x] `uv run --env-file .env uv run pytest tests/api/test_session.py -k "create_dataframe_with_schema or create_dataframe_empty_list" -q` (5 passed, 34 deselected)
- [x] `uv run --env-file .env uv run pytest --collect-only tests/api/test_session.py -q`

#### Phase 1 Manual Verification

None. The behavior is API/data contract behavior covered by automated tests.

> **Implementation Note:** after automated verification passes, record the commands and results in this plan and pause for human confirmation before proceeding to Phase 2.

---

## ✅ Phase 2: Complete input-shape and error contract

### Phase 2 Overview

Finish the public input contract for every supported `DataLike` shape and make schema enforcement strict before Polars' permissive construction behavior can silently accept missing or extra fields.

### Phase 2 Changes Required

#### 2.1 Enforce Column-Oriented Shape Rules

**File**: `src/fenic/api/session/session.py`
**Changes**: Expand `_coerce_to_schema` after Phase 1 so column-oriented inputs must match the schema's top-level name set unless they are zero-column/zero-row schema-only empties.

```diff
@@
 def _coerce_to_schema(
@@
 ) -> pl.DataFrame:
     _validate_explicit_schema(schema)
     target_schema = convert_custom_schema_to_polars_schema(schema)
     ordered_names = schema.column_names()
+    schema_names = set(ordered_names)
     try:
         if pl_df.width == 0 and pl_df.height == 0:
             return _schema_only_empty_frame(schema)
+        data_names = set(pl_df.columns)
+        if not row_oriented and data_names != schema_names:
+            _raise_schema_column_mismatch(schema_names, data_names)
         return pl_df.select(ordered_names).cast(target_schema)
@@
+def _raise_schema_column_mismatch(expected: set[str], actual: set[str]) -> None:
+    missing = sorted(expected - actual)
+    extra = sorted(actual - expected)
+    details = []
+    if missing:
+        details.append(f"missing columns: {missing}")
+    if extra:
+        details.append(f"extra columns: {extra}")
+    raise ValidationError(
+        "Data columns must match the provided schema exactly; " + ", ".join(details)
+    )
```

#### 2.2 Enforce Row-Oriented Shape Rules Before Polars Construction

**File**: `src/fenic/api/session/session.py`
**Changes**: In `_normalize_data_like_to_polars`, validate every non-empty list item is a dict. In `_coerce_to_schema`, check row key extras before selecting/casting and add missing schema columns as nulls.

```diff
@@
         if isinstance(data, list):
@@
-            if not isinstance(data[0], dict):
+            if not all(isinstance(row, dict) for row in data):
                 raise ValidationError(
                     "Cannot create DataFrame from list of non-dict values. Provide a list of dictionaries."
                 )
@@
         if pl_df.width == 0 and pl_df.height == 0:
             return _schema_only_empty_frame(schema)
         data_names = set(pl_df.columns)
-        if not row_oriented and data_names != schema_names:
+        if row_oriented:
+            extra = data_names - schema_names
+            if extra:
+                _raise_schema_column_mismatch(schema_names, data_names)
+        elif data_names != schema_names:
             _raise_schema_column_mismatch(schema_names, data_names)
+        for missing_col in schema_names - data_names:
+            pl_df = pl_df.with_columns(pl.lit(None).alias(missing_col))
         return pl_df.select(ordered_names).cast(target_schema)
```

#### 2.3 Preserve Fenic Error Boundaries

**File**: `src/fenic/api/session/session.py`
**Changes**: Keep contract violations as `ValidationError`; keep physical construction/cast failures as `PlanError` with the original exception as cause.

```diff
@@
-    try:
+    try:
         if pl_df.width == 0 and pl_df.height == 0:
             return _schema_only_empty_frame(schema)
@@
         return pl_df.select(ordered_names).cast(target_schema)
+    except ValidationError:
+        raise
     except Exception as e:
         raise PlanError(f"Failed to create DataFrame with schema {schema}") from e
```

#### 2.4 Add Full API Input and Error Tests

**File**: `tests/api/test_session.py`
**Changes**: Extend the schema-backed test block with pandas, PyArrow, schema-only empty inputs, mismatch errors, uncastable value errors, and duplicate-name validation.

```diff
@@
+def test_create_dataframe_with_schema_from_pandas_and_arrow(local_session):
+    import pyarrow as pa
+
+    schema = Schema([ColumnField("id", IntegerType), ColumnField("name", StringType)])
+    pd_df = pd.DataFrame({"name": ["Alice"], "id": ["1"]})
+    arrow_table = pa.table({"name": ["Bob"], "id": ["2"]})
+
+    assert local_session.create_dataframe(pd_df, schema=schema).to_polars().columns == ["id", "name"]
+    assert local_session.create_dataframe(arrow_table, schema=schema).to_polars().columns == ["id", "name"]
+
+
+@pytest.mark.parametrize(
+    "empty_input",
+    [
+        {},
+        pl.DataFrame(),
+        pd.DataFrame(),
+        pytest.param(None, id="arrow-empty"),
+    ],
+)
+def test_create_dataframe_schema_only_empty_inputs(local_session, empty_input):
+    import pyarrow as pa
+
+    if empty_input is None:
+        empty_input = pa.table({})
+    schema = Schema([ColumnField("id", IntegerType), ColumnField("name", StringType)])
+    df = local_session.create_dataframe(empty_input, schema=schema)
+    assert df.schema == schema
+    assert df.to_polars().shape == (0, 2)
+
+
+def test_create_dataframe_with_schema_column_missing_or_extra_raises(local_session):
+    schema = Schema([ColumnField("id", IntegerType), ColumnField("name", StringType)])
+
+    with pytest.raises(FenicValidationError, match="missing columns"):
+        local_session.create_dataframe({"id": [1]}, schema=schema)
+    with pytest.raises(FenicValidationError, match="extra columns"):
+        local_session.create_dataframe({"id": [1], "name": ["a"], "extra": [1]}, schema=schema)
+
+
+def test_create_dataframe_with_schema_row_extra_key_raises(local_session):
+    schema = Schema([ColumnField("id", IntegerType)])
+    with pytest.raises(FenicValidationError, match="extra columns"):
+        local_session.create_dataframe([{"id": 1}, {"id": 2, "extra": 3}], schema=schema)
+
+
+def test_create_dataframe_with_schema_later_non_dict_raises(local_session):
+    schema = Schema([ColumnField("id", IntegerType)])
+    with pytest.raises(FenicValidationError, match="list of non-dict"):
+        local_session.create_dataframe([{"id": 1}, 2], schema=schema)
+
+
+def test_create_dataframe_with_schema_uncastable_value_raises_plan_error(local_session):
+    schema = Schema([ColumnField("id", IntegerType)])
+    with pytest.raises(PlanError):
+        local_session.create_dataframe({"id": ["not-an-int"]}, schema=schema)
+
+
+def test_create_dataframe_with_schema_duplicate_names_use_plan_validation(local_session):
+    schema = Schema([
+        ColumnField("id", IntegerType),
+        ColumnField("id", StringType),
+    ])
+    with pytest.raises(PlanError, match="Duplicate column names"):
+        local_session.create_dataframe({"id": [1]}, schema=schema)
```

### Phase 2 Success Criteria

#### Phase 2 Automated Verification

- [x] `uv run --env-file .env uv run pytest tests/api/test_session.py -k "create_dataframe_with_schema or create_dataframe_empty" -q` (17 passed, 34 deselected)
- [x] `uv run --env-file .env uv run pytest tests/api/test_session.py -k "create_dataframe" -q` (23 passed, 28 deselected)

#### Phase 2 Manual Verification

None. The phase defines deterministic input/error contracts and covers them with pytest.

> **Implementation Note:** after automated verification passes, record the commands and results in this plan and pause for human confirmation before proceeding to Phase 3.

---

## ✅ Phase 3: Logical schema propagation through planning, serde, and cloud request boundaries

### Phase 3 Overview

Prove explicit schemas are visible to downstream logical validation and serialization. This phase should need little or no production code if Phases 1-2 correctly use `InMemorySource.from_schema`.

### Phase 3 Changes Required

#### 3.1 Add Logical Tagged-String API Tests

**File**: `tests/api/test_session.py`
**Changes**: Import `JsonType` and `MarkdownType`, then assert logical schemas remain tagged while physical storage stays `pl.String`.

```diff
@@
     IntegerType,
+    JsonType,
+    MarkdownType,
@@
+def test_create_dataframe_with_json_schema_exposes_logical_type(local_session):
+    schema = Schema([ColumnField("json_col", JsonType)])
+    df = local_session.create_dataframe({"json_col": ['{"user": "Alice"}']}, schema=schema)
+    assert df.schema == schema
+    assert df.to_polars().schema["json_col"] == pl.String
+
+
+def test_create_dataframe_with_markdown_schema_exposes_logical_type(local_session):
+    schema = Schema([ColumnField("md_col", MarkdownType)])
+    df = local_session.create_dataframe({"md_col": ["# Title"]}, schema=schema)
+    assert df.schema == schema
+    assert df.to_polars().schema["md_col"] == pl.String
```

#### 3.2 Add JSON Function Validation Without Intermediate Cast

**File**: `tests/_backends/local/functions/test_jq.py`
**Changes**: Add `Schema` import and a schema-backed `json.jq` test near the existing jq tests around line 28.

```diff
@@
     JsonType,
+    Schema,
@@
+def test_jq_accepts_create_dataframe_json_schema(local_session):
+    schema = Schema([ColumnField("json_col", JsonType)])
+    df = local_session.create_dataframe(
+        {"json_col": ['{"user": {"name": "Alice"}}', '{"user": {"name": "Bob"}}']},
+        schema=schema,
+    )
+    result = df.select(json.jq(col("json_col"), ".user.name").alias("user_name")).to_polars()
+    assert result.equals(pl.DataFrame({"user_name": [['"Alice"'], ['"Bob"']]}))
```

#### 3.3 Add Markdown Function Validation Without Intermediate Cast

**File**: `tests/_backends/local/functions/test_markdown.py`
**Changes**: Add `Schema` import and a schema-backed `markdown.generate_toc` test before `test_md_generate_toc`.

```diff
@@
     MarkdownType,
+    Schema,
@@
+def test_generate_toc_accepts_create_dataframe_markdown_schema(local_session):
+    schema = Schema([ColumnField("md_col", MarkdownType)])
+    df = local_session.create_dataframe(
+        {"md_col": ["# Title\n\n## Details"]},
+        schema=schema,
+    )
+    result = df.select(markdown.generate_toc(col("md_col")).alias("toc")).to_polars()
+    assert result["toc"].to_list() == ["# Title\n## Details"]
```

#### 3.4 Add Explicit-Schema Plan Serde Coverage

**File**: `tests/_logical_plan/serde/test_plan_serde.py`
**Changes**: Add imports for tagged string/embedding types as needed, then add explicit-schema round-trip tests near `test_basic_plan` around line 247.

```diff
@@
     IntegerType,
+    JsonType,
+    MarkdownType,
     Schema,
@@
+@pytest.mark.parametrize("serde_implementation", serde_implementations)
+def test_inmemory_source_with_explicit_schema_round_trips(
+    local_session,
+    serde_implementation: SupportsLogicalPlanSerde,
+):
+    schema = Schema([
+        ColumnField("id", IntegerType),
+        ColumnField("payload", JsonType),
+    ])
+    df = local_session.create_dataframe({"id": ["1"], "payload": ['{"ok": true}']}, schema=schema)
+    deserialized = _test_plan_serialization(
+        df._logical_plan,
+        local_session._session_state,
+        serde_implementation,
+    )
+    assert deserialized.schema() == schema
+    assert deserialized == df._logical_plan
```

#### 3.5 Add Schema-Sensitive InMemorySource Equality Case

**File**: `tests/_logical_plan/test_plan_equality.py`
**Changes**: Import `JsonType` and add a source equality test next to `test_schema_mismatch` at lines 123-133.

```diff
@@
-from fenic.core.types.datatypes import FloatType, IntegerType, StringType
+from fenic.core.types.datatypes import FloatType, IntegerType, JsonType, StringType
@@
+    def test_inmemory_source_same_physical_data_different_logical_schema_not_equal(self):
+        test_df = pl.DataFrame({"payload": ['{"ok": true}']})
+        string_schema = Schema([ColumnField("payload", StringType)])
+        json_schema = Schema([ColumnField("payload", JsonType)])
+
+        assert InMemorySource.from_schema(test_df, string_schema) != InMemorySource.from_schema(test_df, json_schema)
```

#### 3.6 Add Cloud Request-Boundary Assertion

**File**: `tests/_backends/cloud/test_cloud_execution.py`
**Changes**: Import the production `LogicalPlanSerde`, patch `StartExecution` for a count action, deserialize `request.count.serialized_plan`, and assert the embedded `InMemorySource` logical schema. Anchor near `test_cloud_simple_count` at line 301.

```diff
@@
 from fenic import ColumnField, IntegerType, Schema, StringType, configure_logging
+from fenic.core._logical_plan.plans import InMemorySource
+from fenic.core._serde import LogicalPlanSerde
@@
+def test_cloud_create_dataframe_explicit_schema_serializes_source_schema(cloud_session, mock_engine_service):
+    expected_schema = Schema([
+        ColumnField("id", IntegerType),
+        ColumnField("name", StringType),
+    ])
+
+    async def _start_execution(request: StartExecutionRequest, metadata):
+        plan = LogicalPlanSerde.deserialize(request.count.serialized_plan)
+        assert isinstance(plan, InMemorySource)
+        assert plan.schema() == expected_schema
+        return StartExecutionResponse(execution_id="test-execution-id")
+
+    async def _get_execution_result(execution_id, metadata):
+        return GetExecutionResultResponse(count_result=1)
+
+    mock_engine_service.StartExecution = _start_execution
+    mock_engine_service.GetExecutionResult = _get_execution_result
+
+    df = cloud_session.create_dataframe({"name": ["Alice"], "id": ["1"]}, schema=expected_schema)
+    assert df.count() == 1
+    mock_engine_service.StartExecution = MockEngineService().StartExecution
```

### Phase 4 Success Criteria

#### Phase 3 Automated Verification

- [x] `uv run --env-file .env uv run pytest tests/api/test_session.py tests/_backends/local/functions/test_jq.py tests/_backends/local/functions/test_markdown.py -k "schema" -q` (20 passed, 54 deselected)
- [x] `uv run --env-file .env uv run pytest tests/_logical_plan/serde/test_plan_serde.py -k "explicit_schema or basic_plan" -q` (4 passed, 38 deselected)
- [x] `uv run --env-file .env uv run pytest tests/_logical_plan/test_plan_equality.py -k "schema_mismatch or inmemory_source" -q` (3 passed, 24 deselected)
- [x] `uv sync --extra=cloud --extra=google --extra=anthropic --extra=cohere --extra=mcp` (2nd-party `just` unavailable in env; sync succeeded)
- [x] `uv run --env-file .env uv run pytest tests/_backends/cloud/test_cloud_execution.py -k "explicit_schema or simple_count" -q` (2 passed, 6 deselected)

#### Phase 3 Manual Verification

None. The phase verifies logical schema propagation through tests and mock request inspection.

> **Implementation Note:** after automated verification passes, record the commands and results in this plan and pause for human confirmation before proceeding to Phase 4.

---

## ✅ Phase 4: Local embedding physical preservation

### Phase 4 Overview

Preserve fixed-size `pl.Array` materialization for explicit `EmbeddingType` schema paths during local in-memory execution while keeping current no-schema array normalization and unrelated array-to-list coercions.

### Phase 4 Changes Required

#### 4.1 Pass Logical Source Schema Into Local InMemorySourceExec

**File**: `src/fenic/_backends/local/transpiler/plan_converter.py`
**Changes**: Around lines 162-166, pass `logical.schema()` when lowering `InMemorySource`.

```diff
@@
         elif isinstance(logical, InMemorySource):
             return InMemorySourceExec(
                 df=logical._source,
+                schema=logical.schema(),
                 session_state=self.session_state,
             )
```

#### 4.2 Store Schema on InMemorySourceExec and Preserve It Through Cloning

**File**: `src/fenic/_backends/local/physical_plan/source.py`
**Changes**: Around lines 15-40, import `Schema`, add the constructor parameter, pass it to ingestion coercion, and preserve it in `with_children`.

```diff
@@
 if TYPE_CHECKING:
     from fenic._backends.local.session_state import LocalSessionState
+    from fenic.core.types.schema import Schema
@@
 class InMemorySourceExec(PhysicalPlan):
-    def __init__(self, df: pl.DataFrame, session_state: LocalSessionState):
+    def __init__(self, df: pl.DataFrame, schema: Schema, session_state: LocalSessionState):
         super().__init__(children=[], cache_info=None, session_state=session_state)
         self.df = df
+        self.schema = schema
@@
-        return apply_ingestion_coercions(self.df, coerce_array=True)
+        return apply_ingestion_coercions(
+            self.df,
+            coerce_array=True,
+            logical_schema=self.schema,
+        )
@@
-        return InMemorySourceExec(self.df, self.session_state)
+        return InMemorySourceExec(self.df, self.schema, self.session_state)
```

#### 4.3 Make Ingestion Coercion Schema-Aware Only for Embedding Paths

**File**: `src/fenic/_backends/local/physical_plan/utils.py`
**Changes**: Around lines 1-53, accept `logical_schema: Schema | None`; thread the matching logical dtype into `_build_target_dtype`; preserve `EmbeddingType` as its physical dtype; recurse through matching `StructType` and `ArrayType` paths.

```diff
@@
-from typing import Optional, Union
+from typing import Optional, Union
@@
 import polars as pl
+from fenic.core._utils.schema import convert_custom_dtype_to_polars
+from fenic.core.types.datatypes import ArrayType, EmbeddingType, StructType
+from fenic.core.types.schema import Schema
@@
-def apply_ingestion_coercions(df: pl.DataFrame, coerce_array: bool) -> pl.DataFrame:
+def apply_ingestion_coercions(
+    df: pl.DataFrame,
+    coerce_array: bool,
+    logical_schema: Schema | None = None,
+) -> pl.DataFrame:
@@
     expressions = []
+    logical_fields = (
+        {field.name: field.data_type for field in logical_schema.column_fields}
+        if logical_schema is not None
+        else {}
+    )
     for col_name in df.columns:
         dtype = df[col_name].dtype
-        target_dtype = _build_target_dtype(dtype, coerce_array)
+        target_dtype = _build_target_dtype(
+            dtype,
+            coerce_array,
+            logical_dtype=logical_fields.get(col_name),
+        )
@@
-def _build_target_dtype(dtype: pl.DataType, coerce_array: bool) -> pl.DataType:
-    if isinstance(dtype, pl.Array) and coerce_array:
-        return pl.List(_build_target_dtype(dtype.inner, coerce_array))
+def _build_target_dtype(
+    dtype: pl.DataType,
+    coerce_array: bool,
+    logical_dtype=None,
+) -> pl.DataType:
+    if isinstance(logical_dtype, EmbeddingType):
+        return convert_custom_dtype_to_polars(logical_dtype)
+    if isinstance(dtype, pl.Array) and coerce_array:
+        return pl.List(_build_target_dtype(dtype.inner, coerce_array))
     elif isinstance(dtype, pl.List):
-        return pl.List(_build_target_dtype(dtype.inner, coerce_array))
+        element_logical_dtype = (
+            logical_dtype.element_type if isinstance(logical_dtype, ArrayType) else None
+        )
+        return pl.List(
+            _build_target_dtype(dtype.inner, coerce_array, element_logical_dtype)
+        )
@@
     elif isinstance(dtype, pl.Struct):
+        logical_field_types = (
+            {field.name: field.data_type for field in logical_dtype.struct_fields}
+            if isinstance(logical_dtype, StructType)
+            else {}
+        )
         new_fields = [
-            pl.Field(field.name, _build_target_dtype(field.dtype, coerce_array))
+            pl.Field(
+                field.name,
+                _build_target_dtype(
+                    field.dtype,
+                    coerce_array,
+                    logical_field_types.get(field.name),
+                ),
+            )
             for field in dtype.fields
         ]
```

#### 4.4 Add Ingestion Coercion Unit Tests

**File**: `tests/_backends/test_ingestion_coercion.py`
**Changes**: Import fenic schema/data types and add top-level, mixed, and nested embedding preservation tests after the existing array tests.

```diff
@@
 from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
+from fenic.core.types.datatypes import ArrayType, EmbeddingType, IntegerType, StructField, StructType
+from fenic.core.types.schema import ColumnField, Schema
@@
+def test_array_coercion_preserves_explicit_embedding_schema():
+    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")
+    schema = Schema([ColumnField("embedding", embedding_type)])
+    df = pl.DataFrame(
+        {"embedding": [[1.0, 2.0, 3.0]]},
+        schema={"embedding": pl.Array(pl.Float32, 3)},
+    )
+
+    result = apply_ingestion_coercions(df, coerce_array=True, logical_schema=schema)
+    assert result.schema["embedding"] == pl.Array(pl.Float32, 3)
+
+
+def test_schema_aware_coercion_keeps_unrelated_arrays_as_lists():
+    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")
+    schema = Schema([
+        ColumnField("embedding", embedding_type),
+        ColumnField("array_col", ArrayType(IntegerType)),
+    ])
+    df = pl.DataFrame(
+        {"embedding": [[1.0, 2.0, 3.0]], "array_col": [[1, 2, 3]]},
+        schema={"embedding": pl.Array(pl.Float32, 3), "array_col": pl.Array(pl.Int64, 3)},
+    )
+
+    result = apply_ingestion_coercions(df, coerce_array=True, logical_schema=schema)
+    assert result.schema["embedding"] == pl.Array(pl.Float32, 3)
+    assert result.schema["array_col"] == pl.List(pl.Int64)
+
+
+def test_schema_aware_coercion_preserves_nested_embedding_in_struct():
+    embedding_type = EmbeddingType(dimensions=2, embedding_model="test")
+    schema = Schema([
+        ColumnField("payload", StructType([
+            StructField("embedding", embedding_type),
+            StructField("values", ArrayType(IntegerType)),
+        ]))
+    ])
+    df = pl.DataFrame(
+        {"payload": [{"embedding": [1.0, 2.0], "values": [1, 2]}]},
+        schema={"payload": pl.Struct([
+            pl.Field("embedding", pl.Array(pl.Float32, 2)),
+            pl.Field("values", pl.Array(pl.Int64, 2)),
+        ])},
+    )
+
+    result = apply_ingestion_coercions(df, coerce_array=True, logical_schema=schema)
+    assert result.schema["payload"] == pl.Struct([
+        pl.Field("embedding", pl.Array(pl.Float32, 2)),
+        pl.Field("values", pl.List(pl.Int64)),
+    ])
```

#### 4.5 Extend Transpiler Source Test

**File**: `tests/_backends/local/test_transpiler.py`
**Changes**: Around lines 83-91, create an explicit schema source and assert the physical node stores it.

```diff
@@
-from fenic.core.types import IntegerType
+from fenic.core.types import ColumnField, IntegerType, Schema
@@
 def test_convert_source_plan(local_session):
     df = pl.DataFrame({"a": [1, 2, 3]})
-    source = InMemorySource(df, local_session._session_state)
+    schema = Schema([ColumnField("a", IntegerType)])
+    source = InMemorySource.from_schema(df, schema)
     plan_converter = PlanConverter(local_session._session_state)
@@
     assert isinstance(physical, InMemorySourceExec)
+    assert physical.schema == schema
```

#### 4.6 Add Public Embedding Schema Materialization Test

**File**: `tests/api/test_session.py`
**Changes**: Import `EmbeddingType` and add an API-level regression that checks both logical and physical schema.

```diff
@@
     ColumnField,
+    EmbeddingType,
@@
+def test_create_dataframe_with_embedding_schema_preserves_polars_array(local_session):
+    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")
+    schema = Schema([ColumnField("embedding", embedding_type)])
+
+    df = local_session.create_dataframe(
+        {"embedding": [[1.0, 2.0, 3.0]]},
+        schema=schema,
+    )
+
+    assert df.schema == schema
+    assert df.to_polars().schema["embedding"] == pl.Array(pl.Float32, 3)
```

#### 4.7 Add Embedding Explicit-Schema Serde Regression

**File**: `tests/_logical_plan/serde/test_plan_serde.py`
**Changes**: Extend imports and add a serde test near the explicit-schema round-trip from Phase 3.

```diff
@@
     ColumnField,
+    EmbeddingType,
@@
+@pytest.mark.parametrize("serde_implementation", serde_implementations)
+def test_inmemory_source_with_embedding_schema_round_trips(
+    local_session,
+    serde_implementation: SupportsLogicalPlanSerde,
+):
+    schema = Schema([
+        ColumnField("embedding", EmbeddingType(dimensions=3, embedding_model="test")),
+    ])
+    df = local_session.create_dataframe(
+        {"embedding": [[1.0, 2.0, 3.0]]},
+        schema=schema,
+    )
+    deserialized = _test_plan_serialization(
+        df._logical_plan,
+        local_session._session_state,
+        serde_implementation,
+    )
+    assert deserialized.schema() == schema
```

### Success Criteria

#### Phase 4 Automated Verification

- [x] `uv run pytest tests/_backends/test_ingestion_coercion.py -q` (11 passed, 0 failed, 0 skipped)
- [x] `uv run pytest tests/_backends/local/test_transpiler.py -k "source_plan" -q` (1 passed, 9 deselected)
- [x] `uv run pytest tests/api/test_session.py -k "embedding_schema or create_dataframe_with_schema" -q` (17 passed, 37 deselected)
- [x] `uv run pytest tests/_logical_plan/serde/test_plan_serde.py -k "embedding or explicit_schema" -q` (4 passed, 40 deselected)
- [ ] `just sync=false test-local` (not available in this environment; `just` is unavailable)

#### Phase 4 Manual Verification

None. The behavior is observable through logical schema assertions and Polars physical schema assertions.

> **Implementation Note:** after automated verification passes, record the commands and results in this plan and pause for human confirmation before committing or advancing to review.

## Implementation Notes

**Implemented head:** pending commit

**Spec source:** `docs/td-flow/create-dataframe-schema/plan.md`

**Verification summary:** `uv run --env-file .env uv run pytest tests/api/test_session.py tests/_backends/local/functions/test_jq.py tests/_backends/local/functions/test_markdown.py -k "schema" -q` (22 passed, 53 deselected); `uv run --env-file .env uv run pytest tests/_logical_plan/serde/test_plan_serde.py -k "explicit_schema or basic_plan" -q` (4 passed, 38 deselected); `uv run --env-file .env uv run pytest tests/_logical_plan/test_plan_equality.py -k "schema_mismatch or inmemory_source" -q` (3 passed, 24 deselected); `uv sync --extra=cloud --extra=google --extra=anthropic --extra=cohere --extra=mcp` (success); `uv run --env-file .env uv run pytest tests/_backends/cloud/test_cloud_execution.py -k "explicit_schema or simple_count" -q` (2 passed, 6 deselected); `uv run pytest tests/_backends/test_ingestion_coercion.py -q` (11 passed, 0 failed, 0 skipped); `uv run pytest tests/_backends/local/test_transpiler.py -k "source_plan" -q` (1 passed, 9 deselected); `uv run pytest tests/api/test_session.py -k "embedding_schema or create_dataframe_with_schema" -q` (17 passed, 37 deselected); `uv run pytest tests/_logical_plan/serde/test_plan_serde.py -k "embedding or explicit_schema" -q` (4 passed, 40 deselected); `uv run trunk check --fix` (no issues)

**Deliberate tradeoffs / rejected approaches:** kept embedding preservation narrowly in local ingestion coercion and preserved no-schema array-to-list behavior; avoided broad source recasting by using logical schema only when executing `InMemorySource`.

**Deviations from spec:** `just` recipes could not be executed in this environment, so direct `uv` equivalents were used where needed and logged in checklists.

**Reviewer context:** `just` is unavailable in this environment (`mise.toml` removal); if needed, re-run full local `just` recipes in CI/local environments with `just` installed.

---

## Decision Log

- **[applied]** The plan preserves the structure's four vertical phases and keeps implementation changes concentrated in `Session.create_dataframe`, local source execution, and targeted tests.
- **[applied]** The no-schema path is intentionally left as a direct `InMemorySource.from_session_state` branch so current empty-list rejection, type inference, and generic ingestion coercions remain behavior-compatible.
- **[applied]** Explicit schemas are validated through `InMemorySource.from_schema(pl.DataFrame(), schema)` before converting to a dict-backed Polars schema, preventing duplicate names from being hidden by `convert_custom_schema_to_polars_schema`.
- **[applied]** The local physical seam passes `logical.schema()` into `InMemorySourceExec`, but the utility only uses that schema to preserve `EmbeddingType` paths. This avoids broad recasting of inferred/deserialized sources.
- **[applied]** Verification commands come from `CLAUDE.md` and the real `justfile`; `just -n sync=false test-local`, `just -n sync=false test-cloud`, and `just -n sync=false sync-cloud` validated the parameter form.
- **[applied]** Direct `uv run pytest ... -k ...` commands are used for phase-targeted coverage because `test-local` and `test-cloud` recipes do not expose file or keyword passthrough. This carries recipe-drift risk if global pytest flags change.
- **[applied]** No dedicated typecheck command exists in `just --list`, `pyproject.toml`, or scanned repo config. The plan uses pytest collection, targeted tests, and full local recipe coverage instead.
- **[applied]** plan-eng-review lens completed inline with portability tier 3: minimum change set is acceptable for high-risk scope, no new classes/services are introduced, existing source/schema/coercion seams are reused, and test coverage traces API, logical, serde, cloud request, and local physical paths.
- **[needs-fix]** td-review for `45ccca180a31209ce8be92fcc9b3919cd418080b` found no remaining blocking code-level issues in the reviewed worktree, but the reviewed fixes are not committed to the branch head. Commit the current worktree changes, then run a delta review before `td-land`. Review artifact: `.context/td-review/create-dataframe-schema-45ccca180a31209ce8be92fcc9b3919cd418080b.md`.
- **[approved]** Delta td-review for committed fix head `77c3af7c459788bb6f077fa66288c6df2400ca51` returned Ready to merge: the prior dirty-tree blocker is resolved, target freshness is clean, and focused pytest / `fenic check` / compile verification passed. Review artifact: `.context/td-review/create-dataframe-schema-77c3af7c459788bb6f077fa66288c6df2400ca51.md`.
- **[deferred]** Public documentation updates belong in a follow-up documentation/release pass after behavior lands.

## Open Questions

None.

## Handoff

**Next step (paste into a fresh tab):**

> Use the `td-implement` skill. The plan is approved at
> `docs/td-flow/create-dataframe-schema/plan.md` (track: engineering, size: high-risk).
> Implement Phase 1 from the plan, run the repo's verification, then pause for my
> approval before committing or advancing.
> If this is Codex: I explicitly permit optional subagent use for this phase
> where the skill allows it.

**Approved decisions:** Add `schema: Schema | None = None` to `Session.create_dataframe` / `createDataFrame`; enforce complete top-level schema contracts at session ingestion; allow schema-backed empty inputs; reuse `InMemorySource.from_schema`; preserve logical string-backed types without content validation; pass logical source schema into local in-memory execution only to protect `EmbeddingType` paths from array-to-list coercion.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** Partial schemas, Pydantic schema shortcuts, new cloud protocol fields, a new source node, broad source recasting, tagged-string content validation, and user-facing docs in this implementation pass.
**Evidence summary:** `Session.create_dataframe` currently normalizes supported inputs before `InMemorySource.from_session_state` (`src/fenic/api/session/session.py:186`); `InMemorySource.from_schema` already stores explicit schemas (`src/fenic/core/_logical_plan/plans/source.py:31`); wrapper plan serde already carries logical schemas (`src/fenic/core/_serde/proto/plan_serde.py:48`); `convert_custom_schema_to_polars_schema` maps `EmbeddingType` to fixed-size Polars arrays (`src/fenic/core/_utils/schema.py:185`); local `InMemorySourceExec` currently applies generic array-to-list coercion (`src/fenic/_backends/local/physical_plan/source.py:35`, `src/fenic/_backends/local/physical_plan/utils.py:39`).
**Known weak assumptions:** The exact cloud request field name for the serialized plan should be pressure-tested during Phase 3 implementation against `CloudExecution`; nested embedding preservation should be pressure-tested against Polars' nested array casting behavior.
**Next artifact:** code + Implementation Notes (Implement edits this plan in place with progress markers, then hands off to `td-review`).
**Rollback if:** a phase's code example proves infeasible against the real code - ROLLBACK to structure.
