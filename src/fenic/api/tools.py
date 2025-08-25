"""API-layer generators for automatic MCP tools from DataFrames.

These helpers generate DynamicTool definitions for:
- Schema: dataset column names and types
- Describe: per-column statistics (counts, numeric summaries, simple string summaries)
- Analyze: DuckDB SQL across one or more datasets.

All generated tools return LogicalPlan objects. The MCP server wrapper handles
execution and result formatting.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional, Union

import polars as pl
from mcp.server.fastmcp.exceptions import ValidationError
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated

from fenic.api.dataframe.dataframe import DataFrame
from fenic.api.functions import (
    avg,
    col,
    count,
    stddev,
)
from fenic.api.functions import max as max_
from fenic.api.functions import min as min_
from fenic.api.session.session import Session
from fenic.core._logical_plan.plans import InMemorySource
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.tools import (
    DynamicTool,
)
from fenic.core._utils.structured_outputs import convert_pydantic_model_to_key_descriptions
from fenic.core.error import ConfigurationError
from fenic.core.types.datatypes import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
)


@dataclass
class DatasetSpec:
    """Specification for a dataset exposed to a tool.

    - name: short identifier used in SQL (e.g., {orders}) and displayed to the model
    - description: brief description of dataset contents/semantics for tooldocs
    - df: the DataFrame object
    """
    name: str
    description: str
    df: DataFrame

class ReadFuncParams(BaseModel):
    model_config = ConfigDict(strict=False)

    df_name: str = Field(description="Dataset name to read")
    projection: list[str] | None = Field(description="Columns to project (subset)")
    limit: int | None = Field(description="Max rows to read (accepts number or numeric string)")
    offset: int | None = Field(description="Row offset to start from (requires order_by; accepts number or numeric string)")
    order_by: list[str] | None = Field(description="Columns to order by (required for offset)")
    sort_ascending: bool | None = Field(description="Sort ascending for all order_by columns")

def auto_generate_read_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 50,
) -> DynamicTool:
    """Create a read tool over one or many datasets."""
    if len(datasets) == 0:
        raise ConfigurationError("Cannot create read tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.name: d.df for d in datasets}
    def read_func(params: ReadFuncParams) -> LogicalPlan:
        if isinstance(params, str):
            params = ReadFuncParams.model_validate_json(params)
        if params.df_name not in name_to_df:
            raise ValidationError(f"Unknown DataFrame '{params.df_name}'. Available: {', '.join(name_to_df.keys())}")
        df = name_to_df[params.df_name]

        # projection
        if params.projection:
            missing = [c for c in params.projection if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Column(s) {missing} do not exist in DataFrame. Available columns: {', '.join(df.columns)}"
                )
            df = df.select(*params.projection)

        # order_by when not paginating via OFFSET (to avoid double sorting)
        if params.order_by and params.offset is None:
            missing_order = [c for c in params.order_by if c not in df.columns]
            if missing_order:
                raise ValidationError(
                    f"order_by column(s) {missing_order} do not exist in DataFrame. Available columns: {', '.join(df.columns)}"
                )
            df = df.order_by(params.order_by, ascending=True if params.sort_ascending is None else bool(params.sort_ascending))

        # offset requires order_by and falls back to SQL for OFFSET
        if params.offset is not None:
            if not params.order_by:
                raise ValidationError("offset requires order_by to ensure deterministic paging.")
            # Validate order_by identifiers and reconstruct ORDER BY clause safely
            missing_order = [c for c in params.order_by if c not in df.columns]
            if missing_order:
                raise ValidationError(
                    f"order_by column(s) {missing_order} do not exist in DataFrame. Available columns: {', '.join(df.columns)}"
                )
            direction = "ASC" if (params.sort_ascending is None or params.sort_ascending) else "DESC"
            safe_order_by = ", ".join(params.order_by)
            # Coerce numeric inputs that might arrive as strings
            lim_val = None if params.limit is None else int(str(params.limit))
            off_val = int(str(params.offset))
            base_sql = "SELECT * FROM {src} ORDER BY " + safe_order_by + f" {direction}"
            if lim_val is not None:
                base_sql += f" LIMIT {lim_val}"
            base_sql += f" OFFSET {off_val}"
            df_with_paging = session.sql(base_sql, src=df)
            return df_with_paging._logical_plan

        if params.limit is not None:
            df = df.limit(int(str(params.limit)))
        return df._logical_plan

    schema_description = convert_pydantic_model_to_key_descriptions(ReadFuncParams)
    enriched_description = f"{tool_description}\n\n{schema_description}"
    return DynamicTool(
        name=tool_name,
        description=enriched_description,
        func=read_func,
        result_limit=result_limit,
    )

"""
Replace single search generator with two split generators:
- auto_generate_search_summary_tool
- auto_generate_search_content_tool
"""

def auto_generate_search_summary_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
) -> DynamicTool:
    """Create a grep-like summary tool over one or many datasets (string columns)."""
    if len(datasets) == 0:
        raise ValueError("Cannot create search summary tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.name: d.df for d in datasets}


    def search_summary(
        pattern: Annotated[str, "Regex pattern to search for (use (?i) for case-insensitive)."],
    ) -> LogicalPlan:
        rows: List[Dict[str, object]] = []
        for name, d in name_to_df.items():
            cols = [f.name for f in d.schema.column_fields if f.data_type == StringType]
            if not cols:
                rows.append({"dataset": name, "total_matches": 0})
                continue
            predicate = None
            for c_name in cols:
                this = col(c_name).rlike(pattern)
                predicate = this if predicate is None else (predicate | this)
            total_count = d.filter(predicate).count()
            rows.append({"dataset": name, "total_matches": int(total_count)})

        pl_df = pl.DataFrame(rows)
        return InMemorySource.from_session_state(pl_df, session._session_state)

    return DynamicTool(
        name=tool_name,
        description=tool_description,
        func=search_summary,
        result_limit=None,
    )

class SearchFuncParams(BaseModel):
    df_name: str = Field(description="Dataset name to search (single dataset)")
    pattern: str = Field(description="Regex pattern to search for (use (?i) for case-insensitive).")
    limit: Optional[int] = Field(default=None,description="Maximum rows to return (accepts number or numeric string)")
    include_columns: Optional[str] = Field(default=None,description="Comma-separated column names to include in the search; if omitted, all string columns")
    offset: Optional[int] = Field(default=None,description="Row offset (requires order_by; accepts number or numeric string)")
    order_by: Optional[str] = Field(default=None,description="ORDER BY comma-separated column names (required with offset)")
    sort_ascending: Optional[bool] = Field(default=True,description="Sort ascending")

def auto_generate_search_content_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 100,
) -> DynamicTool:
    """Create a content search tool for a single dataset (string columns)."""
    if len(datasets) == 0:
        raise ValueError("Cannot create search content tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.name: d.df for d in datasets}

    def _string_columns(df: DataFrame, selected: str | None) -> list[str]:
        if selected:
            selected_columns = [c.strip() for c in selected.split(",") if c.strip()]
            missing = [c for c in selected_columns if c not in df.columns]
            if missing:
                raise ValidationError(f"Column(s) {missing} not found. Available: {', '.join(df.columns)}")
            return selected_columns
        return [f.name for f in df.schema.column_fields if f.data_type == StringType]



    def search_rows(params: SearchFuncParams) -> LogicalPlan:
        if isinstance(params, str):
            params = SearchFuncParams.model_validate_json(params)
        if not params.pattern:
            raise ValidationError("Query pattern cannot be empty.")
        if params.df_name not in name_to_df:
            raise ValidationError(f"Unknown DataFrame '{params.df_name}'. Available: {', '.join(name_to_df.keys())}")
        d = name_to_df[params.df_name]
        cols = _string_columns(d, params.include_columns)
        if not cols:
            return d.limit(0)._logical_plan
        predicate = None
        for c_name in cols:
            this = col(c_name).rlike(params.pattern)
            predicate = this if predicate is None else (predicate | this)
        out = d.filter(predicate)

        if params.offset is not None:
            if not params.order_by:
                raise ValidationError("offset requires order_by for deterministic paging")
            # Validate order_by identifiers against actual columns
            missing_order = [c for c in params.order_by if c not in out.columns]
            if missing_order:
                raise ValidationError(
                    f"order_by column(s) {missing_order} do not exist in DataFrame. Available columns: {', '.join(out.columns)}"
                )
            direction = "ASC" if (params.sort_ascending is None or params.sort_ascending) else "DESC"
            safe_order_by = ", ".join(params.order_by)
            lim_val = None if params.limit is None else int(str(params.limit))
            off_val = int(str(params.offset))
            base_sql = "SELECT * FROM {src} ORDER BY " + safe_order_by + f" {direction}"
            if lim_val is not None:
                base_sql += f" LIMIT {lim_val}"
            base_sql += f" OFFSET {off_val}"
            out = session.sql(base_sql, src=out)
        elif params.limit is not None:
            out = out.limit(int(str(params.limit)))

        return out._logical_plan

    schema_description = convert_pydantic_model_to_key_descriptions(SearchFuncParams)
    enriched_description = f"{tool_description}\n\n{schema_description}"
    return DynamicTool(
        name=tool_name,
        description=enriched_description,
        func=search_rows,
        result_limit=result_limit,
    )


def auto_generate_schema_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
) -> DynamicTool:
    """Create a schema tool over one or many datasets.

    - Returns one row per dataset with a column `schema` containing a list of
      {column, type} entries.
    - If `df_name` is provided, returns only that dataset.
    """
    if len(datasets) == 0:
        raise ValueError("Cannot create schema tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.name: d.df for d in datasets}

    def schema_func(
        df_name: Annotated[str | None, "Optional DataFrame name to return a single schema for"] = None,
    ) -> LogicalPlan:
        # Choose subset of datasets
        if df_name is not None:
            if df_name not in name_to_df:
                raise ValidationError(
                    f"Unknown DataFrame '{df_name}'. Available: {', '.join(name_to_df.keys())}"
                )
            selected = {df_name: name_to_df[df_name]}
        else:
            selected = name_to_df

        per_df_frames: List[pl.DataFrame] = []

        for name, d in selected.items():
            # Build a single-row DataFrame with a common list<struct{column,type}> schema column
            schema_entries = [{"column": f.name, "type": str(f.data_type)} for f in d.schema.column_fields]
            df_row = pl.DataFrame({
                "dataset": [name],
                "schema": [schema_entries],
            })
            per_df_frames.append(df_row)

        combined = pl.concat(per_df_frames, how="vertical") if len(per_df_frames) > 1 else per_df_frames[0]
        return InMemorySource.from_session_state(combined, session._session_state)

    # Enhanced description lists datasets and descriptions
    lines: List[str] = [tool_description.strip(), "", "Datasets available:"]
    for spec in datasets:
        if spec.description:
            lines.append(f"- {spec.name}: {spec.description}")
        else:
            lines.append(f"- {spec.name}")
    enhanced_description = "\n".join(lines)

    return DynamicTool(
        name=tool_name,
        description=enhanced_description,
        func=schema_func,
        result_limit=None,
    )



def auto_generate_sql_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 50,
) -> DynamicTool:
    """Create an Analyze tool that executes DuckDB SELECT SQL across datasets.

    - JOINs between the provided datasets are allowed.
    - DDL/DML, CTEs, subqueries, UNION, and multiple top-level queries are not allowed (enforced upstream).
    - The callable returns a LogicalPlan gathered later by the MCP server.
    """
    if len(datasets) == 0:
        raise ConfigurationError("Cannot create SQL tool: no datasets provided.")

    def _assert_full_sql_shape(sql_text: str) -> None:
        text = sql_text.strip().lower()
        if not text.startswith("select"):
            raise ValidationError("Only SELECT is allowed in full_sql")

    def analyze_func(
        full_sql: Annotated[str, "Full SELECT SQL. Refer to DataFrames by name in braces, e.g., {orders}."]
    ) -> LogicalPlan:
        sql_text = full_sql.strip()
        _assert_full_sql_shape(sql_text)
        return session.sql(sql_text, **{spec.name: spec.df for spec in datasets})._logical_plan

    # Enhanced description with dataset names and descriptions
    lines: List[str] = [tool_description.strip(), "", "Datasets available:"]
    for spec in datasets:
        if spec.description:
            lines.append(f"- {spec.name}: {spec.description}")
        else:
            lines.append(f"- {spec.name}")
    if datasets:
        example_name = datasets[0].name
    else:
        example_name = "data"
    lines.extend(
        [
            "",
            "Notes:",
            "- SQL dialect: DuckDB.",
            "- For text search, prefer regular expressions using REGEXP_MATCHES().",
            "- Paging: use ORDER BY to define row order, then LIMIT and OFFSET for pages.",
            "",
            "Examples:",  # nosec B608 - example text only
            f"- SELECT * FROM {{{example_name}}} WHERE REGEXP_MATCHES(message, '(?i)error|fail') LIMIT 100",  # nosec B608 - example text only
            f"- SELECT dept, COUNT(*) AS n FROM {{{example_name}}} WHERE status = 'active' GROUP BY dept HAVING n > 10 ORDER BY n DESC LIMIT 100",  # nosec B608 - example text only
            f"- -- Paging: page 2 of size 50\n  SELECT * FROM {{{example_name}}} ORDER BY created_at DESC LIMIT 50 OFFSET 50",  # nosec B608 - example text only
        ]
    )
    enhanced_description = "\n".join(lines)

    tool = DynamicTool(
        name=tool_name,
        description=enhanced_description,
        func=analyze_func,
        result_limit=result_limit,
    )
    return tool


def _schema_fingerprint(df: DataFrame) -> str:
    hasher = hashlib.sha256()
    for f in df.schema.column_fields:
        hasher.update(f"{f.name}|{str(f.data_type)}".encode("utf-8"))
    return hasher.hexdigest()[:12]


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def auto_generate_describe_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    topk_distinct: int = 10,
) -> DynamicTool:
    """Create a cached Describe tool for one or many datasets.

    Output columns include:
      - dataset, column, type, row_count, non_null_count, null_count
      - min, max, mean, std (for numerics)
      - distinct_count, top_values (JSON) for strings
      - true_count, false_count for booleans
    """
    if len(datasets) == 0:
        raise ValueError("Cannot create describe tool: no datasets provided.")

    def _compute_describe_rows(df: DataFrame, dataset_name: str) -> List[Dict[str, object]]:
        total_rows = df.count()
        preview = df.limit(10000).to_polars()
        rows_list: List[Dict[str, object]] = []
        for field in df.schema.column_fields:
            col_name = field.name
            dtype_str = str(field.data_type)
            nn = df.agg(count(col(col_name)).alias("c")).to_polars().get_column("c")[0]
            null_count = int(total_rows - nn)
            stats: Dict[str, object] = {
                "dataset": dataset_name,
                "column": col_name,
                "type": dtype_str,
                "row_count": int(total_rows),
                "non_null_count": int(nn),
                "null_count": null_count,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "distinct_count": None,
                "top_values": None,
                "true_count": None,
                "false_count": None,
            }
            if field.data_type in (IntegerType, FloatType, DoubleType):
                agg_df = df.agg(
                    min_(col(col_name)).alias("min"),
                    max_(col(col_name)).alias("max"),
                    avg(col(col_name)).alias("mean"),
                    stddev(col(col_name)).alias("std"),
                ).to_polars()
                stats["min"] = agg_df.get_column("min")[0]
                stats["max"] = agg_df.get_column("max")[0]
                stats["mean"] = agg_df.get_column("mean")[0]
                stats["std"] = agg_df.get_column("std")[0]
            elif field.data_type == BooleanType:
                if col_name in preview.columns:
                    s_bool = preview.get_column(col_name).cast(pl.Boolean)
                    stats["true_count"] = int((s_bool).sum())
                    stats["false_count"] = int((~s_bool).sum())
                else:
                    stats["true_count"] = 0
                    stats["false_count"] = 0
            elif field.data_type == StringType:
                if col_name in preview.columns:
                    s = preview.get_column(col_name)
                    try:
                        avg_len = float(s.str.len_chars().mean())
                    except Exception:
                        avg_len = None
                    try:
                        sample_distinct = int(s.n_unique())
                    except Exception:
                        sample_distinct = None
                else:
                    avg_len = None
                    sample_distinct = None
                compute_topk = (
                    (avg_len is not None and avg_len <= 128)
                    and (sample_distinct is not None and sample_distinct <= max(topk_distinct * 10, 200))
                )
                if compute_topk and col_name in preview.columns:
                    vc = preview.get_column(col_name).value_counts(sort=True)
                    stats["distinct_count"] = int(vc.height)
                    val_col = col_name if col_name in vc.columns else vc.columns[0]
                    top_vals: List[Dict[str, object]] = []
                    for i in range(min(topk_distinct, vc.height)):
                        top_vals.append({
                            "value": str(vc.get_column(val_col)[i]),
                            "count": int(vc.get_column("count")[i]),
                        })
                    stats["top_values"] = json.dumps(top_vals)
                else:
                    stats["distinct_count"] = None
                    stats["top_values"] = json.dumps([])
            rows_list.append(stats)
        return rows_list

    def _materialize_dataset_description(df: DataFrame, dataset_name: str, view_name: str) -> None:
        rows_list = _compute_describe_rows(df, dataset_name)
        safe_rows: List[Dict[str, str]] = []
        for row in rows_list:
            safe_rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
        pl_df = pl.DataFrame(safe_rows)
        plan = InMemorySource.from_session_state(pl_df, session._session_state)
        catalog = session._session_state.catalog
        catalog.create_view(view_name, plan, ignore_if_exists=True)

    def _ensure_describe_view_for_dataset(spec: DatasetSpec, tool_key: str, refresh: bool) -> LogicalPlan:
        schema_hash = _schema_fingerprint(spec.df)
        view_name = f"__fenic_describe__{tool_key}__{_sanitize_name(spec.name)}__{schema_hash}"
        catalog = session._session_state.catalog
        if refresh or not catalog.does_view_exist(view_name):
            _materialize_dataset_description(spec.df, spec.name, view_name)
        return catalog.describe_view(view_name)

    def describe_func(
        df_name: Annotated[str | None, "Optional dataset name to describe only that dataset"] = None,
        refresh: Annotated[bool, "Recompute and refresh cached describe view(s)"] = False,
    ) -> LogicalPlan:
        tool_key = _sanitize_name(tool_name)
        # Single dataset branch returns the view plan directly
        if df_name is not None:
            spec = next((d for d in datasets if d.name == df_name), None)
            if spec is None:
                raise ValidationError(f"Unknown dataset '{df_name}'. Available: {', '.join(d.name for d in datasets)}")
            return _ensure_describe_view_for_dataset(spec, tool_key, refresh)

        # Multi-dataset: concatenate cached views (or compute & cache if missing)
        per_df_polars: List[pl.DataFrame] = []
        for spec in datasets:
            # Ensure view exists and read it, then convert to polars for concatenation
            plan = _ensure_describe_view_for_dataset(spec, tool_key, refresh)
            pl_df = session._session_state.execution.collect(plan)[0]
            per_df_polars.append(pl_df)

        combined = pl.concat(per_df_polars, how="vertical") if len(per_df_polars) > 1 else per_df_polars[0]
        return InMemorySource.from_session_state(combined, session._session_state)

    # Enhanced description: list datasets and notes
    lines: List[str] = [tool_description.strip(), "", "Datasets available:"]
    for spec in datasets:
        if spec.description:
            lines.append(f"- {spec.name}: {spec.description}")
        else:
            lines.append(f"- {spec.name}")
    lines.extend(
        [
            "",
            "Notes:",
            "- Results are cached per dataset, tool name, and schema fingerprint; pass refresh=true to recompute.",
            "- Returns per-column stats (numeric, boolean, string).",
        ]
    )
    enhanced_description = "\n".join(lines)

    return DynamicTool(
        name=tool_name,
        description=enhanced_description,
        func=describe_func,
        result_limit=None,
    )


def auto_generate_core_tools(
    datasets: List[DatasetSpec],
    session: Session,
    *,
    tool_group_name: str,
    sql_max_rows: int = 100,
) -> List[DynamicTool]:
    """Generate core tools spanning all datasets: Schema, Describe, Analyze.

    - Schema: list columns/types for any or all datasets
    - Describe: dataset statistics for any or all datasets
    - Analyze: DuckDB SELECT-only SQL across datasets
    """
    group_desc = "; ".join(
        [f"{d.name}: {d.description.strip()}" if d.description else d.name for d in datasets]
    )

    schema_tool = auto_generate_schema_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Schema",
        tool_description="\n\n".join([
            "Show the schema (column names and types) for any or all of the datasets listed below.",
            group_desc,
        ]),
    )

    describe_tool = auto_generate_describe_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Describe",
        tool_description="\n\n".join([
            "Return dataset description: row_count and per-column stats for any or all of the datasets listed below.",
            "Numeric stats: min/max/mean/std; Booleans: true/false counts; Strings: distinct_count and top values.",
            "Results are cached per tool name and schema fingerprint; pass refresh=true to recompute.",
            group_desc,
        ]),
    )

    read_tool = auto_generate_read_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Read",
        tool_description="\n\n".join([
            "Read single dataset rows: subset columns, limit, offset, order_by, sort_ascending.",
            group_desc,
        ]),
        result_limit=sql_max_rows,
    )

    search_summary_tool = auto_generate_search_summary_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Search Summary",
        tool_description="\n\n".join([
            "Perform a substring/regex search across all datasets and return a summary of the number of matches per dataset.",
            group_desc,
        ]),
    )
    search_content_tool = auto_generate_search_content_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Search Content",
        tool_description="\n\n".join([
            "Return matching rows from a single dataset using substring/regex across string columns.",
            group_desc,
        ]),
        result_limit=sql_max_rows,
    )

    analyze_tool = auto_generate_sql_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Analyze",
        tool_description="\n\n".join([
            "Execute DuckDB SELECT-only SQL over the provided datasets.",
            "DDL/DML, CTEs, subqueries, UNION, and multiple top-level queries are not allowed (enforced upstream).",
            "For text search, prefer regular expressions (REGEXP_MATCHES()/REGEXP_EXTRACT()).",
            "Paging: use ORDER BY to define row order, then LIMIT and OFFSET for pages.",
            "JOINs between datasets are allowed. Refer to datasets by name in braces, e.g., {orders}.",
            "Below, the available datasets are listed, by name and description.",
            group_desc,
        ]),
        result_limit=sql_max_rows,
    )

    return [schema_tool, describe_tool, read_tool, search_summary_tool, search_content_tool, analyze_tool]

# -----------------------------
# User-authored Dynamic Tools
# -----------------------------

def _ensure_no_var_args(func: Callable[..., object], *, func_label: str) -> None:
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind.name in {"VAR_POSITIONAL", "VAR_KEYWORD"}:
            raise ValueError(
                f"{func_label} must not use *args or **kwargs for MCP tool introspection."
            )


def create_dynamic_tool(
    tool_name: str,
    tool_description: str,
    source_df: DataFrame,
    tool_func: Callable[[DataFrame], Callable[..., DataFrame]],
    *,
    result_limit: Optional[int] = None,
) -> DynamicTool:
    """Create a DynamicTool by currying a source DataFrame into a user function.

    Usage pattern (closure/factory):
        def make_tool(df: Annotated[DataFrame, "Source dataframe"]):
            def tool(
                param1: Annotated[str, "..."],
                param2: Annotated[int, "..."],
            ) -> DataFrame:
                return df.filter(col("x") == param1).limit(param2)
            return tool

        tool = create_dynamic_tool(
            "my_tool", "...", df, make_tool
        )

    The inner tool's signature (Annotated) is preserved for MCP, and it must not
    use *args/**kwargs. Return either a DataFrame or a LogicalPlan.
    """
    bound_tool_func = tool_func(source_df)
    _ensure_no_var_args(bound_tool_func, func_label="Dynamic tool function")

    @wraps(bound_tool_func)
    def tool_func_wrapper(*args, **kwargs) -> LogicalPlan:
        result_df = bound_tool_func(*args, **kwargs)
        return result_df._logical_plan

    return DynamicTool(
        name=tool_name,
        description=tool_description,
        func=tool_func_wrapper,
        result_limit=result_limit,
    )


def dynamic_tool_from_df(
    source_df: DataFrame,
    *,
    tool_name: str,
    tool_description: str,
    result_limit: Optional[int] = None,
):
    """Decorator to bind a DataFrame to a user-authored tool function.

    Example:
        @dynamic_tool_from_df(df, tool_name="find_rust", tool_description="...")
        def find_rust(
            query: Annotated[str, "Natural language query"],
            limit: Annotated[int, "Max rows"] = 50,
        ) -> DataFrame:
            pred = fc.semantic.predicate("Matches: {{q}}", q=query, bio=fc.col("bio"))
            return df.filter(pred).limit(limit)

    Notes:
    - The decorated function must NOT use *args/**kwargs, and should annotate parameters with Annotated descriptions.
    - Return a DataFrame or LogicalPlan.
    - The returned object is a DynamicTool ready for registration.
    """

    def decorator(func: Callable[..., DataFrame]) -> DynamicTool:
        _ensure_no_var_args(func, func_label="Dynamic tool function")

        @wraps(func)
        def wrapper(*args, **kwargs) -> LogicalPlan:
            result_df = func(*args, **kwargs)
            return result_df._logical_plan

        return DynamicTool(
            name=tool_name,
            description=tool_description,
            func=wrapper,
            result_limit=result_limit,
        )

    return decorator

