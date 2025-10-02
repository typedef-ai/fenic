from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import polars as pl
from typing_extensions import Annotated

from fenic.api import col
from fenic.api.dataframe import DataFrame
from fenic.api.session import Session
from fenic.core._logical_plan import LogicalPlan
from fenic.core._logical_plan.plans import InMemorySource
from fenic.core.error import ConfigurationError, ValidationError
from fenic.core.mcp.types import SystemTool
from fenic.core.types.datatypes import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
)
from fenic.core.types.schema import Schema

LONG_TEXT_COLUMN_THRESHOLD_CHAR_LENGTH = 1024

PROFILE_MAX_SAMPLE_SIZE = 10_000


class ToolDataset:
    """Specification for a dataset exposed to a tool.

    Attributes:
      table_name: name of the table registered in the catalog.
      description: description of the table from the catalog.
    """
    table_name: str
    description: str

    def __init__(
        self,
        table_name: str,
        description: str,
    ):
        self.table_name = table_name
        self.description = description

    def df(self, session: Session) -> DataFrame:
        return session.table(self.table_name)

    def schema(self, session: Session) -> Schema:
        return session.catalog.describe_table(self.table_name).schema


def auto_generate_system_tools_from_tables(
    table_names: list[str],
    session: Session,
    *,
    tool_namespace: Optional[str],
    max_result_limit: int = 100,
) -> List[SystemTool]:
    """Generate Schema/Profile/Read/Search [Content/Summary]/Analyze tools from catalog tables.

    Validates that each table exists and has a non-empty description in catalog metadata.
    """
    if not table_names:
        raise ConfigurationError("At least one table name must be specified for automated system tool creation.")
    datasets = _build_datasets_from_tables(table_names, session)
    return _auto_generate_system_tools(
        datasets,
        session,
        tool_namespace=tool_namespace,
        max_result_limit=max_result_limit,
    )


def _auto_generate_system_tools(
    datasets: List[ToolDataset],
    session: Session,
    *,
    tool_namespace: Optional[str],
    max_result_limit: int = 100,
) -> List[SystemTool]:
    """Generate core tools spanning all datasets: Schema, Profile, Analyze.

    - Schema: list columns/types for any or all datasets
    - Profile: dataset statistics for any or all datasets
    - Read: read rows from a single dataset to sample the data
    - Search Summary: regex search across all datasets and return a summary of the number of matches per dataset
    - Search Content: return matching rows from a single dataset using regex matching across string columns
    - Analyze: DuckDB SELECT-only SQL across datasets
    """
    group_desc = "\n".join(
        [f"{d.table_name}: {d.description.strip()}" if d.description else d.table_name for d in datasets]
    )

    name_to_spec: Dict[str, ToolDataset] = {spec.table_name: spec for spec in datasets}
    generated_tools: List[SystemTool] = [
        _auto_generate_schema_tool(
            name_to_spec,
            session,
            tool_name=f"{tool_namespace} - Schema" if tool_namespace else "Schema",
            tool_description="\n\n".join([
                "Show the schema (column names and types) for any or all of the datasets listed below. This call should be the first step in exploring the available datasets.",
                group_desc,
            ]),
        ), _auto_generate_profile_tool(
            name_to_spec,
            session,
            tool_name=f"{tool_namespace} - Profile" if tool_namespace else "Profile",
            tool_description="\n".join([
                "Return dataset data profile: row_count and per-column stats for any or all of the datasets listed below.",
                "This call should be used as a follow up after calling the `Schema` tool."
                "Numeric stats: min/max/mean/std; Booleans: true/false counts; Strings: distinct_count and top values.",
                "Profiling statistics are calculated across a sample of the original dataset.",
                "Available Datasets:",
                group_desc,
            ]),
        ), _auto_generate_read_tool(
            name_to_spec,
            session,
            tool_name=f"{tool_namespace} - Read" if tool_namespace else "Read",
            tool_description="\n".join([
                "Read rows from a single dataset. Use to sample data, or to execute simple queries over the data that do not require filtering or grouping.",
                "Use `include_columns` and `exclude_columns` to filter columns by name -- this is important to conserve token usage. Use the `Profile` tool to understand the columns and their sizes.",
                "Available datasets:",
                group_desc,
            ]),
            result_limit=max_result_limit,
        ), _auto_generate_search_summary_tool(
            name_to_spec,
            session,
            tool_name=f"{tool_namespace} - Search Summary" if tool_namespace else "Search Summary",
            tool_description="\n".join([
                "Perform a substring/regex search across all datasets and return a summary of the number of matches per dataset.",
                "Available datasets:",
                group_desc,
            ]),
        ), _auto_generate_search_content_tool(
            name_to_spec,
            session,
            tool_name=f"{tool_namespace} - Search Content" if tool_namespace else "Search Content",
            tool_description="\n".join([
                "Return matching rows from a single dataset using substring/regex across string columns.",
                "Available datasets:",
                group_desc,
            ]),
            result_limit=max_result_limit,
        ), _auto_generate_sql_tool(
            name_to_spec,
            session,
            tool_name=f"{tool_namespace} - Analyze" if tool_namespace else "Analyze",
            tool_description="\n".join([
                "Execute Read-Only (SELECT) SQL over the provided datasets using fenic's SQL support.",
                "DDL/DML and multiple top-level queries are not allowed.",
                "For text search, prefer regular expressions (REGEXP_MATCHES()/REGEXP_EXTRACT()).",
                "Paging: use ORDER BY to define row order, then LIMIT and OFFSET for pages.",
                "JOINs between datasets are allowed. Refer to datasets by name in braces, e.g., {orders}.",
                "Below, the available datasets are listed, by name and description.",
                group_desc,
            ]),
            result_limit=max_result_limit,
        )]

    return generated_tools


def _build_datasets_from_tables(table_names: List[str], session: Session) -> List[ToolDataset]:
    """Resolve catalog table names into DatasetSpec list with validated descriptions.

    Raises ConfigurationError if any table is missing or lacks a non-empty description.
    """
    missing_desc: List[str] = []
    missing_tables: List[str] = []
    specs: List[ToolDataset] = []

    for table_name in table_names:
        if not session.catalog.does_table_exist(table_name):
            missing_tables.append(table_name)
            continue
        table_metadata = session.catalog.describe_table(table_name)
        desc = (table_metadata.description or "").strip()
        if not desc:
            missing_desc.append(table_name)
        specs.append(ToolDataset(table_name=table_name, description=desc))

    if missing_tables:
        raise ConfigurationError(
            f"The following tables do not exist: {', '.join(sorted(missing_tables))}"
        )
    if missing_desc:
        raise ConfigurationError(
            "All tables must have a non-empty description to enable automated tool creation. "
            f"Missing descriptions for: {', '.join(sorted(missing_desc))}"
            "Use `session.catalog.set_table_description(table_name, description)` to set the table description."
        )

    return specs


def _auto_generate_read_tool(
    datasets: Dict[str, ToolDataset],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 50,
) -> SystemTool:
    """Create a read tool over one or many datasets."""
    if len(datasets) == 0:
        raise ConfigurationError("Cannot create read tool: no datasets provided.")

    def read_func(
        df_name: Annotated[str, "Dataset name to read rows from."],
        limit: Annotated[Optional[int], "Max rows to read within a page"] = result_limit,
        offset: Annotated[Optional[int], "Row offset to start from (requires order_by)"] = None,
        order_by: Annotated[Optional[str], "Comma separated list of columns to order by (required for offset)"] = None,
        sort_ascending: Annotated[Optional[bool], "Sort ascending for all order_by columns"] = True,
        include_columns: Annotated[Optional[str], "Comma separated list of columns to include in the result"] = None,
        exclude_columns: Annotated[Optional[str], "Comma separated list of columns to exclude from the result"] = None,
    ) -> LogicalPlan:

        if df_name not in datasets:
            raise ValidationError(f"Unknown DataFrame '{df_name}'. Available: {', '.join(datasets.keys())}")
        df = datasets[df_name].df(session)
        order_by = [c.strip() for c in order_by.split(",") if c.strip()] if order_by else None
        available_columns = df.columns
        include_columns = [c.strip() for c in include_columns.split(",") if c.strip()] if include_columns else None
        exclude_columns = [c.strip() for c in exclude_columns.split(",") if c.strip()] if exclude_columns else None
        if include_columns and exclude_columns:
            raise ValidationError("include_columns and exclude_columns cannot be used together.")
        if include_columns:
            filtered_columns = [c for c in include_columns if c in available_columns]
            df = df.select(*filtered_columns)
        if exclude_columns:
            filtered_columns = [c for c in available_columns if c not in exclude_columns]
            df = df.select(*filtered_columns)
        # Apply paging (handles offset+order_by via SQL and optional limit)
        return _apply_paging(
            df,
            session,
            limit=limit,
            offset=offset,
            order_by=order_by,
            sort_ascending=sort_ascending,
        )

    return SystemTool(
        name=tool_name,
        description=tool_description,
        func=read_func,
        max_result_limit=result_limit,
        add_limit_parameter=False,
    )


def _auto_generate_search_summary_tool(
    datasets: Dict[str, ToolDataset],
    session: Session,
    tool_name: str,
    tool_description: str,
) -> SystemTool:
    """Create a grep-like summary tool over one or many datasets (string columns)."""
    if len(datasets) == 0:
        raise ValueError("Cannot create search summary tool: no datasets provided.")

    def search_summary(
        pattern: Annotated[str, "Regex pattern to search for (use (?i) for case-insensitive)."],
    ) -> LogicalPlan:
        rows: List[Dict[str, object]] = []
        for name, dataset in datasets.items():
            df = dataset.df(session)
            cols = [f.name for f in df.schema.column_fields if f.data_type == StringType]
            if not cols:
                rows.append({"dataset": name, "total_matches": 0})
                continue
            predicate = None
            for c_name in cols:
                this = col(c_name).rlike(pattern)
                predicate = this if predicate is None else (predicate | this)

            df = df.filter(predicate)
            total_count = df.count()
            rows.append({"dataset": name, "total_matches": total_count})

        pl_df = pl.DataFrame(rows)
        return InMemorySource.from_session_state(pl_df, session._session_state)

    return SystemTool(
        name=tool_name,
        description=tool_description,
        func=search_summary,
        max_result_limit=None,
    )


def _auto_generate_search_content_tool(
    datasets: Dict[str, ToolDataset],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 100,
) -> SystemTool:
    """Create a content search tool for a single dataset (string columns)."""
    if len(datasets) == 0:
        raise ValidationError("Cannot create search content tool: no datasets provided.")

    def _string_columns(df: DataFrame, selected: Optional[List[str]]) -> List[str]:
        if selected:
            missing = [c for c in selected if c not in df.columns]
            if missing:
                raise ValidationError(f"Column(s) {missing} not found. Available: {', '.join(df.columns)}")
            return selected
        return [f.name for f in df.schema.column_fields if f.data_type == StringType]

    def search_rows(
        df_name: Annotated[str, "Dataset name to search (single dataset)"],
        pattern: Annotated[str, "Regex pattern to search for (use (?i) for case-insensitive)."],
        limit: Annotated[Optional[int], "Max rows to read within a page of search results"] = result_limit,
        offset: Annotated[Optional[int], "Row offset to start from (requires order_by)"] = None,
        order_by: Annotated[
            Optional[str], "Comma separated list of column names to order by (required with offset)"] = None,
        sort_ascending: Annotated[Optional[Union[bool, str]], "Sort ascending"] = True,
        search_columns: Annotated[Optional[
            str], "Comma separated list of column names search within; if omitted, matches in any string coluumn will be returned. Use this to query only specific columns in the search as needed."] = None,
    ) -> LogicalPlan:

        limit = int(limit) if isinstance(limit, str) else limit
        offset = int(offset) if isinstance(offset, str) else offset
        sort_ascending = bool(sort_ascending) if isinstance(sort_ascending, str) else sort_ascending
        search_columns = [c.strip() for c in search_columns.split(",") if c.strip()] if search_columns else None
        order_by = [c.strip() for c in order_by.split(",") if c.strip()] if order_by else None

        if not pattern:
            raise ValidationError("Query pattern cannot be empty.")
        if df_name not in datasets:
            raise ValidationError(f"Unknown DataFrame '{df_name}'. Available: {', '.join(datasets.keys())}")
        df = datasets[df_name].df(session)
        cols = _string_columns(df, search_columns)
        if not cols:
            return df.limit(0)._logical_plan
        predicate = None
        for c_name in cols:
            this = col(c_name).rlike(pattern)
            predicate = this if predicate is None else (predicate | this)
        out = df.filter(predicate)

        return _apply_paging(
            out,
            session,
            limit=limit,
            offset=offset,
            order_by=order_by,
            sort_ascending=sort_ascending,
        )

    return SystemTool(
        name=tool_name,
        description=tool_description,
        func=search_rows,
        max_result_limit=result_limit,
        add_limit_parameter=False,
    )


def _auto_generate_schema_tool(
    datasets: Dict[str, ToolDataset],
    session: Session,
    tool_name: str,
    tool_description: str,
) -> SystemTool:
    """Create a schema tool over one or many datasets.

    - Returns one row per dataset with a column `schema` containing a list of
      {column, type} entries.
    - If `df_name` is provided, returns only that dataset.
    """
    if len(datasets) == 0:
        raise ValueError("Cannot create schema tool: no datasets provided.")

    def schema_func(
        df_name: Annotated[
            str | None, "Optional DataFrame name to return a single schema for. To return schemas for all datasets, OMIT this parameter."] = None,
    ) -> LogicalPlan:
        # sometimes the models get...very confused, and pass the null string instead of `null` or omitting the field entirely
        if not df_name or df_name == "null":
            df_name = None
        # Choose subset of datasets
        if df_name is not None:
            if df_name not in datasets:
                raise ValidationError(
                    f"Unknown DataFrame '{df_name}'. Available: {', '.join(datasets.keys())}"
                )
            selected = {df_name: datasets[df_name]}
        else:
            selected = datasets

        dataset_names: List[str] = []
        dataset_schemas: List[List[Dict[str, str]]] = []

        for name, dataset in selected.items():
            # Build a single-row DataFrame with a common list<struct{column,type}> schema column
            schema_entries = [{"column": f.name, "type": str(f.data_type)} for f in dataset.schema(session).column_fields]
            dataset_names.append(name)
            dataset_schemas.append(schema_entries)

        return InMemorySource.from_session_state(
            pl.DataFrame({
                "dataset": dataset_names,
                "schema": dataset_schemas,
            }),
            session._session_state,
        )

    return SystemTool(
        name=tool_name,
        description=tool_description.strip(),
        func=schema_func,
        max_result_limit=None,
    )


def _auto_generate_sql_tool(
    datasets: Dict[str, ToolDataset],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 100,
) -> SystemTool:
    """Create an Analyze tool that executes DuckDB SELECT SQL across datasets.

    - JOINs between the provided datasets are allowed.
    - DDL/DML and multiple top-level queries are not allowed (enforced in `session.sql()`).
    - The callable returns a LogicalPlan gathered later by the MCP server.
    """
    if len(datasets) == 0:
        raise ConfigurationError("Cannot create SQL tool: no datasets provided.")

    def analyze_func(
        full_sql: Annotated[
            str, "Full SELECT SQL. Refer to DataFrames by name in braces, e.g., `SELECT * FROM {orders}`. JOINs between the provided datasets are allowed. SQL dialect: DuckDB. DDL/DML and multiple top-level queries are not allowed"]
    ) -> LogicalPlan:
        return session.sql(full_sql.strip(), **{spec.table_name: spec.df(session) for spec in datasets.values()})._logical_plan

    # Enhanced description with dataset names and descriptions
    lines: List[str] = [tool_description.strip()]
    if datasets:
        example_name = next(iter(datasets.keys()))
    else:
        example_name = "data"
    lines.extend(
        [
            "\n\nNotes:\n",
            "- SQL dialect: DuckDB.\n",
            "- For text search, prefer regular expressions using REGEXP_MATCHES().\n",
            "- Paging: use ORDER BY to define row order, then LIMIT and OFFSET for pages.\n",
            f"- Results are limited to {result_limit} rows, use LIMIT/OFFSET to paginate when receiving a result set of {result_limit} or more rows.\n",
            "Examples:\n",
            f"- SELECT * FROM {{{example_name}}} WHERE REGEXP_MATCHES(message, '(?i)error|fail') LIMIT {result_limit}", # nosec B608 - example text only
            f"- SELECT dept, COUNT(*) AS n FROM {{{example_name}}} WHERE status = 'active' GROUP BY dept HAVING n > 10 ORDER BY n DESC LIMIT {result_limit}", # nosec B608 - example text only
            f"- Paging: page 2 of size {result_limit}\n  SELECT * FROM {{{example_name}}} ORDER BY created_at DESC LIMIT {result_limit} OFFSET {result_limit}", # nosec B608 - example text only
        ]
    )
    enhanced_description = "\n".join(lines)

    tool = SystemTool(
        name=tool_name,
        description=enhanced_description,
        func=analyze_func,
        max_result_limit=result_limit,
        add_limit_parameter=False,
    )
    return tool


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def _apply_paging(
    df: DataFrame,
    session: Session,
    *,
    limit: int | None,
    offset: int | None,
    order_by: list[str] | None,
    sort_ascending: bool | None,
) -> LogicalPlan:
    """Apply ordering, limit, and offset via a single SQL statement.

    - If offset is provided, order_by must also be provided to ensure deterministic paging.
    - Validates that all order_by columns exist.
    - Builds: SELECT * FROM {src} [ORDER BY ...] [LIMIT N] [OFFSET M]
    - When no ordering/limit/offset are provided, returns the original plan.
    """
    if order_by:
        missing_order = [c for c in order_by if c not in df.columns]
        if missing_order:
            raise ValidationError(
                f"order_by column(s) {missing_order} do not exist in DataFrame. Available columns: {', '.join(df.columns)}"
            )

    if offset is not None and not order_by:
        raise ValidationError("offset requires order_by to ensure deterministic paging.")

    if order_by is None and limit is None and offset is None:
        return df._logical_plan

    direction = "ASC" if (sort_ascending is None or sort_ascending) else "DESC"
    lim_val = None if limit is None else int(str(limit))
    off_val = None if offset is None else int(str(offset))

    base_sql = "SELECT * FROM {src}"
    if order_by:
        safe_order_by = ", ".join(order_by)
        base_sql += " ORDER BY " + safe_order_by + f" {direction}"  # nosec B608
    if lim_val is not None:
        base_sql += f" LIMIT {lim_val}"
    if off_val is not None:
        base_sql += f" OFFSET {off_val}"

    df_with_paging = session.sql(base_sql, src=df)
    return df_with_paging._logical_plan


@dataclass
class NumericStats:
    min: float
    max: float
    mean: float
    std_dev: float
    median: float
    quantile_25: float
    quantile_75: float

@dataclass
class TopValues:
    value: str
    count: int

@dataclass
class StringStats:
    avg_length_chars: float
    distinct_count: int
    top_values: List[TopValues]
    example_values: List[str]

@dataclass
class BooleanStats:
    true_rows: int
    false_rows: int

@dataclass
class ProfileRow:
    dataset_name: str
    column_name: str
    data_type: str
    total_rows: int
    sample_size: int
    sample_percentage_of_original: float
    null_row_count: int
    non_null_row_count: int
    percent_rows_contains_null: float
    semantic_type: Literal["identifier", "categorical", "continuous", "text", "boolean", "unknown"]
    cardinality: Literal["unique", "low", "medium", "high", "unknown"]
    hints: List[str]
    numeric_stats: Optional[NumericStats]
    string_stats: Optional[StringStats]
    boolean_stats: Optional[BooleanStats]

def _auto_generate_profile_tool(
    datasets: Dict[str, ToolDataset],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    topk_distinct: int = 10,
) -> SystemTool:
    """Create a cached Profile tool for one or many datasets.

    Output columns include:
      - dataset, column, type, row_count, non_null_count, null_count
      - min, max, mean, std (for numerics)
      - distinct_count, top_values (JSON) for strings
      - true_count, false_count for booleans
    """
    if len(datasets) == 0:
        raise ValueError("Cannot create profile tool: no datasets provided.")

    def profile_func(
        df_name: Annotated[
            str | None, "Optional DataFrame name to return a single profile for. To return profiles for all datasets, omit this parameter."] = None,
    ) -> LogicalPlan:
        # sometimes the models get...very confused, and pass the null string instead of `null` or omitting the field entirely
        if not df_name or df_name == "null":
            df_name = None
        # Single dataset branch returns the view plan directly
        if df_name is not None:
            spec = datasets.get(df_name)
            if spec is None:
                raise ValidationError(
                    f"Unknown dataset '{df_name}'. Available: {', '.join(datasets.keys())}")
            profile_df = _compute_profile_for_dataset(session, spec, topk_distinct)
            return profile_df._logical_plan

        # Multi-dataset: concatenate cached views (or compute & cache if missing)
        profile_df = None
        for spec in datasets.values():
            df = _compute_profile_for_dataset(session, spec, topk_distinct)
            if not profile_df:
                profile_df = df
            else:
                profile_df = profile_df.union(df)

        return profile_df._logical_plan

    return SystemTool(
        name=tool_name,
        description=tool_description,
        func=profile_func,
        max_result_limit=None,
    )

def _compute_profile_for_dataset(
    session: Session,
    spec: ToolDataset,
    topk_distinct: int,
) -> DataFrame:
    df_rows = _compute_profile_rows(
        spec.df(session),
        spec.table_name,
        topk_distinct,
    )
    # Enforce struct dtypes so columns don't collapse to Null when all values are None
    top_values_struct = pl.Struct([pl.Field("value", pl.Utf8), pl.Field("count", pl.Int64)])
    numeric_struct = pl.Struct([
        pl.Field("min", pl.Float64),
        pl.Field("max", pl.Float64),
        pl.Field("mean", pl.Float64),
        pl.Field("std_dev", pl.Float64),
        pl.Field("median", pl.Float64),
        pl.Field("quantile_25", pl.Float64),
        pl.Field("quantile_75", pl.Float64),
    ])
    boolean_struct = pl.Struct([
        pl.Field("true_rows", pl.Int64),
        pl.Field("false_rows", pl.Int64),
    ])
    string_struct = pl.Struct([
        pl.Field("avg_length_chars", pl.Float64),
        pl.Field("distinct_count", pl.Int64),
        pl.Field("top_values", pl.List(top_values_struct)),
        pl.Field("example_values", pl.List(pl.Utf8)),
    ])

    pl_df = pl.DataFrame(
        df_rows,
        schema_overrides={
            "numeric_stats": numeric_struct,
            "boolean_stats": boolean_struct,
            "string_stats": string_struct,
        },
    )

    return DataFrame._from_logical_plan(
        InMemorySource.from_session_state(pl_df, session._session_state),
        session._session_state,
    )


def _compute_profile_rows(
    df: DataFrame,
    dataset_name: str,
    topk_distinct: int,
) -> List[dict[str, Any]]:
    pl_df = df.to_polars()
    total_rows = pl_df.height
    sampled_df = pl_df.sample(min(total_rows, PROFILE_MAX_SAMPLE_SIZE))

    # Build a single batched select of aggregations for all columns
    exprs: List[pl.Expr] = []
    numeric_types = (IntegerType, FloatType, DoubleType)

    # Alias builders for convenience
    def a_nulls(name: str) -> str:
        return f"nulls__{name}"

    def a_mean(name: str) -> str:
        return f"mean__{name}"

    def a_min(name: str) -> str:
        return f"min__{name}"

    def a_max(name: str) -> str:
        return f"max__{name}"

    def a_median(name: str) -> str:
        return f"median__{name}"

    def a_std(name: str) -> str:
        return f"std__{name}"

    def a_true(name: str) -> str:
        return f"true__{name}"

    def a_strlen_mean(name: str) -> str:
        return f"strlen_mean__{name}"

    def a_n_unique(name: str) -> str:
        return f"n_unique__{name}"

    def a_quantile_25(name: str) -> str:
        return f"quantile_25__{name}"

    def a_quantile_75(name: str) -> str:
        return f"quantile_75__{name}"

    for field in df.schema.column_fields:
        col_name = field.name
        # Common null counts
        exprs.append(pl.col(col_name).is_null().sum().alias(a_nulls(col_name)))
        # type-specific aggregations
        if field.data_type in numeric_types:
            exprs.extend([
                pl.col(col_name).mean().alias(a_mean(col_name)),
                pl.col(col_name).min().alias(a_min(col_name)),
                pl.col(col_name).max().alias(a_max(col_name)),
                pl.col(col_name).median().alias(a_median(col_name)),
                pl.col(col_name).std().alias(a_std(col_name)),
                pl.col(col_name).quantile(0.25).alias(a_quantile_25(col_name)),
                pl.col(col_name).quantile(0.75).alias(a_quantile_75(col_name)),
            ])
        elif field.data_type == BooleanType:
            # Count of true values (nulls treated as False to avoid null sums)
            exprs.append(pl.col(col_name).drop_nulls().sum().alias(a_true(col_name)))
        elif field.data_type == StringType:
            exprs.extend([
                pl.col(col_name).str.len_chars().mean().alias(a_strlen_mean(col_name)),
                pl.col(col_name).n_unique().alias(a_n_unique(col_name)),
            ])

    agg_row: Dict[str, object] = {}
    if exprs:
        agg_df = sampled_df.select(exprs)
        agg_row = agg_df.to_dicts()[0] if agg_df.height > 0 else {}

    rows_list: List[dict[str, Any]] = []
    sample_size = sampled_df.height
    for field in df.schema.column_fields:
        col_name = field.name
        dtype_str = str(field.data_type)
        null_count_val = agg_row.get(a_nulls(col_name), 0)
        null_count = int(null_count_val) if null_count_val else 0
        non_null_count = sample_size - null_count
        numeric_stats = NumericStats(
            min=agg_row.get(a_min(col_name)),
            max=agg_row.get(a_max(col_name)),
            mean=agg_row.get(a_mean(col_name)),
            std_dev=agg_row.get(a_std(col_name)),
            median=agg_row.get(a_median(col_name)),
            quantile_25=agg_row.get(a_quantile_25(col_name)),
            quantile_75=agg_row.get(a_quantile_75(col_name)),
        )

        string_stats = StringStats(
            avg_length_chars=agg_row.get(a_strlen_mean(col_name)),
            distinct_count=agg_row.get(a_n_unique(col_name)),
            top_values=None,
            example_values=None,
        )
        boolean_stats = BooleanStats(
            true_rows=agg_row.get(a_true(col_name)) if agg_row.get(a_true(col_name)) is not None else None,
            false_rows=(sample_size - agg_row.get(a_true(col_name))) if agg_row.get(
                a_true(col_name)) is not None else None,
        )
        stats = ProfileRow(
            dataset_name=dataset_name,
            column_name=col_name,
            data_type=dtype_str,
            total_rows=total_rows,
            sample_size=sample_size,
            sample_percentage_of_original=round((float(sample_size) / total_rows) * 100, 1),
            percent_rows_contains_null=round(((null_count / float(sample_size)) * 100) if sample_size > 0 else 0.0,
                                             1),
            null_row_count=null_count,
            non_null_row_count=non_null_count,
            hints=[],
            cardinality="unknown",
            semantic_type="unknown",
            boolean_stats=boolean_stats,
            numeric_stats=numeric_stats,
            string_stats=string_stats,
        )
        if field.data_type in numeric_types:
            stats.semantic_type = "continuous"
            stats.cardinality = "high"
            # Check if it might be an identifier
            if "id" in col_name.lower() or col_name.lower().endswith("_id"):
                stats.semantic_type = "identifier"

        elif field.data_type == BooleanType:
            stats.semantic_type = "boolean"
            stats.cardinality = "low"
        elif field.data_type == StringType:
            # Determine cardinality and semantic type
            if stats.string_stats.distinct_count is not None:
                distinct_ratio = (stats.string_stats.distinct_count / sample_size) if sample_size > 0 else 0
                if distinct_ratio > 0.95:
                    stats.cardinality = "unique"
                    stats.semantic_type = "identifier" if (
                                stats.string_stats.avg_length_chars is not None and stats.string_stats.avg_length_chars < 50) else "text"
                elif stats.string_stats.distinct_count <= 10:
                    stats.cardinality = "low"
                    stats.semantic_type = "categorical"
                elif stats.string_stats.distinct_count <= 100:
                    stats.cardinality = "medium"
                    stats.semantic_type = "categorical"
                else:
                    stats.cardinality = "high"
                    stats.semantic_type = "text"

            if stats.string_stats.avg_length_chars is not None and stats.string_stats.avg_length_chars > LONG_TEXT_COLUMN_THRESHOLD_CHAR_LENGTH:
                stats.hints.extend([
                    "has_long_text",
                    "consider_excluding_from_read",
                    "prefer_search_content",
                    "use_row_limit",
                    "prefer_analyze_regex_match"
                ])

            # Add cardinality-based recommendations
            if stats.cardinality == "low":
                stats.hints.extend([
                    "use_for_aggregations",
                    "use_for_filtering",
                ])
            elif stats.cardinality == "medium":
                stats.hints.extend([
                    "use_for_aggregations",
                    "prefer_search_content",
                    "prefer_row_limit",
                    "prefer_analyze_regex_match"
                ])
            elif stats.cardinality == "unique":
                stats.hints.extend([
                    "high_cardinality",
                    "unique_values",
                ])
            elif stats.cardinality == "high":
                stats.hints.extend([
                    "high_cardinality",
                ])

            compute_topk = (
                    (
                                stats.string_stats.avg_length_chars is not None and stats.string_stats.avg_length_chars <= LONG_TEXT_COLUMN_THRESHOLD_CHAR_LENGTH) and
                    (stats.string_stats.distinct_count is not None and stats.string_stats.distinct_count <= max(topk_distinct * 10,
                                                                                                    200))
            )

            if compute_topk:
                vc = sampled_df.get_column(col_name).value_counts(sort=True)
                val_col = col_name if col_name in vc.columns else vc.columns[0]
                top_vals: List[TopValues] = []
                for i in range(min(topk_distinct, vc.height)):
                    top_vals.append(
                        TopValues(
                            value=vc.get_column(val_col)[i],
                            count=vc.get_column("count")[i],
                        )
                    )
                stats.string_stats.top_values = top_vals
            else:
                # No top-k: still provide a few sample values when strings aren't too long.
                # Use a conservative threshold to avoid dumping giant text fields.
                if stats.percent_rows_contains_null < 100:
                    s_non_null = sampled_df.get_column(col_name).drop_nulls()
                    k = min(3, int(s_non_null.len()))
                    sampled = s_non_null.sample(
                        n=k,
                        with_replacement=False,
                        shuffle=True
                    ).to_list()
                    # there are very few values in the sample, so it's not too much of a performance hit to truncate them in python instead of polars
                    stats.string_stats.example_values = [
                        (v[:LONG_TEXT_COLUMN_THRESHOLD_CHAR_LENGTH] + f"... (truncated {len(v) - LONG_TEXT_COLUMN_THRESHOLD_CHAR_LENGTH} characters)")
                        if isinstance(v, str) and len(v) > LONG_TEXT_COLUMN_THRESHOLD_CHAR_LENGTH
                        else v for v in sampled
                    ]
        stats.hints = list(set(stats.hints))
        rows_list.append(asdict(stats))
    return rows_list
