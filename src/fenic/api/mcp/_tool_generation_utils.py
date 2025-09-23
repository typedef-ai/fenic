from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import polars as pl
from typing_extensions import Annotated

from fenic.api import col
from fenic.api.dataframe import DataFrame
from fenic.api.functions import avg, stddev
from fenic.api.functions import max as max_
from fenic.api.functions import min as min_
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

PROFILE_MAX_SAMPLE_SIZE = 10_000


@dataclass
class DatasetSpec:
    """Specification for a dataset exposed to a tool.

    Attributes:
      table_name: name of the table registered in the catalog.
      description: description of the table from the catalog.
      df: the fenic DataFrame object with the table data.
    """
    table_name: str
    description: str
    df: DataFrame

def _auto_generate_system_tools(
    datasets: List[DatasetSpec],
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

    schema_tool = _auto_generate_schema_tool(
        datasets,
        session,
        tool_name=f"{tool_namespace} - Schema" if tool_namespace else "Schema",
        tool_description="\n\n".join([
            "Show the schema (column names and types) for any or all of the datasets listed below. This call should be the first step in exploring the available datasets.",
            group_desc,
        ]),
    )

    profile_tool = _auto_generate_profile_tool(
        datasets,
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
    )

    read_tool = _auto_generate_read_tool(
        datasets,
        session,
        tool_name=f"{tool_namespace} - Read" if tool_namespace else "Read",
        tool_description="\n".join([
            "Read rows from a single dataset. Use to sample data, or to execute simple queries over the data that do not require filtering or grouping.",
            "Use `include_columns` and `exclude_columns` to filter columns by name -- this is important to conserve token usage. Use the `Profile` tool to understand the columns and their sizes.",
            "Available datasets:",
            group_desc,
        ]),
        result_limit=max_result_limit,
    )

    search_summary_tool = _auto_generate_search_summary_tool(
        datasets,
        session,
        tool_name=f"{tool_namespace} - Search Summary" if tool_namespace else "Search Summary",
        tool_description="\n".join([
            "Perform a substring/regex search across all datasets and return a summary of the number of matches per dataset.",
            "Available datasets:",
            group_desc,
        ]),
    )
    search_content_tool = _auto_generate_search_content_tool(
        datasets,
        session,
        tool_name=f"{tool_namespace} - Search Content" if tool_namespace else "Search Content",
        tool_description="\n".join([
            "Return matching rows from a single dataset using substring/regex across string columns.",
            "Available datasets:",
            group_desc,
        ]),
        result_limit=max_result_limit,
    )

    analyze_tool = _auto_generate_sql_tool(
        datasets,
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
    )

    return [schema_tool, profile_tool, read_tool, search_summary_tool, search_content_tool, analyze_tool]


def _build_datasets_from_tables(table_names: List[str], session: Session) -> List[DatasetSpec]:
    """Resolve catalog table names into DatasetSpec list with validated descriptions.

    Raises ConfigurationError if any table is missing or lacks a non-empty description.
    """
    missing_desc: List[str] = []
    missing_tables: List[str] = []
    specs: List[DatasetSpec] = []

    for table_name in table_names:
        if not session.catalog.does_table_exist(table_name):
            missing_tables.append(table_name)
            continue
        table_metadata = session.catalog.describe_table(table_name)
        desc = (table_metadata.description or "").strip()
        if not desc:
            missing_desc.append(table_name)
        df = session.table(table_name)
        specs.append(DatasetSpec(table_name=table_name, description=desc, df=df))

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
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 50,
) -> SystemTool:
    """Create a read tool over one or many datasets."""
    if len(datasets) == 0:
        raise ConfigurationError("Cannot create read tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.table_name: d.df for d in datasets}

    def _validate_columns(
        available_columns: List[str],
        original_columns: List[str],
        filtered_columns: List[str],
    ) -> None:
        if not filtered_columns:
            raise ValidationError(f"Column(s) {original_columns} not found. Available: {', '.join(available_columns)}")
        if len(filtered_columns) != len(original_columns):
            invalid_columns = [c for c in original_columns if c not in filtered_columns]
            raise ValidationError(f"Column(s) {invalid_columns} not found. Available: {', '.join(available_columns)}")

    async def read_func(
        df_name: Annotated[str, "Dataset name to read rows from."],
        limit: Annotated[Optional[int], "Max rows to read within a page"] = result_limit,
        offset: Annotated[Optional[int], "Row offset to start from (requires order_by)"] = None,
        order_by: Annotated[Optional[str], "Comma separated list of columns to order by (required for offset)"] = None,
        sort_ascending: Annotated[Optional[bool], "Sort ascending for all order_by columns"] = True,
        include_columns: Annotated[Optional[str], "Comma separated list of columns to include in the result"] = None,
        exclude_columns: Annotated[Optional[str], "Comma separated list of columns to exclude from the result"] = None,
    ) -> LogicalPlan:

        if df_name not in name_to_df:
            raise ValidationError(f"Unknown DataFrame '{df_name}'. Available: {', '.join(name_to_df.keys())}")
        df = name_to_df[df_name]
        order_by = [c.strip() for c in order_by.split(",") if c.strip()] if order_by else None
        available_columns = df.columns
        include_columns = [c.strip() for c in include_columns.split(",") if c.strip()] if include_columns else None
        exclude_columns = [c.strip() for c in exclude_columns.split(",") if c.strip()] if exclude_columns else None
        if include_columns:
            filtered_columns = [c for c in include_columns if c in available_columns]
            _validate_columns(available_columns, include_columns, filtered_columns)
            df = df.select(*filtered_columns)
        if exclude_columns:
            filtered_columns = [c for c in available_columns if c not in exclude_columns]
            _validate_columns(available_columns, exclude_columns, filtered_columns)
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
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
) -> SystemTool:
    """Create a grep-like summary tool over one or many datasets (string columns)."""
    if len(datasets) == 0:
        raise ValueError("Cannot create search summary tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.table_name: d.df for d in datasets}

    async def search_summary(
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

    return SystemTool(
        name=tool_name,
        description=tool_description,
        func=search_summary,
        max_result_limit=None,
    )


def _auto_generate_search_content_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 100,
) -> SystemTool:
    """Create a content search tool for a single dataset (string columns)."""
    if len(datasets) == 0:
        raise ValidationError("Cannot create search content tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.table_name: d.df for d in datasets}

    def _string_columns(df: DataFrame, selected: Optional[List[str]]) -> List[str]:
        if selected:
            missing = [c for c in selected if c not in df.columns]
            if missing:
                raise ValidationError(f"Column(s) {missing} not found. Available: {', '.join(df.columns)}")
            return selected
        return [f.name for f in df.schema.column_fields if f.data_type == StringType]

    async def search_rows(
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
        if df_name not in name_to_df:
            raise ValidationError(f"Unknown DataFrame '{df_name}'. Available: {', '.join(name_to_df.keys())}")
        d = name_to_df[df_name]
        cols = _string_columns(d, search_columns)
        if not cols:
            return d.limit(0)._logical_plan
        predicate = None
        for c_name in cols:
            this = col(c_name).rlike(pattern)
            predicate = this if predicate is None else (predicate | this)
        out = d.filter(predicate)

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
    datasets: List[DatasetSpec],
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

    name_to_df: Dict[str, DataFrame] = {d.table_name: d.df for d in datasets}

    async def schema_func(
        df_name: Annotated[
            str | None, "Optional DataFrame name to return a single schema for. To return schemas for all datasets, OMIT this parameter."] = None,
    ) -> LogicalPlan:
        # sometimes the models get...very confused, and pass the null string instead of `null` or omitting the field entirely
        if not df_name or df_name == "null":
            df_name = None
        # Choose subset of datasets
        if df_name is not None:
            if df_name not in name_to_df:
                raise ValidationError(
                    f"Unknown DataFrame '{df_name}'. Available: {', '.join(name_to_df.keys())}"
                )
            selected = {df_name: name_to_df[df_name]}
        else:
            selected = name_to_df

        dataset_names: List[str] = []
        dataset_schemas: List[List[Dict[str, str]]] = []

        for name, d in selected.items():
            # Build a single-row DataFrame with a common list<struct{column,type}> schema column
            schema_entries = [{"column": f.name, "type": str(f.data_type)} for f in d.schema.column_fields]
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
    datasets: List[DatasetSpec],
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

    async def analyze_func(
        full_sql: Annotated[
            str, "Full SELECT SQL. Refer to DataFrames by name in braces, e.g., `SELECT * FROM {orders}`. JOINs between the provided datasets are allowed. SQL dialect: DuckDB. DDL/DML and multiple top-level queries are not allowed"]
    ) -> LogicalPlan:
        return session.sql(full_sql.strip(), **{spec.table_name: spec.df for spec in datasets})._logical_plan

    # Enhanced description with dataset names and descriptions
    lines: List[str] = [tool_description.strip()]
    if datasets:
        example_name = datasets[0].table_name
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


def _schema_fingerprint(df: DataFrame) -> str:
    hasher = hashlib.sha256()
    for f in df.schema.column_fields:
        hasher.update(f"{f.name}|{str(f.data_type)}".encode("utf-8"))
    return hasher.hexdigest()[:12]


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
class _ProfileRow:
    dataset_name: str
    column_name: str
    data_type: str
    total_rows: int
    percent_rows_contains_null: float
    semantic_type: Literal["identifier", "categorical", "continuous", "text", "boolean", "unknown"]
    cardinality: Literal["unique", "low", "medium", "high", "unknown"]
    usage_recommendations: List[str]
    numeric_min: Optional[float]
    numeric_max: Optional[float]
    numeric_mean: Optional[float]
    numeric_std_dev: Optional[float]
    string_avg_length: Optional[float]
    string_distinct_count: Optional[int]
    string_top_values: Optional[List[Dict[str, Union[int, str]]]]
    string_example_values: Optional[List[str]]
    boolean_true_rows: Optional[int]
    boolean_false_rows: Optional[int]


def _auto_generate_profile_tool(
    datasets: List[DatasetSpec],
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
    tool_key = _sanitize_name(tool_name)

    async def profile_func(
        df_name: Annotated[
            str | None, "Optional DataFrame name to return a single profile for. To return profiles for all datasets, omit this parameter."] = None,
    ) -> LogicalPlan:
        # sometimes the models get...very confused, and pass the null string instead of `null` or omitting the field entirely
        if not df_name or df_name == "null":
            df_name = None
        # Single dataset branch returns the view plan directly
        if df_name is not None:
            spec = next((d for d in datasets if d.table_name == df_name), None)
            if spec is None:
                raise ValidationError(
                    f"Unknown dataset '{df_name}'. Available: {', '.join(d.table_name for d in datasets)}")
            return await _ensure_profile_view_for_dataset(session, tool_key, spec, topk_distinct)

        # Multi-dataset: concatenate cached views (or compute & cache if missing)
        profile_df = None
        for spec in datasets:
            plan = await _ensure_profile_view_for_dataset(session, tool_key, spec, topk_distinct)
            df = DataFrame._from_logical_plan(plan, session_state=session._session_state)
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


async def _ensure_profile_view_for_dataset(
    session: Session,
    tool_key: str,
    spec: DatasetSpec,
    topk_distinct: int,
) -> LogicalPlan:
    schema_hash = _schema_fingerprint(spec.df)
    view_name = f"__fenic_profile__{tool_key}__{_sanitize_name(spec.table_name)}__{schema_hash}"
    catalog = session._session_state.catalog
    if not catalog.does_view_exist(view_name):
        profile_rows = await _compute_profile_rows(
            spec.df,
            spec.table_name,
            topk_distinct,
        )
        view_plan = InMemorySource.from_session_state(
            pl.DataFrame(profile_rows), session._session_state,
        )
        catalog.create_view(view_name, view_plan)
    return catalog.get_view_plan(view_name)


async def _compute_profile_rows(
    df: DataFrame,
    dataset_name: str,
    topk_distinct: int,
) -> List[_ProfileRow]:
    pl_df = df.to_polars()
    total_rows = pl_df.height
    sampled_df = pl_df.sample(min(total_rows, PROFILE_MAX_SAMPLE_SIZE))
    rows_list: List[_ProfileRow] = []
    for field in df.schema.column_fields:
        col_name = field.name
        dtype_str = str(field.data_type)
        null_count = sampled_df.select(pl.col(col_name).is_null().sum()).item()
        non_null_count = sampled_df.height - null_count
        stats = _ProfileRow(
            dataset_name=dataset_name,
            column_name=col_name,
            data_type=dtype_str,
            total_rows=total_rows,
            percent_rows_contains_null=round((non_null_count / float(sampled_df.height) * 100) if sampled_df.height > 0 else 0, 1),
            usage_recommendations=[],
            cardinality="unknown",
            semantic_type="unknown",
            boolean_true_rows=None,
            boolean_false_rows=None,
            numeric_min=None,
            numeric_max=None,
            numeric_mean=None,
            numeric_std_dev=None,
            string_avg_length=None,
            string_distinct_count=None,
            string_top_values=None,
            string_example_values=None,
        )
        if field.data_type in (IntegerType, FloatType, DoubleType):
            agg_df = df.agg(
                min_(col(col_name)).alias("min"),
                max_(col(col_name)).alias("max"),
                avg(col(col_name)).alias("mean"),
                stddev(col(col_name)).alias("std"),
            ).to_pylist()
            stats.numeric_min = agg_df[0]["min"]
            stats.numeric_max = agg_df[0]["max"]
            stats.numeric_mean = agg_df[0]["mean"]
            stats.numeric_std_dev = agg_df[0]["std"]
            stats.semantic_type = "continuous"
            stats.cardinality = "high"
            # Check if it might be an identifier
            if "id" in col_name.lower() or col_name.lower().endswith("_id"):
                stats.semantic_type = "identifier"
        elif field.data_type == BooleanType:
            stats.semantic_type = "boolean"
            stats.cardinality = "low"
            s_bool = sampled_df.get_column(col_name)
            stats.boolean_true_rows = int((s_bool).sum())
            stats.boolean_false_rows = int((~s_bool).sum())
        elif field.data_type == StringType:
            s = sampled_df.get_column(col_name)
            stats.string_avg_length = s.str.len_chars().mean()
            stats.string_distinct_count = s.n_unique()
            stats.string_top_values = None
            stats.string_example_values = None

            # Determine cardinality and semantic type
            if stats.string_distinct_count is not None:
                distinct_ratio = stats.string_distinct_count / len(sampled_df) if len(sampled_df) > 0 else 0
                if distinct_ratio > 0.95:
                    stats.cardinality = "unique"
                    stats.semantic_type = "identifier" if stats.string_avg_length and stats.string_avg_length < 50 else "text"
                elif stats.string_distinct_count <= 10:
                    stats.cardinality = "low"
                    stats.semantic_type = "categorical"
                elif stats.string_distinct_count <= 100:
                    stats.cardinality = "medium"
                    stats.semantic_type = "categorical"
                else:
                    stats.cardinality = "high"
                    stats.semantic_type = "text"

            # Record the average length for agent awareness
            if stats.string_avg_length is not None and stats.string_avg_length > 1024:
                stats.usage_recommendations.append(
                    "This column appears to contain long text. "
                    "When using the 'Analyze' tool ensure LIMIT/ORDER BY is set. "
                    "Exclude this column from `Read` unless the result limit is very low. "
                    "To find relevant rows based on data from this column, consider the `Search Content` tool."
                )
            compute_topk = (
                stats.string_avg_length <= 512 and
                stats.string_distinct_count <= max(topk_distinct * 10, 200)
            )

            if compute_topk:
                vc = sampled_df.get_column(col_name).value_counts(sort=True)
                val_col = col_name if col_name in vc.columns else vc.columns[0]
                top_vals: List[Dict[str, Union[int, str]]] = []
                for i in range(min(topk_distinct, vc.height)):
                    top_vals.append(
                        {
                            "count": vc.get_column("count")[i],
                            "value": vc.get_column(val_col)[i],
                        }
                    )
                stats.string_top_values = top_vals
            else:
                # No top-k: still provide a few sample values when strings aren't too long.
                # Use a conservative threshold to avoid dumping giant text fields.
                # Add cardinality-based recommendations
                if stats.cardinality == "low":
                    stats.usage_recommendations.append(
                        f"Low cardinality categorical column ({stats.string_distinct_count} values). "
                        f"Excellent for GROUP BY, filtering, and aggregations."
                    )
                elif stats.cardinality == "medium":
                    stats.usage_recommendations.append(
                        f"Medium cardinality column ({stats.string_distinct_count} values). "
                        f"Good for filtering and grouping. Consider using `Search Content` or `Analyze` w/REGEXP_MATCHES for text matching."
                    )
                elif stats.cardinality == "unique":
                    stats.usage_recommendations.append(
                        "This column appears to have mostly unique values. "
                        "May be an identifier or free-text field. Use Search Content for text matching."
                    )
                else:
                    stats.usage_recommendations.append(
                        "This column has high cardinality. "
                        "To find relevant rows based on data from this column, consider the `Search Content` tool."
                    )
                if stats.percent_rows_contains_null > 0:
                    s_non_null = sampled_df.get_column(col_name).drop_nulls()
                    k = min(3, int(s_non_null.len()))
                    sampled = s_non_null.sample(
                        n=k,
                        with_replacement=False,
                        shuffle=True
                    ).str.slice(0, length=512).to_list()
                    stats.string_example_values = sampled
        rows_list.append(stats)
    return rows_list


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
