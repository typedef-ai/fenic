"""API-layer generators for automatic MCP tools from DataFrames.

These helpers generate DynamicTool definitions for:
- Schema: dataset column names and types
- Profile: per-column statistics (counts, numeric summaries, simple string summaries)
- Analyze: DuckDB SQL across one or more datasets.

All generated tools return LogicalPlan objects. The MCP server wrapper handles
execution and result formatting.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import polars as pl
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
from fenic.core._utils.schema import convert_custom_dtype_to_polars
from fenic.core.error import ConfigurationError, ValidationError
from fenic.core.mcp.types import DynamicToolDefinition, TableFormat
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
    table_name: str
    description: str
    df: DataFrame

@dataclass
class ToolGenerationConfig:
    """Configuration for automated tool generation.

    Args:
        table_names: List of table names.
        tool_group_name: Name of the tool group.
        sql_max_rows: Maximum number of rows to be returned from SQL queries.
    """

    table_names: List[str]
    tool_group_name: str
    sql_max_rows: int = 100

def auto_generate_core_tools_from_tables(
    table_names: List[str],
    session: Session,
    *,
    tool_group_name: str,
    sql_max_rows: int = 100,
) -> List[DynamicToolDefinition]:
    """Generate Schema/Profile/Read/Search/Analyze tools from catalog tables.

    Validates that each table exists and has a non-empty description in catalog metadata.
    """
    datasets = _build_datasets_from_tables(table_names, session)
    return _auto_generate_core_tools(
        datasets,
        session,
        tool_group_name=tool_group_name,
        sql_max_rows=sql_max_rows,
    )

def fenic_tool(
    tool_name: str,
    tool_description: str,
    max_result_limit: Optional[int] = None,
    default_table_format: TableFormat = "markdown",
    read_only: bool = True,
    idempotent: bool = True,
    destructive: bool = False,
    open_world: bool = False,
) -> Callable[[Callable[..., DataFrame]], DynamicToolDefinition]:
    """Decorator to bind a DataFrame to a user-authored tool function.

    Args:
        tool_name: The name of the tool.
        tool_description: The description of the tool.
        max_result_limit: The maximum number of results to return.
        default_table_format: The default table format to return.
        read_only: A hint to provide to the model that the tool does not modify its environment.
        idempotent: A hint to provide to the model that calling the tool multiple times with the same input will always return the same result (redundant if read_only is True).
        destructive: A hint to provide to the model that the tool may destructively modify its environment.
        open_world: A hint to provide to the model that the tool may interact with an "open world" of external entities outside of the MCP server's environment.

    Example:
        @dynamic_tool(tool_name="find_rust", tool_description="...")
        def find_rust(
            query: Annotated[str, "Natural language query"],
        ) -> DataFrame:
            pred = fc.semantic.predicate("Matches: {{q}} Data: {{bio}}", q=fc.lit(query), bio=fc.col("bio"))
            return df.filter(pred)

        mcp_server = fc.create_mcp_server(
            local_session,
            "...",
            dynamic_tools=[find_rust],
        )
        fc.run_mcp_server_sync(mcp_server)

    Example: Creating an open-world tool that reaches out to an external API. The open_world flag indicates to the model that the tool may interact with an "open world" of external entities
        @fenic_tool(tool_name="search_knowledge_base", tool_description="...", open_world=True)
        def search_knowledge_base(
            query: Annotated[str, "Knowledge base search query"],
        ) -> DataFrame:
            results = requests.get(...)
            return fc.create_dataframe(results)

    Notes:
    - The decorated function MUST NOT use *args/**kwargs
    - The decorated function MUST return a fenic DataFrame.
    - The decorated function SHOULD annotate parameters with `Annotated` types and descriptions.
    - The returned object is a DynamicTool ready for registration.
    - A `limit` parameter is automatically added to the function signature, which can be used to limit the number of rows returned up to the tool's `max_result_limit`.
    - A `table_format` parameter is automatically added to the function signature, which can be used to specify the format of the returned data (markdown, structured)
    """

    def decorator(func: Callable[..., DataFrame]) -> DynamicToolDefinition:
        _ensure_no_var_args(func, func_label=tool_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> LogicalPlan:
            result_df = func(*args, **kwargs)
            return result_df._logical_plan

        return DynamicToolDefinition(
            name=tool_name,
            description=tool_description,
            max_result_limit=max_result_limit,
            default_table_format=default_table_format,
            read_only=read_only,
            idempotent=idempotent,
            destructive=destructive,
            open_world=open_world,
            _func=wrapper,
        )

    return decorator

def _ensure_no_var_args(func: Callable[..., object], *, func_label: str) -> None:
    sig = inspect.signature(func)
    for p in sig.parameters.values():
        if p.kind.name in {"VAR_POSITIONAL", "VAR_KEYWORD"}:
            raise ValueError(
                f"{func_label} must not use *args or **kwargs for MCP tool introspection."
            )


def _auto_generate_read_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 50,
) -> DynamicToolDefinition:
    """Create a read tool over one or many datasets."""
    if len(datasets) == 0:
        raise ConfigurationError("Cannot create read tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.table_name: d.df for d in datasets}
    def read_func(
        df_name: Annotated[str, "Dataset name to read rows from."],
        limit: Annotated[Optional[Union[int, str]], "Max rows to read within a page"] = result_limit,
        offset: Annotated[Optional[Union[int, str]], "Row offset to start from (requires order_by)"] = None,
        order_by: Annotated[Optional[str], "Comma separated list of columns to order by (required for offset)"] = None,
        sort_ascending: Annotated[Optional[Union[bool, str]], "Sort ascending for all order_by columns"] = True,
        include_columns: Annotated[Optional[str], "Comma separated list of columns to include in the result"] = None,
        exclude_columns: Annotated[Optional[str], "Comma separated list of columns to exclude from the result"] = None,
    ) -> LogicalPlan:

        if df_name not in name_to_df:
            raise ValidationError(f"Unknown DataFrame '{df_name}'. Available: {', '.join(name_to_df.keys())}")
        df = name_to_df[df_name]
        limit = int(limit) if isinstance(limit, str) else limit
        offset = int(offset) if isinstance(offset, str) else offset
        sort_ascending = bool(sort_ascending) if isinstance(sort_ascending, str) else sort_ascending
        order_by = [c.strip() for c in order_by.split(",") if c.strip()] if order_by else None
        include_columns = [c.strip() for c in include_columns.split(",") if c.strip()] if include_columns else None
        exclude_columns = [c.strip() for c in exclude_columns.split(",") if c.strip()] if exclude_columns else None
        if include_columns:
            df = df.select(*include_columns)
        if exclude_columns:
            df = df.select(*[c for c in df.columns if c not in exclude_columns])
        # Apply paging (handles offset+order_by via SQL and optional limit)
        return _apply_paging(
            df,
            session,
            limit=limit,
            offset=offset,
            order_by=order_by,
            sort_ascending=sort_ascending,
        )

    return DynamicToolDefinition(
        name=tool_name,
        description=tool_description,
        _func=read_func,
        max_result_limit=result_limit,
        add_limit_parameter=False,
    )

def _auto_generate_search_summary_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
) -> DynamicToolDefinition:
    """Create a grep-like summary tool over one or many datasets (string columns)."""
    if len(datasets) == 0:
        raise ValueError("Cannot create search summary tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.table_name: d.df for d in datasets}


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

    return DynamicToolDefinition(
        name=tool_name,
        description=tool_description,
        _func=search_summary,
        max_result_limit=None,
    )

def auto_generate_search_content_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 100,
) -> DynamicToolDefinition:
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

    def search_rows(
        df_name: Annotated[str, "Dataset name to search (single dataset)"],
        pattern: Annotated[str, "Regex pattern to search for (use (?i) for case-insensitive)."],
        limit: Annotated[Optional[Union[int, str]], "Max rows to read within a page of search results"] = result_limit,
        offset: Annotated[Optional[Union[int, str]], "Row offset to start from (requires order_by)"] = None,
        order_by: Annotated[Optional[str], "Comma separated list of column names to order by (required with offset)"] = None,
        sort_ascending: Annotated[Optional[Union[bool, str]], "Sort ascending"] = True,
        search_columns: Annotated[Optional[str], "Comma separated list of column names search within; if omitted, matches in any string coluumn will be returned. Use this to query only specific columns in the search as needed."] = None,
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

    return DynamicToolDefinition(
        name=tool_name,
        description=tool_description,
        _func=search_rows,
        max_result_limit=result_limit,
        add_limit_parameter=False,
    )

def _auto_generate_schema_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
) -> DynamicToolDefinition:
    """Create a schema tool over one or many datasets.

    - Returns one row per dataset with a column `schema` containing a list of
      {column, type} entries.
    - If `df_name` is provided, returns only that dataset.
    """
    if len(datasets) == 0:
        raise ValueError("Cannot create schema tool: no datasets provided.")

    name_to_df: Dict[str, DataFrame] = {d.table_name: d.df for d in datasets}

    def schema_func(
        df_name: Annotated[str | None, "Optional DataFrame name to return a single schema for. To return schemas for all datasets, OMIT this parameter."] = None,
    ) -> LogicalPlan:
         # sometimes the models get...very confused, and pass the null string instead of `null` or omitting the field entirely
        if df_name == "null":
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
            schema_entries = [{"column": f.name, "type": str(convert_custom_dtype_to_polars(f.data_type))} for f in d.schema.column_fields]
            dataset_names.append(name)
            dataset_schemas.append(schema_entries)

        return InMemorySource.from_session_state(
            pl.DataFrame({
                "dataset": dataset_names,
                "schema": dataset_schemas,
            }),
            session._session_state,
        )

    # Enhanced description lists datasets and descriptions
    lines: List[str] = [tool_description.strip(), "", "Datasets available:"]
    for spec in datasets:
        if spec.description:
            lines.append(f"- {spec.table_name}: {spec.description}")
        else:
            lines.append(f"- {spec.table_name}")
    enhanced_description = "\n".join(lines)

    return DynamicToolDefinition(
        name=tool_name,
        description=enhanced_description,
        _func=schema_func,
        max_result_limit=None,
    )

def _auto_generate_sql_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    result_limit: int = 50,
) -> DynamicToolDefinition:
    """Create an Analyze tool that executes DuckDB SELECT SQL across datasets.

    - JOINs between the provided datasets are allowed.
    - DDL/DML, CTEs, subqueries, UNION, and multiple top-level queries are not allowed (enforced upstream).
    - The callable returns a LogicalPlan gathered later by the MCP server.
    """
    if len(datasets) == 0:
        raise ConfigurationError("Cannot create SQL tool: no datasets provided.")

    def analyze_func(
        full_sql: Annotated[str, "Full SELECT SQL. Refer to DataFrames by name in braces, e.g., {orders}."]
    ) -> LogicalPlan:
        return session.sql(full_sql.strip(), **{spec.table_name: spec.df for spec in datasets})._logical_plan

    # Enhanced description with dataset names and descriptions
    lines: List[str] = [tool_description.strip(), "", "Datasets available:"]
    for spec in datasets:
        if spec.description:
            lines.append(f"- {spec.table_name}: {spec.description}")
        else:
            lines.append(f"- {spec.table_name}")
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
            f"- Returns a maximum of {result_limit} rows.\n",
            "Examples:\n",  # nosec B608 - example text only
            f"- SELECT * FROM {example_name} WHERE REGEXP_MATCHES(message, '(?i)error|fail') LIMIT {result_limit}",  # nosec B608 - example text only
            f"- SELECT dept, COUNT(*) AS n FROM {example_name} WHERE status = 'active' GROUP BY dept HAVING n > 10 ORDER BY n DESC LIMIT {result_limit}",  # nosec B608 - example text only
            f"- Paging: page 2 of size {result_limit}\n  SELECT * FROM {example_name} ORDER BY created_at DESC LIMIT {result_limit} OFFSET {result_limit}",  # nosec B608 - example text only
        ]
    )
    enhanced_description = "\n".join(lines)

    tool = DynamicToolDefinition(
        name=tool_name,
        description=enhanced_description,
        _func=analyze_func,
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
        base_sql += " ORDER BY " + safe_order_by + f" {direction}"  #nosec B608
    if lim_val is not None:
        base_sql += f" LIMIT {lim_val}"
    if off_val is not None:
        base_sql += f" OFFSET {off_val}"

    df_with_paging = session.sql(base_sql, src=df)
    return df_with_paging._logical_plan


def _auto_generate_profile_tool(
    datasets: List[DatasetSpec],
    session: Session,
    tool_name: str,
    tool_description: str,
    *,
    topk_distinct: int = 10,
) -> DynamicToolDefinition:
    """Create a cached Profile tool for one or many datasets.

    Output columns include:
      - dataset, column, type, row_count, non_null_count, null_count
      - min, max, mean, std (for numerics)
      - distinct_count, top_values (JSON) for strings
      - true_count, false_count for booleans
    """
    if len(datasets) == 0:
        raise ValueError("Cannot create profile tool: no datasets provided.")

    def _compute_profile_rows(df: DataFrame, dataset_name: str) -> List[Dict[str, object]]:
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
        rows_list = _compute_profile_rows(df, dataset_name)
        safe_rows: List[Dict[str, str]] = []
        for row in rows_list:
            safe_rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
        pl_df = pl.DataFrame(safe_rows)
        plan = InMemorySource.from_session_state(pl_df, session._session_state)
        catalog = session._session_state.catalog
        catalog.create_view(view_name, plan, ignore_if_exists=True)

    def _ensure_profile_view_for_dataset(spec: DatasetSpec, tool_key: str, refresh: bool) -> LogicalPlan:
        schema_hash = _schema_fingerprint(spec.df)
        view_name = f"__fenic_profile__{tool_key}__{_sanitize_name(spec.table_name)}__{schema_hash}"
        catalog = session._session_state.catalog
        if refresh or not catalog.does_view_exist(view_name):
            _materialize_dataset_description(spec.df, spec.table_name, view_name)
        return catalog.get_view_plan(view_name)

    def profile_func(
        df_name: Annotated[str | None, "Optional DataFrame name to return a single profile for. To return profiles for all datasets, omit this parameter."] = None,
        refresh: Annotated[bool, "Recompute and refresh cached profile view(s)"] = False,
    ) -> LogicalPlan:
        # sometimes the models get...very confused, and pass the null string instead of `null` or omitting the field entirely
        if df_name == "null":
            df_name = None
        tool_key = _sanitize_name(tool_name)
        # Single dataset branch returns the view plan directly
        if df_name is not None:
            spec = next((d for d in datasets if d.table_name == df_name), None)
            if spec is None:
                raise ValidationError(f"Unknown dataset '{df_name}'. Available: {', '.join(d.table_name for d in datasets)}")
            return _ensure_profile_view_for_dataset(spec, tool_key, refresh)

        # Multi-dataset: concatenate cached views (or compute & cache if missing)
        per_df_polars: List[pl.DataFrame] = []
        for spec in datasets:
            # Ensure view exists and read it, then convert to polars for concatenation
            plan = _ensure_profile_view_for_dataset(spec, tool_key, refresh)
            pl_df = session._session_state.execution.collect(plan)[0]
            per_df_polars.append(pl_df)

        combined = pl.concat(per_df_polars, how="vertical") if len(per_df_polars) > 1 else per_df_polars[0]
        return InMemorySource.from_session_state(combined, session._session_state)

    # Enhanced description: list datasets and notes
    lines: List[str] = [tool_description.strip(), "", "Datasets available:"]
    for spec in datasets:
        if spec.description:
            lines.append(f"- {spec.table_name}: {spec.description}")
        else:
            lines.append(f"- {spec.table_name}")
    lines.extend(
        [
            "",
            "Notes:",
            "- Results are cached per dataset, tool name, and schema fingerprint; pass refresh=true to recompute.",
            "- Returns per-column stats (numeric, boolean, string).",
        ]
    )
    enhanced_description = "\n".join(lines)

    return DynamicToolDefinition(
        name=tool_name,
        description=enhanced_description,
        _func=profile_func,
        max_result_limit=None,
    )


def _auto_generate_core_tools(
    datasets: List[DatasetSpec],
    session: Session,
    *,
    tool_group_name: str,
    sql_max_rows: int = 100,
) -> List[DynamicToolDefinition]:
    """Generate core tools spanning all datasets: Schema, Profile, Analyze.

    - Schema: list columns/types for any or all datasets
    - Profile: dataset statistics for any or all datasets
    - Read: read rows from a single dataset to sample the data
    - Search Summary: regex search across all datasets and return a summary of the number of matches per dataset
    - Search Content: return matching rows from a single dataset using regex matching across string columns
    - Analyze: DuckDB SELECT-only SQL across datasets
    """
    group_desc = "; ".join(
        [f"{d.table_name}: {d.description.strip()}" if d.description else d.table_name for d in datasets]
    )

    schema_tool = _auto_generate_schema_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Schema",
        tool_description="\n\n".join([
            "Show the schema (column names and types) for any or all of the datasets listed below.",
            group_desc,
        ]),
    )

    profile_tool = _auto_generate_profile_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Profile",
        tool_description="\n\n".join([
            "Return dataset data profile: row_count and per-column stats for any or all of the datasets listed below.",
            "Numeric stats: min/max/mean/std; Booleans: true/false counts; Strings: distinct_count and top values.",
            "Results are cached per tool name and schema fingerprint; pass refresh=true to recompute.",
            group_desc,
        ]),
    )

    read_tool = _auto_generate_read_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Read",
        tool_description="\n\n".join([
            "Read rows from a single dataset. Use to sample data, or to execute simple queries over the data that do not require filtering or grouping.",
            "Available datasets:\n",
            group_desc,
        ]),
        result_limit=sql_max_rows,
    )

    search_summary_tool = _auto_generate_search_summary_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Search Summary",
        tool_description="\n\n".join([
            "Perform a substring/regex search across all datasets and return a summary of the number of matches per dataset.",
            "Available datasets:\n",
            group_desc,
        ]),
    )
    search_content_tool = auto_generate_search_content_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Search Content",
        tool_description="\n\n".join([
            "Return matching rows from a single dataset using substring/regex across string columns.",
            "Available datasets:",
            group_desc,
        ]),
        result_limit=sql_max_rows,
    )

    analyze_tool = _auto_generate_sql_tool(
        datasets,
        session,
        tool_name=f"{tool_group_name} - Analyze",
        tool_description="\n\n".join([
            "Execute Read-Only (SELECT) SQL over the provided datasets using fenic's SQL support.",
            "DDL/DML, CTEs, subqueries, UNION, and multiple top-level queries are not allowed (enforced upstream).",
            "For text search, prefer regular expressions (REGEXP_MATCHES()/REGEXP_EXTRACT()).",
            "Paging: use ORDER BY to define row order, then LIMIT and OFFSET for pages.",
            "JOINs between datasets are allowed. Refer to datasets by name in braces, e.g., {orders}.",
            "Below, the available datasets are listed, by name and description.",
            group_desc,
        ]),
        result_limit=sql_max_rows,
    )

    return [schema_tool, profile_tool, read_tool, search_summary_tool, search_content_tool, analyze_tool]


def _build_datasets_from_tables(table_names: List[str], session: Session) -> List[DatasetSpec]:
    """Resolve catalog table names into DatasetSpec list with validated descriptions.

    Raises ConfigurationError if any table is missing or lacks a non-empty description.
    """
    if len(table_names) == 0:
        raise ConfigurationError("No tables provided for tool generation.")

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
