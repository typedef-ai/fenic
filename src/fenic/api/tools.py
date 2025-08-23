"""API-layer generators for automatic MCP tools from DataFrames.

This module builds optional-parameter filter tools and standalone semantic tools
by inspecting a DataFrame schema, then delegates to core plumbing to create
ResolvedTool objects.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List

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
from fenic.core._logical_plan.plans import SQL, InMemorySource
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.tools import (
    DynamicTool,
)
from fenic.core.types.datatypes import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
)


def auto_generate_schema_tool(
    df: DataFrame,
    name: str,
    description: str,
) -> DynamicTool:
    """Generate a dataset schema tool that returns all columns and types (no limit).

    The output rows have columns: `column` and `type`.
    """

    def schema_func() -> LogicalPlan:
        schema_rows = [
            {"column": f.name, "type": str(f.data_type)} for f in df.schema.column_fields
        ]
        pl_df = pl.DataFrame(schema_rows)
        return InMemorySource.from_session_state(pl_df, df._session_state)

    enhanced_description = "\n\n".join(
        [
            description.strip(),
            "Notes:",
            "- Returns the full schema (no row limit).",
            "- Columns: column, type.",
        ]
    )

    return DynamicTool(
        name=name,
        description=enhanced_description,
        func=schema_func,
        result_limit=None,
    )



def auto_generate_sql_tool(
    df: DataFrame,
    name: str,
    description: str,
    *,
    result_limit: int = 50,
) -> DynamicTool:
    """Generate an Analyze tool that accepts a full SELECT-only DuckDB SQL over {df}.

    DDL/DML, JOINs, CTEs, subqueries, UNION, and multiple tables are not allowed (enforced upstream).
    """
    column_names = [f.name for f in df.schema.column_fields]
    if not column_names:
        raise ValueError("Cannot create SQL tool: DataFrame has no columns.")

    def _assert_full_sql_shape(sql_text: str) -> None:
        text = sql_text.strip().lower()
        if not text.startswith("select"):
            raise ValueError("Only SELECT is allowed in full_sql")

    def analyze_func(
        full_sql: Annotated[str, "Full SELECT SQL. Must reference the DataFrame as {df}."]
    ) -> LogicalPlan:
        sql_text = full_sql.strip()
        _assert_full_sql_shape(sql_text)
        query = sql_text
        # Restricted to SELECT-only over a single source {df}
        plan = SQL.from_session_state([df._logical_plan], ["df"], query, df._session_state)  # nosec B608
        return plan

    enhanced_description = "\n\n".join(
        [
            description.strip(),
            "Notes:",
            "- SQL dialect: DuckDB.",
            "- For text search, prefer regular expressions using REGEXP_MATCHES().",
            "- Paging: use ORDER BY to define row order, then LIMIT and OFFSET for pages.",
            "",
            "Examples:",
            "- SELECT * FROM {df} WHERE REGEXP_MATCHES(message, '(?i)error|fail') LIMIT 100",
            "- SELECT dept, COUNT(*) AS n FROM {df} WHERE status = 'active' GROUP BY dept HAVING n > 10 ORDER BY n DESC LIMIT 100",
            "- -- Paging: page 2 of size 50\n  SELECT * FROM {df} ORDER BY created_at DESC LIMIT 50 OFFSET 50",
        ]
    )

    tool = DynamicTool(
        name=name,
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


def auto_generate_profile_tool(
    df: DataFrame,
    name: str,
    description: str,
    *,
    topk_distinct: int = 10,
) -> DynamicTool:
    """Generate a cached Describe tool with dataset-level profile info.

    Columns in result:
      - column, type, row_count, non_null_count, null_count
      - min, max, mean, std (for numerics)
      - distinct_count, top_values (JSON) for strings [NOTE: computed via group_by; improve with a distinct operator]
      - true_count, false_count for booleans
    """

    def profile_func(
        refresh: Annotated[bool, "Recompute and refresh cached profile view"] = False,
    ) -> LogicalPlan:
        tool_key = _sanitize_name(name)
        schema_hash = _schema_fingerprint(df)
        view_name = f"__fenic_profile__{tool_key}__{schema_hash}"

        catalog = df._session_state.catalog
        if not refresh and catalog.does_view_exist(view_name):
            return catalog.describe_view(view_name)

        # Compute row_count
        total_rows = df.count()

        # Sample approach: operate on a preview for some stats; fallback to small materialization for string top-k
        preview = df.limit(10000)
        rows = preview.to_polars()
        rows_list: List[Dict[str, object]] = []
        for field in df.schema.column_fields:
            col_name = field.name
            dtype_str = str(field.data_type)

            # Non-null count
            nn = df.agg(count(col(col_name)).alias("c")).to_polars().get_column("c")[0]
            null_count = int(total_rows - nn)

            stats: Dict[str, object] = {
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

            # Numeric stats
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

            # Boolean stats (computed from sample to avoid heavy full scans)
            elif field.data_type == BooleanType:
                if col_name in rows.columns:
                    s_bool = rows.get_column(col_name).cast(pl.Boolean)
                    true_cnt = int((s_bool).sum())
                    # Use vectorized negation for boolean Series
                    false_cnt = int((~s_bool).sum())
                else:
                    true_cnt = 0
                    false_cnt = 0
                stats["true_count"] = true_cnt
                stats["false_count"] = false_cnt

            # String stats
            elif field.data_type == StringType:
                # Heuristics to avoid overwhelming results for large text columns
                # Use a sample to estimate average length and distinctness
                if col_name in rows.columns:
                    s = rows.get_column(col_name)
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

                # Only compute top-K for reasonably short text with manageable distincts
                compute_topk = (
                    (avg_len is not None and avg_len <= 128) and
                    (sample_distinct is not None and sample_distinct <= max(topk_distinct * 10, 200))
                )

                if compute_topk and col_name in rows.columns:
                    vc = rows.get_column(col_name).value_counts(sort=True)
                    distinct_cnt = int(vc.height)
                    stats["distinct_count"] = distinct_cnt
                    top_vals: List[Dict[str, object]] = []
                    val_col = col_name if col_name in vc.columns else vc.columns[0]
                    for i in range(min(topk_distinct, vc.height)):
                        top_vals.append({
                            "value": str(vc.get_column(val_col)[i]),
                            "count": int(vc.get_column("count")[i]),
                        })
                    stats["top_values"] = json.dumps(top_vals)
                else:
                    # Skip heavy distincts for large text; leave fields empty
                    stats["distinct_count"] = None
                    stats["top_values"] = json.dumps([])

            rows_list.append(stats)

        # Materialize profile as a view for caching
        # Ensure all columns are valid (avoid Null dtype) by stringifying values
        safe_rows: List[Dict[str, str]] = []
        for row in rows_list:
            safe_row: Dict[str, str] = {}
            for k, v in row.items():
                safe_row[k] = "" if v is None else str(v)
            safe_rows.append(safe_row)
        pl_df = pl.DataFrame(safe_rows)
        plan = InMemorySource.from_session_state(pl_df, df._session_state)
        catalog = df._session_state.catalog
        catalog.create_view(view_name, plan, ignore_if_exists=True)
        return catalog.describe_view(view_name)

    enhanced_description = "\n\n".join(
        [
            description.strip(),
            "Notes:",
            "- Results are cached per tool name and schema fingerprint; pass refresh=true to recompute.",
            (
                "- Returns per-column stats:\n"
                "  * Numeric: min, max, mean, std\n"
                "  * Boolean: true_count, false_count\n"
                "  * String: distinct_count, top_values (best-effort from sample)"
            ),
        ]
    )

    return DynamicTool(
        name=name,
        description=enhanced_description,
        func=profile_func,
        result_limit=None,
    )


def auto_generate_core_tools(
    df: DataFrame,
    dataset_name: str,
    dataset_description: str,
    *,
    sql_max_rows: int = 100,
) -> List[DynamicTool]:
    """Generate the three core MCP tools for a dataset: Schema, Describe, Analyze.

    - Schema: returns all columns and types
    - Describe: cached dataset profile (row count and per-column stats)
    - Analyze: full DuckDB SELECT-only SQL over {df}, with a maximum row cap
    """
    dataset_desc = dataset_description.strip()

    schema_tool = auto_generate_schema_tool(
        df,
        name=f"{dataset_name} - Schema",
        description="\n\n".join([
            dataset_desc,
            "Lists all columns and types for this dataset. Returns the full schema (no row limit).",
        ]),
    )

    describe_tool = auto_generate_profile_tool(
        df,
        name=f"{dataset_name} - Describe",
        description="\n\n".join([
            dataset_desc,
            (
                "Return dataset profile: row_count and per-column stats.\n"
                "Numerics: min/max/mean/std; Booleans: true/false counts; Strings: distinct_count and top values.\n"
                "Results are cached per tool name and schema fingerprint; pass refresh=true to recompute."
            ),
        ]),
    )

    analyze_tool = auto_generate_sql_tool(
        df,
        name=f"{dataset_name} - Analyze",
        description="\n\n".join([
            dataset_desc,
            (
                "Execute DuckDB SELECT-only SQL over this dataset referenced as {df}.\n"
                "DDL/DML, JOINs, CTEs, subqueries, UNION, and multiple tables are not allowed.\n"
                "For text search, prefer regular expressions (REGEXP_MATCHES()/REGEXP_EXTRACT())."
            ),
        ]),
        result_limit=sql_max_rows,
    )

    return [schema_tool, describe_tool, analyze_tool]
