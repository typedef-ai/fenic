"""API-layer generators for automatic MCP tools from DataFrames.

This module builds optional-parameter filter tools and standalone semantic tools
by inspecting a DataFrame schema, then delegates to core plumbing to create
ResolvedTool objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from fenic.api.dataframe.dataframe import DataFrame
from fenic.api.functions import array_contains, coalesce, col, lit, semantic, tool_param
from fenic.core._logical_plan.tools import (
    ResolvedTool,
    create_unresolved_tool,
    resolve_tool,
)
from fenic.core._logical_plan.tools import (
    ToolParam as CoreToolParam,
)
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
)

ToolParameterType = Union[str, int, float, bool, list, dict]


def auto_generate_filter_tool(
    df: DataFrame,
    name: str,
    description: str,
    *,
    result_limit: int = 50,
    include_string_contains: bool = True,
    include_array_contains: bool = True,
    allowed_values_map: Optional[Dict[str, List[ToolParameterType]]] = None,
) -> Optional[ResolvedTool]:
    """Auto-generate an optional-parameter filter tool for primitive/array columns.

    Supported types:
    - integer/float (range: min_<col>, max_<col>)
    - boolean (equality: <col>)
    - string (equality: <col>, optional contains: <col>_contains)
    - arrays of primitive (single element contains: <col>_contains)

    Returns None if no eligible columns are present in the DataFrame schema.
    """
    allowed_values_map = allowed_values_map or {}

    predicates: List = []
    tool_params: List[CoreToolParam] = []

    for field in df.schema.column_fields:
        col_name = field.name
        dtype = field.data_type

        if dtype in (IntegerType, FloatType, DoubleType):
            min_param = f"min_{col_name}"
            max_param = f"max_{col_name}"
            tool_params.append(CoreToolParam(name=min_param, description=f"Minimum {col_name}", has_default=True, default_value=None))
            tool_params.append(CoreToolParam(name=max_param, description=f"Maximum {col_name}", has_default=True, default_value=None))
            predicates.append(coalesce(col(col_name) >= tool_param(min_param, dtype), lit(True)))
            predicates.append(coalesce(col(col_name) <= tool_param(max_param, dtype), lit(True)))
            continue

        if dtype == BooleanType:
            param_name = col_name
            tool_params.append(CoreToolParam(name=param_name, description=f"{col_name}?", has_default=True, default_value=None))
            predicates.append(coalesce(col(col_name) == tool_param(param_name, dtype), lit(True)))
            continue

        if dtype == StringType:
            eq_allowed = allowed_values_map.get(col_name)
            tool_params.append(
                CoreToolParam(
                    name=col_name,
                    description=f"{col_name} equals",
                    allowed_values=eq_allowed,
                    has_default=True,
                    default_value=None,
                )
            )
            predicates.append(coalesce(col(col_name) == tool_param(col_name, dtype), lit(True)))
            if include_string_contains:
                contains_param = f"{col_name}_contains"
                tool_params.append(CoreToolParam(name=contains_param, description=f"{col_name} contains", has_default=True, default_value=None))
                predicates.append(coalesce(col(col_name).contains(tool_param(contains_param, dtype)), lit(True)))
            continue

        if isinstance(dtype, ArrayType) and dtype.element_type in (IntegerType, FloatType, DoubleType, StringType, BooleanType):
            if include_array_contains:
                arr_param = f"{col_name}_contains"
                tool_params.append(CoreToolParam(name=arr_param, description=f"{col_name} contains element", has_default=True, default_value=None))
                predicates.append(coalesce(array_contains(col(col_name), tool_param(arr_param, dtype.element_type)), lit(True)))
            continue

    if not tool_params:
        return None

    if not predicates:
        query = df.filter(lit(True))._logical_plan
    else:
        conj = predicates[0]
        for p in predicates[1:]:
            conj = conj & p
        query = df.filter(conj)._logical_plan

    unresolved = create_unresolved_tool(name=name, description=description, params=tool_params, result_limit=result_limit)
    return resolve_tool(unresolved, query)


def auto_generate_semantic_tool(
    df: DataFrame,
    name: str,
    description: str,
    *,
    result_limit: int = 50,
    semantic_param_name: str = "semantic_query",
    semantic_columns: Optional[List[str]] = None,
    semantic_prompt_prefix: Optional[str] = None,
) -> Optional[ResolvedTool]:
    """Auto-generate a standalone semantic query tool over selected columns (string columns by default)."""
    if semantic_columns is None:
        semantic_columns = [f.name for f in df.schema.column_fields if f.data_type == StringType]

    if not semantic_columns:
        return None

    lines = []
    if semantic_prompt_prefix:
        lines.append(semantic_prompt_prefix.strip())
    else:
        lines.append("The following is a search query over this dataset. Determine if the row matches the query.")
    lines.append("QUERY: {{query}}")
    lines.append("ROW DATA:")
    for col_name in semantic_columns:
        lines.append(f"{col_name.upper()}: {{{{{col_name}}}}}")
    prompt = "\n".join(lines)

    tool_params = [CoreToolParam(name=semantic_param_name, description="Natural language query")]
    semantic_vars = {c: col(c) for c in semantic_columns}
    semantic_pred = semantic.predicate(prompt, strict=False, query=tool_param(semantic_param_name, StringType), **semantic_vars)
    filtered_plan = df.filter(semantic_pred)._logical_plan

    unresolved = create_unresolved_tool(name=name, description=description, params=tool_params, result_limit=result_limit)
    return resolve_tool(unresolved, filtered_plan)
