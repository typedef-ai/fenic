from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fenic.api.dataframe import DataFrame

from typing import Dict, List, Tuple, Union

from fenic.api.column import Column
from fenic.api.functions import (
    approx_count_distinct,
    avg,
    col,
    count,
    count_distinct,
    max,
    min,
    sum,
    sum_distinct,
)
from fenic.core._logical_plan.expressions import AliasExpr
from fenic.core._logical_plan.expressions.base import AggregateExpr


class BaseGroupedData:
    """Base class for aggregation methods shared between GroupedData and SemanticallyGroupedData."""

    def __init__(self, df: "DataFrame"):
        self._df = df

    def _process_agg_dict(self, agg_dict: Dict[str, str]) -> List[Column]:
        """Process dictionary-style aggregation specifications."""
        agg_funcs = {
            "sum": sum,
            "avg": avg,
            "min": min,
            "max": max,
            "count": count,
            "mean": avg,
            "count_distinct": count_distinct,
            "approx_count_distinct": approx_count_distinct,
            "sum_distinct": sum_distinct,
        }
        agg_exprs = []
        for col_name, func_name in agg_dict.items():
            if func_name not in agg_funcs:
                raise ValueError(
                    f"Unsupported aggregation function: {func_name}. "
                    f"Supported functions are: {list(agg_funcs.keys())}"
                )
            agg_expr = agg_funcs[func_name](col(col_name))
            agg_expr = agg_expr.alias(f"{func_name}({col_name})")
            agg_exprs.append(agg_expr)
        return agg_exprs

    def _process_agg_exprs(self, cols: Tuple[Column, ...]) -> List[AliasExpr]:
        """Process Column-style aggregation expressions."""
        agg_exprs = []
        for column in cols:
            if isinstance(column._logical_expr, AggregateExpr):
                column = column.alias(str(column._logical_expr))
                agg_exprs.append(column._logical_expr)
            elif isinstance(column._logical_expr, AliasExpr) and isinstance(
                column._logical_expr.expr, AggregateExpr
            ):
                agg_exprs.append(column._logical_expr)
            else:
                raise ValueError(
                    f"Expression {column._logical_expr} is not an aggregation. "
                    "Aggregation expressions must use aggregate functions like sum(), avg(), min(), max(), count(). "
                    "For example: df.agg(sum('col'), avg('col2'))"
                )
        return agg_exprs

    def _validate_agg_exprs(self, *exprs: Union[Column, Dict[str, str]]) -> None:
        """Validate aggregation expressions."""
        if not exprs:
            raise ValueError("At least one aggregation expression is required")

        if len(exprs) == 1 and isinstance(exprs[0], dict):
            agg_dict = exprs[0]
            for col_name, agg_func in agg_dict.items():
                if not isinstance(col_name, str):
                    raise TypeError(
                        f"Dictionary keys must be strings (column names), got {type(col_name)}"
                    )
                if not isinstance(agg_func, str):
                    raise TypeError(
                        f"Dictionary values must be strings (aggregation function names), got {type(agg_func)} for key '{col_name}'"
                    )
            return

        for i, expr in enumerate(exprs):
            if not isinstance(expr, Column):
                raise TypeError(
                    f"Aggregation expressions must be Column objects, got {type(expr)} at position {i}"
                )
