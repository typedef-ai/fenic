from pathlib import Path

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]


def count_tokens(
    expr: IntoExpr,
) -> pl.Expr:
    """Count either characters or words in text.

    Args:
        expr: Input text column

    Returns:
        pl.Expr: Expression with counts as UInt32
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="count_tokens",
        args=expr,
        is_elementwise=True,
    )


@pl.api.register_expr_namespace("tokenization")
class Tokenization:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def count_tokens(
        self,
    ) -> pl.Expr:
        """Count tokens using tiktoken."""
        return count_tokens(self._expr)
