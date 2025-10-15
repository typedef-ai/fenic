from pathlib import Path

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]


@pl.api.register_expr_namespace("regexp")
class Regexp:
    """Namespace for regular expression operations on Polars expressions."""

    def __init__(self, expr: pl.Expr) -> None:
        """Initialize a Regexp Namespace with a Polars expression.

        Args:
            expr: A Polars expression containing the text data for regex operations.
        """
        self.expr = expr

    def instr(self, pattern: IntoExpr, idx: IntoExpr) -> pl.Expr:
        """Find the position of a regex match in a string.

        Args:
            pattern: Regular expression pattern to search for.
            idx: Capture group index (0 for whole match, 1+ for capture groups).

        Returns:
            1-based position of the match, or 0 if no match found, or null if input is null.
        """
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="regexp_instr",
            args=[self.expr, pattern, idx],
            is_elementwise=True,
        )

    def extract_all(self, pattern: IntoExpr, idx: IntoExpr) -> pl.Expr:
        """Extract all matches of a regex pattern, optionally from a specific capture group.

        Args:
            pattern: Regular expression pattern to search for.
            idx: Capture group index (0 for whole match, 1+ for capture groups).

        Returns:
            List of all matches, or empty list if no matches, or null if input is null.
        """
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="regexp_extract_all",
            args=[self.expr, pattern, idx],
            is_elementwise=True,
        )

