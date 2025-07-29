from pathlib import Path

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]



@pl.api.register_expr_namespace("fuzz")
class Fuzz:
    """Namespace for fuzzy string matching operations on Polars expressions."""

    def __init__(self, expr: pl.Expr) -> None:
        """Initialize a Fuzz Namespace with a Polars expression.

        Args:
            expr: A Polars expression containing the data for fuzzy string matching.
        """
        self.expr = expr

    def normalized_indel_similarity(self, other: IntoExpr) -> pl.Expr:
        """Compute the Indel similarity between two strings."""
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="normalized_indel_similarity",
            args=[self.expr, other],
            is_elementwise=True,
        )

    def normalized_levenshtein_similarity(self, other: IntoExpr) -> pl.Expr:
        """Compute the Levenshtein similarity between two strings."""
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="normalized_levenshtein_similarity",
            args=[self.expr, other],
            is_elementwise=True,
        )

    def normalized_damerau_levenshtein_similarity(self, other: IntoExpr) -> pl.Expr:
        """Compute the Damerau-Levenshtein similarity between two strings."""
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="normalized_damerau_levenshtein_similarity",
            args=[self.expr, other],
            is_elementwise=True,
        )

    def normalized_jarowinkler_similarity(self, other: IntoExpr) -> pl.Expr:
        """Compute the Jaro-Winkler similarity between two strings."""
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="normalized_jarowinkler_similarity",
            args=[self.expr, other],
            is_elementwise=True,
        )

    def normalized_jaro_similarity(self, other: IntoExpr) -> pl.Expr:
        """Compute the Jaro similarity between two strings."""
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="normalized_jaro_similarity",
            args=[self.expr, other],
            is_elementwise=True,
        )

    def normalized_hamming_similarity(self, other: IntoExpr) -> pl.Expr:
        """Compute the Hamming similarity between two strings."""
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="normalized_hamming_similarity",
            args=[self.expr, other],
            is_elementwise=True,
        )
