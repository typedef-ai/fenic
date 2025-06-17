from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]


@pl.api.register_expr_namespace("json")
class Json:
    """Namespace for JSON-related operations on Polars expressions."""

    def __init__(self, expr: pl.Expr) -> None:
        """Initialize a Json Namespace with a Polars expression.

        Args:
            expr: A Polars expression for JSON processing.
        """
        self.expr = expr

    def jq(
        self,
        query: str,
    ) -> pl.Expr:
        """Apply a JQ query to each JSON string in the column.

        This expression applies the provided JQ filter to each row in the column,
        assuming each row is a valid JSON string. If a row is invalid JSON or if
        the query fails, the result is set to null.

        The output is a list of strings for each row, corresponding to the results
        of applying the JQ query.

        Args:
            query: A valid JQ filter expression to apply to each JSON value.

        Returns:
            A Polars expression returning a `List[str]` column with the query results.
        """
        kwargs = {
            "query": query,
        }
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="jq_expr",
            args=self.expr,
            kwargs=kwargs,
            is_elementwise=True,
        )
