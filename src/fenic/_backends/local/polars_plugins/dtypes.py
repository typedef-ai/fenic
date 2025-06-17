from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]


@pl.api.register_expr_namespace("dtypes")
class Dtypes:
    """Namespace for Fenic data type operations on Polars expressions."""

    def __init__(self, expr: pl.Expr) -> None:
        """Initialize a Json Namespace with a Polars expression.

        Args:
            expr: A Polars expression for JSON processing.
        """
        self.expr = expr

    def cast(
        self,
        source_dtype: str,
        dest_dtype: str,
    ) -> pl.Expr:
        """Cast a Polars expression to a Fenic data type.

        Args:
            source_dtype: A JSON string representing a Fenic data type.
            dest_dtype: A JSON string representing a Fenic data type.

        Returns:
            A Polars expression returning a column with the casted values.
        """
        kwargs = {
            "source_dtype": source_dtype,
            "dest_dtype": dest_dtype,
        }
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="cast_expr",
            args=self.expr,
            kwargs=kwargs,
            is_elementwise=True,
        )
