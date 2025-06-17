from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]

@pl.api.register_expr_namespace("markdown")
class MarkdownExtractor:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def to_json(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="md_to_json_expr",
            args=self._expr,
            is_elementwise=True,
        )
