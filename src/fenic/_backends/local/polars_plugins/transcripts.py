from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]


@pl.api.register_expr_namespace("transcript")
class TranscriptExtractor:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def parse(self, format: str) -> pl.Expr:
        """Parse the transcript into a structured format with unified schema."""
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="ts_parse_expr",
            args=self._expr,
            kwargs={"format": format},
            is_elementwise=True,
        )
