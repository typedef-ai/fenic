from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parents[3]


@pl.api.register_expr_namespace("jinja")
class Jinja:
    """Namespace for Jinja template rendering operations on Polars expressions."""

    def __init__(self, expr: pl.Expr) -> None:
        """Initialize a Jinja Namespace with a Polars expression.

        Args:
            expr: A Polars expression containing the data for template rendering.
                 Expected to be a struct expression with fields matching template variables.
        """
        self.expr = expr

    def render(
        self,
        template: str,
        strict: bool,
    ) -> pl.Expr:
        """Render a Jinja template using values from the struct expression.

        This expression applies the provided Jinja template to each row in the struct column,
        using the specified variable names to extract values from the struct fields.

        Args:
            template: A Jinja2 template string to render for each row.
                     Example: "Hello {{ user.name }}! You have {{ item_length }} items."

        Returns:
            A Polars expression returning a `String` column with the rendered templates.

        Example:
            ```python
            # Given a struct column with fields "user" and "items"
            result = (
                pl.struct([pl.col("user"), pl.col("item_length")])
                .jinja.render(
                    template="Hello {{ user.name }}! You have {{ item_length }} items.",
                )
            )
            ```

        Notes:
            - Template syntax and variable access is validated by VariableTree before execution
            - Null struct values result in null output
            - Template rendering errors result in null output for that row
            - The template has access to all Jinja2 built-in filters and functions
        """
        kwargs = {
            "template": template,
            "strict": strict,
        }
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="jinja_render",
            args=self.expr,
            kwargs=kwargs,
            is_elementwise=True,
        )
