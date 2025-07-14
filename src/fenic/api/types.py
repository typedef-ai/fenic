"""Types for providing natural language instructions to semantic functions."""

from typing import Dict

from pydantic import BaseModel, Field

from fenic.api.column import Column
from fenic.core._logical_plan import LogicalExpr
from fenic.core._logical_plan.expressions.basic import AliasExpr, ColumnExpr
from fenic.core._utils import misc as misc_utils
from fenic.core.error import ValidationError
from fenic.core.types.instruction import NamedExpr, ResolvedInstructionTemplate


class InstructionTemplate(BaseModel):
    """A template for semantic operations that allows binding complex column expressions to placeholders.

    InstructionTemplate enables semantic functions to work with arbitrary column expressions
    instead of just simple column names. It combines a natural language instruction with
    bindings that map placeholder names to column expressions.

    This is particularly useful for:
    - Creating rich contextual information by concatenating multiple columns
    - Applying conditional logic to determine what text to analyze
    - Combining literal values with column data
    - Pre-processing column values before semantic analysis

    Args:
        instruction: A string containing the natural language instruction with placeholders
            in curly braces (e.g., "Analyze {customer_info} for {issue_type}").
        **bindings: Keyword arguments where each key is a placeholder name and each value
            is a Column expression that will be substituted for that placeholder.

    Attributes:
        instruction: The template instruction string with placeholders.
        bindings: Dictionary mapping placeholder names to Column expressions.

    Example: Basic usage with concatenation
        ```python
        from fenic.api.functions.text import concat
        from fenic.api.functions import lit, col

        template = InstructionTemplate(
            "Analyze feedback from {customer_context}",
            customer_context=concat(col("name"), lit(" ("), col("tier"), lit(")"))
        )

        # Use with semantic functions
        df.select(semantic.map(template))
        ```

    Example: Conditional expressions
        ```python
        template = InstructionTemplate(
            "Process this {priority_message}",
            priority_message=when(col("urgent") == True,
                                  concat(lit("URGENT: "), col("message"))
                             ).otherwise(col("message"))
        )

        df.filter(semantic.predicate(template))
        ```

    Example: Multiple bindings
        ```python
        template = InstructionTemplate(
            "Given customer {customer_info} and their issue: {issue_details}, generate a response",
            customer_info=concat(col("first_name"), lit(" "), col("last_name"),
                               lit(" ("), col("subscription_tier"), lit(")")),
            issue_details=concat(col("category"), lit(": "), col("description"))
        )

        df.select(semantic.map(template))
        ```

    Example: Reusing templates
        ```python
        # Create template once
        enriched_template = InstructionTemplate(
            "Analyze {enriched_content}",
            enriched_content=concat(col("title"), lit(" - "), col("body"))
        )

        # Use across multiple operations
        df.select(semantic.map(enriched_template).alias("summary"))
        df.filter(semantic.predicate(enriched_template))
        ```

    Note:
        - Placeholder names must match the binding keys exactly
        - Each placeholder in the instruction should have a corresponding binding
        - If a placeholder has no binding, it will be treated as a simple column reference
        - Column expressions are automatically aliased to match placeholder names
        - Templates can be reused across multiple semantic operations
    """
    instruction: str = Field(..., description="Natural language instruction string with placeholders in {braces}")
    bindings: Dict[str, Column] = Field(default_factory=dict, description="Dictionary mapping placeholder names to Column expressions")

    model_config = {"arbitrary_types_allowed": True}
    def __init__(self, instruction: str, **bindings: Column):
        """Initialize an InstructionTemplate with instruction and column bindings.

        Args:
            instruction: Natural language instruction string with placeholders in {braces}.
            **bindings: Keyword arguments mapping placeholder names to Column expressions.
        """
        super().__init__(instruction=instruction, bindings=dict(bindings))

    def to_resolved_template(self) -> ResolvedInstructionTemplate:
        """Convert the InstructionTemplate to a ResolvedInstructionTemplate.

        Returns:
            ResolvedInstructionTemplate with placeholders replaced by column expressions.
        """
        children = self._resolve_children()
        return ResolvedInstructionTemplate(self.instruction, children)

    @staticmethod
    def _auto_alias_logical_expr(logical_expr: LogicalExpr, placeholder_column_name: str) -> NamedExpr:
        if isinstance(logical_expr, AliasExpr):
            # Validate that the alias matches the key
            if logical_expr.name != placeholder_column_name:
                raise ValidationError(
                    f"Alias name must match the key. Expected '{placeholder_column_name}', got '{logical_expr.name}'")
            return logical_expr
        else:
            aliased_expr = AliasExpr(logical_expr, placeholder_column_name)
            return aliased_expr

    def _resolve_children(self) -> list[NamedExpr]:
        """Resolve placeholders in the instruction to named column expressions.

        This method parses the instruction string to find all placeholders (text in {braces}),
        then resolves each placeholder to either:
        1. A bound column expression from the bindings dictionary, or
        2. A simple column reference if no binding exists

        The method automatically handles:
        - Deduplication: Multiple occurrences of the same placeholder use a single expression
        - Auto-aliasing: Unaliased expressions get aliased to match the placeholder name
        - Alias validation: Existing aliases must match the placeholder name exactly

        Returns:
            List of named expressions (AliasExpr or ColumnExpr) for each unique placeholder
            in the instruction, in the order they first appear.

        Raises:
            ValidationError: If an aliased expression has a name that doesn't match the placeholder.

        Example:
            ```python
            template = InstructionTemplate(
                "Process {data} and {data} again",  # {data} appears twice
                data=concat(col("a"), col("b"))
            )
            # resolve_children() returns a single expression for "data", not two
            ```
        """
        placeholder_keys = misc_utils.parse_instruction(self.instruction)
        # Deduplicate placeholder keys to avoid creating multiple expressions for the same placeholder
        unique_placeholder_keys = list(dict.fromkeys(placeholder_keys))  # Preserves order
        children: list[NamedExpr] = []

        for placeholder_key in unique_placeholder_keys:
            if placeholder_key in self.bindings:
                children.append(
                    self._auto_alias_logical_expr(
                        self.bindings[placeholder_key]._logical_expr,
                        placeholder_key
                    )
                )
            else:
                # Use column reference for missing keys
                children.append(ColumnExpr(placeholder_key))

        return children