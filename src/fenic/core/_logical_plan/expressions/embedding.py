"""Embedding operation expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction
from fenic.core.error import ValidationError
from fenic.core.types import (
    ColumnField,
    SemanticSimilarityMetric,
)


class EmbeddingNormalizeExpr(ScalarFunction):
    """Expression for normalizing embedding vectors to unit length."""

    function_name = "embedding.normalize"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.dimensions = None
        
        super().__init__(expr)

    def __str__(self) -> str:
        return f"embedding.normalize({self.expr})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        # Call parent to handle signature validation (which ensures EmbeddingType)
        super().to_column_field(plan)
        # Get the actual input field to extract dimensions
        input_field = self.expr.to_column_field(plan)
        self.dimensions = input_field.data_type.dimensions
        # Return same EmbeddingType - normalization preserves the model
        return ColumnField(name=str(self), data_type=input_field.data_type)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class EmbeddingSimilarityExpr(ScalarFunction):
    """Expression for computing similarity between embedding vectors."""

    function_name = "embedding.compute_similarity"

    def __init__(self, expr: LogicalExpr, other: Union[LogicalExpr, np.ndarray], metric: SemanticSimilarityMetric):
        self.expr = expr
        self.other = other
        self.metric = metric

        # Pass appropriate arguments to signature validation
        if isinstance(other, LogicalExpr):
            super().__init__(expr, other)  # Both embedding inputs
        else:
            super().__init__(expr)  # Only main embedding input

    def __str__(self) -> str:
        if isinstance(self.other, LogicalExpr):
            return f"embedding.compute_similarity({self.expr}, {self.other}, metric={self.metric})"
        else:
            return f"embedding.compute_similarity({self.expr}, query_vector, metric={self.metric})"

    def _validate_query_vector_dimensions(self, plan: LogicalPlan):
        """Validate query vector dimensions match embedding dimensions."""
        if not isinstance(self.other, LogicalExpr):
            # Column vs query vector - check dimensions
            input_field = self.expr.to_column_field(plan)
            if input_field.data_type.dimensions != len(self.other):
                raise ValidationError(
                    f"Query vector dimensions ({len(self.other)}) must match embedding dimensions ({input_field.data_type.dimensions})"
                )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        # If comparison field is a query vector, validate its dimensions
        self._validate_query_vector_dimensions(plan)
        # Call parent to handle signature validation
        result = super().to_column_field(plan)
        return result

    def children(self) -> List[LogicalExpr]:
        if isinstance(self.other, LogicalExpr):
            return [self.expr, self.other]
        return [self.expr]
