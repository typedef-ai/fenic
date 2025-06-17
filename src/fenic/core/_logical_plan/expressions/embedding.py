"""Embedding operation expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types import (
    ColumnField,
    EmbeddingType,
    FloatType,
    SemanticSimilarityMetric,
)


class EmbeddingNormalizeExpr(LogicalExpr):
    """Expression for normalizing embedding vectors to unit length."""

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.dimensions = None

    def __str__(self) -> str:
        return f"embedding.normalize({self.expr})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if not isinstance(input_field.data_type, EmbeddingType):
            raise TypeMismatchError(
                EmbeddingType, input_field.data_type, "embedding.normalize()"
            )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        input_field = self.expr.to_column_field(plan)
        self.dimensions = input_field.data_type.dimensions
        # Return same EmbeddingType - normalization preserves the model
        return ColumnField(name=str(self), data_type=input_field.data_type)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class EmbeddingSimilarityExpr(LogicalExpr):
    """Expression for computing similarity between embedding vectors."""

    def __init__(self, expr: LogicalExpr, other: Union[LogicalExpr, np.ndarray], metric: SemanticSimilarityMetric):
        self.expr = expr
        self.other = other
        self.metric = metric

    def __str__(self) -> str:
        if isinstance(self.other, LogicalExpr):
            return f"embedding.compute_similarity({self.expr}, {self.other}, metric={self.metric})"
        else:
            return f"embedding.compute_similarity({self.expr}, query_vector, metric={self.metric})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if not isinstance(input_field.data_type, EmbeddingType):
            raise TypeMismatchError(
                EmbeddingType, input_field.data_type, "embedding.compute_similarity()"
            )

        if isinstance(self.other, LogicalExpr):
            # Column vs column - must have same EmbeddingType
            other_field = self.other.to_column_field(plan)
            if not isinstance(other_field.data_type, EmbeddingType):
                raise TypeMismatchError(
                    EmbeddingType, other_field.data_type, "embedding.compute_similarity() second argument"
                )
            if input_field.data_type != other_field.data_type:
                raise TypeMismatchError.from_message(
                    f"Embedding types must match for embedding.compute_similarity(): {input_field.data_type} != {other_field.data_type}."
                )
        else:
            # Column vs query vector - just check dimensions
            if input_field.data_type.dimensions != len(self.other):
                raise ValidationError(
                    f"Query vector dimensions ({len(self.other)}) must match embedding dimensions ({input_field.data_type.dimensions})"
                )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=FloatType)

    def children(self) -> List[LogicalExpr]:
        if isinstance(self.other, LogicalExpr):
            return [self.expr, self.other]
        return [self.expr]
