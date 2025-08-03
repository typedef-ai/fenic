"""Embedding operation expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import numpy as np

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    UnparameterizedExpr,
    ValidatedSignature,
)
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core.error import ValidationError
from fenic.core.types import (
    ColumnField,
    SemanticSimilarityMetric,
)


class EmbeddingNormalizeExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression for normalizing embedding vectors to unit length."""

    function_name = "embedding.normalize"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.dimensions = None
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        # Use validator to handle signature validation (which ensures EmbeddingType)
        super().to_column_field(plan, session_state)
        # Get the actual input field to extract dimensions
        input_field = self.expr.to_column_field(plan, session_state)
        self.dimensions = input_field.data_type.dimensions
        # Return same EmbeddingType - normalization preserves the model
        return ColumnField(name=str(self), data_type=input_field.data_type)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class EmbeddingSimilarityExpr(ValidatedSignature, LogicalExpr):
    """Expression for computing similarity between embedding vectors."""

    function_name = "embedding.compute_similarity"

    def __init__(self, expr: LogicalExpr, other: Union[LogicalExpr, np.ndarray], metric: SemanticSimilarityMetric):
        self.expr = expr
        self.other = other
        self.metric = metric
        self._validator = SignatureValidator(self.function_name)
        if isinstance(self.other, LogicalExpr):
            self._children =  [self.expr, self.other]
        else:
            self._children = [self.expr]

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def __str__(self) -> str:
        if isinstance(self.other, LogicalExpr):
            return f"embedding.compute_similarity({self.expr}, {self.other}, metric={self.metric})"
        else:
            return f"embedding.compute_similarity({self.expr}, query_vector, metric={self.metric})"

    def _validate_query_vector_dimensions(self, plan: LogicalPlan, session_state: BaseSessionState):
        """Validate query vector dimensions match embedding dimensions."""
        if not isinstance(self.other, LogicalExpr):
            # Column vs query vector - check dimensions
            input_field = self.expr.to_column_field(plan, session_state)
            if input_field.data_type.dimensions != len(self.other):
                raise ValidationError(
                    f"Query vector dimensions ({len(self.other)}) must match embedding dimensions ({input_field.data_type.dimensions})"
                )

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        # If comparison field is a query vector, validate its dimensions
        self._validate_query_vector_dimensions(plan, session_state)
        # Use mixin's validation by calling super()
        return super().to_column_field(plan, session_state)

    def children(self) -> List[LogicalExpr]:
        return self._children

    def _eq_specific(self, other: EmbeddingSimilarityExpr) -> bool:
        # Check metric (always needs to be compared)
        if self.metric != other.metric:
            return False

        # Check the type of self.other vs other.other
        if isinstance(self.other, LogicalExpr) != isinstance(other.other, LogicalExpr):
            return False

        # If both are numpy arrays, compare them
        if isinstance(self.other, np.ndarray):
            # Both are numpy arrays (we know from check above)
            return np.array_equal(self.other, other.other)

        # Both are LogicalExpr - will be compared via children
        return True
