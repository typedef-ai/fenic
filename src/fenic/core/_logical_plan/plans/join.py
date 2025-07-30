from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    ColumnExpr,
    LogicalExpr,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.utils import validate_completion_parameters
from fenic.core.error import TypeMismatchError
from fenic.core.types import (
    ColumnField,
    DoubleType,
    EmbeddingType,
    JoinExampleCollection,
    Schema,
    StringType,
)
from fenic.core.types.enums import SemanticSimilarityMetric

SIMILARITY_SCORE_COL_NAME = "_similarity_score"


class Join(LogicalPlan):
    def __init__(
            self,
            left: LogicalPlan,
            right: LogicalPlan,
            left_on: List[LogicalExpr],
            right_on: List[LogicalExpr],
            how: str,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._left = left
        self._right = right
        self._left_on = left_on
        self._right_on = right_on
        self._how = how
        super().__init__(session_state, schema)

    @classmethod
    def from_session_state(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: List[LogicalExpr],
        right_on: List[LogicalExpr],
        how: str,
        session_state: BaseSessionState) -> Join:
        return Join(left, right, left_on, right_on, how, session_state, None)

    @classmethod
    def from_schema(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: List[LogicalExpr],
        right_on: List[LogicalExpr],
        how: str,
        schema: Schema) -> Join:
        return Join(left, right, left_on, right_on, how, None, schema)

    def children(self) -> List[LogicalPlan]:
        return [self._left, self._right]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        for left_on, right_on in zip(self._left_on, self._right_on, strict=False):
            left_type = left_on.to_column_field(self._left, session_state).data_type
            right_type = right_on.to_column_field(self._right, session_state).data_type
            if left_type != right_type:
                raise TypeMismatchError.from_message(
                    f"Join condition: Cannot compare '{left_on.name}' ({left_type}) with "
                    f"'{right_on.name}' ({right_type}). Join on equality comparison requires the same type "
                    f"on both sides."
                )

        # Handle cross and outer joins - include all columns from both DataFrames
        if self._how == "cross" or self._how == "full":
            return Schema(self._left.schema().column_fields + self._right.schema().column_fields)

        if self._how == "right":
            primary_df = self._right
            secondary_df = self._left
            primary_join_cols = self._right_on
            secondary_join_cols = self._left_on
        else:  # left, inner
            primary_df = self._left
            secondary_df = self._right
            primary_join_cols = self._left_on
            secondary_join_cols = self._right_on

        # Get regular column names (ignoring derived columns)
        primary_column_join_names = [col.name for col in primary_join_cols if isinstance(col, ColumnExpr)]
        secondary_column_join_names = [col.name for col in secondary_join_cols if isinstance(col, ColumnExpr)]

        # Start with all columns from primary DataFrame
        primary_fields = primary_df.schema().column_fields

        # Add columns from secondary DataFrame, excluding regular join columns that appear in both
        secondary_fields_to_add = []
        for field in secondary_df.schema().column_fields:
            # Skip if this is a regular column join that appears in both primary and secondary
            if field.name in primary_column_join_names and field.name in secondary_column_join_names:
                continue
            secondary_fields_to_add.append(field)

        # Combine based on join type
        if self._how == "right":
            primary_fields = secondary_fields_to_add + primary_fields
        else:
            primary_fields = primary_fields + secondary_fields_to_add

        return Schema(primary_fields)

    def _repr(self) -> str:
        return f"Join(how={self._how}, left_on={', '.join(str(expr) for expr in self._left_on)}, right_on={', '.join(str(expr) for expr in self._right_on)})"

    def left_on(self) -> List[LogicalExpr]:
        return self._left_on

    def right_on(self) -> List[LogicalExpr]:
        return self._right_on

    def how(self) -> str:
        return self._how

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 2:
            raise ValueError("Join must have exactly two children")
        result = Join.from_session_state(children[0], children[1], self._left_on, self._right_on, self._how, session_state)
        result.set_cache_info(self.cache_info)
        return result


class BaseSemanticJoin(LogicalPlan, ABC):
    def __init__(
            self,
            left: LogicalPlan,
            right: LogicalPlan,
            left_on: LogicalExpr,
            right_on: LogicalExpr,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._left = left
        self._right = right
        self._left_on = left_on
        self._right_on = right_on
        super().__init__(session_state, schema)

    @abstractmethod
    def _validate_columns(self) -> None:
        pass

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        self._validate_columns()
        return Schema(
            self._left.schema().column_fields + self._right.schema().column_fields
        )

    def left_on(self) -> LogicalExpr:
        return self._left_on

    def right_on(self) -> LogicalExpr:
        return self._right_on

    def children(self) -> List[LogicalPlan]:
        return [self._left, self._right]

    @abstractmethod
    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        raise NotImplementedError("Subclasses must implement with_children")


class SemanticJoin(BaseSemanticJoin):
    def __init__(
        self,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: ColumnExpr,
        right_on: ColumnExpr,
        join_instruction: str,
        temperature: float = 0.0,
        model_alias: Optional[str] = None,
        examples: Optional[JoinExampleCollection] = None,
        session_state: Optional[BaseSessionState] = None,
        schema: Optional[Schema] = None,
    ):
        self._join_instruction = join_instruction
        self._examples = examples
        self.temperature = temperature
        self.model_alias = model_alias
        super().__init__(left, right, left_on, right_on, session_state, schema)
        if session_state:
            validate_completion_parameters(model_alias, session_state.session_config, temperature)

    @classmethod
    def from_session_state(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: List[LogicalExpr],
        right_on: List[LogicalExpr],
        join_instruction: str,
        temperature: float = 0.0,
        model_alias: Optional[str] = None,
        examples: Optional[JoinExampleCollection] = None,
        session_state: BaseSessionState = None) -> Join:
        return SemanticJoin(left,
                right,
                left_on,
                right_on,
                join_instruction,
                temperature,
                model_alias,
                examples,
                session_state,
                None)

    @classmethod
    def from_schema(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: List[LogicalExpr],
        right_on: List[LogicalExpr],
        join_instruction: str,
        temperature: float,
        model_alias: Optional[str] = None,
        examples: Optional[JoinExampleCollection] = None,
        schema: Optional[Schema] = None) -> Join:
        return SemanticJoin(left,
                right,
                left_on,
                right_on,
                join_instruction,
                temperature,
                model_alias,
                examples,
                None,
                Schema)


    def _validate_columns(self) -> None:
        left_schema = self._left.schema()
        right_schema = self._right.schema()
        left_columns = {field.name: field for field in left_schema.column_fields}
        right_columns = {field.name: field for field in right_schema.column_fields}

        if self._left_on.name not in left_columns:
            raise ValueError(
                f"Column '{self._left_on.name}' not found in left DataFrame. "
                f"Available columns: {', '.join(sorted(left_columns.keys()))}"
            )
        if self._right_on.name not in right_columns:
            raise ValueError(
                f"Column '{self._right_on.name}' not found in right DataFrame. "
                f"Available columns: {', '.join(sorted(right_columns.keys()))}"
            )
        if left_columns[self._left_on.name].data_type != StringType:
            raise TypeMismatchError(
                left_columns[self._left_on.name].data_type,
                StringType,
                f"Cannot apply semantic.join on non-string column '{self._left_on.name}' (left side)",
            )
        if right_columns[self._right_on.name].data_type != StringType:
            raise TypeMismatchError(
                right_columns[self._right_on.name].data_type,
                StringType,
                f"Cannot apply semantic.join on non-string column '{self._right_on.name}' (right side)",
            )

    def join_instruction(self) -> str:
        return self._join_instruction

    def examples(self) -> Optional[JoinExampleCollection]:
        return self._examples

    def _repr(self) -> str:
        return (
            f"SemanticJoin(left_on={self._left_on.name}, "
            f"right_on={self._right_on.name}, join_instruction={self._join_instruction})"
        )

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 2:
            raise ValueError(f"SemanticJoin expects 2 children, got {len(children)}")

        result = SemanticJoin.from_session_state(
            left=children[0],
            right=children[1],
            left_on=self._left_on,
            right_on=self._right_on,
            join_instruction=self._join_instruction,
            examples=self._examples,
            temperature=self.temperature,
            model_alias=self.model_alias,
            session_state=session_state,
        )
        result.set_cache_info(self.cache_info)
        return result


class SemanticSimilarityJoin(BaseSemanticJoin):
    def __init__(
        self,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: LogicalExpr,
        right_on: LogicalExpr,
        k: int,
        similarity_metric: SemanticSimilarityMetric,
        similarity_score_column: Optional[str] = None,
        session_state: Optional[BaseSessionState] = None,
        schema: Optional[Schema] = None,
    ):
        self._k = k
        self._similarity_metric = similarity_metric
        self._similarity_score_column = similarity_score_column
        super().__init__(left, right, left_on, right_on, session_state, schema)

    @classmethod
    def from_session_state(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: List[LogicalExpr],
        right_on: List[LogicalExpr],
        k: int,
        similarity_metric: SemanticSimilarityMetric,
        similarity_score_column: Optional[str] = None,
        session_state: BaseSessionState = None) -> Join:
        return SemanticSimilarityJoin(left,
                right,
                left_on,
                right_on,
                k,
                similarity_metric,
                similarity_score_column,
                session_state,
                None)

    @classmethod
    def from_schema(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: List[LogicalExpr],
        right_on: List[LogicalExpr],
        k: int,
        similarity_metric: SemanticSimilarityMetric,
        similarity_score_column: Optional[str] = None,
        schema: Optional[Schema] = None) -> Join:
        return SemanticSimilarityJoin(left,
                right,
                left_on,
                right_on,
                k,
                similarity_metric,
                similarity_score_column,
                None,
                schema)

    def _validate_columns(self, session_state: BaseSessionState) -> None:
        left_dtype = self._left_on.to_column_field(self._left, session_state).data_type
        if not isinstance(left_dtype, EmbeddingType):
            raise TypeMismatchError.from_message(
                f"Cannot apply semantic.sim_join on non embeddings type left join key column '{self._left_on.name}' with type {left_dtype}",
            )
        right_dtype = self._right_on.to_column_field(self._right, session_state).data_type
        if not left_dtype == right_dtype:
            raise TypeMismatchError.from_message(
                f"Cannot apply semantic.sim_join with mismatched types: left column '{self._left_on.name}' has type {left_dtype}, right column '{self._right_on.name}' has type {right_dtype}",
            )

    def k(self) -> int:
        return self._k

    def similarity_metric(self) -> SemanticSimilarityMetric:
        return self._similarity_metric

    def similarity_score_column(self) -> Optional[str]:
        return self._similarity_score_column

    def _repr(self) -> str:
        return (
            f"SemanticSimilarityJoin(left_on={self._left_on.name}, "
            f"right_on={self._right_on.name}, k={self._k}, "
        )

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        self._validate_columns(session_state)
        # add scores field if requested by user.
        additional_fields = []
        if self._similarity_score_column:
            additional_fields.append(
                ColumnField(
                    name=self._similarity_score_column,
                    data_type=DoubleType,
                )
            )
        return Schema(
            self._left.schema().column_fields
            + self._right.schema().column_fields
            + additional_fields
        )

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 2:
            raise ValueError(
                f"SemanticSimilarityJoin expects 2 children, got {len(children)}"
            )

        result = SemanticSimilarityJoin.from_session_state(
            left=children[0],
            right=children[1],
            left_on=self._left_on,
            right_on=self._right_on,
            k=self._k,
            similarity_metric=self._similarity_metric,
            similarity_score_column=self._similarity_score_column,
            session_state=session_state,
        )
        result.set_cache_info(self.cache_info)
        return result
