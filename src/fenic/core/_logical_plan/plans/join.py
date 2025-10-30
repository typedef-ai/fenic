from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from fenic._constants import LEFT_ON_KEY, RIGHT_ON_KEY
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    ColumnExpr,
    LogicalExpr,
)
from fenic.core._logical_plan.jinja_validation import VariableTree
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._logical_plan.utils import (
    validate_completion_parameters,
    validate_scalar_expr,
)
from fenic.core.error import InternalError, TypeMismatchError, ValidationError
from fenic.core.types import (
    ColumnField,
    DoubleType,
    EmbeddingType,
    JoinExampleCollection,
    Schema,
)
from fenic.core.types.enums import SemanticSimilarityMetric


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
        for left_on_expr in left_on:
            validate_scalar_expr(left_on_expr, "join")
        for right_on_expr in right_on:
            validate_scalar_expr(right_on_expr, "join")
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
        return Join(left, right, left_on, right_on, how, session_state)

    @classmethod
    def from_schema(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: List[LogicalExpr],
        right_on: List[LogicalExpr],
        how: str,
        schema: Schema) -> Join:
        return Join(left, right, left_on, right_on, how, schema=schema)

    def children(self) -> List[LogicalPlan]:
        return [self._left, self._right]

    def exprs(self) -> List[LogicalExpr]:
        return list(self._left_on) + list(self._right_on)

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

    def _eq_specific(self, other: Join) -> bool:
        return (
            self._left_on == other._left_on
            and self._right_on == other._right_on
            and self._how == other._how
        )


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
    def _validate_columns(self, session_state: BaseSessionState) -> None:
        pass

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        self._validate_columns(session_state)
        return Schema(
            self._left.schema().column_fields + self._right.schema().column_fields
        )

    def left_on(self) -> LogicalExpr:
        return self._left_on

    def right_on(self) -> LogicalExpr:
        return self._right_on

    def children(self) -> List[LogicalPlan]:
        return [self._left, self._right]

    def exprs(self) -> List[LogicalExpr]:
        return [self._left_on, self._right_on]

    @abstractmethod
    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        raise NotImplementedError("Subclasses must implement with_children")


class SemanticJoin(BaseSemanticJoin):
    def __init__(
        self,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: LogicalExpr,
        right_on: LogicalExpr,
        jinja_template: str,
        strict: bool,
        temperature: float = 0.0,
        model_alias: Optional[ResolvedModelAlias] = None,
        examples: Optional[JoinExampleCollection] = None,
        request_timeout: Optional[float] = None,
        session_state: Optional[BaseSessionState] = None,
        schema: Optional[Schema] = None,
    ):
        self._jinja_template = jinja_template
        self._strict = strict
        self._examples = examples
        self.temperature = temperature
        self.model_alias = model_alias
        self.request_timeout = request_timeout
        validate_scalar_expr(left_on, "semantic.join")
        validate_scalar_expr(right_on, "semantic.join")
        super().__init__(left, right, left_on, right_on, session_state, schema)
        if session_state:
            validate_completion_parameters(model_alias, session_state.session_config, temperature)

    @classmethod
    def from_session_state(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: LogicalExpr,
        right_on: LogicalExpr,
        jinja_template: str,
        strict: bool,
        temperature: float = 0.0,
        model_alias: Optional[ResolvedModelAlias] = None,
        examples: Optional[JoinExampleCollection] = None,
        request_timeout: Optional[float] = None,
        session_state: BaseSessionState = None) -> SemanticJoin:
        return SemanticJoin(left,
                right,
                left_on,
                right_on,
                jinja_template,
                strict,
                temperature,
                model_alias,
                examples,
                request_timeout,
                session_state)

    @classmethod
    def from_schema(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: LogicalExpr,
        right_on: LogicalExpr,
        jinja_template: str,
        strict: bool,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
        examples: Optional[JoinExampleCollection] = None,
        request_timeout: Optional[float] = None,
        schema: Optional[Schema] = None) -> SemanticJoin:
        return SemanticJoin(left,
                right,
                left_on,
                right_on,
                jinja_template,
                strict,
                temperature,
                model_alias,
                examples,
                request_timeout,
                schema=schema)


    def _validate_columns(self, session_state: BaseSessionState) -> None:
        variable_tree = VariableTree.from_jinja_template(self._jinja_template)
        variables = variable_tree.variables
        if set(variables) != {"left_on", "right_on"}:
            raise ValidationError(
                "The `predicate` argument to `semantic.join` must contain exactly the variables 'left_on' and 'right_on'. "
                f"Got: {list(variables)}"
            )

        left_on_dtype = self._left_on.to_column_field(self._left, session_state).data_type
        right_on_dtype = self._right_on.to_column_field(self._right, session_state).data_type
        if self._examples:
            self._examples._validate_against_join_types(left_on_dtype, right_on_dtype)
        variable_tree.validate_jinja_variable(LEFT_ON_KEY, left_on_dtype)
        variable_tree.validate_jinja_variable(RIGHT_ON_KEY, right_on_dtype)

    def jinja_template(self) -> str:
        return self._jinja_template

    def strict(self) -> bool:
        return self._strict

    def examples(self) -> Optional[JoinExampleCollection]:
        return self._examples

    def _repr(self) -> str:
        return (
            f"SemanticJoin(left_on={self._left_on}, "
            f"right_on={self._right_on}, jinja_template={self._jinja_template})"
        )

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 2:
            raise InternalError(f"SemanticJoin expects 2 children, got {len(children)}")

        result = SemanticJoin.from_session_state(
            left=children[0],
            right=children[1],
            left_on=self._left_on,
            right_on=self._right_on,
            jinja_template=self._jinja_template,
            strict=self._strict,
            examples=self._examples,
            temperature=self.temperature,
            model_alias=self.model_alias,
            request_timeout=self.request_timeout,
            session_state=session_state,
        )
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: SemanticJoin) -> bool:
        return (
            self._left_on == other._left_on
            and self._right_on == other._right_on
            and self._jinja_template == other._jinja_template
            and self._strict == other._strict
            and self._examples == other._examples
            and self.temperature == other.temperature
            and self.model_alias == other.model_alias
            and self.request_timeout == other.request_timeout
        )


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
        request_timeout: Optional[float] = None,
        session_state: Optional[BaseSessionState] = None,
        schema: Optional[Schema] = None,
    ):
        self._k = k
        self._similarity_metric = similarity_metric
        self._similarity_score_column = similarity_score_column
        self.request_timeout = request_timeout
        validate_scalar_expr(left_on, "semantic.sim_join")
        validate_scalar_expr(right_on, "semantic.sim_join")
        super().__init__(left, right, left_on, right_on, session_state, schema)

    @classmethod
    def from_session_state(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: LogicalExpr,
        right_on: LogicalExpr,
        k: int,
        similarity_metric: SemanticSimilarityMetric,
        similarity_score_column: Optional[str] = None,
        request_timeout: Optional[float] = None,
        session_state: BaseSessionState = None) -> SemanticSimilarityJoin:
        return SemanticSimilarityJoin(left,
                right,
                left_on,
                right_on,
                k,
                similarity_metric,
                similarity_score_column,
                request_timeout,
                session_state)

    @classmethod
    def from_schema(cls,
        left: LogicalPlan,
        right: LogicalPlan,
        left_on: LogicalExpr,
        right_on: LogicalExpr,
        k: int,
        similarity_metric: SemanticSimilarityMetric,
        similarity_score_column: Optional[str] = None,
            request_timeout: Optional[float] = None,
        schema: Optional[Schema] = None) -> SemanticSimilarityJoin:
        return SemanticSimilarityJoin(left,
                right,
                left_on,
                right_on,
                k,
                similarity_metric,
                similarity_score_column,
                request_timeout,
                schema=schema)

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
            raise InternalError(
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
            request_timeout=self.request_timeout,
            session_state=session_state,
        )
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: SemanticSimilarityJoin) -> bool:
        return (
            self._left_on == other._left_on
            and self._right_on == other._right_on
            and self._k == other._k
            and self._similarity_metric == other._similarity_metric
            and self._similarity_score_column == other._similarity_score_column
            and self.request_timeout == other.request_timeout
        )
