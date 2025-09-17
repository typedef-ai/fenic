from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.local.semantic_operators import Join as SemanticJoin
from fenic._backends.local.semantic_operators import SimJoin as SemanticSimJoin
from fenic._backends.local.semantic_operators.sim_join import (
    DISTANCE_COL_NAME,
    LEFT_ON_COL_NAME,
    RIGHT_ON_COL_NAME,
)
from fenic._constants import LEFT_ON_KEY, RIGHT_ON_KEY
from fenic.core._logical_plan.plans import CacheInfo
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core.error import InternalError
from fenic.core.types import JoinExampleCollection
from fenic.core.types.enums import JoinType, SemanticSimilarityMetric

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import (
    PhysicalPlan,
    _with_lineage_uuid,
)
from fenic._backends.local.physical_plan.utils import (
    normalize_column_before_join,
    restore_column_after_join,
)

logger = logging.getLogger(__name__)


class JoinExec(PhysicalPlan):
    def __init__(
        self,
        left: PhysicalPlan,
        right: PhysicalPlan,
        left_on_exprs: List[pl.Expr],
        right_on_exprs: List[pl.Expr],
        how: JoinType,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__(
            [left, right], cache_info=cache_info, session_state=session_state
        )
        self.left_on_exprs = left_on_exprs
        self.right_on_exprs = right_on_exprs
        self.how = how

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 2:
            raise ValueError("Unreachable: JoinExec expects 2 children")
        left_df = child_dfs[0]
        right_df = child_dfs[1]

        # Set join keys based on join type
        left_on = self.left_on_exprs if self.how != "cross" else None
        right_on = self.right_on_exprs if self.how != "cross" else None

        return left_df.join(
            other=right_df,
            left_on=left_on,
            right_on=right_on,
            how=self.how
        )

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 2:
            raise InternalError("Unreachable: JoinExec expects 2 children")
        return JoinExec(
            left=children[0],
            right=children[1],
            left_on_exprs=self.left_on_exprs,
            right_on_exprs=self.right_on_exprs,
            how=self.how,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        left_operator, left_df = self.children[0].build_node_lineage(leaf_nodes)
        right_operator, right_df = self.children[1].build_node_lineage(leaf_nodes)

        left_df = left_df.rename({"_uuid": "_left_uuid"})
        right_df = right_df.rename({"_uuid": "_right_uuid"})

        joined_df = self.execute_node([left_df, right_df])
        materialize_df = _with_lineage_uuid(joined_df)
        backwards_df_left = materialize_df.select(["_uuid", "_left_uuid"]).rename(
            {"_left_uuid": "_backwards_uuid"}
        )
        backwards_df_right = materialize_df.select(["_uuid", "_right_uuid"]).rename(
            {"_right_uuid": "_backwards_uuid"}
        )

        materialize_df = materialize_df.drop(["_left_uuid", "_right_uuid"])
        operator = self._build_binary_operator_lineage(
            materialize_df=materialize_df,
            left_child=(left_operator, backwards_df_left),
            right_child=(right_operator, backwards_df_right),
        )
        return operator, materialize_df


class SemanticJoinExec(PhysicalPlan):
    def __init__(
        self,
        left: PhysicalPlan,
        right: PhysicalPlan,
        left_on: Union[str, pl.Expr],
        right_on: Union[str, pl.Expr],
        jinja_template: str,
        strict: bool,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
        model_alias: Optional[ResolvedModelAlias] = None,
        temperature = 0.0,
        examples: Optional[JoinExampleCollection] = None,
    ):
        super().__init__(
            [left, right], cache_info=cache_info, session_state=session_state
        )
        self.examples = examples
        self.jinja_template = jinja_template
        self.strict = strict
        self.left_on = left_on
        self.right_on = right_on
        self.temperature = temperature
        self.model_alias = model_alias

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 2:
            raise ValueError("Unreachable: SemanticJoinExec expects 2 children")

        left_df, right_df = child_dfs

        # Normalize both join sides to standard column names
        left_df, maybe_left_name = normalize_column_before_join(
            left_df, self.left_on, LEFT_ON_KEY
        )
        right_df, maybe_right_name = normalize_column_before_join(
            right_df, self.right_on, RIGHT_ON_KEY
        )

        result = SemanticJoin(
            left_df,
            right_df,
            self.jinja_template,
            self.strict,
            self.session_state.get_language_model(self.model_alias),
            examples=self.examples,
            temperature=self.temperature,
            model_alias=self.model_alias,
        ).execute()

        # Restore original column names or drop temporary columns
        result = restore_column_after_join(
            result, maybe_left_name, LEFT_ON_KEY
        )
        result = restore_column_after_join(
            result, maybe_right_name, RIGHT_ON_KEY
        )

        return result


    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 2:
            raise InternalError("Unreachable: SemanticJoinExec expects 2 children")
        return SemanticJoinExec(
            left=children[0],
            right=children[1],
            left_on=self.left_on,
            right_on=self.right_on,
            jinja_template=self.jinja_template,
            strict=self.strict,
            cache_info=self.cache_info,
            session_state=self.session_state,
            model_alias=self.model_alias,
            temperature=self.temperature,
            examples=self.examples,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        left_operator, left_df = self.children[0].build_node_lineage(leaf_nodes)
        right_operator, right_df = self.children[1].build_node_lineage(leaf_nodes)

        left_df = left_df.rename({"_uuid": "_left_uuid"})
        right_df = right_df.rename({"_uuid": "_right_uuid"})

        joined_df = self.execute_node([left_df, right_df])

        materialize_df = _with_lineage_uuid(joined_df)
        backwards_df_left = materialize_df.select(["_uuid", "_left_uuid"]).rename(
            {"_left_uuid": "_backwards_uuid"}
        )
        backwards_df_right = materialize_df.select(["_uuid", "_right_uuid"]).rename(
            {"_right_uuid": "_backwards_uuid"}
        )

        materialize_df = materialize_df.drop(["_left_uuid", "_right_uuid"])
        operator = self._build_binary_operator_lineage(
            materialize_df=materialize_df,
            left_child=(left_operator, backwards_df_left),
            right_child=(right_operator, backwards_df_right),
        )

        return operator, materialize_df


class SemanticSimilarityJoinExec(PhysicalPlan):
    def __init__(
        self,
        left: PhysicalPlan,
        right: PhysicalPlan,
        left_on: Union[str, pl.Expr],
        right_on: Union[str, pl.Expr],
        k: int,
        similarity_metric: SemanticSimilarityMetric,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
        similarity_score_column: Optional[str] = None,
    ):
        super().__init__(
            [left, right], cache_info=cache_info, session_state=session_state
        )
        self.left_on = left_on
        self.right_on = right_on
        self.k = k
        self.similarity_metric = similarity_metric
        self.similarity_score_column = similarity_score_column

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 2:
            raise ValueError(
                "Unreachable: SemanticSimilarityJoinExec expects 2 children"
            )

        left_df, right_df = child_dfs

        # Normalize both join sides to standard column names
        left_df, maybe_left_name = normalize_column_before_join(
            left_df, self.left_on, LEFT_ON_COL_NAME
        )
        right_df, maybe_right_name = normalize_column_before_join(
            right_df, self.right_on, RIGHT_ON_COL_NAME
        )

        # TODO(rohitrastogi): Avoid regenerating embeddings if semantic index already exists
        result = SemanticSimJoin(left_df, right_df, self.k, self.similarity_metric).execute()

        if self.similarity_score_column:
            result = result.rename({DISTANCE_COL_NAME: self.similarity_score_column})
        else:
            result = result.drop(DISTANCE_COL_NAME)

        # Restore original column names or drop temporary columns
        result = restore_column_after_join(
            result, maybe_left_name, LEFT_ON_COL_NAME
        )
        result = restore_column_after_join(
            result, maybe_right_name, RIGHT_ON_COL_NAME
        )

        return result

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 2:
            raise InternalError("Unreachable: SemanticSimilarityJoinExec expects 2 children")
        return SemanticSimilarityJoinExec(
            left=children[0],
            right=children[1],
            left_on=self.left_on,
            right_on=self.right_on,
            k=self.k,
            similarity_metric=self.similarity_metric,
            cache_info=self.cache_info,
            session_state=self.session_state,
            similarity_score_column=self.similarity_score_column,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        left_operator, left_df = self.children[0].build_node_lineage(leaf_nodes)
        right_operator, right_df = self.children[1].build_node_lineage(leaf_nodes)

        left_df = left_df.rename({"_uuid": "_left_uuid"})
        right_df = right_df.rename({"_uuid": "_right_uuid"})

        joined_df = self.execute_node([left_df, right_df])

        materialize_df = _with_lineage_uuid(joined_df)
        backwards_df_left = materialize_df.select(["_uuid", "_left_uuid"]).rename(
            {"_left_uuid": "_backwards_uuid"}
        )
        backwards_df_right = materialize_df.select(["_uuid", "_right_uuid"]).rename(
            {"_right_uuid": "_backwards_uuid"}
        )

        materialize_df = materialize_df.drop(["_left_uuid", "_right_uuid"])
        operator = self._build_binary_operator_lineage(
            materialize_df=materialize_df,
            left_child=(left_operator, backwards_df_left),
            right_child=(right_operator, backwards_df_right),
        )
        return operator, materialize_df
