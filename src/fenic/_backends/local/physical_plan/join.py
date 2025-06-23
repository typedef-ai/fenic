from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.local.semantic_operators import Join as SemanticJoin
from fenic._backends.local.semantic_operators import SimJoin as SemanticSimJoin
from fenic._backends.local.semantic_operators.sim_join import (
    DISTANCE_COL_NAME,
    LEFT_ON_COL_NAME,
    RIGHT_ON_COL_NAME,
)
from fenic.core._logical_plan.plans import CacheInfo
from fenic.core.types import JoinExampleCollection
from fenic.core.types.enums import JoinType, SemanticSimilarityMetric

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import (
    PhysicalPlan,
    _with_lineage_uuid,
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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
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

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        left_operator, left_df = self.children[0]._build_lineage(leaf_nodes)
        right_operator, right_df = self.children[1]._build_lineage(leaf_nodes)

        left_df = left_df.rename({"_uuid": "_left_uuid"})
        right_df = right_df.rename({"_uuid": "_right_uuid"})

        joined_df = self._execute([left_df, right_df])
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
        left_on_name: str,
        right_on_name: str,
        join_instruction: str,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
        model_alias: str,
        temperature = 0.0,
        examples: Optional[JoinExampleCollection] = None,
    ):
        super().__init__(
            [left, right], cache_info=cache_info, session_state=session_state
        )
        self.examples = examples
        self.join_instruction = join_instruction
        self.left_on_name = left_on_name
        self.right_on_name = right_on_name
        self.temperature = temperature
        self.model_alias = model_alias

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 2:
            raise ValueError("Unreachable: SemanticJoinExec expects 2 children")

        left_df = child_dfs[0]
        right_df = child_dfs[1]
        return SemanticJoin(
            left_df,
            right_df,
            self.left_on_name,
            self.right_on_name,
            self.join_instruction,
            self.session_state.get_language_model(self.model_alias),
            examples=self.examples,
            temperature=self.temperature,
        ).execute()

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        left_operator, left_df = self.children[0]._build_lineage(leaf_nodes)
        right_operator, right_df = self.children[1]._build_lineage(leaf_nodes)

        left_df = left_df.rename({"_uuid": "_left_uuid"})
        right_df = right_df.rename({"_uuid": "_right_uuid"})

        joined_df = self._execute([left_df, right_df])

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
        left_on: str | pl.Expr,
        right_on: str | pl.Expr,
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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 2:
            raise ValueError(
                "Unreachable: SemanticSimilarityJoinExec expects 2 children"
            )

        left_df, right_df = child_dfs

        # Normalize both join sides to standard column names
        left_df, left_was_expr, left_orig_name = self._normalize_column(
            left_df, self.left_on, LEFT_ON_COL_NAME
        )
        right_df, right_was_expr, right_orig_name = self._normalize_column(
            right_df, self.right_on, RIGHT_ON_COL_NAME
        )

        # TODO(rohitrastogi): Avoid regenerating embeddings if semantic index already exists
        result = SemanticSimJoin(left_df, right_df, self.k, self.similarity_metric).execute()

        if self.similarity_score_column:
            result = result.rename({DISTANCE_COL_NAME: self.similarity_score_column})
        else:
            result = result.drop(DISTANCE_COL_NAME)

        # Restore original column names or drop temporary columns
        result = self._restore_column(
            result, left_was_expr, left_orig_name, LEFT_ON_COL_NAME
        )
        result = self._restore_column(
            result, right_was_expr, right_orig_name, RIGHT_ON_COL_NAME
        )

        return result

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        left_operator, left_df = self.children[0]._build_lineage(leaf_nodes)
        right_operator, right_df = self.children[1]._build_lineage(leaf_nodes)

        left_df = left_df.rename({"_uuid": "_left_uuid"})
        right_df = right_df.rename({"_uuid": "_right_uuid"})

        joined_df = self._execute([left_df, right_df])

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

    @staticmethod
    def _normalize_column(
        df: pl.DataFrame, col: str | pl.Expr, alias: str
    ) -> tuple[pl.DataFrame, bool, str]:
        """Normalize a column (by expression or name) to a given alias.

        Returns:
            - New DataFrame
            - Whether it was an expression (i.e., needs to be dropped later)
            - Original name if it was a string
        """
        if isinstance(col, pl.Expr):
            return df.with_columns(col.alias(alias)), True, None
        else:
            return df.rename({col: alias}), False, col

    @staticmethod
    def _restore_column(
        df: pl.DataFrame, was_expr: bool, original_name: Optional[str], alias: str
    ) -> pl.DataFrame:
        """Restore a column to its original name or drop a temporary column.

        Args:
            df: DataFrame to restore
            was_expr: Whether the column was an expression (i.e., needs to be dropped)
            original_name: Original name of the column if it was a string
            alias: Alias of the column
        """
        if was_expr:
            return df.drop(alias)
        else:
            return df.rename({alias: original_name})
