from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.local.semantic_operators import Cluster
from fenic.core._logical_plan.plans import CacheInfo

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import (
    PhysicalPlan,
    _with_lineage_uuid,
)


class AggregateExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        group_exprs: List[pl.Expr],
        agg_exprs: List[pl.Expr],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.group_exprs = group_exprs
        self.agg_exprs = agg_exprs

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: AggregateExec expects 1 child")
        child_df = child_dfs[0]
        if not self.group_exprs:
            df = child_df.group_by(pl.lit(1).alias("_dummy_group_id")).agg(
                self.agg_exprs
            )
            df = df.drop("_dummy_group_id")
            return df
        else:
            return child_df.group_by(self.group_exprs).agg(self.agg_exprs)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0]._build_lineage(leaf_nodes)
        agg_exprs_with_uuid = self.agg_exprs + [
            pl.col("_uuid").alias("_backwards_uuid")
        ]

        if not self.group_exprs:
            materialize_df = (
                child_df.group_by(pl.lit(1).alias("_dummy_group_id"))
                .agg(agg_exprs_with_uuid)
                .drop("_dummy_group_id")
            )
        else:
            materialize_df = child_df.group_by(self.group_exprs).agg(
                agg_exprs_with_uuid
            )

        materialize_df = _with_lineage_uuid(materialize_df)
        backwards_df = materialize_df.select(["_uuid", "_backwards_uuid"]).explode(
            "_backwards_uuid"
        )
        materialize_df = materialize_df.drop("_backwards_uuid")

        operator = self._build_unary_operator_lineage(
            materialize_df=materialize_df,
            child=(child_operator, backwards_df),
        )
        return operator, materialize_df


class SemanticAggregateExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        group_expr: pl.Expr,
        group_expr_name: str,
        agg_exprs: List[pl.Expr],
        num_clusters: int,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.group_expr = group_expr
        self.group_expr_name = group_expr_name
        self.agg_exprs = agg_exprs
        self.num_clusters = num_clusters

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: SemanticAggregateExec expects 1 child")
        child_df = child_dfs[0]
        child_df = child_df.with_columns(self.group_expr.alias(self.group_expr_name))

        # TODO (DY): Determine what to use for number of iterations (niter)
        clustered_df = Cluster(
            child_df,
            self.group_expr_name,
            self.num_clusters,
            self.session_state.app_name,
        ).execute()
        df = clustered_df.group_by("_cluster_id").agg(self.agg_exprs)
        return df

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0]._build_lineage(leaf_nodes)
        child_df = child_df.with_columns(self.group_expr.alias(self.group_expr_name))
        clustered_df = Cluster(
            child_df,
            self.group_expr_name,
            self.num_clusters,
            self.session_state.app_name,
        ).execute()

        agg_exprs_with_uuid = self.agg_exprs + [
            pl.col("_uuid").alias("_backwards_uuid")
        ]
        materialize_df = clustered_df.group_by("_cluster_id").agg(agg_exprs_with_uuid)
        materialize_df = _with_lineage_uuid(materialize_df)
        backwards_df = materialize_df.select(["_uuid", "_backwards_uuid"]).explode(
            "_backwards_uuid"
        )
        materialize_df = materialize_df.drop("_backwards_uuid")

        operator = self._build_unary_operator_lineage(
            materialize_df=materialize_df,
            child=(child_operator, backwards_df),
        )
        return operator, materialize_df
