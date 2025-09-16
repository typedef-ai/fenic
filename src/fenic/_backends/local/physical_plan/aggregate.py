from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic.core._logical_plan.plans import CacheInfo

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import (
    PhysicalPlan,
    _with_lineage_uuid,
)
from fenic.core.error import InternalError


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

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
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

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: AggregateExec expects 1 child")
        return AggregateExec(
            child=children[0],
            group_exprs=self.group_exprs,
            agg_exprs=self.agg_exprs,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0].build_node_lineage(leaf_nodes)
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
