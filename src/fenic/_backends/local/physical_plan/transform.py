from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple
from typing import Union as TypeUnion

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic._backends.local.semantic_operators.cluster import Cluster
from fenic.core._logical_plan.plans import CacheInfo, CentroidInfo
from fenic.core.error import ExecutionError, InternalError

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import (
    PhysicalPlan,
    _with_lineage_uuid,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SeriesLiteralCheck:
    column_name: str
    expected_length: int


class ProjectionExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        projections: List[pl.Expr],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
        series_literal_checks: Optional[List[SeriesLiteralCheck]] = None,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.projections = projections
        self.series_literal_checks = series_literal_checks or []

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: ProjectionExec expects 1 child")
        child_df = child_dfs[0]
        if self.series_literal_checks:
            df_height = child_df.height
            for check in self.series_literal_checks:
                if check.expected_length != df_height:
                    raise ExecutionError(
                        f"Column '{check.column_name}' was created from a Series of length "
                        f"{check.expected_length}, but the DataFrame has {df_height} rows. "
                        "Series data must match the DataFrame height."
                    )
        return child_df.select(self.projections)


    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: ProjectionExec expects 1 child")
        return ProjectionExec(
            child=children[0],
            projections=self.projections,
            cache_info=self.cache_info,
            session_state=self.session_state,
            series_literal_checks=self.series_literal_checks,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0].build_node_lineage(leaf_nodes)

        materialize_df = child_df.select([*self.projections, pl.col("_uuid")])

        backwards_df = materialize_df.select(["_uuid"])
        backwards_df = backwards_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid")
        )

        operator = self._build_unary_operator_lineage(
            materialize_df=materialize_df,
            child=(child_operator, backwards_df),
        )
        return operator, materialize_df


class FilterExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        predicate: pl.Expr,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.predicate = predicate

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: FilterExec expects 1 child")
        return child_dfs[0].filter(self.predicate)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: FilterExec expects 1 child")
        return FilterExec(
            child=children[0],
            predicate=self.predicate,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)


class UnionExec(PhysicalPlan):
    def __init__(
        self,
        children: List[PhysicalPlan],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__(children, cache_info=cache_info, session_state=session_state)

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 2:
            raise ValueError("Unreachable: UnionExec expects exactly two children")

        left_df = child_dfs[0]
        right_df = child_dfs[1]

        # Align right dataframe columns with left dataframe
        right_df_aligned = right_df.select(left_df.columns)
        combined = pl.concat([left_df, right_df_aligned], how="vertical")
        return combined

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 2:
            raise InternalError("Unreachable: UnionExec expects exactly two children")
        return UnionExec(
            children=children,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        if len(self.children) != 2:
            raise ValueError("Unreachable: UnionExec expects exactly two children")

        left_operator, left_df = self.children[0].build_node_lineage(leaf_nodes)
        right_operator, right_df = self.children[1].build_node_lineage(leaf_nodes)

        new_uuids = [uuid.uuid4().hex for _ in range(left_df.height + right_df.height)]

        left_df = left_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid"),
            pl.Series("_uuid", new_uuids[: left_df.height]),
        )
        right_df = right_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid"),
            pl.Series("_uuid", new_uuids[left_df.height :]),
        )

        materialize_df = self.execute_node([left_df, right_df])

        left_backwards = left_df.select(["_uuid", "_backwards_uuid"])
        right_backwards = right_df.select(["_uuid", "_backwards_uuid"])
        materialize_df = materialize_df.drop("_backwards_uuid")

        operator = self._build_binary_operator_lineage(
            materialize_df=materialize_df,
            left_child=(left_operator, left_backwards),
            right_child=(right_operator, right_backwards),
        )
        return operator, materialize_df


class ExplodeExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        physical_expr: pl.Expr,
        col_name: str,
        keep_null_and_empty: bool,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.physical_expr = physical_expr
        self.col_name = col_name
        self.keep_null_and_empty = keep_null_and_empty

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: ExplodeExec expects 1 child")
        child_df = child_dfs[0]
        child_df = child_df.with_columns(self.physical_expr)
        exploded_df = child_df.explode(self.col_name)
        # Filter out nulls unless keep_null_and_empty is True
        if not self.keep_null_and_empty:
            exploded_df = exploded_df.filter(pl.col(self.col_name).is_not_null())
        return exploded_df

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: ExplodeExec expects 1 child")
        return ExplodeExec(
            child=children[0],
            physical_expr=self.physical_expr,
            col_name=self.col_name,
            keep_null_and_empty=self.keep_null_and_empty,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0].build_node_lineage(leaf_nodes)
        exploded_df = child_df.explode(self.col_name)
        # Filter out nulls unless keep_null_and_empty is True
        if not self.keep_null_and_empty:
            exploded_df = exploded_df.filter(pl.col(self.col_name).is_not_null())
        exploded_df = exploded_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid"),
        )
        exploded_df = _with_lineage_uuid(exploded_df)
        backwards_df = exploded_df.select(["_uuid", "_backwards_uuid"])

        materialize_df = exploded_df.drop("_backwards_uuid")

        operator = self._build_unary_operator_lineage(
            materialize_df=materialize_df,
            child=(child_operator, backwards_df),
        )

        return operator, materialize_df


class ExplodeWithIndexExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        physical_expr: pl.Expr,
        col_name: str,
        index_name: str,
        value_name: str,
        keep_null_and_empty: bool,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.physical_expr = physical_expr
        self.col_name = col_name
        self.index_name = index_name
        self.value_name = value_name
        self.keep_null_and_empty = keep_null_and_empty

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: ExplodeWithIndexExec expects 1 child")
        child_df = child_dfs[0]

        # Add the array column if it's an expression
        child_df = child_df.with_columns(self.physical_expr)

        # Add a temporary row index to track original rows
        child_df = child_df.with_row_index("__explode_row_id")

        # Explode the array column
        exploded_df = child_df.explode(self.col_name)

        if self.keep_null_and_empty:
            # For outer explode, we need to handle null/empty arrays specially
            # Add the position column, but set it to null for null/empty arrays
            exploded_df = exploded_df.with_columns(
                pl.when(pl.col(self.col_name).is_not_null())
                .then(pl.int_range(pl.len(), dtype=pl.Int64).over("__explode_row_id", mapping_strategy="group_to_rows"))
                .otherwise(None)
                .alias(self.index_name)
            )
        else:
            # Filter out nulls in the exploded column for regular explode
            exploded_df = exploded_df.filter(pl.col(self.col_name).is_not_null())
            # Add the position column (0-based index within each original row)
            exploded_df = exploded_df.with_columns(
                pl.int_range(pl.len(), dtype=pl.Int64)
                .over("__explode_row_id", mapping_strategy="group_to_rows")
                .alias(self.index_name)
            )

        # Drop the temporary row index
        exploded_df = exploded_df.drop("__explode_row_id")

        # Rename the exploded column if needed
        if self.value_name != self.col_name:
            exploded_df = exploded_df.rename({self.col_name: self.value_name})

        return exploded_df

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: ExplodeWithIndexExec expects 1 child")
        return ExplodeWithIndexExec(
            child=children[0],
            physical_expr=self.physical_expr,
            col_name=self.col_name,
            index_name=self.index_name,
            value_name=self.value_name,
            keep_null_and_empty=self.keep_null_and_empty,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0].build_node_lineage(leaf_nodes)
        exploded_df = child_df.explode(self.col_name)
        # Filter out nulls unless keep_null_and_empty is True
        if not self.keep_null_and_empty:
            exploded_df = exploded_df.filter(pl.col(self.col_name).is_not_null())
        exploded_df = exploded_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid"),
        )
        exploded_df = _with_lineage_uuid(exploded_df)
        backwards_df = exploded_df.select(["_uuid", "_backwards_uuid"])

        materialize_df = exploded_df.drop("_backwards_uuid")

        operator = self._build_unary_operator_lineage(
            materialize_df=materialize_df,
            child=(child_operator, backwards_df),
        )
        return operator, materialize_df


class LimitExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        n: int,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.n = n

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: LimitExec expects 1 child")

        df = child_dfs[0]
        if self.n > 0:
            return df.limit(self.n)
        else:
            return pl.DataFrame(schema=df.schema)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: LimitExec expects 1 child")
        return LimitExec(
            child=children[0],
            n=self.n,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)


class DropDuplicatesExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        subset: List[str],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.subset = subset

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: DropDuplicates expects 1 child")

        df = child_dfs[0]

        current_subset = None
        if len(self.subset) > 0:
            current_subset = self.subset

        return df.unique(subset=current_subset)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: DropDuplicatesExec expects 1 child")
        return DropDuplicatesExec(
            child=children[0],
            subset=self.subset,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)


class SortExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        cols: List[pl.Expr],
        descending: TypeUnion[bool, List[bool]],
        nulls_last: List[bool],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.cols = cols
        self.descending = descending
        self.nulls_last = nulls_last

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: Sort expects 1 child")

        df = child_dfs[0]

        return df.sort(
            self.cols, descending=self.descending, nulls_last=self.nulls_last
        )

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: SortExec expects 1 child")
        return SortExec(
            child=children[0],
            cols=self.cols,
            descending=self.descending,
            nulls_last=self.nulls_last,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)


class UnnestExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        col_names: List[str],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.col_names = col_names

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: UnnestExec expects 1 child")
        return child_dfs[0].unnest(self.col_names)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: UnnestExec expects 1 child")
        return UnnestExec(
            child=children[0],
            col_names=self.col_names,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)

class SQLExec(PhysicalPlan):
    def __init__(
        self,
        children: List[PhysicalPlan],
        query: str,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
        view_names: List[str],
    ):
        super().__init__(children, cache_info=cache_info, session_state=session_state)
        if len(children) != len(view_names):
            raise InternalError("Unreachable: SQLExec expects 1 child")
        self.query = query
        self.view_names = view_names

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        cursor = self.session_state.intermediate_df_client.db_conn.cursor()
        for child_df, view_name in zip(child_dfs, self.view_names, strict=False):
            cursor.register(view_name, child_df)
        try:
            pl_result = cursor.execute(self.query).pl()
            return apply_ingestion_coercions(pl_result, coerce_array=True)
        finally:
            for view_name in self.view_names:
                try:
                    cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
                except Exception:
                    logger.error(f"Failed to drop view: {view_name}")
                    pass

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != len(self.children):
            raise InternalError("Unreachable: SQLExec expects 1 child")
        return SQLExec(
            children=children,
            query=self.query,
            cache_info=self.cache_info,
            session_state=self.session_state,
            view_names=self.view_names,
        )

    def build_node_lineage(
        self,
        _leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        # Lineage can work with SQLExec, but the traversal API needs to support more than two children.
        # Currently, when traversing the plan backwards, the API only allows traversing left or right children.
        raise NotImplementedError("Lineage not supported for SQLExec")

class SemanticClusterExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        by_expr: pl.Expr,
        by_expr_name: str,
        num_clusters: int,
        max_iter: int,
        num_init: int,
        label_column: str,
        centroid_info: Optional[CentroidInfo],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.by_expr = by_expr
        self.by_expr_name = by_expr_name
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.num_init = num_init
        self.label_column = label_column
        self.centroid_info = centroid_info

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: SemanticClusterExec expects 1 child")
        child_df = child_dfs[0]
        child_df = child_df.with_columns(self.by_expr.alias(self.by_expr_name))

        # Perform clustering and add cluster metadata columns
        clustered_df = Cluster(
            child_df,
            self.by_expr_name,
            num_clusters=self.num_clusters,
            max_iter=self.max_iter,
            num_init=self.num_init,
            label_column=self.label_column,
            centroid_info=self.centroid_info,
        ).execute()

        # Remove the temporary column we added for clustering if it wasn't in the original
        if self.by_expr_name not in child_dfs[0].columns:
            clustered_df = clustered_df.drop(self.by_expr_name)

        return clustered_df

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: SemanticClusterExec expects 1 child")
        return SemanticClusterExec(
            child=children[0],
            by_expr=self.by_expr,
            by_expr_name=self.by_expr_name,
            num_clusters=self.num_clusters,
            max_iter=self.max_iter,
            num_init=self.num_init,
            label_column=self.label_column,
            centroid_info=self.centroid_info,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)


class MergedDuckDBExec(PhysicalPlan):
    def __init__(
        self,
        merge_root: PhysicalPlan,
        children: List[PhysicalPlan],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__(children, cache_info=cache_info, session_state=session_state)
        self.merge_root = merge_root

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        """
        Execute the merged DuckDB plan.
        Note: child_dfs contains the DataFrame results from executing all leaf nodes
        in the subtree rooted at merge_root, in the same order they would be
        encountered during a depth-first traversal. This ordering guarantee allows
        us to consume DataFrames sequentially as we traverse the tree.
        """
        pass

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != len(self.children):
            raise InternalError("Inconsistent number of children for MergedDuckDBExec")
        return MergedDuckDBExec(
            merge_root=self.merge_root,
            children=children,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        raise NotImplementedError("Lineage not supported for SQLExec")
