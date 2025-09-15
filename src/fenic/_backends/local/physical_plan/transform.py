from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from typing import Union as TypeUnion

import polars as pl
from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic._backends.local.semantic_operators.cluster import Cluster
from fenic.core._logical_plan.plans import CacheInfo, CentroidInfo
from fenic.core.error import InternalError
from fenic.core._utils.misc import replace_sql_query_placeholders, generate_unique_arrow_view_name

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import (
    PhysicalPlan,
    _with_lineage_uuid,
    DuckDBNodeMixin,
)
from fenic._backends.local.physical_plan.sink import DuckDBTableSinkExec


logger = logging.getLogger(__name__)

class ProjectionExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        projections: List[pl.Expr],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.projections = projections

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: ProjectionExec expects 1 child")
        return child_dfs[0].select(self.projections)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0]._build_lineage(leaf_nodes)

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

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: ProjectionExec expects 1 child")
        return ProjectionExec(
            child=children[0],
            projections=self.projections,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: FilterExec expects 1 child")
        return child_dfs[0].filter(self.predicate)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: FilterExec expects 1 child")
        return FilterExec(
            child=children[0],
            predicate=self.predicate,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

class UnionExec(PhysicalPlan):
    def __init__(
        self,
        children: List[PhysicalPlan],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__(children, cache_info=cache_info, session_state=session_state)

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 2:
            raise ValueError("Unreachable: UnionExec expects exactly two children")

        left_df = child_dfs[0]
        right_df = child_dfs[1]

        # Align right dataframe columns with left dataframe
        right_df_aligned = right_df.select(left_df.columns)
        combined = pl.concat([left_df, right_df_aligned], how="vertical")
        return combined

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        if len(self.children) != 2:
            raise ValueError("Unreachable: UnionExec expects exactly two children")

        left_operator, left_df = self.children[0]._build_lineage(leaf_nodes)
        right_operator, right_df = self.children[1]._build_lineage(leaf_nodes)

        new_uuids = [uuid.uuid4().hex for _ in range(left_df.height + right_df.height)]

        left_df = left_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid"),
            pl.Series("_uuid", new_uuids[: left_df.height]),
        )
        right_df = right_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid"),
            pl.Series("_uuid", new_uuids[left_df.height :]),
        )

        materialize_df = self._execute([left_df, right_df])

        left_backwards = left_df.select(["_uuid", "_backwards_uuid"])
        right_backwards = right_df.select(["_uuid", "_backwards_uuid"])
        materialize_df = materialize_df.drop("_backwards_uuid")

        operator = self._build_binary_operator_lineage(
            materialize_df=materialize_df,
            left_child=(left_operator, left_backwards),
            right_child=(right_operator, right_backwards),
        )
        return operator, materialize_df

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 2:
            raise InternalError("Unreachable: UnionExec expects exactly two children")
        return UnionExec(
            children=children,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

class ExplodeExec(PhysicalPlan):
    def __init__(
        self,
        child: PhysicalPlan,
        physical_expr: pl.Expr,
        col_name: str,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__([child], cache_info=cache_info, session_state=session_state)
        self.physical_expr = physical_expr
        self.col_name = col_name

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: ExplodeExec expects 1 child")
        child_df = child_dfs[0]
        child_df = child_df.with_columns(self.physical_expr)
        exploded_df = child_df.explode(self.col_name)
        # Optionally filter out rows where the exploded column is null.
        return exploded_df.filter(pl.col(self.col_name).is_not_null())

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        child_operator, child_df = self.children[0]._build_lineage(leaf_nodes)
        exploded_df = child_df.explode(self.col_name)
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

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: ExplodeExec expects 1 child")
        return ExplodeExec(
            child=children[0],
            physical_expr=self.physical_expr,
            col_name=self.col_name,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: LimitExec expects 1 child")

        df = child_dfs[0]
        if self.n > 0:
            return df.limit(self.n)
        else:
            return pl.DataFrame(schema=df.schema)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: LimitExec expects 1 child")
        return LimitExec(
            child=children[0],
            n=self.n,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )


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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: DropDuplicates expects 1 child")

        df = child_dfs[0]

        current_subset = None
        if len(self.subset) > 0:
            current_subset = self.subset

        return df.unique(subset=current_subset)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: DropDuplicatesExec expects 1 child")
        return DropDuplicatesExec(
            child=children[0],
            subset=self.subset,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )


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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: Sort expects 1 child")

        df = child_dfs[0]

        return df.sort(
            self.cols, descending=self.descending, nulls_last=self.nulls_last
        )

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)

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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise ValueError("Unreachable: UnnestExec expects 1 child")
        return child_dfs[0].unnest(self.col_names)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)

class SQLExec(PhysicalPlan):
    def __init__(
        self,
        template_name_to_plan: Dict[str, PhysicalPlan],
        templated_query: str,
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__(list(template_name_to_plan.values()), cache_info=cache_info, session_state=session_state)
        self.template_name_to_plan = template_name_to_plan
        self.templated_query = templated_query

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        resolved_query, template_name_to_view_name = replace_sql_query_placeholders(self.templated_query, list(self.template_name_to_plan.keys()))
        view_name_to_df = zip(template_name_to_view_name.values(), child_dfs, strict=True)
        cursor = self.session_state.db_client.cursor()
        for view_name, child_df in view_name_to_df:
            cursor.register(view_name, child_df)
        try:
            arrow_result = cursor.execute(resolved_query).arrow()
            return apply_ingestion_coercions(pl.from_arrow(arrow_result))
        finally:
            for arrow_view_name in self.arrow_view_names:
                try:
                    cursor.execute(f"DROP VIEW IF EXISTS {arrow_view_name}")
                except Exception:
                    logger.error(f"Failed to drop view: {arrow_view_name}")
                    pass

    def get_sql_query(self, view_names: List[str]) -> str:
        return replace_sql_query_placeholders(
            self.templated_query,
            self.template_name_to_plan.keys(),
            view_names,
        )[0]

    def _build_lineage(
        self,
        _leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        # Lineage can work with SQLExec, but the traversal API needs to support more than two children.
        # Currently, when traversing the plan backwards, the API only allows traversing left or right children.
        raise NotImplementedError("Lineage not supported for SQLExec")

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != len(self.template_name_to_plan):
            raise InternalError("Unreachable: SQLExec expects 1 child")
        return SQLExec(
            template_name_to_plan=self.template_name_to_plan,
            templated_query=self.templated_query,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
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

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        return self._build_row_subset_lineage(leaf_nodes)

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

class MergedDuckDBExec(PhysicalPlan, DuckDBNodeMixin):
    def __init__(
        self,
        merge_root: PhysicalPlan,
        children: List[PhysicalPlan],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        super().__init__(children, cache_info=cache_info, session_state=session_state)
        self.merge_root = merge_root

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        """
        Execute the merged DuckDB plan.

        Note: child_dfs contains the DataFrame results from executing all leaf nodes
        in the subtree rooted at merge_root, in the same order they would be
        encountered during a depth-first traversal. This ordering guarantee allows
        us to consume DataFrames sequentially as we traverse the tree.
        """
        cursor = self.session_state.db_client.cursor()
        created_views = []
        df_index = 0

        def create_view_for_node(node: PhysicalPlan) -> str:
            nonlocal df_index

            # If it's not a DuckDB node, register the DataFrame as a view
            if not isinstance(node, DuckDBNodeMixin):
                if df_index >= len(child_dfs):
                    raise InternalError("Ran out of DataFrames while processing nodes")

                view_name = generate_unique_arrow_view_name()
                cursor.register(view_name, child_dfs[df_index])
                df_index += 1
                created_views.append(view_name)
                return view_name

            # If it's a DuckDB node with no children, create view from its SQL
            if len(node.children) == 0:
                view_name = generate_unique_arrow_view_name()
                sql = node.get_sql_query([])
                cursor.execute(f"CREATE TEMPORARY VIEW {view_name} AS SELECT * FROM {sql}")
                created_views.append(view_name)
                return view_name

            # If it's a DuckDB node with children, process children first
            child_views = []
            for child in node.children:
                child_view = create_view_for_node(child)
                child_views.append(child_view)

            # Create view for this node using its children's views
            view_name = generate_unique_arrow_view_name()
            sql = node.get_sql_query(child_views)
            cursor.execute(f"CREATE TEMPORARY VIEW {view_name} AS {sql}")
            created_views.append(view_name)
            return view_name

        try:
            # Special handling for table sink operations
            if isinstance(self.merge_root, DuckDBTableSinkExec):
                input_view = create_view_for_node(self.merge_root.children[0])
                ddl = self.merge_root.get_sql_query([input_view])
                cursor.execute(ddl)
                return pl.DataFrame()  # Table operations return empty DataFrame

            # Normal query execution
            root_view = create_view_for_node(self.merge_root)
            arrow_result = cursor.execute(f"SELECT * FROM {root_view}").arrow()
            return apply_ingestion_coercions(pl.from_arrow(arrow_result))

        finally:
            # Clean up all temporary views
            for view in created_views:
                cursor.execute(f"DROP VIEW IF EXISTS {view}")

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        pass

    def get_sql_query(self, view_names: List[str]) -> str:
        raise InternalError("MergedDuckDBExec does not support get_sql_query")

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != len(self.children):
            raise InternalError("Inconsistent number of children for MergedDuckDBExec")
        return MergedDuckDBExec(
            merge_root=self.merge_root,
            children=children,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )
