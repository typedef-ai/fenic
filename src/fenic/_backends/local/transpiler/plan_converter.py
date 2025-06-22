from __future__ import annotations

from typing import TYPE_CHECKING

from fenic._backends.local.physical_plan import (
    AggregateExec,
    DropDuplicatesExec,
    DuckDBTableSinkExec,
    DuckDBTableSourceExec,
    ExplodeExec,
    FileSinkExec,
    FileSourceExec,
    FilterExec,
    InMemorySourceExec,
    JoinExec,
    LimitExec,
    PhysicalPlan,
    ProjectionExec,
    SemanticClusterExec,
    SemanticJoinExec,
    SemanticSimilarityJoinExec,
    SortExec,
    SQLExec,
    UnionExec,
    UnnestExec,
)
from fenic.core._logical_plan.expressions import (
    ColumnExpr,
)
from fenic.core._logical_plan.optimizer import (
    LogicalPlanOptimizer,
    MergeFiltersRule,
    NotFilterPushdownRule,
    SemanticFilterRewriteRule,
)
from fenic.core._logical_plan.plans import (
    SQL,
    Aggregate,
    DropDuplicates,
    Explode,
    FileSink,
    FileSource,
    Filter,
    InMemorySource,
    Join,
    Limit,
    LogicalPlan,
    Projection,
    SemanticCluster,
    SemanticJoin,
    SemanticSimilarityJoin,
    Sort,
    TableSink,
    TableSource,
    Union,
    Unnest,
)

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.transpiler.expr_converter import (
    ExprConverter,
)


class PlanConverter:
    def __init__(self, session_state: LocalSessionState):
        self.session_state = session_state
        self.expr_converter = ExprConverter(session_state)

    def convert(
        self,
        logical: LogicalPlan,
    ) -> PhysicalPlan:
        # Note the order of the rules is important here.
        # NotFilterPushdownRule() and MergeFiltersRule() can be applied
        # in any order, but both must be applied before SemanticFilterRewriteRule()
        # for SemanticFilterRewriteRule() to produce optimal plans.
        logical = (
            LogicalPlanOptimizer(
                [NotFilterPushdownRule(), MergeFiltersRule(), SemanticFilterRewriteRule()]
            )
            .optimize(logical)
            .plan
        )
        if isinstance(logical, Projection):
            child_physical = self.convert(
                logical.children()[0]
            )
            physical_exprs = [
                self.expr_converter.convert(log_expr)
                for log_expr in logical.exprs()
            ]
            return ProjectionExec(
                child_physical,
                physical_exprs,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Filter):
            child_physical = self.convert(
                logical.children()[0]
            )
            physical_expr = self.expr_converter.convert(
                logical.predicate()
            )

            return FilterExec(
                child_physical,
                physical_expr,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Union):
            children_physical = [
                self.convert(child)
                for child in logical.children()
            ]
            return UnionExec(
                children_physical,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, FileSource):
            return FileSourceExec(
                paths=logical._paths,
                file_format=logical._file_format,
                session_state=self.session_state,
                options=logical._options,
            )
        elif isinstance(logical, InMemorySource):
            return InMemorySourceExec(
                df=logical._source,
                session_state=self.session_state,
            )
        elif isinstance(logical, TableSource):
            return DuckDBTableSourceExec(
                table_name=logical._table_name,
                session_state=self.session_state,
            )
        elif isinstance(logical, Limit):
            child_physical = self.convert(
                logical.children()[0]
            )
            return LimitExec(
                child_physical,
                logical.n,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Aggregate):
            child_physical = self.convert(
                logical.children()[0]
            )
            physical_group_exprs = [
                self.expr_converter.convert(log_expr)
                for log_expr in logical.group_exprs()
            ]
            physical_agg_exprs = [
                self.expr_converter.convert(log_expr)
                for log_expr in logical.agg_exprs()
            ]
            return AggregateExec(
                child_physical,
                physical_group_exprs,
                physical_agg_exprs,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Join):
            left_logical = logical.children()[0]
            right_logical = logical.children()[1]

            left_physical = self.convert(
                left_logical
            )
            right_physical = self.convert(
                right_logical
            )
            left_on_exprs = [
                self.expr_converter.convert(log_expr, with_alias=False)
                for log_expr in logical.left_on()
            ]
            right_on_exprs = [
                self.expr_converter.convert(log_expr, with_alias=False)
                for log_expr in logical.right_on()
            ]
            return JoinExec(
                left_physical,
                right_physical,
                left_on_exprs,
                right_on_exprs,
                logical.how(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, SemanticJoin):
            left_physical = self.convert(
                logical.children()[0]
            )
            right_physical = self.convert(
                logical.children()[1]
            )

            return SemanticJoinExec(
                left_physical,
                right_physical,
                logical.left_on().name,
                logical.right_on().name,
                logical.join_instruction(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
                examples=logical.examples(),
                temperature=logical.temperature,
                model_alias=logical.model_alias
            )

        elif isinstance(logical, SemanticSimilarityJoin):
            left_physical = self.convert(
                logical.children()[0]
            )
            right_physical = self.convert(
                logical.children()[1]
            )
            return SemanticSimilarityJoinExec(
                left_physical,
                right_physical,
                (
                    logical.left_on().name
                    if isinstance(logical.left_on(), ColumnExpr)
                    else self.expr_converter.convert(
                        logical.left_on()
                    )
                ),
                (
                    logical.right_on().name
                    if isinstance(logical.right_on(), ColumnExpr)
                    else self.expr_converter.convert(
                        logical.right_on()
                    )
                ),
                logical.k(),
                logical.similarity_metric(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
                return_similarity_scores=logical.return_similarity_scores(),
            )

        elif isinstance(logical, SemanticCluster):
            child_physical = self.convert(
                logical.children()[0]
            )
            physical_group_expr = self.expr_converter.convert(
                logical.by_expr()
            )
            return SemanticClusterExec(
                child_physical,
                physical_group_expr,
                str(logical.by_expr()),
                logical.num_clusters(),
                logical.centroid_dimensions(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Explode):
            child_logical = logical.children()[0]
            physical_expr = self.expr_converter.convert(
                logical._expr
            )
            child_physical = self.convert(
                child_logical
            )
            target_field = logical._expr.to_column_field(child_logical)
            return ExplodeExec(
                child_physical,
                physical_expr,
                target_field.name,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, DropDuplicates):
            child_logical = logical.children()[0]
            child_physical = self.convert(
                child_logical
            )

            return DropDuplicatesExec(
                child_physical,
                logical._subset(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Sort):
            child_logical = logical.children()[0]
            child_physical = self.convert(
                child_logical
            )

            descending_list = []
            physical_col_exprs = []
            nulls_last_list = []

            for sort_expr in logical.sort_exprs():
                # sort dataframe op will convert all columns to SortExprs
                # read the sort orders and nulls_last info from the sort_expr
                # and convert the underlying column expression to a physical expression
                descending_list.append(not sort_expr.ascending)
                nulls_last_list.append(sort_expr.nulls_last)
                physical_col_exprs.append(
                    self.expr_converter.convert(
                        sort_expr.column_expr()
                    )
                )

            return SortExec(
                child_physical,
                physical_col_exprs,
                descending_list,
                nulls_last_list,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Unnest):
            child_logical = logical.children()[0]
            child_physical = self.convert(
                child_logical
            )
            return UnnestExec(
                child_physical,
                logical.col_names(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, FileSink):
            child_physical = self.convert(
                logical.child
            )
            return FileSinkExec(
                child=child_physical,
                path=logical.path,
                file_type=logical.sink_type,
                mode=logical.mode,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, TableSink):
            child_physical = self.convert(
                logical.child
            )
            return DuckDBTableSinkExec(
                child=child_physical,
                table_name=logical.table_name,
                mode=logical.mode,
                cache_info=logical.cache_info,
                session_state=self.session_state,
                schema=logical.schema(),
            )

        elif isinstance(logical, SQL):
            return SQLExec(
                children=[self.convert(child) for child in logical.children()],
                query=logical.resolved_query,
                cache_info=logical.cache_info,
                session_state=self.session_state,
                arrow_view_names=logical.view_names,
            )
