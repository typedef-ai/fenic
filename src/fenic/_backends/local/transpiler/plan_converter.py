from __future__ import annotations

from typing import TYPE_CHECKING

from fenic._backends.local.physical_plan import (
    AggregateExec,
    CacheReadExec,
    DocSourceExec,
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
from fenic._backends.local.physical_plan.optimizer import (
    MergeDuckDBNodesRule,
    PhysicalPlanOptimizer,
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
    DocSource,
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
from fenic.core.error import InternalError


class PlanConverter:
    def __init__(self, session_state: LocalSessionState):
        self.session_state = session_state
        self.expr_converter = ExprConverter(session_state)
        self.logical_optimizer = LogicalPlanOptimizer(session_state, [
            NotFilterPushdownRule(),
            MergeFiltersRule(),
            SemanticFilterRewriteRule(),
        ])
        self.physical_optimizer = PhysicalPlanOptimizer(
            session_state,
            [MergeDuckDBNodesRule()]
        )

    def _optimize_physical_plan(self, physical_plan: PhysicalPlan) -> PhysicalPlan:
        """Apply physical plan optimizations."""
        result = self.physical_optimizer.optimize(physical_plan)
        return result.plan

    def convert(
        self,
        logical: LogicalPlan,
    ) -> PhysicalPlan:
        """Convert logical plan to optimized physical plan."""

        # Apply logical optimizations
        # Note the order of the rules is important here.
        # NotFilterPushdownRule() and MergeFiltersRule() can be applied
        # in any order, but both must be applied before SemanticFilterRewriteRule()
        # for SemanticFilterRewriteRule() to produce optimal plans.
        logical = (
            self.logical_optimizer
            .optimize(logical)
            .plan
        )

        # Convert to physical plan
        physical = self._convert_to_physical(logical)

        # Apply physical optimizations (always enabled)
        optimized_result = self.physical_optimizer.optimize(physical)

        return optimized_result.plan

    def _convert_to_physical(
        self,
        logical: LogicalPlan,
    ) -> PhysicalPlan:
        if logical.cache_info and self.session_state.db_client.intermediate.is_df_cached(logical.cache_info.cache_key):
            return CacheReadExec(
                cache_key=logical.cache_info.cache_key,
                session_state=self.session_state,
            )

        if isinstance(logical, Projection):
            child_physical = self._convert_to_physical(
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
            child_physical = self._convert_to_physical(
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
                self._convert_to_physical(child)
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
        elif isinstance(logical, DocSource):
            return DocSourceExec(
                paths=logical._paths,
                valid_file_extension=logical._valid_file_extension,
                exclude=logical._exclude,
                recursive=logical._recursive,
                session_state=self.session_state,
            )
        elif isinstance(logical, Limit):
            child_physical = self._convert_to_physical(
                logical.children()[0]
            )
            return LimitExec(
                child_physical,
                logical.n,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Aggregate):
            child_physical = self._convert_to_physical(
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

            left_physical = self._convert_to_physical(
                left_logical
            )
            right_physical = self._convert_to_physical(
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
            left_physical = self._convert_to_physical(
                logical.children()[0]
            )
            right_physical = self._convert_to_physical(
                logical.children()[1]
            )

            return SemanticJoinExec(
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
                logical.jinja_template(),
                logical.strict(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
                examples=logical.examples(),
                temperature=logical.temperature,
                model_alias=logical.model_alias
            )

        elif isinstance(logical, SemanticSimilarityJoin):
            left_physical = self._convert_to_physical(
                logical.children()[0]
            )
            right_physical = self._convert_to_physical(
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
                similarity_score_column=logical.similarity_score_column(),
            )

        elif isinstance(logical, SemanticCluster):
            child_physical = self._convert_to_physical(
                logical.children()[0]
            )
            physical_by_expr = self.expr_converter.convert(
                logical.by_expr()
            )
            return SemanticClusterExec(
                child_physical,
                physical_by_expr,
                str(logical.by_expr()),
                num_clusters=logical.num_clusters(),
                max_iter=logical.max_iter(),
                num_init=logical.num_init(),
                label_column=logical.label_column(),
                centroid_info=logical.centroid_info(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, Explode):
            child_logical = logical.children()[0]
            physical_expr = self.expr_converter.convert(
                logical._expr
            )
            child_physical = self._convert_to_physical(
                child_logical
            )
            target_field = logical._expr.to_column_field(child_logical, self.session_state)
            return ExplodeExec(
                child_physical,
                physical_expr,
                target_field.name,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, DropDuplicates):
            child_logical = logical.children()[0]
            child_physical = self._convert_to_physical(
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
            child_physical = self._convert_to_physical(
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
            child_physical = self._convert_to_physical(
                child_logical
            )
            return UnnestExec(
                child_physical,
                logical.col_names(),
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        elif isinstance(logical, FileSink):
            child_physical = self._convert_to_physical(
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
            child_physical = self._convert_to_physical(
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
                template_name_to_plan={template_name: self._convert_to_physical(plan) for template_name, plan in logical.template_name_to_plan.items()},
                templated_query=logical.templated_query,
                cache_info=logical.cache_info,
                session_state=self.session_state,
            )

        else:
            raise InternalError(f"Unsupported logical plan type: {type(logical)}")
