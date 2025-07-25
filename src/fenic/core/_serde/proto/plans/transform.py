"""Transform plan serialization/deserialization."""


from fenic.core._logical_plan.plans.transform import (
    SQL,
    DropDuplicates,
    Explode,
    Filter,
    Limit,
    Projection,
    SemanticCluster,
    Sort,
    Union,
    Unnest,
)
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    serialize_logical_plan,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    DropDuplicatesProto,
    ExplodeProto,
    FilterProto,
    LimitProto,
    LogicalPlanProto,
    ProjectionProto,
    SemanticClusterProto,
    SortProto,
    SQLProto,
    UnionProto,
    UnnestProto,
)

# =============================================================================
# Projection
# =============================================================================


@serialize_logical_plan.register
def _serialize_projection(
    projection: Projection, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a projection."""
    proto = ProjectionProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, projection._input),
        exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, projection._exprs),
        schema=context.serialize_fenic_schema(projection.schema()),
    )
    return LogicalPlanProto(projection=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_projection(projection: ProjectionProto, context: SerdeContext) -> Projection:
    """Deserialize a Projection LogicalPlan Node."""
    return Projection.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, projection.input),
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, projection.exprs),
        schema=context.deserialize_fenic_schema(projection.schema),
    )


# =============================================================================
# Filter
# =============================================================================


@serialize_logical_plan.register
def _serialize_filter(
    filter_plan: Filter, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a filter."""
    proto = FilterProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, filter_plan._input),
        predicate=context.serialize_logical_expr(SerdeContext.EXPR, filter_plan._predicate),
        schema=context.serialize_fenic_schema(filter_plan.schema()),
    )
    return LogicalPlanProto(filter=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_filter(filter_proto: FilterProto, context: SerdeContext) -> Filter:
    """Deserialize a Filter LogicalPlan Node."""
    return Filter.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, filter_proto.input),
        predicate=context.deserialize_logical_expr(SerdeContext.EXPR, filter_proto.predicate),
        schema=context.deserialize_fenic_schema(filter_proto.schema),
    )


# =============================================================================
# Union
# =============================================================================


@serialize_logical_plan.register
def _serialize_union(union: Union, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a union."""
    proto = UnionProto(
        inputs=context.serialize_logical_plan_list(SerdeContext.INPUTS, union._inputs),
        schema=context.serialize_fenic_schema(union.schema()),
    )
    return LogicalPlanProto(union=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_union(union: UnionProto, context: SerdeContext) -> Union:
    """Deserialize a Union LogicalPlan Node."""
    return Union.from_schema(
        inputs=context.deserialize_logical_plan_list(SerdeContext.INPUTS, union.inputs),
        schema=context.deserialize_fenic_schema(union.schema),
    )


# =============================================================================
# Limit
# =============================================================================


@serialize_logical_plan.register
def _serialize_limit(limit: Limit, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a limit."""
    proto = LimitProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, limit._input),
        n=limit.n,
        schema=context.serialize_fenic_schema(limit.schema()),
    )
    return LogicalPlanProto(limit=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_limit(limit: LimitProto, context: SerdeContext) -> Limit:
    """Deserialize a Limit LogicalPlan Node."""
    return Limit.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, limit.input),
        n=limit.n,
        schema=context.deserialize_fenic_schema(limit.schema),
    )


# =============================================================================
# Explode
# =============================================================================


@serialize_logical_plan.register
def _serialize_explode(
    explode: Explode, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize an explode."""
    proto = ExplodeProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, explode._input),
        expr=context.serialize_logical_expr(SerdeContext.EXPR, explode._expr),
        schema=context.serialize_fenic_schema(explode.schema()),
    )
    return LogicalPlanProto(explode=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_explode(explode: ExplodeProto, context: SerdeContext) -> Explode:
    """Deserialize an Explode LogicalPlan Node."""
    return Explode.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, explode.input),
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, explode.expr),
        schema=context.deserialize_fenic_schema(explode.schema),
    )


# =============================================================================
# DropDuplicates
# =============================================================================


@serialize_logical_plan.register
def _serialize_drop_duplicates(
    drop_duplicates: DropDuplicates, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a drop duplicates."""
    proto = DropDuplicatesProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, drop_duplicates._input),
        subset=context.serialize_logical_expr_list(SerdeContext.EXPRS, drop_duplicates.subset),
        schema=context.serialize_fenic_schema(drop_duplicates.schema()),
    )
    return LogicalPlanProto(drop_duplicates=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_drop_duplicates(
    drop_duplicates: DropDuplicatesProto,
    context: SerdeContext,
) -> DropDuplicates:
    """Deserialize a DropDuplicates LogicalPlan Node."""
    return DropDuplicates.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, drop_duplicates.input),
        subset=context.deserialize_logical_expr_list(SerdeContext.EXPRS, drop_duplicates.subset),
        schema=context.deserialize_fenic_schema(drop_duplicates.schema),
    )


# =============================================================================
# Sort
# =============================================================================


@serialize_logical_plan.register
def _serialize_sort(sort: Sort, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a sort."""
    proto = SortProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, sort._input),
        sort_exprs=context.serialize_logical_expr_list("sort_exprs", sort._sort_exprs),
        schema=context.serialize_fenic_schema(sort.schema()),
    )
    return LogicalPlanProto(sort=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_sort(sort: SortProto, context: SerdeContext) -> Sort:
    """Deserialize a Sort LogicalPlan Node."""
    return Sort.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, sort.input),
        sort_exprs=context.deserialize_logical_expr_list("sort_exprs", sort.sort_exprs),
        schema=context.deserialize_fenic_schema(sort.schema),
    )


# =============================================================================
# Unnest
# =============================================================================


@serialize_logical_plan.register
def _serialize_unnest(
    unnest: Unnest, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize an unnest."""
    proto = UnnestProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, unnest._input),
        exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, unnest._exprs),
        schema=context.serialize_fenic_schema(unnest.schema()),
    )
    return LogicalPlanProto(unnest=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_unnest(unnest: UnnestProto, context: SerdeContext) -> Unnest:
    """Deserialize an Unnest LogicalPlan Node."""
    return Unnest.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, unnest.input),
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, unnest.exprs),
        schema=context.deserialize_fenic_schema(unnest.schema),
    )


# =============================================================================
# SQL
# =============================================================================


@serialize_logical_plan.register
def _serialize_sql(sql: SQL, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a SQL plan."""
    proto = SQLProto(
        inputs=context.serialize_logical_plan_list(SerdeContext.INPUTS, sql._inputs),
        template_names=sql._template_names,
        templated_query=sql._templated_query,
        schema=context.serialize_fenic_schema(sql.schema()),
    )
    return LogicalPlanProto(sql=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_sql(sql: SQLProto, context: SerdeContext) -> SQL:
    """Deserialize a SQL LogicalPlan Node."""
    return SQL.from_schema(
        inputs=context.deserialize_logical_plan_list(SerdeContext.INPUTS, sql.inputs),
        template_names=list(sql.template_names),
        templated_query=sql.templated_query,
        schema=context.deserialize_fenic_schema(sql.schema),
    )


# =============================================================================
# SemanticCluster
# =============================================================================


@serialize_logical_plan.register
def _serialize_semantic_cluster(
    semantic_cluster: SemanticCluster, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a semantic cluster."""
    proto = SemanticClusterProto(
        input=context.serialize_logical_plan(SerdeContext.INPUT, semantic_cluster._input),
        by_expr=context.serialize_logical_expr(SerdeContext.EXPR, semantic_cluster._by_expr),
        num_init=semantic_cluster._num_init,
        num_clusters=semantic_cluster._num_clusters,
        max_iter=semantic_cluster._max_iter,
        label_column=semantic_cluster._label_column,
        centroid_column=semantic_cluster._centroid_column or None,
        schema=context.serialize_fenic_schema(semantic_cluster.schema()),
    )
    return LogicalPlanProto(semantic_cluster=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_semantic_cluster(
    semantic_cluster_proto: SemanticClusterProto,
    context: SerdeContext,
) -> SemanticCluster:
    """Deserialize a SemanticCluster LogicalPlan Node."""
    return SemanticCluster.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, semantic_cluster_proto.input),
        by_expr=context.deserialize_logical_expr(SerdeContext.EXPR, semantic_cluster_proto.by_expr),
        num_clusters=semantic_cluster_proto.num_clusters,
        max_iter=semantic_cluster_proto.max_iter,
        num_init=semantic_cluster_proto.num_init,
        label_column=semantic_cluster_proto.label_column,
        centroid_column=semantic_cluster_proto.centroid_column or None,
        schema=context.deserialize_fenic_schema(semantic_cluster_proto.schema),
    )
