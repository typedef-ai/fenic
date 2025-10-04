"""Transform plan serialization/deserialization."""


from fenic.core._logical_plan.plans.transform import (
    SQL,
    DropDuplicates,
    Explode,
    ExplodeWithIndex,
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
    _serialize_logical_plan_helper,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    DropDuplicatesProto,
    ExplodeProto,
    ExplodeWithIndexProto,
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
from fenic.core.types.schema import Schema

# =============================================================================
# Projection
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_projection(
    projection: Projection, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a projection (wrapper)."""
    return LogicalPlanProto(
        projection=ProjectionProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, projection._input),
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, projection._exprs),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_projection(projection: ProjectionProto, context: SerdeContext, schema: Schema) -> Projection:
    """Deserialize a Projection LogicalPlan Node."""
    return Projection.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, projection.input),
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, projection.exprs),
        schema=schema,
    )


# =============================================================================
# Filter
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_filter(
    filter_plan: Filter, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a filter (wrapper)."""
    return LogicalPlanProto(
        filter=FilterProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, filter_plan._input),
            predicate=context.serialize_logical_expr(SerdeContext.EXPR, filter_plan._predicate),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_filter(filter_proto: FilterProto, context: SerdeContext, schema: Schema) -> Filter:
    """Deserialize a Filter LogicalPlan Node."""
    return Filter.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, filter_proto.input),
        predicate=context.deserialize_logical_expr(SerdeContext.EXPR, filter_proto.predicate),
        schema=schema,
    )


# =============================================================================
# Union
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_union(union: Union, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a union (wrapper)."""
    return LogicalPlanProto(
        union=UnionProto(
            inputs=context.serialize_logical_plan_list(SerdeContext.INPUTS, union._inputs),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_union(union: UnionProto, context: SerdeContext, schema: Schema) -> Union:
    """Deserialize a Union LogicalPlan Node."""
    return Union.from_schema(
        inputs=context.deserialize_logical_plan_list(SerdeContext.INPUTS, union.inputs),
        schema=schema,
    )


# =============================================================================
# Limit
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_limit(limit: Limit, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a limit (wrapper)."""
    return LogicalPlanProto(
        limit=LimitProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, limit._input),
            n=limit.n,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_limit(limit: LimitProto, context: SerdeContext, schema: Schema) -> Limit:
    """Deserialize a Limit LogicalPlan Node."""
    return Limit.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, limit.input),
        n=limit.n,
        schema=schema,
    )


# =============================================================================
# Explode
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_explode(
    explode: Explode, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize an explode (wrapper)."""
    return LogicalPlanProto(
        explode=ExplodeProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, explode._input),
            expr=context.serialize_logical_expr(SerdeContext.EXPR, explode._expr),
            keep_null_and_empty=explode.keep_null_and_empty,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_explode(explode: ExplodeProto, context: SerdeContext, schema: Schema) -> Explode:
    """Deserialize an Explode LogicalPlan Node."""
    return Explode.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, explode.input),
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, explode.expr),
        schema=schema,
        keep_null_and_empty=explode.keep_null_and_empty,
    )


# =============================================================================
# ExplodeWithIndex
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_explode_with_index(
    explode_with_index: ExplodeWithIndex, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize an explode_with_index (wrapper)."""
    return LogicalPlanProto(
        explode_with_index=ExplodeWithIndexProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, explode_with_index._input),
            expr=context.serialize_logical_expr(SerdeContext.EXPR, explode_with_index._expr),
            index_name=explode_with_index.index_name,
            value_name=explode_with_index.value_name,
            keep_null_and_empty=explode_with_index.keep_null_and_empty,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_explode_with_index(explode_with_index: ExplodeWithIndexProto, context: SerdeContext, schema: Schema) -> ExplodeWithIndex:
    """Deserialize an ExplodeWithIndex LogicalPlan Node."""
    return ExplodeWithIndex.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, explode_with_index.input),
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, explode_with_index.expr),
        index_name=explode_with_index.index_name,
        value_name=explode_with_index.value_name or None,
        schema=schema,
        keep_null_and_empty=explode_with_index.keep_null_and_empty,
    )


# =============================================================================
# DropDuplicates
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_drop_duplicates(
    drop_duplicates: DropDuplicates, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a drop duplicates (wrapper)."""
    return LogicalPlanProto(
        drop_duplicates=DropDuplicatesProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, drop_duplicates._input),
            subset=context.serialize_logical_expr_list("subset", drop_duplicates.subset),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_drop_duplicates(
    drop_duplicates: DropDuplicatesProto,
    context: SerdeContext,
    schema: Schema,
) -> DropDuplicates:
    """Deserialize a DropDuplicates LogicalPlan Node."""
    return DropDuplicates.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, drop_duplicates.input),
        subset=context.deserialize_logical_expr_list("subset", drop_duplicates.subset),
        schema=schema,
    )


# =============================================================================
# Sort
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_sort(sort: Sort, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a sort (wrapper)."""
    return LogicalPlanProto(
        sort=SortProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, sort._input),
            sort_exprs=context.serialize_logical_expr_list("sort_exprs", sort._sort_exprs),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_sort(sort: SortProto, context: SerdeContext, schema: Schema) -> Sort:
    """Deserialize a Sort LogicalPlan Node."""
    return Sort.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, sort.input),
        sort_exprs=context.deserialize_logical_expr_list("sort_exprs", sort.sort_exprs),
        schema=schema,
    )


# =============================================================================
# Unnest
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_unnest(
    unnest: Unnest, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize an unnest (wrapper)."""
    return LogicalPlanProto(
        unnest=UnnestProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, unnest._input),
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, unnest._exprs),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_unnest(unnest: UnnestProto, context: SerdeContext, schema: Schema) -> Unnest:
    """Deserialize an Unnest LogicalPlan Node."""
    return Unnest.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, unnest.input),
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, unnest.exprs),
        schema=schema,
    )


# =============================================================================
# SQL
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_sql(sql: SQL, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a SQL (wrapper)."""
    return LogicalPlanProto(
        sql=SQLProto(
            inputs=context.serialize_logical_plan_list(SerdeContext.INPUTS, sql._inputs),
            template_names=sql._template_names,
            templated_query=sql._templated_query,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_sql(sql: SQLProto, context: SerdeContext, schema: Schema) -> SQL:
    """Deserialize a SQL LogicalPlan Node."""
    return SQL.from_schema(
        inputs=context.deserialize_logical_plan_list(SerdeContext.INPUTS, sql.inputs),
        template_names=sql.template_names,
        templated_query=sql.templated_query,
        schema=schema,
    )


# =============================================================================
# SemanticCluster
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_semantic_cluster(
    semantic_cluster: SemanticCluster, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a semantic cluster (wrapper)."""
    return LogicalPlanProto(
        semantic_cluster=SemanticClusterProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, semantic_cluster._input),
            by_expr=context.serialize_logical_expr("by_expr", semantic_cluster._by_expr),
            num_clusters=semantic_cluster._num_clusters,
            max_iter=semantic_cluster._max_iter,
            num_init=semantic_cluster._num_init,
            label_column=semantic_cluster._label_column,
            centroid_column=semantic_cluster._centroid_column,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_semantic_cluster(
    semantic_cluster_proto: SemanticClusterProto,
    context: SerdeContext,
    schema: Schema,
) -> SemanticCluster:
    """Deserialize a SemanticCluster LogicalPlan Node."""
    return SemanticCluster.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, semantic_cluster_proto.input),
        by_expr=context.deserialize_logical_expr("by_expr", semantic_cluster_proto.by_expr),
        num_clusters=semantic_cluster_proto.num_clusters,
        max_iter=semantic_cluster_proto.max_iter,
        num_init=semantic_cluster_proto.num_init,
        label_column=semantic_cluster_proto.label_column,
        centroid_column=semantic_cluster_proto.centroid_column or None,
        schema=schema,
    )
