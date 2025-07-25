"""Transform plan serialization/deserialization."""

from typing import Optional

from fenic.core._interfaces.session_state import BaseSessionState
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
    input_proto = context.serialize_logical_plan(
        SerdeContext.INPUT, projection._input
    )
    exprs_protos = context.serialize_logical_expr_list(
        SerdeContext.EXPRS, projection._exprs
    )
    proto = ProjectionProto(input=input_proto, exprs=exprs_protos)
    return LogicalPlanProto(projection=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_projection(projection: ProjectionProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a Projection LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(
        SerdeContext.INPUT, projection.input, session_state=session_state
    )
    exprs = context.deserialize_logical_expr_list(
        SerdeContext.EXPRS, projection.exprs
    )
    result = Projection(input=input_plan, exprs=exprs)
    result.session_state = session_state
    return result


# =============================================================================
# Filter
# =============================================================================


@serialize_logical_plan.register
def _serialize_filter(
    filter_plan: Filter, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a filter."""
    input_proto = context.serialize_logical_plan(
        SerdeContext.INPUT, filter_plan._input
    )
    predicate_proto = context.serialize_logical_expr(
        "predicate", filter_plan._predicate
    )
    proto = FilterProto(input=input_proto, predicate=predicate_proto)
    return LogicalPlanProto(filter=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_filter(filter_proto: FilterProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a Filter LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(
        SerdeContext.INPUT, filter_proto.input, session_state=session_state
    )
    predicate = context.deserialize_logical_expr(
        "predicate", filter_proto.predicate
    )
    result = Filter(input=input_plan, predicate=predicate)
    result.session_state = session_state
    return result


# =============================================================================
# Union
# =============================================================================


@serialize_logical_plan.register
def _serialize_union(union: Union, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a union."""
    inputs_protos = context.serialize_logical_plan_list(
        SerdeContext.INPUTS, union._inputs
    )
    proto = UnionProto(inputs=inputs_protos)
    return LogicalPlanProto(union=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_union(union: UnionProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a Union LogicalPlan Node."""
    inputs = context.deserialize_logical_plan_list(SerdeContext.INPUTS, union.inputs)
    result = Union(inputs=inputs)
    result.session_state = session_state
    return result


# =============================================================================
# Limit
# =============================================================================


@serialize_logical_plan.register
def _serialize_limit(limit: Limit, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a limit."""
    input_proto = context.serialize_logical_plan(SerdeContext.INPUT, limit._input)
    proto = LimitProto(input=input_proto, n=limit.n)
    return LogicalPlanProto(limit=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_limit(limit: LimitProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a Limit LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(SerdeContext.INPUT, limit.input, session_state=session_state)
    result = Limit(input=input_plan, n=limit.n)
    result.session_state = session_state
    return result


# =============================================================================
# Explode
# =============================================================================


@serialize_logical_plan.register
def _serialize_explode(
    explode: Explode, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize an explode."""
    input_proto = context.serialize_logical_plan(SerdeContext.INPUT, explode._input)
    expr_proto = context.serialize_logical_expr(SerdeContext.EXPR, explode._expr)
    proto = ExplodeProto(input=input_proto, expr=expr_proto)
    return LogicalPlanProto(explode=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_explode(explode: ExplodeProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize an Explode LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(SerdeContext.INPUT, explode.input, session_state=session_state)
    expr = context.deserialize_logical_expr(SerdeContext.EXPR, explode.expr)
    result = Explode(input=input_plan, expr=expr)
    result.session_state = session_state
    return result


# =============================================================================
# DropDuplicates
# =============================================================================


@serialize_logical_plan.register
def _serialize_drop_duplicates(
    drop_duplicates: DropDuplicates, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a drop duplicates."""
    input_proto = context.serialize_logical_plan(
        SerdeContext.INPUT, drop_duplicates._input
    )
    subset_protos = context.serialize_logical_expr_list(
        SerdeContext.EXPRS, drop_duplicates.subset
    )
    proto = DropDuplicatesProto(input=input_proto, subset=subset_protos)
    return LogicalPlanProto(drop_duplicates=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_drop_duplicates(
    drop_duplicates: DropDuplicatesProto,
    context: SerdeContext,
    session_state: Optional[BaseSessionState] = None,
):
    """Deserialize a DropDuplicates LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(
        SerdeContext.INPUT, drop_duplicates.input, session_state=session_state
    )
    subset = context.deserialize_logical_expr_list(
        SerdeContext.EXPRS, drop_duplicates.subset
    )
    result = DropDuplicates(input=input_plan, subset=subset)
    return result


# =============================================================================
# Sort
# =============================================================================


@serialize_logical_plan.register
def _serialize_sort(sort: Sort, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a sort."""
    input_proto = context.serialize_logical_plan(SerdeContext.INPUT, sort._input)
    sort_exprs_protos = context.serialize_logical_expr_list(
        SerdeContext.EXPRS, sort._sort_exprs
    )
    proto = SortProto(input=input_proto, sort_exprs=sort_exprs_protos)
    return LogicalPlanProto(sort=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_sort(sort: SortProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a Sort LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(SerdeContext.INPUT, sort.input, session_state=session_state)
    sort_exprs = context.deserialize_logical_expr_list(
        SerdeContext.EXPRS, sort.sort_exprs
    )
    result = Sort(input=input_plan, sort_exprs=sort_exprs)
    return result


# =============================================================================
# Unnest
# =============================================================================


@serialize_logical_plan.register
def _serialize_unnest(
    unnest: Unnest, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize an unnest."""
    input_proto = context.serialize_logical_plan(SerdeContext.INPUT, unnest._input)
    exprs_protos = context.serialize_logical_expr_list(
        SerdeContext.EXPRS, unnest._exprs
    )
    proto = UnnestProto(input=input_proto, exprs=exprs_protos)
    return LogicalPlanProto(unnest=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_unnest(unnest: UnnestProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize an Unnest LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(SerdeContext.INPUT, unnest.input, session_state=session_state)
    exprs = context.deserialize_logical_expr_list(SerdeContext.EXPRS, unnest.exprs)
    result = Unnest(input=input_plan, exprs=exprs)
    return result


# =============================================================================
# SQL
# =============================================================================


@serialize_logical_plan.register
def _serialize_sql(sql: SQL, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a SQL plan."""
    inputs_protos = context.serialize_logical_plan_list(
        SerdeContext.INPUTS, sql._inputs
    )
    proto = SQLProto(
        inputs=inputs_protos,
        template_names=sql._template_names,
        templated_query=sql._templated_query,
    )
    return LogicalPlanProto(sql=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_sql(sql: SQLProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a SQL LogicalPlan Node."""
    inputs = context.deserialize_logical_plan_list(SerdeContext.INPUTS, sql.inputs, session_state=session_state)
    result = SQL(
        inputs=inputs,
        template_names=list(sql.template_names),
        templated_query=sql.templated_query,
    )
    result.session_state = session_state
    return result


# =============================================================================
# SemanticCluster
# =============================================================================


@serialize_logical_plan.register
def _serialize_semantic_cluster(
    semantic_cluster: SemanticCluster, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a semantic cluster."""
    input_proto = context.serialize_logical_plan(
        SerdeContext.INPUT, semantic_cluster._input
    )
    by_expr_proto = context.serialize_logical_expr(
        SerdeContext.EXPR, semantic_cluster._by_expr
    )
    proto = SemanticClusterProto(
        input=input_proto,
        by_expr=by_expr_proto,
        num_init=semantic_cluster._num_init,
        num_clusters=semantic_cluster._num_clusters,
        max_iter=semantic_cluster._max_iter,
        label_column=semantic_cluster._label_column,
        centroid_column=semantic_cluster._centroid_column or "",
    )
    return LogicalPlanProto(semantic_cluster=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_semantic_cluster(
    semantic_cluster_proto: SemanticClusterProto,
    context: SerdeContext,
    session_state: Optional[BaseSessionState] = None,
):
    """Deserialize a SemanticCluster LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(
        SerdeContext.INPUT, semantic_cluster_proto.input, session_state=session_state
    )
    by_expr = context.deserialize_logical_expr(
        "by_expr", semantic_cluster_proto.by_expr
    )
    centroid_column = (
        semantic_cluster_proto.centroid_column if semantic_cluster_proto.centroid_column else None
    )
    result = SemanticCluster(
        input=input_plan,
        by_expr=by_expr,
        num_clusters=semantic_cluster_proto.num_clusters,
        max_iter=semantic_cluster_proto.max_iter,
        num_init=semantic_cluster_proto.num_init,
        label_column=semantic_cluster_proto.label_column,
        centroid_column=centroid_column,
    )
    result.session_state = session_state
    return result
