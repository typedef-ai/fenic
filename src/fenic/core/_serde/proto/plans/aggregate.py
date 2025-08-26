"""Aggregate plan serialization/deserialization."""

from fenic.core._logical_plan.plans.aggregate import Aggregate
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    _serialize_logical_plan_helper,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import AggregateProto, LogicalPlanProto
from fenic.core.types.schema import Schema

# =============================================================================
# Aggregate
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_aggregate(
    aggregate: Aggregate,
    context: SerdeContext,
) -> LogicalPlanProto:
    """Serialize an aggregate (wrapper)."""
    return LogicalPlanProto(
        aggregate=AggregateProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, aggregate._input),
            group_exprs=context.serialize_logical_expr_list("group_exprs", aggregate._group_exprs),
            agg_exprs=context.serialize_logical_expr_list("agg_exprs", aggregate._agg_exprs),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_aggregate(
    aggregate: AggregateProto,
    context: SerdeContext,
    schema: Schema,\
) -> Aggregate:
    """Deserialize an Aggregate LogicalPlan Node."""
    return Aggregate.from_schema(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, aggregate.input),
        group_exprs=context.deserialize_logical_expr_list("group_exprs", aggregate.group_exprs),
        agg_exprs=context.deserialize_logical_expr_list("agg_exprs", aggregate.agg_exprs),
        schema=schema,
    )
