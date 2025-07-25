"""Aggregate plan serialization/deserialization."""

from typing import Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.aggregate import Aggregate
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    serialize_logical_plan,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import AggregateProto, LogicalPlanProto

# =============================================================================
# Aggregate
# =============================================================================


@serialize_logical_plan.register
def _serialize_aggregate(
    aggregate: Aggregate, context: SerdeContext,
) -> LogicalPlanProto:
    """Serialize an aggregate."""
    input_proto = context.serialize_logical_plan(SerdeContext.INPUT, aggregate._input)
    group_exprs_protos = context.serialize_logical_expr_list(
        "group_exprs", aggregate._group_exprs
    )
    agg_exprs_protos = context.serialize_logical_expr_list(
        "agg_exprs", aggregate._agg_exprs
    )
    proto = AggregateProto(
        input=input_proto, group_exprs=group_exprs_protos, agg_exprs=agg_exprs_protos
    )
    return LogicalPlanProto(aggregate=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_aggregate(aggregate: AggregateProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize an Aggregate LogicalPlan Node."""
    input_plan = context.deserialize_logical_plan(SerdeContext.INPUT, aggregate.input, session_state=session_state)
    group_exprs = context.deserialize_logical_expr_list(
        "group_exprs", aggregate.group_exprs
    )
    agg_exprs = context.deserialize_logical_expr_list(
        "agg_exprs", aggregate.agg_exprs
    )
    result = Aggregate(input=input_plan, group_exprs=group_exprs, agg_exprs=agg_exprs)
    return result
