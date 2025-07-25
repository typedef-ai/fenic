"""Join plan serialization/deserialization."""

from typing import Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.join import (
    Join,
    SemanticJoin,
    SemanticSimilarityJoin,
)
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    serialize_logical_plan,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    JoinExampleCollectionProto,
    JoinExampleProto,
    JoinProto,
    LogicalPlanProto,
    SemanticJoinProto,
    SemanticSimilarityJoinProto,
)
from fenic.core.types.semantic_examples import JoinExample, JoinExampleCollection

# =============================================================================
# Join
# =============================================================================


@serialize_logical_plan.register
def _serialize_join(join: Join, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a join."""
    left_proto = context.serialize_logical_plan(SerdeContext.LEFT, join._left)
    right_proto = context.serialize_logical_plan(SerdeContext.RIGHT, join._right)
    left_keys_protos = context.serialize_logical_expr_list(
        "left_keys", join._left_on
    )
    right_keys_protos = context.serialize_logical_expr_list(
        "right_keys", join._right_on
    )
    proto = JoinProto(
        left=left_proto,
        right=right_proto,
        left_keys=left_keys_protos,
        right_keys=right_keys_protos,
        join_type=join._how,
    )
    return LogicalPlanProto(join=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_join(join: JoinProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a Join LogicalPlan Node."""
    left = context.deserialize_logical_plan(SerdeContext.LEFT, join.left, session_state=session_state)
    right = context.deserialize_logical_plan(SerdeContext.RIGHT, join.right, session_state=session_state)
    left_on = context.deserialize_logical_expr_list(
        "left_keys", join.left_keys
    )
    right_on = context.deserialize_logical_expr_list(
        "right_keys", join.right_keys
    )
    result = Join(
        left=left, right=right, left_on=left_on, right_on=right_on, how=join.join_type
    )
    result.session_state = session_state
    return result


# =============================================================================
# SemanticJoin
# =============================================================================


@serialize_logical_plan.register
def _serialize_semantic_join(
    semantic_join: SemanticJoin, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a semantic join."""
    left_proto = context.serialize_logical_plan(
        SerdeContext.LEFT, semantic_join._left
    )
    right_proto = context.serialize_logical_plan(
        SerdeContext.RIGHT, semantic_join._right
    )
    left_on_proto = context.serialize_logical_expr(
        "left_on", semantic_join._left_on
    )
    right_on_proto = context.serialize_logical_expr(
        "right_on", semantic_join._right_on
    )


    examples: Optional[JoinExampleCollectionProto] = None
    if semantic_join.examples():
        examples = JoinExampleCollectionProto(
            examples=[
                JoinExampleProto(
                    left=example.left,
                    right=example.right,
                    output=example.output,
                )
                for example in semantic_join.examples().examples
            ]
        )

    proto = SemanticJoinProto(
        left=left_proto,
        right=right_proto,
        left_on=left_on_proto,
        right_on=right_on_proto,
        join_instruction=semantic_join.join_instruction(),
        temperature=semantic_join.temperature,
        model_alias=semantic_join.model_alias,
        examples=examples,
    )
    return LogicalPlanProto(semantic_join=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_semantic_join(
    semantic_join: SemanticJoinProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None
):
    """Deserialize a SemanticJoin LogicalPlan Node."""
    left = context.deserialize_logical_plan(SerdeContext.LEFT, semantic_join.left, session_state=session_state)
    right = context.deserialize_logical_plan(SerdeContext.RIGHT, semantic_join.right, session_state=session_state)
    left_on = context.deserialize_logical_expr(
        "left_on", semantic_join.left_on
    )
    right_on = context.deserialize_logical_expr(
        "right_on", semantic_join.right_on
    )

    # Handle complex types deserialization if needed
    examples = None
    if semantic_join.examples.examples:
        examples = JoinExampleCollection(
            examples=[
                JoinExample(
                    left=example.left,
                    right=example.right,
                    output=example.output,
                ) for example in semantic_join.examples.examples
            ]
        )

    result = SemanticJoin(
        left=left,
        right=right,
        left_on=left_on,
        right_on=right_on,
        join_instruction=semantic_join.join_instruction,
        temperature=semantic_join.temperature,
        model_alias=semantic_join.model_alias if semantic_join.model_alias else None,
        examples=examples,
    )
    result.session_state = session_state
    return result


# =============================================================================
# SemanticSimilarityJoin
# =============================================================================


@serialize_logical_plan.register
def _serialize_semantic_similarity_join(
    semantic_similarity_join: SemanticSimilarityJoin,
    context: SerdeContext,
) -> LogicalPlanProto:
    """Serialize a semantic similarity join."""
    left_proto = context.serialize_logical_plan(
        SerdeContext.LEFT, semantic_similarity_join._left
    )
    right_proto = context.serialize_logical_plan(
        SerdeContext.RIGHT, semantic_similarity_join._right
    )
    left_on_proto = context.serialize_logical_expr(
        "left_on", semantic_similarity_join._left_on
    )
    right_on_proto = context.serialize_logical_expr(
        "right_on", semantic_similarity_join._right_on
    )
    # SemanticSimilarityMetric is a string literal type, not an enum
    similarity_metric = semantic_similarity_join.similarity_metric()
    similarity_score_column = (
        semantic_similarity_join.similarity_score_column()
        if semantic_similarity_join.similarity_score_column()
        else None
    )
    proto = SemanticSimilarityJoinProto(
        left=left_proto,
        right=right_proto,
        left_on=left_on_proto,
        right_on=right_on_proto,
        k=semantic_similarity_join.k(),
        similarity_metric=similarity_metric,
        similarity_score_column=similarity_score_column,
    )
    return LogicalPlanProto(semantic_similarity_join=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_semantic_similarity_join(
    proto: SemanticSimilarityJoinProto,
    context: SerdeContext,
    session_state: Optional[BaseSessionState] = None,
):
    """Deserialize a SemanticSimilarityJoin LogicalPlan Node."""
    left = context.deserialize_logical_plan(
        SerdeContext.LEFT, proto.left, session_state=session_state
    )
    right = context.deserialize_logical_plan(
        SerdeContext.RIGHT, proto.right, session_state=session_state
    )
    left_on = context.deserialize_logical_expr(
        "left_on", proto.left_on
    )
    right_on = context.deserialize_logical_expr(
        "right_on", proto.right_on
    )
    # SemanticSimilarityMetric is a string literal type, not an enum
    similarity_metric = proto.similarity_metric
    similarity_score_column = (
        proto.similarity_score_column
        if proto.similarity_score_column
        else None
    )
    result = SemanticSimilarityJoin(
        left=left,
        right=right,
        left_on=left_on,
        right_on=right_on,
        k=proto.k,
        similarity_metric=similarity_metric,
        similarity_score_column=similarity_score_column,
    )
    return result
