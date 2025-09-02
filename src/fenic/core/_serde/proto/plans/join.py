"""Join plan serialization/deserialization."""

from typing import Optional

from fenic.core._logical_plan.plans.join import (
    Join,
    SemanticJoin,
    SemanticSimilarityJoin,
)
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    _serialize_logical_plan_helper,
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
from fenic.core.types.enums import SemanticSimilarityMetric
from fenic.core.types.schema import Schema
from fenic.core.types.semantic_examples import JoinExample, JoinExampleCollection

# =============================================================================
# Join
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_join(join: Join, context: SerdeContext) -> LogicalPlanProto:
    """Serialize a join (wrapper)."""
    return LogicalPlanProto(
        join=JoinProto(
            left=context.serialize_logical_plan(SerdeContext.LEFT, join._left),
            right=context.serialize_logical_plan(SerdeContext.RIGHT, join._right),
            left_on=context.serialize_logical_expr_list("left_on", join._left_on),
            right_on=context.serialize_logical_expr_list("right_on", join._right_on),
            join_type=join._how,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_join(join: JoinProto, context: SerdeContext, schema: Schema) -> Join:
    """Deserialize a Join LogicalPlan Node."""
    return Join.from_schema(
        left=context.deserialize_logical_plan(SerdeContext.LEFT, join.left),
        right=context.deserialize_logical_plan(SerdeContext.RIGHT, join.right),
        left_on=context.deserialize_logical_expr_list("left_on", join.left_on),
        right_on=context.deserialize_logical_expr_list("right_on", join.right_on),
        how=join.join_type,
        schema=schema,
    )


# =============================================================================
# SemanticJoin
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_semantic_join(
    semantic_join: SemanticJoin, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a semantic join (wrapper)."""
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

    return LogicalPlanProto(
        semantic_join=SemanticJoinProto(
            left=context.serialize_logical_plan(SerdeContext.LEFT, semantic_join._left),
            right=context.serialize_logical_plan(SerdeContext.RIGHT, semantic_join._right),
            left_on=context.serialize_logical_expr("left_on", semantic_join._left_on),
            right_on=context.serialize_logical_expr("right_on", semantic_join._right_on),
            jinja_template=semantic_join.jinja_template(),
            strict=semantic_join.strict(),
            temperature=semantic_join.temperature,
            model_alias=context.serialize_resolved_model_alias("model_alias", semantic_join.model_alias) if semantic_join.model_alias else None,
            examples=examples,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_semantic_join(
    semantic_join: SemanticJoinProto, context: SerdeContext, schema: Schema
) -> SemanticJoin:
    """Deserialize a SemanticJoin LogicalPlan Node."""
    examples: Optional[JoinExampleCollection] = None
    if semantic_join.HasField("examples"):
        examples = JoinExampleCollection(
            examples=[
                JoinExample(
                    left=example.left,
                    right=example.right,
                    output=example.output,
                )
                for example in semantic_join.examples.examples
            ]
        )

    return SemanticJoin.from_schema(
        left=context.deserialize_logical_plan(SerdeContext.LEFT, semantic_join.left),
        right=context.deserialize_logical_plan(SerdeContext.RIGHT, semantic_join.right),
        left_on=context.deserialize_logical_expr("left_on", semantic_join.left_on),
        right_on=context.deserialize_logical_expr("right_on", semantic_join.right_on),
        jinja_template=semantic_join.jinja_template,
        strict=semantic_join.strict,
        temperature=semantic_join.temperature,
        model_alias=context.deserialize_resolved_model_alias("model_alias", semantic_join.model_alias) if semantic_join.HasField("model_alias") else None,
        examples=examples,
        schema=schema,
    )


# =============================================================================
# SemanticSimilarityJoin
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_semantic_similarity_join(
    semantic_similarity_join: SemanticSimilarityJoin,
    context: SerdeContext,
) -> LogicalPlanProto:
    """Serialize a semantic similarity join (wrapper)."""
    return LogicalPlanProto(
        semantic_similarity_join=SemanticSimilarityJoinProto(
            left=context.serialize_logical_plan(SerdeContext.LEFT, semantic_similarity_join._left),
            right=context.serialize_logical_plan(SerdeContext.RIGHT, semantic_similarity_join._right),
            left_on=context.serialize_logical_expr("left_on", semantic_similarity_join._left_on),
            right_on=context.serialize_logical_expr("right_on", semantic_similarity_join._right_on),
            k=semantic_similarity_join._k,
            similarity_metric=context.serialize_python_literal(
                "similarity_metric",
                semantic_similarity_join._similarity_metric,
                SemanticSimilarityJoinProto.SemanticSimilarityMetric,
            ),
            similarity_score_column=semantic_similarity_join._similarity_score_column,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_semantic_similarity_join(
    semantic_similarity_join: SemanticSimilarityJoinProto,
    context: SerdeContext,
    schema: Schema,
) -> SemanticSimilarityJoin:
    """Deserialize a SemanticSimilarityJoin LogicalPlan Node."""
    return SemanticSimilarityJoin.from_schema(
        left=context.deserialize_logical_plan(SerdeContext.LEFT, semantic_similarity_join.left),
        right=context.deserialize_logical_plan(SerdeContext.RIGHT, semantic_similarity_join.right),
        left_on=context.deserialize_logical_expr("left_on", semantic_similarity_join.left_on),
        right_on=context.deserialize_logical_expr("right_on", semantic_similarity_join.right_on),
        k=semantic_similarity_join.k,
        similarity_metric=context.deserialize_python_literal(
            "similarity_metric",
            semantic_similarity_join.similarity_metric,
            SemanticSimilarityMetric,
            SemanticSimilarityJoinProto.SemanticSimilarityMetric,
        ),
        similarity_score_column=semantic_similarity_join.similarity_score_column if semantic_similarity_join.HasField("similarity_score_column") else None,
        schema=schema,
    )
