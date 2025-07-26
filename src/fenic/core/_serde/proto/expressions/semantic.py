"""Semantic/LLM expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.semantic import (
    AnalyzeSentimentExpr,
    EmbeddingsExpr,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticPredExpr,
    SemanticReduceExpr,
    SemanticSummarizeExpr,
)

# Import the main serialize/deserialize functions from parent
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    AnalyzeSentimentExprProto,
    ClassifyExampleCollectionProto,
    ClassifyExampleProto,
    EmbeddingsExprProto,
    KeyPointsProto,
    LogicalExprProto,
    MapExampleCollectionProto,
    MapExampleProto,
    ParagraphProto,
    PredicateExampleCollectionProto,
    PredicateExampleProto,
    SemanticClassifyExprProto,
    SemanticExtractExprProto,
    SemanticMapExprProto,
    SemanticPredExprProto,
    SemanticReduceExprProto,
    SemanticSummarizeExprProto,
    SummarizationFormatProto,
)
from fenic.core.types.semantic_examples import (
    ClassifyExample,
    ClassifyExampleCollection,
    MapExample,
    MapExampleCollection,
    PredicateExample,
    PredicateExampleCollection,
)
from fenic.core.types.summarize import KeyPoints, Paragraph

# =============================================================================
# SemanticMapExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_map_expr(logical: SemanticMapExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a semantic map expression."""
    examples_proto = (
        MapExampleCollectionProto(
            examples=[
                MapExampleProto(
                    input=example.input,
                    output=example.output,
                )
                for example in logical.examples.examples
            ]
        )
        if logical.examples
        else None
    )

    output_schema_proto = (
        context.serialize_pydantic_model_type("response_format", logical.response_format)
        if logical.response_format
        else None
    )

    return LogicalExprProto(
        semantic_map=SemanticMapExprProto(
            instruction=logical.instruction,
            exprs=context.serialize_logical_expr_list("exprs", logical.exprs),
            max_tokens=logical.max_tokens,
            temperature=logical.temperature,
            model_alias=logical.model_alias,
            response_format=output_schema_proto,
            examples=examples_proto,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_semantic_map_expr(
    logical_proto: SemanticMapExprProto,
    context: SerdeContext,
) -> SemanticMapExpr:
    """Deserialize a semantic map expression."""
    examples = MapExampleCollection(
        examples=[
            MapExample(
                input=example.input,
                output=example.output,
            )
            for example in logical_proto.examples.examples
        ]
    ) if logical_proto.examples.examples else None

    return SemanticMapExpr(
        instruction=logical_proto.instruction,
        max_tokens=logical_proto.max_tokens,
        temperature=logical_proto.temperature,
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
        response_format=context.deserialize_pydantic_model_type("response_format", logical_proto.response_format) if logical_proto.response_format else None,
        examples=examples,
    )


# =============================================================================
# SemanticExtractExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_extract_expr(logical: SemanticExtractExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a semantic extract expression."""
    schema_proto = (
        context.serialize_pydantic_model_type("schema", logical.schema) if logical.schema else None
    )

    return LogicalExprProto(
        semantic_extract=SemanticExtractExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            schema=schema_proto,
            max_tokens=logical.max_tokens,
            temperature=logical.temperature,
            model_alias=logical.model_alias,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_semantic_extract_expr(
    logical_proto: SemanticExtractExprProto,
    context: SerdeContext,
) -> SemanticExtractExpr:
    """Deserialize a semantic extract expression."""
    return SemanticExtractExpr(
        expr=context.deserialize_logical_expr("expr", logical_proto.expr),
        schema=context.deserialize_pydantic_model_type("schema", logical_proto.schema),
        max_tokens=logical_proto.max_tokens,
        temperature=logical_proto.temperature,
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
    )


# =============================================================================
# SemanticPredExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_pred_expr(logical: SemanticPredExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a semantic predicate expression."""
    examples_proto = (
        PredicateExampleCollectionProto(
            examples=[
                PredicateExampleProto(input=example.input, output=example.output)
                for example in logical.examples.examples
            ]
        )
        if logical.examples
        else None
    )

    return LogicalExprProto(
        semantic_pred=SemanticPredExprProto(
            instruction=logical.instruction,
            temperature=logical.temperature,
            model_alias=logical.model_alias,
            examples=examples_proto,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_semantic_pred_expr(
    logical_proto: SemanticPredExprProto,
    context: SerdeContext,
) -> SemanticPredExpr:
    """Deserialize a semantic predicate expression."""
    examples = PredicateExampleCollection(
        examples=[
            PredicateExample(
                input=example.input,
                output=example.output,
            )
            for example in logical_proto.examples.examples
        ]
    ) if logical_proto.examples.examples else None

    return SemanticPredExpr(
        instruction=logical_proto.instruction,
        temperature=logical_proto.temperature,
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
        examples=examples,
    )


# =============================================================================
# SemanticReduceExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_reduce_expr(logical: SemanticReduceExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a semantic reduce expression."""
    return LogicalExprProto(
        semantic_reduce=SemanticReduceExprProto(
            instruction=logical.instruction,
            max_tokens=logical.max_tokens,
            temperature=logical.temperature,
            model_alias=logical.model_alias,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_semantic_reduce_expr(
    logical_proto: SemanticReduceExprProto,
    context: SerdeContext,
) -> SemanticReduceExpr:
    """Deserialize a semantic reduce expression."""
    return SemanticReduceExpr(
        instruction=logical_proto.instruction,
        max_tokens=logical_proto.max_tokens,
        temperature=logical_proto.temperature,
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
    )


# =============================================================================
# SemanticClassifyExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_classify_expr(
    logical: SemanticClassifyExpr,
    context: SerdeContext,
) -> LogicalExprProto:
    """Serialize a semantic classify expression."""
    examples_proto = (
        ClassifyExampleCollectionProto(
            examples=[
                ClassifyExampleProto(input=example.input, output=example.output)
                for example in logical.examples.examples
            ]
        )
        if logical.examples
        else None
    )

    return LogicalExprProto(
        semantic_classify=SemanticClassifyExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            classes=[
                context.serialize_resolved_class_definition("class_definition", class_definition)
                for class_definition in logical.classes
            ],
            temperature=logical.temperature,
            model_alias=logical.model_alias,
            examples=examples_proto,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_semantic_classify_expr(
    logical_proto: SemanticClassifyExprProto,
    context: SerdeContext,
) -> SemanticClassifyExpr:
    """Deserialize a semantic classify expression."""
    examples = ClassifyExampleCollection(
        examples=[
            ClassifyExample(
                input=example.input,
                output=example.output,
            )
            for example in logical_proto.examples.examples
        ]
    ) if logical_proto.examples.examples else None

    return SemanticClassifyExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        classes=[
            context.deserialize_resolved_class_definition("class_definition", class_definition)
            for class_definition in logical_proto.classes
        ],
        temperature=logical_proto.temperature,
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
        examples=examples,
    )


# =============================================================================
# AnalyzeSentimentExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_analyze_sentiment_expr(
    logical: AnalyzeSentimentExpr,
    context: SerdeContext,
) -> LogicalExprProto:
    """Serialize an analyze sentiment expression."""
    return LogicalExprProto(
        analyze_sentiment=AnalyzeSentimentExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            temperature=logical.temperature,
            model_alias=logical.model_alias,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_analyze_sentiment_expr(
    logical_proto: AnalyzeSentimentExprProto,
    context: SerdeContext,
) -> AnalyzeSentimentExpr:
    """Deserialize an analyze sentiment expression."""
    return AnalyzeSentimentExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        temperature=logical_proto.temperature,
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
    )


# =============================================================================
# EmbeddingsExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_embeddings_expr(logical: EmbeddingsExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize an embeddings expression."""
    return LogicalExprProto(
        embeddings=EmbeddingsExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            model_alias=logical.model_alias,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_embeddings_expr(
    logical_proto: EmbeddingsExprProto,
    context: SerdeContext,
) -> EmbeddingsExpr:
    """Deserialize an embeddings expression."""
    return EmbeddingsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
    )


# =============================================================================
# SemanticSummarizeExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_summarize_expr(
    logical: SemanticSummarizeExpr,
    context: SerdeContext,
) -> LogicalExprProto:
    """Serialize a semantic summarize expression."""
    if isinstance(logical.format, KeyPoints):
        format_proto = SummarizationFormatProto(
            key_points=KeyPointsProto(max_points=logical.format.max_points)
        )
    elif isinstance(logical.format, Paragraph):
        format_proto = SummarizationFormatProto(
            paragraph=ParagraphProto(max_words=logical.format.max_words)
        )
    else:
        raise ValueError(f"Unsupported summarize format: {type(logical.format)}")

    return LogicalExprProto(
        semantic_summarize=SemanticSummarizeExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            format=format_proto,
            temperature=logical.temperature,
            model_alias=logical.model_alias,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_semantic_summarize_expr(
    logical_proto: SemanticSummarizeExprProto,
    context: SerdeContext,
) -> SemanticSummarizeExpr:
    """Deserialize a semantic summarize expression."""
    if logical_proto.format.HasField("key_points"):
        summary_format = KeyPoints(max_points=logical_proto.format.key_points.max_points)
    elif logical_proto.format.HasField("paragraph"):
        summary_format = Paragraph(max_words=logical_proto.format.paragraph.max_words)
    else:
        raise ValueError(f"Unsupported summarize format: {logical_proto.format.WhichOneof('format')}")

    return SemanticSummarizeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        format=summary_format,
        temperature=logical_proto.temperature,
        model_alias=logical_proto.model_alias if logical_proto.model_alias else None,
    )
