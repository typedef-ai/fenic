"""Semantic/LLM expression serialization/deserialization."""

from pydantic import BaseModel

from fenic.core._logical_plan.expressions.semantic import (
    AnalyzeSentimentExpr,
    EmbeddingsExpr,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticParsePDFExpr,
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
    SemanticParsePDFExprProto,
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
    def serialize_map_example(example: MapExample) -> MapExampleProto:
        input_proto = {
            key: context.serialize_scalar_value(key, value)
            for key, value in example.input.items()
        }
        # If the output is a BaseModel, serialize it to JSON. This does not preserve the schema, but this is what we'd be
        # doing anyway when transforming the MapExample into a FewShotExample, so nothing should be lost.
        serialized_output = example.output.model_dump_json() if isinstance(example.output, BaseModel) else example.output
        return MapExampleProto(
            input=input_proto,
            output=serialized_output,
        )

    examples_proto = (
        MapExampleCollectionProto(
            examples=[
                serialize_map_example(example)
                for example in logical.examples.examples
            ]
        )
        if logical.examples
        else None
    )

    output_schema_proto = (
        context.serialize_resolved_response_format("response_format", logical.response_format)
        if logical.response_format
        else None
    )

    return LogicalExprProto(
        semantic_map=SemanticMapExprProto(
            template=logical.template,
            strict=logical.strict,
            exprs=context.serialize_logical_expr_list("exprs", logical.exprs),
            max_tokens=logical.max_tokens,
            temperature=logical.temperature,
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
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
    def deserialize_map_example(example: MapExampleProto) -> MapExample:
        input_dict = {
            key: context.deserialize_scalar_value(key, value)
            for key, value in example.input.items()
        }
        return MapExample(input=input_dict, output=example.output)

    examples = MapExampleCollection(
        examples=[
            deserialize_map_example(example)
            for example in logical_proto.examples.examples
        ]
    ) if logical_proto.examples.examples else None

    return SemanticMapExpr(
        jinja_template=logical_proto.template,
        strict=logical_proto.strict,
        exprs=context.deserialize_logical_expr_list("exprs", logical_proto.exprs),
        max_tokens=logical_proto.max_tokens,
        temperature=logical_proto.temperature,
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
        response_format=context.deserialize_resolved_response_format("response_format", logical_proto.response_format) if logical_proto.HasField("response_format") else None,
        examples=examples,
    )


# =============================================================================
# SemanticExtractExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_extract_expr(logical: SemanticExtractExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a semantic extract expression."""
    return LogicalExprProto(
        semantic_extract=SemanticExtractExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            response_format=context.serialize_resolved_response_format("response_format", logical.response_format),
            max_tokens=logical.max_tokens,
            temperature=logical.temperature,
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias)
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
        response_format=context.deserialize_resolved_response_format("response_format", logical_proto.response_format),
        max_tokens=logical_proto.max_tokens,
        temperature=logical_proto.temperature,
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
    )


# =============================================================================
# SemanticPredExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_pred_expr(logical: SemanticPredExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a semantic predicate expression."""
    def serialize_predicate_example(example: PredicateExample) -> PredicateExampleProto:
        input_proto = {
            key: context.serialize_scalar_value(key, value)
            for key, value in example.input.items()
        }
        return PredicateExampleProto(input=input_proto, output=example.output)

    examples_proto = (
        PredicateExampleCollectionProto(
            examples=[
                serialize_predicate_example(example)
                for example in logical.examples.examples
            ]
        )
        if logical.examples
        else None
    )

    return LogicalExprProto(
        semantic_pred=SemanticPredExprProto(
            template=logical.template,
            strict=logical.strict,
            exprs=context.serialize_logical_expr_list("exprs", logical.exprs),
            temperature=logical.temperature,
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
            examples=examples_proto,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_semantic_pred_expr(
    logical_proto: SemanticPredExprProto,
    context: SerdeContext,
) -> SemanticPredExpr:
    """Deserialize a semantic predicate expression."""
    def deserialize_predicate_example(example: PredicateExampleProto) -> PredicateExample:
        input_dict = {
            key: context.deserialize_scalar_value(key, value)
            for key, value in example.input.items()
        }
        return PredicateExample(input=input_dict, output=example.output)

    examples = PredicateExampleCollection(
        examples=[
            deserialize_predicate_example(example)
            for example in logical_proto.examples.examples
        ]
    ) if logical_proto.examples.examples else None

    return SemanticPredExpr(
        jinja_template=logical_proto.template,
        strict=logical_proto.strict,
        exprs=context.deserialize_logical_expr_list("exprs", logical_proto.exprs),
        temperature=logical_proto.temperature,
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
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
            input_expr=context.serialize_logical_expr("input_expr", logical.input_expr),
            group_context_exprs=context.serialize_logical_expr_list("group_context_exprs", list(logical.group_context_exprs.values())),
            order_by_exprs=context.serialize_logical_expr_list("order_by_exprs", logical.order_by_exprs),
            max_tokens=logical.max_tokens,
            temperature=logical.temperature,
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
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
        input_expr=context.deserialize_logical_expr("input_expr", logical_proto.input_expr),
        group_context_exprs=context.deserialize_logical_expr_list("group_context_exprs", logical_proto.group_context_exprs),
        order_by_exprs=context.deserialize_logical_expr_list("order_by_exprs", logical_proto.order_by_exprs),
        max_tokens=logical_proto.max_tokens,
        temperature=logical_proto.temperature,
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
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
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
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
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
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
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
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
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
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
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
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
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
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
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
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
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
    )

# =============================================================================
# SemanticParsePDFExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_semantic_parse_pdf_expr(logical: SemanticParsePDFExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a semantic parse PDF expression."""
    return LogicalExprProto(
        semantic_parse_pdf=SemanticParsePDFExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            model_alias=context.serialize_resolved_model_alias("model_alias", logical.model_alias),
            page_separator=logical.page_separator,
            describe_images=logical.describe_images,
        )
    )

@_deserialize_logical_expr_helper.register
def _deserialize_semantic_parse_pdf_expr(
    logical_proto: SemanticParsePDFExprProto,
    context: SerdeContext,
) -> SemanticParsePDFExpr:
    """Deserialize a semantic parse PDF expression."""
    return SemanticParsePDFExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        model_alias=context.deserialize_resolved_model_alias("model_alias", logical_proto.model_alias) if logical_proto.HasField("model_alias") else None,
        page_separator=logical_proto.page_separator,
        describe_images=logical_proto.describe_images,
    )