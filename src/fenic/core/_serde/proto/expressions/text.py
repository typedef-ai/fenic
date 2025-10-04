"""Text processing expression serialization/deserialization."""

# Import additional types for text expressions
from fenic.core._logical_plan.expressions.text import (
    ArrayJoinExpr,
    ByteLengthExpr,
    ChunkCharacterSet,
    ChunkLengthFunction,
    ConcatExpr,
    ContainsAnyExpr,
    ContainsExpr,
    CountTokensExpr,
    EndsWithExpr,
    FuzzyRatioExpr,
    FuzzyTokenSetRatioExpr,
    FuzzyTokenSortRatioExpr,
    ILikeExpr,
    JinjaExpr,
    LikeExpr,
    RecursiveTextChunkExpr,
    RecursiveTextChunkExprConfiguration,
    RegexpCountExpr,
    RegexpExtractAllExpr,
    RegexpExtractExpr,
    RegexpInstrExpr,
    RegexpSplitExpr,
    RegexpSubstrExpr,
    ReplaceExpr,
    RLikeExpr,
    SplitPartExpr,
    StartsWithExpr,
    StringCasingExpr,
    StripCharsExpr,
    StrLengthExpr,
    TextChunkExpr,
    TextChunkExprConfiguration,
    TextractExpr,
    TsParseExpr,
)

# Import the main serialize/deserialize functions from parent
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    ArrayJoinExprProto,
    ByteLengthExprProto,
    ChunkCharacterSetProto,
    ChunkLengthFunctionProto,
    ConcatExprProto,
    ContainsAnyExprProto,
    ContainsExprProto,
    CountTokensExprProto,
    EndsWithExprProto,
    FuzzyRatioExprProto,
    FuzzySimilarityMethodProto,
    FuzzyTokenSetRatioExprProto,
    FuzzyTokenSortRatioExprProto,
    ILikeExprProto,
    JinjaExprProto,
    LikeExprProto,
    LogicalExprProto,
    RecursiveTextChunkExprProto,
    RegexpCountExprProto,
    RegexpExtractAllExprProto,
    RegexpExtractExprProto,
    RegexpInstrExprProto,
    RegexpSplitExprProto,
    RegexpSubstrExprProto,
    ReplaceExprProto,
    RLikeExprProto,
    SplitPartExprProto,
    StartsWithExprProto,
    StringCasingExprProto,
    StripCharsExprProto,
    StrLengthExprProto,
    TextChunkExprProto,
    TextractExprProto,
    TsParseExprProto,
)

# Import enum types for literal serialization
from fenic.core.types.enums import (
    StringCasingType,
    StripCharsSide,
    TranscriptFormatType,
)

# =============================================================================
# TextractExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_textract_expr(
    logical: TextractExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a textract expression."""
    return LogicalExprProto(
        textract=TextractExprProto(
            input_expr=context.serialize_logical_expr("expr", logical.input_expr),
            template=logical.template,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_textract_expr(
    logical_proto: TextractExprProto, context: SerdeContext
) -> TextractExpr:
    """Deserialize a textract expression."""
    return TextractExpr(
        input_expr=context.deserialize_logical_expr("expr", logical_proto.input_expr),
        template=logical_proto.template,
    )


# =============================================================================
# TextChunkExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_text_chunk_expr(
    logical: TextChunkExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a text chunk expression."""
    def serialize_chunking_configuration(
        configuration: TextChunkExprConfiguration,
    ) -> TextChunkExprProto.TextChunkExprConfiguration:
        return TextChunkExprProto.TextChunkExprConfiguration(
            desired_chunk_size=configuration.desired_chunk_size,
            chunk_overlap_percentage=configuration.chunk_overlap_percentage,
            chunk_length_function_name=context.serialize_enum_value(
                "chunk_length_function_name",
                configuration.chunk_length_function_name,
                ChunkLengthFunctionProto,
            ),
        )

    return LogicalExprProto(
        text_chunk=TextChunkExprProto(
            expr=context.serialize_logical_expr("expr", logical.input_expr),
            configuration=serialize_chunking_configuration(logical.chunking_configuration),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_text_chunk_expr(
    logical_proto: TextChunkExprProto, context: SerdeContext
) -> TextChunkExpr:
    """Deserialize a text chunk expression."""
    def deserialize_chunking_configuration(
        configuration: TextChunkExprProto.TextChunkExprConfiguration,
    ) -> TextChunkExprConfiguration:
        return TextChunkExprConfiguration(
            desired_chunk_size=configuration.desired_chunk_size,
            chunk_overlap_percentage=configuration.chunk_overlap_percentage,
            chunk_length_function_name=context.deserialize_enum_value(
                "chunk_length_function_name",
                ChunkLengthFunction,
                ChunkLengthFunctionProto,
                configuration.chunk_length_function_name,
            ),
        )

    return TextChunkExpr(
        input_expr=context.deserialize_logical_expr(
            SerdeContext.EXPR, logical_proto.expr
        ),
        chunking_configuration=deserialize_chunking_configuration(
            logical_proto.configuration
        ),
    )


# =============================================================================
# RecursiveTextChunkExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_recursive_text_chunk_expr(
    logical: RecursiveTextChunkExpr,
    context: SerdeContext,
) -> LogicalExprProto:
    """Serialize a recursive text chunk expression."""
    def serialize_chunking_configuration(
        configuration: RecursiveTextChunkExprConfiguration,
    ) -> RecursiveTextChunkExprProto.RecursiveTextChunkExprConfiguration:
        return RecursiveTextChunkExprProto.RecursiveTextChunkExprConfiguration(
            desired_chunk_size=configuration.desired_chunk_size,
            chunk_overlap_percentage=configuration.chunk_overlap_percentage,
            chunk_length_function_name=context.serialize_enum_value(
                "chunk_length_function_name",
                configuration.chunk_length_function_name,
                ChunkLengthFunctionProto,
            ),
            chunking_character_set_name=context.serialize_enum_value(
                "chunking_character_set_name",
                configuration.chunking_character_set_name,
                ChunkCharacterSetProto,
            ),
            chunking_character_set_custom_characters=configuration.chunking_character_set_custom_characters,
        )

    return LogicalExprProto(
        recursive_text_chunk=RecursiveTextChunkExprProto(
            input_expr=context.serialize_logical_expr("expr", logical.input_expr),
            configuration=serialize_chunking_configuration(logical.chunking_configuration),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_recursive_text_chunk_expr(
    logical_proto: RecursiveTextChunkExprProto,
    context: SerdeContext,
) -> RecursiveTextChunkExpr:
    """Deserialize a recursive text chunk expression."""
    def deserialize_chunking_configuration(
        configuration: RecursiveTextChunkExprProto.RecursiveTextChunkExprConfiguration,
    ) -> RecursiveTextChunkExprConfiguration:
        chunking_character_set_name = context.deserialize_enum_value(
            "chunking_character_set_name",
            ChunkCharacterSet,
            ChunkCharacterSetProto,
            configuration.chunking_character_set_name,
        )
        return RecursiveTextChunkExprConfiguration(
            desired_chunk_size=configuration.desired_chunk_size,
            chunk_overlap_percentage=configuration.chunk_overlap_percentage,
            chunk_length_function_name=context.deserialize_enum_value(
                "chunk_length_function_name",
                ChunkLengthFunction,
                ChunkLengthFunctionProto,
                configuration.chunk_length_function_name,
            ),
            chunking_character_set_name=chunking_character_set_name,
            chunking_character_set_custom_characters=configuration.chunking_character_set_custom_characters if chunking_character_set_name == ChunkCharacterSet.CUSTOM else None,
        )

    return RecursiveTextChunkExpr(
        input_expr=context.deserialize_logical_expr(
            SerdeContext.EXPR, logical_proto.input_expr
        ),
        chunking_configuration=deserialize_chunking_configuration(
            logical_proto.configuration
        ),
    )


# =============================================================================
# CountTokensExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_count_tokens_expr(
    logical: CountTokensExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a count tokens expression."""
    return LogicalExprProto(
        count_tokens=CountTokensExprProto(
            input_expr=context.serialize_logical_expr("expr", logical.input_expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_count_tokens_expr(
    logical_proto: CountTokensExprProto,
    context: SerdeContext,
) -> CountTokensExpr:
    """Deserialize a count tokens expression."""
    return CountTokensExpr(
        input_expr=context.deserialize_logical_expr(
            SerdeContext.EXPR, logical_proto.input_expr
        )
    )


# =============================================================================
# ConcatExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_concat_expr(
    logical: ConcatExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a concat expression."""
    return LogicalExprProto(
        concat=ConcatExprProto(
            exprs=context.serialize_logical_expr_list("exprs", logical.exprs)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_concat_expr(
    logical_proto: ConcatExprProto, context: SerdeContext
) -> ConcatExpr:
    """Deserialize a concat expression."""
    return ConcatExpr(
        exprs=context.deserialize_logical_expr_list(
            SerdeContext.EXPRS, logical_proto.exprs
        )
    )


# =============================================================================
# ArrayJoinExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_join_expr(
    logical: ArrayJoinExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize an array join expression."""
    return LogicalExprProto(
        array_join=ArrayJoinExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            delimiter=logical.delimiter,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_join_expr(
    logical_proto: ArrayJoinExprProto, context: SerdeContext
) -> ArrayJoinExpr:
    """Deserialize an array join expression."""
    return ArrayJoinExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        delimiter=logical_proto.delimiter,
    )


# =============================================================================
# ContainsExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_contains_expr(
    logical: ContainsExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a contains expression."""
    return LogicalExprProto(
        contains=ContainsExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substr=context.serialize_logical_expr("substr", logical.substr),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_contains_expr(
    logical_proto: ContainsExprProto, context: SerdeContext
) -> ContainsExpr:
    """Deserialize a contains expression."""
    return ContainsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        substr=context.deserialize_logical_expr(
            SerdeContext.SUBSTR, logical_proto.substr
        ),
    )


# =============================================================================
# ContainsAnyExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_contains_any_expr(
    logical: ContainsAnyExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a contains any expression."""
    return LogicalExprProto(
        contains_any=ContainsAnyExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substrs=logical.substrs,
            case_insensitive=logical.case_insensitive,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_contains_any_expr(
    logical_proto: ContainsAnyExprProto,
    context: SerdeContext,
) -> ContainsAnyExpr:
    """Deserialize a contains any expression."""
    return ContainsAnyExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        substrs=list(logical_proto.substrs),
        case_insensitive=logical_proto.case_insensitive,
    )


# =============================================================================
# RLikeExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_rlike_expr(
    logical: RLikeExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize an rlike expression."""
    return LogicalExprProto(
        rlike=RLikeExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            pattern=context.serialize_logical_expr(SerdeContext.PATTERN, logical.pattern),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_rlike_expr(
    logical_proto: RLikeExprProto, context: SerdeContext
) -> RLikeExpr:
    """Deserialize an rlike expression."""
    return RLikeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.PATTERN, logical_proto.pattern),
    )


# =============================================================================
# LikeExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_like_expr(logical: LikeExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a like expression."""
    return LogicalExprProto(
        like=LikeExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            pattern=context.serialize_logical_expr(SerdeContext.PATTERN, logical.pattern),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_like_expr(
    logical_proto: LikeExprProto, context: SerdeContext
) -> LikeExpr:
    """Deserialize a like expression."""
    return LikeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.PATTERN, logical_proto.pattern),
    )


# =============================================================================
# ILikeExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_ilike_expr(
    logical: ILikeExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize an ilike expression."""
    return LogicalExprProto(
        ilike=ILikeExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=context.serialize_logical_expr(SerdeContext.PATTERN, logical.pattern),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_ilike_expr(
    logical_proto: ILikeExprProto, context: SerdeContext
) -> ILikeExpr:
    """Deserialize an ilike expression."""
    return ILikeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.PATTERN, logical_proto.pattern),
    )


# =============================================================================
# TsParseExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_ts_parse_expr(
    logical: TsParseExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a timestamp parse expression."""
    return LogicalExprProto(
        ts_parse=TsParseExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            format=context.serialize_python_literal(
                "format", logical.format, TsParseExprProto.TranscriptFormatType
            ),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_ts_parse_expr(
    logical_proto: TsParseExprProto, context: SerdeContext
) -> TsParseExpr:
    """Deserialize a timestamp parse expression."""
    return TsParseExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        format=context.deserialize_python_literal(
            "format",
            logical_proto.format,
            TranscriptFormatType,
            TsParseExprProto.TranscriptFormatType,
        ),
    )


# =============================================================================
# StartsWithExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_starts_with_expr(
    logical: StartsWithExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a starts with expression."""
    return LogicalExprProto(
        starts_with=StartsWithExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substr=context.serialize_logical_expr("substr", logical.substr),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_starts_with_expr(
    logical_proto: StartsWithExprProto, context: SerdeContext
) -> StartsWithExpr:
    """Deserialize a starts with expression."""
    return StartsWithExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        substr=context.deserialize_logical_expr(
            SerdeContext.SUBSTR, logical_proto.substr
        ),
    )


# =============================================================================
# EndsWithExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_ends_with_expr(
    logical: EndsWithExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize an ends with expression."""
    return LogicalExprProto(
        ends_with=EndsWithExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substr=context.serialize_logical_expr("substr", logical.substr),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_ends_with_expr(
    logical_proto: EndsWithExprProto, context: SerdeContext
) -> EndsWithExpr:
    """Deserialize an ends with expression."""
    return EndsWithExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        substr=context.deserialize_logical_expr(
            SerdeContext.SUBSTR, logical_proto.substr
        ),
    )


# =============================================================================
# RegexpSplitExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_regexp_split_expr(
    logical: RegexpSplitExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a regexp split expression."""
    return LogicalExprProto(
        regexp_split=RegexpSplitExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=logical.pattern,
            limit=logical.limit,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_regexp_split_expr(
    logical_proto: RegexpSplitExprProto,
    context: SerdeContext,
) -> RegexpSplitExpr:
    """Deserialize a regexp split expression."""
    return RegexpSplitExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=logical_proto.pattern,
        limit=logical_proto.limit,
    )


# =============================================================================
# RegexpCountExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_regexp_count_expr(
    logical: RegexpCountExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a regexp count expression."""
    return LogicalExprProto(
        regexp_count=RegexpCountExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=context.serialize_logical_expr("pattern", logical.pattern),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_regexp_count_expr(
    logical_proto: RegexpCountExprProto,
    context: SerdeContext,
) -> RegexpCountExpr:
    """Deserialize a regexp count expression."""
    return RegexpCountExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.pattern),
    )


# =============================================================================
# RegexpExtractExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_regexp_extract_expr(
    logical: RegexpExtractExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a regexp extract expression."""
    return LogicalExprProto(
        regexp_extract=RegexpExtractExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=context.serialize_logical_expr("pattern", logical.pattern),
            idx=context.serialize_logical_expr("idx", logical.idx),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_regexp_extract_expr(
    logical_proto: RegexpExtractExprProto,
    context: SerdeContext,
) -> RegexpExtractExpr:
    """Deserialize a regexp extract expression."""
    return RegexpExtractExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.pattern),
        idx=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.idx),
    )


# =============================================================================
# RegexpExtractAllExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_regexp_extract_all_expr(
    logical: RegexpExtractAllExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a regexp extract all expression."""
    return LogicalExprProto(
        regexp_extract_all=RegexpExtractAllExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=context.serialize_logical_expr("pattern", logical.pattern),
            idx=context.serialize_logical_expr("idx", logical.idx),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_regexp_extract_all_expr(
    logical_proto: RegexpExtractAllExprProto,
    context: SerdeContext,
) -> RegexpExtractAllExpr:
    """Deserialize a regexp extract all expression."""
    return RegexpExtractAllExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.pattern),
        idx=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.idx),
    )


# =============================================================================
# RegexpInstrExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_regexp_instr_expr(
    logical: RegexpInstrExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a regexp instr expression."""
    return LogicalExprProto(
        regexp_instr=RegexpInstrExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=context.serialize_logical_expr("pattern", logical.pattern),
            idx=context.serialize_logical_expr("idx", logical.idx),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_regexp_instr_expr(
    logical_proto: RegexpInstrExprProto,
    context: SerdeContext,
) -> RegexpInstrExpr:
    """Deserialize a regexp instr expression."""
    return RegexpInstrExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.pattern),
        idx=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.idx),
    )


# =============================================================================
# RegexpSubstrExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_regexp_substr_expr(
    logical: RegexpSubstrExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a regexp substr expression."""
    return LogicalExprProto(
        regexp_substr=RegexpSubstrExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=context.serialize_logical_expr("pattern", logical.pattern),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_regexp_substr_expr(
    logical_proto: RegexpSubstrExprProto,
    context: SerdeContext,
) -> RegexpSubstrExpr:
    """Deserialize a regexp substr expression."""
    return RegexpSubstrExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.pattern),
    )


# =============================================================================
# SplitPartExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_split_part_expr(
    logical: SplitPartExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a split part expression."""
    return LogicalExprProto(
        split_part=SplitPartExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            delimiter=context.serialize_logical_expr("delimiter", logical.delimiter),
            part_number=context.serialize_logical_expr(
                "part_number", logical.part_number
            ),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_split_part_expr(
    logical_proto: SplitPartExprProto, context: SerdeContext
) -> SplitPartExpr:
    """Deserialize a split part expression."""
    return SplitPartExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        delimiter=context.deserialize_logical_expr(
            "delimiter", logical_proto.delimiter
        ),
        part_number=context.deserialize_logical_expr(
            "part_number", logical_proto.part_number
        ),
    )


# =============================================================================
# StringCasingExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_string_casing_expr(
    logical: StringCasingExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a string casing expression."""
    return LogicalExprProto(
        string_casing=StringCasingExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            case=context.serialize_python_literal(
                "case", logical.case, StringCasingExprProto.StringCasingType
            ),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_string_casing_expr(
    logical_proto: StringCasingExprProto,
    context: SerdeContext,
) -> StringCasingExpr:
    """Deserialize a string casing expression."""
    return StringCasingExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        case=context.deserialize_python_literal(
            "case",
            logical_proto.case,
            StringCasingType,
            StringCasingExprProto.StringCasingType,
        ),
    )


# =============================================================================
# StripCharsExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_strip_chars_expr(
    logical: StripCharsExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a strip chars expression."""
    return LogicalExprProto(
        strip_chars=StripCharsExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            chars=context.serialize_logical_expr("chars", logical.chars)
            if logical.chars
            else None,
            side=context.serialize_python_literal(
                "side", logical.side, StripCharsExprProto.StripCharsSide
            ),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_strip_chars_expr(
    logical_proto: StripCharsExprProto, context: SerdeContext
) -> StripCharsExpr:
    """Deserialize a strip chars expression."""
    return StripCharsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        chars=context.deserialize_logical_expr("chars", logical_proto.chars)
        if logical_proto.chars
        else None,
        side=context.deserialize_python_literal(
            "side",
            logical_proto.side,
            StripCharsSide,
            StripCharsExprProto.StripCharsSide,
        ),
    )


# =============================================================================
# ReplaceExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_replace_expr(
    logical: ReplaceExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a replace expression."""
    return LogicalExprProto(
        replace=ReplaceExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            search=context.serialize_logical_expr("search", logical.search),
            replacement=context.serialize_logical_expr(
                "replacement", logical.replacement
            ),
            literal=logical.literal,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_replace_expr(
    logical_proto: ReplaceExprProto, context: SerdeContext
) -> ReplaceExpr:
    """Deserialize a replace expression."""
    return ReplaceExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        search=context.deserialize_logical_expr("search", logical_proto.search),
        replacement=context.deserialize_logical_expr(
            "replacement", logical_proto.replacement
        ),
        literal=logical_proto.literal,
    )


# =============================================================================
# StrLengthExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_str_length_expr(
    logical: StrLengthExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a string length expression."""
    return LogicalExprProto(
        str_length=StrLengthExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_str_length_expr(
    logical_proto: StrLengthExprProto, context: SerdeContext
) -> StrLengthExpr:
    """Deserialize a string length expression."""
    return StrLengthExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ByteLengthExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_byte_length_expr(
    logical: ByteLengthExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a byte length expression."""
    return LogicalExprProto(
        byte_length=ByteLengthExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_byte_length_expr(
    logical_proto: ByteLengthExprProto, context: SerdeContext
) -> ByteLengthExpr:
    """Deserialize a byte length expression."""
    return ByteLengthExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# JinjaExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_jinja_expr(
    logical: JinjaExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a jinja expression."""
    return LogicalExprProto(
        jinja=JinjaExprProto(
            exprs=context.serialize_logical_expr_list("exprs", logical.exprs),
            template=logical.template,
            strict=logical.strict,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_jinja_expr(
    logical_proto: JinjaExprProto, context: SerdeContext
) -> JinjaExpr:
    """Deserialize a jinja expression."""
    return JinjaExpr(
        exprs=context.deserialize_logical_expr_list("exprs", logical_proto.exprs),
        template=logical_proto.template,
        strict=logical_proto.strict,
    )


# =============================================================================
# Fuzzy Matching Expressions
# =============================================================================


@serialize_logical_expr.register
def _serialize_fuzzy_ratio_expr(
    logical: FuzzyRatioExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a fuzzy ratio expression."""
    return LogicalExprProto(
        fuzzy_ratio=FuzzyRatioExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            other=context.serialize_logical_expr(SerdeContext.OTHER, logical.other),
            method=context.serialize_python_literal(
                "method", logical.method, FuzzySimilarityMethodProto
            ),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_fuzzy_ratio_expr(
    logical_proto: FuzzyRatioExprProto, context: SerdeContext
) -> FuzzyRatioExpr:
    """Deserialize a fuzzy ratio expression."""
    from fenic.core.types.enums import FuzzySimilarityMethod

    return FuzzyRatioExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        other=context.deserialize_logical_expr(SerdeContext.OTHER, logical_proto.other),
        method=context.deserialize_python_literal(
            "method",
            logical_proto.method,
            FuzzySimilarityMethod,
            FuzzySimilarityMethodProto,
        ),
    )


@serialize_logical_expr.register
def _serialize_fuzzy_token_sort_ratio_expr(
    logical: FuzzyTokenSortRatioExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a fuzzy token sort ratio expression."""
    return LogicalExprProto(
        fuzzy_token_sort_ratio=FuzzyTokenSortRatioExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            other=context.serialize_logical_expr(SerdeContext.OTHER, logical.other),
            method=context.serialize_python_literal(
                "method", logical.method, FuzzySimilarityMethodProto
            ),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_fuzzy_token_sort_ratio_expr(
    logical_proto: FuzzyTokenSortRatioExprProto, context: SerdeContext
) -> FuzzyTokenSortRatioExpr:
    """Deserialize a fuzzy token sort ratio expression."""
    from fenic.core.types.enums import FuzzySimilarityMethod

    return FuzzyTokenSortRatioExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        other=context.deserialize_logical_expr(SerdeContext.OTHER, logical_proto.other),
        method=context.deserialize_python_literal(
            "method",
            logical_proto.method,
            FuzzySimilarityMethod,
            FuzzySimilarityMethodProto,
        ),
    )


@serialize_logical_expr.register
def _serialize_fuzzy_token_set_ratio_expr(
    logical: FuzzyTokenSetRatioExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a fuzzy token set ratio expression."""
    return LogicalExprProto(
        fuzzy_token_set_ratio=FuzzyTokenSetRatioExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            other=context.serialize_logical_expr(SerdeContext.OTHER, logical.other),
            method=context.serialize_python_literal(
                "method", logical.method, FuzzySimilarityMethodProto
            ),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_fuzzy_token_set_ratio_expr(
    logical_proto: FuzzyTokenSetRatioExprProto, context: SerdeContext
) -> FuzzyTokenSetRatioExpr:
    """Deserialize a fuzzy token set ratio expression."""
    from fenic.core.types.enums import FuzzySimilarityMethod

    return FuzzyTokenSetRatioExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        other=context.deserialize_logical_expr(SerdeContext.OTHER, logical_proto.other),
        method=context.deserialize_python_literal(
            "method",
            logical_proto.method,
            FuzzySimilarityMethod,
            FuzzySimilarityMethodProto,
        ),
    )
