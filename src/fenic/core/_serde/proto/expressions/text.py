"""Text processing expression serialization/deserialization."""

# Import additional types for text expressions
from fenic.core._logical_plan.expressions import ChunkCharacterSet, ChunkLengthFunction
from fenic.core._logical_plan.expressions.text import (
    ArrayJoinExpr,
    ByteLengthExpr,
    ConcatExpr,
    ContainsAnyExpr,
    ContainsExpr,
    CountTokensExpr,
    EndsWithExpr,
    ILikeExpr,
    LikeExpr,
    RecursiveTextChunkExpr,
    RegexpSplitExpr,
    ReplaceExpr,
    RLikeExpr,
    SplitPartExpr,
    StartsWithExpr,
    StringCasingExpr,
    StripCharsExpr,
    StrLengthExpr,
    TextChunkExpr,
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
    ILikeExprProto,
    LikeExprProto,
    LogicalExprProto,
    RecursiveTextChunkExprProto,
    RegexpSplitExprProto,
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


@serialize_logical_expr.register
def _serialize_textract_expr(logical: TextractExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a textract expression."""
    return LogicalExprProto(
        textract=TextractExprProto(
            input_expr=context.serialize_logical_expr(
                "expr", logical.input_expr
            ),
            template=logical.template,
        )
    )


@serialize_logical_expr.register
def _serialize_text_chunk_expr(logical: TextChunkExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a text chunk expression."""
    return LogicalExprProto(
        text_chunk=TextChunkExprProto(
            expr=context.serialize_logical_expr(
                "expr", logical.input_expr
            ),
            configuration=context.serialize_text_chunk_expr_configuration(
                logical.chunk_configuration
            ),
        )
    )


@serialize_logical_expr.register
def _serialize_recursive_text_chunk_expr(
    logical: RecursiveTextChunkExpr,
    context: SerdeContext,
) -> LogicalExprProto:
    """Serialize a recursive text chunk expression."""
    return LogicalExprProto(
        recursive_text_chunk=RecursiveTextChunkExprProto(
            input_expr=context.serialize_logical_expr(
                "expr", logical.input_expr
            ),
            configuration=context.serialize_recursive_text_chunk_expr_configuration(
                logical.chunking_configuration
            ),
        )
    )


@serialize_logical_expr.register
def _serialize_count_tokens_expr(logical: CountTokensExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a count tokens expression."""
    return LogicalExprProto(
        count_tokens=CountTokensExprProto(
            input_expr=context.serialize_logical_expr(
                "expr", logical.input_expr
            )
        )
    )


@serialize_logical_expr.register
def _serialize_concat_expr(logical: ConcatExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a concat expression."""
    return LogicalExprProto(
        concat=ConcatExprProto(
            exprs=context.serialize_logical_expr_list(
                "exprs", logical.exprs
            )
        )
    )


@serialize_logical_expr.register
def _serialize_array_join_expr(logical: ArrayJoinExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize an array join expression."""
    return LogicalExprProto(
        array_join=ArrayJoinExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            delimiter=logical.delimiter,
        )
    )


@serialize_logical_expr.register
def _serialize_contains_expr(logical: ContainsExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a contains expression."""
    return LogicalExprProto(
        contains=ContainsExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substr=context.serialize_logical_expr("substr", logical.substr),
        )
    )


@serialize_logical_expr.register
def _serialize_contains_any_expr(logical: ContainsAnyExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a contains any expression."""
    return LogicalExprProto(
        contains_any=ContainsAnyExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substrs=logical.substrs,
            case_insensitive=logical.case_insensitive,
        )
    )


@serialize_logical_expr.register
def _serialize_rlike_expr(logical: RLikeExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize an rlike expression."""
    return LogicalExprProto(
        rlike=RLikeExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=logical.pattern,
        )
    )


@serialize_logical_expr.register
def _serialize_like_expr(logical: LikeExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a like expression."""
    return LogicalExprProto(
        like=LikeExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=logical.raw_pattern,
        )
    )


@serialize_logical_expr.register
def _serialize_ilike_expr(logical: ILikeExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize an ilike expression."""
    return LogicalExprProto(
        ilike=ILikeExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=logical.raw_pattern,
        )
    )


@serialize_logical_expr.register
def _serialize_ts_parse_expr(logical: TsParseExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a timestamp parse expression."""
    return LogicalExprProto(
        ts_parse=TsParseExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            format=logical.format,
        )
    )


@serialize_logical_expr.register
def _serialize_starts_with_expr(logical: StartsWithExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a starts with expression."""
    return LogicalExprProto(
        starts_with=StartsWithExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substr=context.serialize_logical_expr("substr", logical.substr),
        )
    )


@serialize_logical_expr.register
def _serialize_ends_with_expr(logical: EndsWithExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize an ends with expression."""
    return LogicalExprProto(
        ends_with=EndsWithExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            substr=context.serialize_logical_expr("substr", logical.substr),
        )
    )


@serialize_logical_expr.register
def _serialize_regexp_split_expr(logical: RegexpSplitExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a regexp split expression."""
    return LogicalExprProto(
        regexp_split=RegexpSplitExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            pattern=logical.pattern,
            limit=logical.limit,
        )
    )


@serialize_logical_expr.register
def _serialize_split_part_expr(logical: SplitPartExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a split part expression."""
    return LogicalExprProto(
        split_part=SplitPartExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            delimiter=context.serialize_logical_expr(
                "delimiter", logical.delimiter
            ),
            part_number=context.serialize_logical_expr(
                "part_number", logical.part_number
            ),
        )
    )


@serialize_logical_expr.register
def _serialize_string_casing_expr(logical: StringCasingExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a string casing expression."""
    return LogicalExprProto(
        string_casing=StringCasingExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            case=logical.case,
        )
    )


@serialize_logical_expr.register
def _serialize_strip_chars_expr(logical: StripCharsExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a strip chars expression."""
    return LogicalExprProto(
        strip_chars=StripCharsExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr),
            chars=context.serialize_logical_expr("chars", logical.chars)
            if logical.chars
            else None,
        )
    )


@serialize_logical_expr.register
def _serialize_replace_expr(logical: ReplaceExpr, context: SerdeContext) -> LogicalExprProto:
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


@serialize_logical_expr.register
def _serialize_str_length_expr(logical: StrLengthExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a string length expression."""
    return LogicalExprProto(
        str_length=StrLengthExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr)
        )
    )


@serialize_logical_expr.register
def _serialize_byte_length_expr(logical: ByteLengthExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a byte length expression."""
    return LogicalExprProto(
        byte_length=ByteLengthExprProto(
            expr=context.serialize_logical_expr("expr", logical.expr)
        )
    )


# Register text expression deserializers
@_deserialize_logical_expr_helper.register
def _deserialize_textract_expr(logical_proto: TextractExprProto, context: SerdeContext) -> TextractExpr:
    """Deserialize a textract expression."""
    return TextractExpr(
        input_expr=context.deserialize_logical_expr(
            "expr", logical_proto.input_expr
        ),
        template=logical_proto.template,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_text_chunk_expr(logical_proto: TextChunkExprProto, context: SerdeContext) -> TextChunkExpr:
    """Deserialize a text chunk expression."""
    return TextChunkExpr(
        input_expr=context.deserialize_logical_expr(
            SerdeContext.EXPR, logical_proto.expr
        ),
        desired_chunk_size=logical_proto.configuration.desired_chunk_size,
        chunk_overlap_percentage=logical_proto.configuration.chunk_overlap_percentage,
        chunk_length_function_name=context.deserialize_enum_value(SerdeContext.CHUNK_LENGTH_FUNCTION_NAME, ChunkLengthFunction, ChunkLengthFunctionProto, logical_proto.configuration.chunk_length_function_name),
    )


@_deserialize_logical_expr_helper.register
def _deserialize_recursive_text_chunk_expr(
    logical_proto: RecursiveTextChunkExprProto,
    context: SerdeContext,
) -> RecursiveTextChunkExpr:
    """Deserialize a recursive text chunk expression."""
    return RecursiveTextChunkExpr(
        input_expr=context.deserialize_logical_expr(
            SerdeContext.EXPR, logical_proto.input_expr
        ),
        desired_chunk_size=logical_proto.configuration.desired_chunk_size,
        chunk_overlap_percentage=logical_proto.configuration.chunk_overlap_percentage,
        chunk_length_function_name=context.deserialize_enum_value("chunk_length_function_name", ChunkLengthFunction, ChunkLengthFunctionProto, logical_proto.configuration.chunk_length_function_name),
        chunking_character_set_name=context.deserialize_enum_value("chunking_character_set_name", ChunkCharacterSet, ChunkCharacterSetProto, logical_proto.configuration.chunking_character_set_name),
        chunking_character_set_custom_characters=logical_proto.configuration.chunking_character_set_custom_characters
        if logical_proto.configuration.chunking_character_set_custom_characters
        else None,
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


@_deserialize_logical_expr_helper.register
def _deserialize_concat_expr(logical_proto: ConcatExprProto, context: SerdeContext) -> ConcatExpr:
    """Deserialize a concat expression."""
    return ConcatExpr(
        exprs=context.deserialize_logical_expr_list(
            SerdeContext.EXPRS, logical_proto.exprs
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_join_expr(logical_proto: ArrayJoinExprProto, context: SerdeContext) -> ArrayJoinExpr:
    """Deserialize an array join expression."""
    return ArrayJoinExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        delimiter=logical_proto.delimiter,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_contains_expr(logical_proto: ContainsExprProto, context: SerdeContext) -> ContainsExpr:
    """Deserialize a contains expression."""
    return ContainsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        substr=context.deserialize_logical_expr(
            SerdeContext.SUBSTR, logical_proto.substr
        ),
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


@_deserialize_logical_expr_helper.register
def _deserialize_rlike_expr(logical_proto: RLikeExprProto, context: SerdeContext) -> RLikeExpr:
    """Deserialize an rlike expression."""
    return RLikeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=logical_proto.pattern,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_like_expr(logical_proto: LikeExprProto, context: SerdeContext) -> LikeExpr:
    """Deserialize a like expression."""
    return LikeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=logical_proto.pattern,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_ilike_expr(logical_proto: ILikeExprProto, context: SerdeContext) -> ILikeExpr:
    """Deserialize an ilike expression."""
    return ILikeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        pattern=logical_proto.pattern,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_ts_parse_expr(logical_proto: TsParseExprProto, context: SerdeContext) -> TsParseExpr:
    """Deserialize a timestamp parse expression."""
    return TsParseExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        format=logical_proto.format,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_starts_with_expr(logical_proto: StartsWithExprProto, context: SerdeContext) -> StartsWithExpr:
    """Deserialize a starts with expression."""
    return StartsWithExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        substr=context.deserialize_logical_expr(
            SerdeContext.SUBSTR, logical_proto.substr
        ),
    )


@_deserialize_logical_expr_helper.register
def _deserialize_ends_with_expr(logical_proto: EndsWithExprProto, context: SerdeContext) -> EndsWithExpr:
    """Deserialize an ends with expression."""
    return EndsWithExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        substr=context.deserialize_logical_expr(
            SerdeContext.SUBSTR, logical_proto.substr
        ),
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


@_deserialize_logical_expr_helper.register
def _deserialize_split_part_expr(logical_proto: SplitPartExprProto, context: SerdeContext) -> SplitPartExpr:
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


@_deserialize_logical_expr_helper.register
def _deserialize_string_casing_expr(
    logical_proto: StringCasingExprProto,
    context: SerdeContext,
) -> StringCasingExpr:
    """Deserialize a string casing expression."""
    return StringCasingExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        case=logical_proto.case,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_strip_chars_expr(logical_proto: StripCharsExprProto, context: SerdeContext) -> StripCharsExpr:
    """Deserialize a strip chars expression."""
    return StripCharsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        chars=context.deserialize_logical_expr("chars", logical_proto.chars)
        if logical_proto.chars
        else None,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_replace_expr(logical_proto: ReplaceExprProto, context: SerdeContext) -> ReplaceExpr:
    """Deserialize a replace expression."""
    return ReplaceExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        search=context.deserialize_logical_expr(
            "search", logical_proto.search
        ),
        replacement=context.deserialize_logical_expr(
            "replacement", logical_proto.replacement
        ),
        literal=logical_proto.literal,
    )


@_deserialize_logical_expr_helper.register
def _deserialize_str_length_expr(logical_proto: StrLengthExprProto, context: SerdeContext) -> StrLengthExpr:
    """Deserialize a string length expression."""
    return StrLengthExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


@_deserialize_logical_expr_helper.register
def _deserialize_byte_length_expr(logical_proto: ByteLengthExprProto, context: SerdeContext) -> ByteLengthExpr:
    """Deserialize a byte length expression."""
    return ByteLengthExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )
