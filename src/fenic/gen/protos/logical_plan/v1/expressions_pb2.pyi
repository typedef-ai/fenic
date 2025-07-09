from fenic.gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic.gen.protos.logical_plan.v1 import enums_pb2 as _enums_pb2
from fenic.gen.protos.logical_plan.v1 import complex_types_pb2 as _complex_types_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogicalExprProto(_message.Message):
    __slots__ = ("column", "literal", "alias", "sort", "index", "array", "struct", "cast", "coalesce", "is_null", "array_length", "array_contains", "arithmetic", "boolan", "equality_comparison", "numeric_comparison", "semantic_map", "semantic_extract", "semantic_pred", "semantic_reduce", "semantic_classify", "analyze_sentiment", "embeddings", "semantic_summarize", "embedding_normalize", "embedding_similarity", "textract", "text_chunk", "recursive_text_chunk", "count_tokens", "concat", "array_join", "contains", "contains_any", "rlike", "like", "ilike", "ts_parse", "starts_with", "ends_with", "regexp_split", "split_part", "string_casing", "strip_chars", "replace", "str_length", "jq", "json_type", "json_contains", "md_to_json", "md_get_code_blocks", "md_generate_toc", "md_extract_header_chunks", "when", "otherwise")
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    LITERAL_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    SORT_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    STRUCT_FIELD_NUMBER: _ClassVar[int]
    CAST_FIELD_NUMBER: _ClassVar[int]
    NOT_FIELD_NUMBER: _ClassVar[int]
    COALESCE_FIELD_NUMBER: _ClassVar[int]
    IN_FIELD_NUMBER: _ClassVar[int]
    IS_NULL_FIELD_NUMBER: _ClassVar[int]
    ARRAY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    ARRAY_CONTAINS_FIELD_NUMBER: _ClassVar[int]
    ARITHMETIC_FIELD_NUMBER: _ClassVar[int]
    BOOLAN_FIELD_NUMBER: _ClassVar[int]
    EQUALITY_COMPARISON_FIELD_NUMBER: _ClassVar[int]
    NUMERIC_COMPARISON_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_MAP_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_EXTRACT_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_PRED_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_REDUCE_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_CLASSIFY_FIELD_NUMBER: _ClassVar[int]
    ANALYZE_SENTIMENT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_SUMMARIZE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    TEXTRACT_FIELD_NUMBER: _ClassVar[int]
    TEXT_CHUNK_FIELD_NUMBER: _ClassVar[int]
    RECURSIVE_TEXT_CHUNK_FIELD_NUMBER: _ClassVar[int]
    COUNT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CONCAT_FIELD_NUMBER: _ClassVar[int]
    ARRAY_JOIN_FIELD_NUMBER: _ClassVar[int]
    CONTAINS_FIELD_NUMBER: _ClassVar[int]
    CONTAINS_ANY_FIELD_NUMBER: _ClassVar[int]
    RLIKE_FIELD_NUMBER: _ClassVar[int]
    LIKE_FIELD_NUMBER: _ClassVar[int]
    ILIKE_FIELD_NUMBER: _ClassVar[int]
    TS_PARSE_FIELD_NUMBER: _ClassVar[int]
    STARTS_WITH_FIELD_NUMBER: _ClassVar[int]
    ENDS_WITH_FIELD_NUMBER: _ClassVar[int]
    REGEXP_SPLIT_FIELD_NUMBER: _ClassVar[int]
    SPLIT_PART_FIELD_NUMBER: _ClassVar[int]
    STRING_CASING_FIELD_NUMBER: _ClassVar[int]
    STRIP_CHARS_FIELD_NUMBER: _ClassVar[int]
    REPLACE_FIELD_NUMBER: _ClassVar[int]
    STR_LENGTH_FIELD_NUMBER: _ClassVar[int]
    JQ_FIELD_NUMBER: _ClassVar[int]
    JSON_TYPE_FIELD_NUMBER: _ClassVar[int]
    JSON_CONTAINS_FIELD_NUMBER: _ClassVar[int]
    MD_TO_JSON_FIELD_NUMBER: _ClassVar[int]
    MD_GET_CODE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    MD_GENERATE_TOC_FIELD_NUMBER: _ClassVar[int]
    MD_EXTRACT_HEADER_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    WHEN_FIELD_NUMBER: _ClassVar[int]
    OTHERWISE_FIELD_NUMBER: _ClassVar[int]
    column: ColumnExprProto
    literal: LiteralExprProto
    alias: AliasExprProto
    sort: SortExprProto
    index: IndexExprProto
    array: ArrayExprProto
    struct: StructExprProto
    cast: CastExprProto
    coalesce: CoalesceExprProto
    is_null: IsNullExprProto
    array_length: ArrayLengthExprProto
    array_contains: ArrayContainsExprProto
    arithmetic: BinaryExprProto
    boolan: BinaryExprProto
    equality_comparison: BinaryExprProto
    numeric_comparison: BinaryExprProto
    semantic_map: SemanticMapExprProto
    semantic_extract: SemanticExtractExprProto
    semantic_pred: SemanticPredExprProto
    semantic_reduce: SemanticReduceExprProto
    semantic_classify: SemanticClassifyExprProto
    analyze_sentiment: AnalyzeSentimentExprProto
    embeddings: EmbeddingsExprProto
    semantic_summarize: SemanticSummarizeExprProto
    embedding_normalize: EmbeddingNormalizeExprProto
    embedding_similarity: EmbeddingSimilarityExprProto
    textract: TextractExprProto
    text_chunk: TextChunkExprProto
    recursive_text_chunk: RecursiveTextChunkExprProto
    count_tokens: CountTokensExprProto
    concat: ConcatExprProto
    array_join: ArrayJoinExprProto
    contains: ContainsExprProto
    contains_any: ContainsAnyExprProto
    rlike: RLikeExprProto
    like: LikeExprProto
    ilike: ILikeExprProto
    ts_parse: TsParseExprProto
    starts_with: StartsWithExprProto
    ends_with: EndsWithExprProto
    regexp_split: RegexpSplitExprProto
    split_part: SplitPartExprProto
    string_casing: StringCasingExprProto
    strip_chars: StripCharsExprProto
    replace: ReplaceExprProto
    str_length: StrLengthExprProto
    jq: JqExprProto
    json_type: JsonTypeExprProto
    json_contains: JsonContainsExprProto
    md_to_json: MdToJsonExprProto
    md_get_code_blocks: MdGetCodeBlocksExprProto
    md_generate_toc: MdGenerateTocExprProto
    md_extract_header_chunks: MdExtractHeaderChunksProto
    when: WhenExprProto
    otherwise: OtherwiseExprProto
    def __init__(self, column: _Optional[_Union[ColumnExprProto, _Mapping]] = ..., literal: _Optional[_Union[LiteralExprProto, _Mapping]] = ..., alias: _Optional[_Union[AliasExprProto, _Mapping]] = ..., sort: _Optional[_Union[SortExprProto, _Mapping]] = ..., index: _Optional[_Union[IndexExprProto, _Mapping]] = ..., array: _Optional[_Union[ArrayExprProto, _Mapping]] = ..., struct: _Optional[_Union[StructExprProto, _Mapping]] = ..., cast: _Optional[_Union[CastExprProto, _Mapping]] = ..., coalesce: _Optional[_Union[CoalesceExprProto, _Mapping]] = ..., is_null: _Optional[_Union[IsNullExprProto, _Mapping]] = ..., array_length: _Optional[_Union[ArrayLengthExprProto, _Mapping]] = ..., array_contains: _Optional[_Union[ArrayContainsExprProto, _Mapping]] = ..., arithmetic: _Optional[_Union[BinaryExprProto, _Mapping]] = ..., boolan: _Optional[_Union[BinaryExprProto, _Mapping]] = ..., equality_comparison: _Optional[_Union[BinaryExprProto, _Mapping]] = ..., numeric_comparison: _Optional[_Union[BinaryExprProto, _Mapping]] = ..., semantic_map: _Optional[_Union[SemanticMapExprProto, _Mapping]] = ..., semantic_extract: _Optional[_Union[SemanticExtractExprProto, _Mapping]] = ..., semantic_pred: _Optional[_Union[SemanticPredExprProto, _Mapping]] = ..., semantic_reduce: _Optional[_Union[SemanticReduceExprProto, _Mapping]] = ..., semantic_classify: _Optional[_Union[SemanticClassifyExprProto, _Mapping]] = ..., analyze_sentiment: _Optional[_Union[AnalyzeSentimentExprProto, _Mapping]] = ..., embeddings: _Optional[_Union[EmbeddingsExprProto, _Mapping]] = ..., semantic_summarize: _Optional[_Union[SemanticSummarizeExprProto, _Mapping]] = ..., embedding_normalize: _Optional[_Union[EmbeddingNormalizeExprProto, _Mapping]] = ..., embedding_similarity: _Optional[_Union[EmbeddingSimilarityExprProto, _Mapping]] = ..., textract: _Optional[_Union[TextractExprProto, _Mapping]] = ..., text_chunk: _Optional[_Union[TextChunkExprProto, _Mapping]] = ..., recursive_text_chunk: _Optional[_Union[RecursiveTextChunkExprProto, _Mapping]] = ..., count_tokens: _Optional[_Union[CountTokensExprProto, _Mapping]] = ..., concat: _Optional[_Union[ConcatExprProto, _Mapping]] = ..., array_join: _Optional[_Union[ArrayJoinExprProto, _Mapping]] = ..., contains: _Optional[_Union[ContainsExprProto, _Mapping]] = ..., contains_any: _Optional[_Union[ContainsAnyExprProto, _Mapping]] = ..., rlike: _Optional[_Union[RLikeExprProto, _Mapping]] = ..., like: _Optional[_Union[LikeExprProto, _Mapping]] = ..., ilike: _Optional[_Union[ILikeExprProto, _Mapping]] = ..., ts_parse: _Optional[_Union[TsParseExprProto, _Mapping]] = ..., starts_with: _Optional[_Union[StartsWithExprProto, _Mapping]] = ..., ends_with: _Optional[_Union[EndsWithExprProto, _Mapping]] = ..., regexp_split: _Optional[_Union[RegexpSplitExprProto, _Mapping]] = ..., split_part: _Optional[_Union[SplitPartExprProto, _Mapping]] = ..., string_casing: _Optional[_Union[StringCasingExprProto, _Mapping]] = ..., strip_chars: _Optional[_Union[StripCharsExprProto, _Mapping]] = ..., replace: _Optional[_Union[ReplaceExprProto, _Mapping]] = ..., str_length: _Optional[_Union[StrLengthExprProto, _Mapping]] = ..., jq: _Optional[_Union[JqExprProto, _Mapping]] = ..., json_type: _Optional[_Union[JsonTypeExprProto, _Mapping]] = ..., json_contains: _Optional[_Union[JsonContainsExprProto, _Mapping]] = ..., md_to_json: _Optional[_Union[MdToJsonExprProto, _Mapping]] = ..., md_get_code_blocks: _Optional[_Union[MdGetCodeBlocksExprProto, _Mapping]] = ..., md_generate_toc: _Optional[_Union[MdGenerateTocExprProto, _Mapping]] = ..., md_extract_header_chunks: _Optional[_Union[MdExtractHeaderChunksProto, _Mapping]] = ..., when: _Optional[_Union[WhenExprProto, _Mapping]] = ..., otherwise: _Optional[_Union[OtherwiseExprProto, _Mapping]] = ..., **kwargs) -> None: ...

class ColumnExprProto(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class LiteralExprProto(_message.Message):
    __slots__ = ("string_value", "int_value", "double_value", "bool_value", "bytes_value", "data_type")
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BYTES_VALUE_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    string_value: str
    int_value: int
    double_value: float
    bool_value: bool
    bytes_value: bytes
    data_type: _datatypes_pb2.DataTypeProto
    def __init__(self, string_value: _Optional[str] = ..., int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., bool_value: bool = ..., bytes_value: _Optional[bytes] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataTypeProto, _Mapping]] = ...) -> None: ...

class AliasExprProto(_message.Message):
    __slots__ = ("expr", "name")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    name: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., name: _Optional[str] = ...) -> None: ...

class SortExprProto(_message.Message):
    __slots__ = ("expr", "order")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    order: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., order: _Optional[str] = ...) -> None: ...

class IndexExprProto(_message.Message):
    __slots__ = ("expr", "int_index", "string_index", "expr_index")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    INT_INDEX_FIELD_NUMBER: _ClassVar[int]
    STRING_INDEX_FIELD_NUMBER: _ClassVar[int]
    EXPR_INDEX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    int_index: int
    string_index: str
    expr_index: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., int_index: _Optional[int] = ..., string_index: _Optional[str] = ..., expr_index: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class ArrayExprProto(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExprProto]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExprProto, _Mapping]]] = ...) -> None: ...

class StructExprProto(_message.Message):
    __slots__ = ("exprs", "field_names")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAMES_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExprProto]
    field_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExprProto, _Mapping]]] = ..., field_names: _Optional[_Iterable[str]] = ...) -> None: ...

class CastExprProto(_message.Message):
    __slots__ = ("expr", "dest_type")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DEST_TYPE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    dest_type: _datatypes_pb2.DataTypeProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., dest_type: _Optional[_Union[_datatypes_pb2.DataTypeProto, _Mapping]] = ...) -> None: ...

class NotExprProto(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class CoalesceExprProto(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExprProto]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExprProto, _Mapping]]] = ...) -> None: ...

class InExprProto(_message.Message):
    __slots__ = ("expr", "other")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    other: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., other: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class IsNullExprProto(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class ArrayLengthExprProto(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class ArrayContainsExprProto(_message.Message):
    __slots__ = ("expr", "value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    value: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., value: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class BinaryExprProto(_message.Message):
    __slots__ = ("left", "right", "operator")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExprProto
    right: LogicalExprProto
    operator: _enums_pb2.OperatorProto
    def __init__(self, left: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., right: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., operator: _Optional[_Union[_enums_pb2.OperatorProto, str]] = ...) -> None: ...

class SemanticMapExprProto(_message.Message):
    __slots__ = ("instruction", "exprs", "max_tokens", "temperature", "model_alias", "response_format", "examples")
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    instruction: str
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExprProto]
    max_tokens: int
    temperature: float
    model_alias: str
    response_format: _complex_types_pb2.PydanticModelType
    examples: _complex_types_pb2.MapExampleCollectionProto
    def __init__(self, instruction: _Optional[str] = ..., exprs: _Optional[_Iterable[_Union[LogicalExprProto, _Mapping]]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ..., response_format: _Optional[_Union[_complex_types_pb2.PydanticModelType, _Mapping]] = ..., examples: _Optional[_Union[_complex_types_pb2.MapExampleCollectionProto, _Mapping]] = ...) -> None: ...

class SemanticExtractExprProto(_message.Message):
    __slots__ = ("expr", "schema", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    schema: _complex_types_pb2.PydanticModelType
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., schema: _Optional[_Union[_complex_types_pb2.PydanticModelType, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class SemanticPredExprProto(_message.Message):
    __slots__ = ("expr", "predicate", "temperature", "model_alias", "examples")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    predicate: str
    temperature: float
    model_alias: str
    examples: _complex_types_pb2.PredicateExampleCollectionProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., predicate: _Optional[str] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ..., examples: _Optional[_Union[_complex_types_pb2.PredicateExampleCollectionProto, _Mapping]] = ...) -> None: ...

class SemanticReduceExprProto(_message.Message):
    __slots__ = ("expr", "instruction", "max_tokens", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    instruction: str
    max_tokens: int
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., instruction: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class SemanticClassifyExprProto(_message.Message):
    __slots__ = ("expr", "labels", "temperature", "model_alias", "examples")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    labels: _containers.RepeatedScalarFieldContainer[str]
    temperature: float
    model_alias: str
    examples: _complex_types_pb2.ClassifyExampleCollectionProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., labels: _Optional[_Iterable[str]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ..., examples: _Optional[_Union[_complex_types_pb2.ClassifyExampleCollectionProto, _Mapping]] = ...) -> None: ...

class AnalyzeSentimentExprProto(_message.Message):
    __slots__ = ("expr", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class EmbeddingsExprProto(_message.Message):
    __slots__ = ("exprs", "model_alias")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExprProto]
    model_alias: str
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExprProto, _Mapping]]] = ..., model_alias: _Optional[str] = ...) -> None: ...

class SemanticSummarizeExprProto(_message.Message):
    __slots__ = ("expr", "format", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    format: _complex_types_pb2.SummarizationFormatProto
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., format: _Optional[_Union[_complex_types_pb2.SummarizationFormatProto, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class EmbeddingNormalizeExprProto(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class EmbeddingSimilarityExprProto(_message.Message):
    __slots__ = ("expr", "other_expr", "query_vector", "metric")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_EXPR_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    other_expr: LogicalExprProto
    query_vector: _complex_types_pb2.NumpyArray
    metric: _enums_pb2.SemanticSimilarityMetric
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., other_expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., query_vector: _Optional[_Union[_complex_types_pb2.NumpyArray, _Mapping]] = ..., metric: _Optional[_Union[_enums_pb2.SemanticSimilarityMetric, str]] = ...) -> None: ...

class TextractExprProto(_message.Message):
    __slots__ = ("exprs", "patterns")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    PATTERNS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExprProto]
    patterns: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExprProto, _Mapping]]] = ..., patterns: _Optional[_Iterable[str]] = ...) -> None: ...

class TextChunkExprProto(_message.Message):
    __slots__ = ("expr", "chunk_size", "overlap", "length_function", "character_set")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_SET_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    chunk_size: int
    overlap: int
    length_function: _enums_pb2.ChunkLengthFunctionProto
    character_set: _enums_pb2.ChunkCharacterSetProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., chunk_size: _Optional[int] = ..., overlap: _Optional[int] = ..., length_function: _Optional[_Union[_enums_pb2.ChunkLengthFunctionProto, str]] = ..., character_set: _Optional[_Union[_enums_pb2.ChunkCharacterSetProto, str]] = ...) -> None: ...

class RecursiveTextChunkExprProto(_message.Message):
    __slots__ = ("expr", "chunk_size", "overlap", "length_function", "character_set")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_SET_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    chunk_size: int
    overlap: int
    length_function: _enums_pb2.ChunkLengthFunctionProto
    character_set: _enums_pb2.ChunkCharacterSetProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., chunk_size: _Optional[int] = ..., overlap: _Optional[int] = ..., length_function: _Optional[_Union[_enums_pb2.ChunkLengthFunctionProto, str]] = ..., character_set: _Optional[_Union[_enums_pb2.ChunkCharacterSetProto, str]] = ...) -> None: ...

class CountTokensExprProto(_message.Message):
    __slots__ = ("input_expr", "model_alias")
    INPUT_EXPR_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    input_expr: LogicalExprProto
    model_alias: str
    def __init__(self, input_expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., model_alias: _Optional[str] = ...) -> None: ...

class ConcatExprProto(_message.Message):
    __slots__ = ("exprs", "separator")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    SEPARATOR_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExprProto]
    separator: str
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExprProto, _Mapping]]] = ..., separator: _Optional[str] = ...) -> None: ...

class ArrayJoinExprProto(_message.Message):
    __slots__ = ("expr", "separator")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SEPARATOR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    separator: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., separator: _Optional[str] = ...) -> None: ...

class ContainsExprProto(_message.Message):
    __slots__ = ("expr", "substring", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTRING_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    substring: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., substring: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class ContainsAnyExprProto(_message.Message):
    __slots__ = ("expr", "substrings", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTRINGS_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    substrings: _containers.RepeatedScalarFieldContainer[str]
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., substrings: _Optional[_Iterable[str]] = ..., case_sensitive: bool = ...) -> None: ...

class RLikeExprProto(_message.Message):
    __slots__ = ("expr", "pattern", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    pattern: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., pattern: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class LikeExprProto(_message.Message):
    __slots__ = ("expr", "pattern", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    pattern: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., pattern: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class ILikeExprProto(_message.Message):
    __slots__ = ("expr", "pattern")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    pattern: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., pattern: _Optional[str] = ...) -> None: ...

class TsParseExprProto(_message.Message):
    __slots__ = ("expr", "format")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    format: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., format: _Optional[str] = ...) -> None: ...

class StartsWithExprProto(_message.Message):
    __slots__ = ("expr", "prefix", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    prefix: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., prefix: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class EndsWithExprProto(_message.Message):
    __slots__ = ("expr", "suffix", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    suffix: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., suffix: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class RegexpSplitExprProto(_message.Message):
    __slots__ = ("expr", "pattern", "limit")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    pattern: str
    limit: int
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., pattern: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class SplitPartExprProto(_message.Message):
    __slots__ = ("expr", "delimiter", "index")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DELIMITER_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    delimiter: str
    index: int
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., delimiter: _Optional[str] = ..., index: _Optional[int] = ...) -> None: ...

class StringCasingExprProto(_message.Message):
    __slots__ = ("expr", "case_type")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CASE_TYPE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    case_type: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., case_type: _Optional[str] = ...) -> None: ...

class StripCharsExprProto(_message.Message):
    __slots__ = ("expr", "chars", "side")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CHARS_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    chars: str
    side: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., chars: _Optional[str] = ..., side: _Optional[str] = ...) -> None: ...

class ReplaceExprProto(_message.Message):
    __slots__ = ("expr", "old_value", "new_value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OLD_VALUE_FIELD_NUMBER: _ClassVar[int]
    NEW_VALUE_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    old_value: str
    new_value: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., old_value: _Optional[str] = ..., new_value: _Optional[str] = ..., **kwargs) -> None: ...

class StrLengthExprProto(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class JqExprProto(_message.Message):
    __slots__ = ("expr", "query")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    query: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., query: _Optional[str] = ...) -> None: ...

class JsonTypeExprProto(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class JsonContainsExprProto(_message.Message):
    __slots__ = ("expr", "key")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    key: str
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., key: _Optional[str] = ...) -> None: ...

class MdToJsonExprProto(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class MdGetCodeBlocksExprProto(_message.Message):
    __slots__ = ("expr", "language", "include_text")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_TEXT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    language: str
    include_text: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., language: _Optional[str] = ..., include_text: bool = ...) -> None: ...

class MdGenerateTocExprProto(_message.Message):
    __slots__ = ("expr", "max_depth", "include_links")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_LINKS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    max_depth: int
    include_links: bool
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., max_depth: _Optional[int] = ..., include_links: bool = ...) -> None: ...

class MdExtractHeaderChunksProto(_message.Message):
    __slots__ = ("expr", "max_chunk_size", "overlap")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    MAX_CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    max_chunk_size: int
    overlap: int
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., max_chunk_size: _Optional[int] = ..., overlap: _Optional[int] = ...) -> None: ...

class WhenExprProto(_message.Message):
    __slots__ = ("expr", "condition", "value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    condition: LogicalExprProto
    value: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., condition: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., value: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...

class OtherwiseExprProto(_message.Message):
    __slots__ = ("expr", "value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExprProto
    value: LogicalExprProto
    def __init__(self, expr: _Optional[_Union[LogicalExprProto, _Mapping]] = ..., value: _Optional[_Union[LogicalExprProto, _Mapping]] = ...) -> None: ...
