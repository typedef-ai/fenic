from fenic.gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic.gen.protos.logical_plan.v1 import enums_pb2 as _enums_pb2
from fenic.gen.protos.logical_plan.v1 import complex_types_pb2 as _complex_types_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogicalExpr(_message.Message):
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
    column: ColumnExpr
    literal: LiteralExpr
    alias: AliasExpr
    sort: SortExpr
    index: IndexExpr
    array: ArrayExpr
    struct: StructExpr
    cast: CastExpr
    coalesce: CoalesceExpr
    is_null: IsNullExpr
    array_length: ArrayLengthExpr
    array_contains: ArrayContainsExpr
    arithmetic: BinaryExpr
    boolan: BinaryExpr
    equality_comparison: BinaryExpr
    numeric_comparison: BinaryExpr
    semantic_map: SemanticMapExpr
    semantic_extract: SemanticExtractExpr
    semantic_pred: SemanticPredExpr
    semantic_reduce: SemanticReduceExpr
    semantic_classify: SemanticClassifyExpr
    analyze_sentiment: AnalyzeSentimentExpr
    embeddings: EmbeddingsExpr
    semantic_summarize: SemanticSummarizeExpr
    embedding_normalize: EmbeddingNormalizeExpr
    embedding_similarity: EmbeddingSimilarityExpr
    textract: TextractExpr
    text_chunk: TextChunkExpr
    recursive_text_chunk: RecursiveTextChunkExpr
    count_tokens: CountTokensExpr
    concat: ConcatExpr
    array_join: ArrayJoinExpr
    contains: ContainsExpr
    contains_any: ContainsAnyExpr
    rlike: RLikeExpr
    like: LikeExpr
    ilike: ILikeExpr
    ts_parse: TsParseExpr
    starts_with: StartsWithExpr
    ends_with: EndsWithExpr
    regexp_split: RegexpSplitExpr
    split_part: SplitPartExpr
    string_casing: StringCasingExpr
    strip_chars: StripCharsExpr
    replace: ReplaceExpr
    str_length: StrLengthExpr
    jq: JqExpr
    json_type: JsonTypeExpr
    json_contains: JsonContainsExpr
    md_to_json: MdToJsonExpr
    md_get_code_blocks: MdGetCodeBlocksExpr
    md_generate_toc: MdGenerateTocExpr
    md_extract_header_chunks: MdExtractHeaderChunks
    when: WhenExpr
    otherwise: OtherwiseExpr
    def __init__(self, column: _Optional[_Union[ColumnExpr, _Mapping]] = ..., literal: _Optional[_Union[LiteralExpr, _Mapping]] = ..., alias: _Optional[_Union[AliasExpr, _Mapping]] = ..., sort: _Optional[_Union[SortExpr, _Mapping]] = ..., index: _Optional[_Union[IndexExpr, _Mapping]] = ..., array: _Optional[_Union[ArrayExpr, _Mapping]] = ..., struct: _Optional[_Union[StructExpr, _Mapping]] = ..., cast: _Optional[_Union[CastExpr, _Mapping]] = ..., coalesce: _Optional[_Union[CoalesceExpr, _Mapping]] = ..., is_null: _Optional[_Union[IsNullExpr, _Mapping]] = ..., array_length: _Optional[_Union[ArrayLengthExpr, _Mapping]] = ..., array_contains: _Optional[_Union[ArrayContainsExpr, _Mapping]] = ..., arithmetic: _Optional[_Union[BinaryExpr, _Mapping]] = ..., boolan: _Optional[_Union[BinaryExpr, _Mapping]] = ..., equality_comparison: _Optional[_Union[BinaryExpr, _Mapping]] = ..., numeric_comparison: _Optional[_Union[BinaryExpr, _Mapping]] = ..., semantic_map: _Optional[_Union[SemanticMapExpr, _Mapping]] = ..., semantic_extract: _Optional[_Union[SemanticExtractExpr, _Mapping]] = ..., semantic_pred: _Optional[_Union[SemanticPredExpr, _Mapping]] = ..., semantic_reduce: _Optional[_Union[SemanticReduceExpr, _Mapping]] = ..., semantic_classify: _Optional[_Union[SemanticClassifyExpr, _Mapping]] = ..., analyze_sentiment: _Optional[_Union[AnalyzeSentimentExpr, _Mapping]] = ..., embeddings: _Optional[_Union[EmbeddingsExpr, _Mapping]] = ..., semantic_summarize: _Optional[_Union[SemanticSummarizeExpr, _Mapping]] = ..., embedding_normalize: _Optional[_Union[EmbeddingNormalizeExpr, _Mapping]] = ..., embedding_similarity: _Optional[_Union[EmbeddingSimilarityExpr, _Mapping]] = ..., textract: _Optional[_Union[TextractExpr, _Mapping]] = ..., text_chunk: _Optional[_Union[TextChunkExpr, _Mapping]] = ..., recursive_text_chunk: _Optional[_Union[RecursiveTextChunkExpr, _Mapping]] = ..., count_tokens: _Optional[_Union[CountTokensExpr, _Mapping]] = ..., concat: _Optional[_Union[ConcatExpr, _Mapping]] = ..., array_join: _Optional[_Union[ArrayJoinExpr, _Mapping]] = ..., contains: _Optional[_Union[ContainsExpr, _Mapping]] = ..., contains_any: _Optional[_Union[ContainsAnyExpr, _Mapping]] = ..., rlike: _Optional[_Union[RLikeExpr, _Mapping]] = ..., like: _Optional[_Union[LikeExpr, _Mapping]] = ..., ilike: _Optional[_Union[ILikeExpr, _Mapping]] = ..., ts_parse: _Optional[_Union[TsParseExpr, _Mapping]] = ..., starts_with: _Optional[_Union[StartsWithExpr, _Mapping]] = ..., ends_with: _Optional[_Union[EndsWithExpr, _Mapping]] = ..., regexp_split: _Optional[_Union[RegexpSplitExpr, _Mapping]] = ..., split_part: _Optional[_Union[SplitPartExpr, _Mapping]] = ..., string_casing: _Optional[_Union[StringCasingExpr, _Mapping]] = ..., strip_chars: _Optional[_Union[StripCharsExpr, _Mapping]] = ..., replace: _Optional[_Union[ReplaceExpr, _Mapping]] = ..., str_length: _Optional[_Union[StrLengthExpr, _Mapping]] = ..., jq: _Optional[_Union[JqExpr, _Mapping]] = ..., json_type: _Optional[_Union[JsonTypeExpr, _Mapping]] = ..., json_contains: _Optional[_Union[JsonContainsExpr, _Mapping]] = ..., md_to_json: _Optional[_Union[MdToJsonExpr, _Mapping]] = ..., md_get_code_blocks: _Optional[_Union[MdGetCodeBlocksExpr, _Mapping]] = ..., md_generate_toc: _Optional[_Union[MdGenerateTocExpr, _Mapping]] = ..., md_extract_header_chunks: _Optional[_Union[MdExtractHeaderChunks, _Mapping]] = ..., when: _Optional[_Union[WhenExpr, _Mapping]] = ..., otherwise: _Optional[_Union[OtherwiseExpr, _Mapping]] = ..., **kwargs) -> None: ...

class ColumnExpr(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class LiteralExpr(_message.Message):
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
    data_type: _datatypes_pb2.DataType
    def __init__(self, string_value: _Optional[str] = ..., int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., bool_value: bool = ..., bytes_value: _Optional[bytes] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ...) -> None: ...

class AliasExpr(_message.Message):
    __slots__ = ("expr", "name")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    name: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., name: _Optional[str] = ...) -> None: ...

class SortExpr(_message.Message):
    __slots__ = ("expr", "order")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    order: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., order: _Optional[str] = ...) -> None: ...

class IndexExpr(_message.Message):
    __slots__ = ("expr", "int_index", "string_index", "expr_index")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    INT_INDEX_FIELD_NUMBER: _ClassVar[int]
    STRING_INDEX_FIELD_NUMBER: _ClassVar[int]
    EXPR_INDEX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    int_index: int
    string_index: str
    expr_index: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., int_index: _Optional[int] = ..., string_index: _Optional[str] = ..., expr_index: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

class StructExpr(_message.Message):
    __slots__ = ("exprs", "field_names")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAMES_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    field_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., field_names: _Optional[_Iterable[str]] = ...) -> None: ...

class CastExpr(_message.Message):
    __slots__ = ("expr", "dest_type")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DEST_TYPE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    dest_type: _datatypes_pb2.DataType
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., dest_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ...) -> None: ...

class NotExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class CoalesceExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

class InExpr(_message.Message):
    __slots__ = ("expr", "other")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    other: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., other: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class IsNullExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayLengthExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayContainsExpr(_message.Message):
    __slots__ = ("expr", "value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    value: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., value: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class BinaryExpr(_message.Message):
    __slots__ = ("left", "right", "operator")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    operator: _enums_pb2.Operator
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ..., operator: _Optional[_Union[_enums_pb2.Operator, str]] = ...) -> None: ...

class SemanticMapExpr(_message.Message):
    __slots__ = ("instruction", "exprs", "max_tokens", "temperature", "model_alias", "response_format", "examples")
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    instruction: str
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    max_tokens: int
    temperature: float
    model_alias: str
    response_format: _complex_types_pb2.PydanticModelType
    examples: _complex_types_pb2.MapExampleCollection
    def __init__(self, instruction: _Optional[str] = ..., exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ..., response_format: _Optional[_Union[_complex_types_pb2.PydanticModelType, _Mapping]] = ..., examples: _Optional[_Union[_complex_types_pb2.MapExampleCollection, _Mapping]] = ...) -> None: ...

class SemanticExtractExpr(_message.Message):
    __slots__ = ("expr", "schema", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    schema: _complex_types_pb2.PydanticModelType
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., schema: _Optional[_Union[_complex_types_pb2.PydanticModelType, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class SemanticPredExpr(_message.Message):
    __slots__ = ("expr", "predicate", "temperature", "model_alias", "examples")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    predicate: str
    temperature: float
    model_alias: str
    examples: _complex_types_pb2.PredicateExampleCollection
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., predicate: _Optional[str] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ..., examples: _Optional[_Union[_complex_types_pb2.PredicateExampleCollection, _Mapping]] = ...) -> None: ...

class SemanticReduceExpr(_message.Message):
    __slots__ = ("expr", "instruction", "max_tokens", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    instruction: str
    max_tokens: int
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., instruction: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class SemanticClassifyExpr(_message.Message):
    __slots__ = ("expr", "labels", "temperature", "model_alias", "examples")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    labels: _containers.RepeatedScalarFieldContainer[str]
    temperature: float
    model_alias: str
    examples: _complex_types_pb2.ClassifyExampleCollection
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., labels: _Optional[_Iterable[str]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ..., examples: _Optional[_Union[_complex_types_pb2.ClassifyExampleCollection, _Mapping]] = ...) -> None: ...

class AnalyzeSentimentExpr(_message.Message):
    __slots__ = ("expr", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class EmbeddingsExpr(_message.Message):
    __slots__ = ("exprs", "model_alias")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    model_alias: str
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., model_alias: _Optional[str] = ...) -> None: ...

class SemanticSummarizeExpr(_message.Message):
    __slots__ = ("expr", "format", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    format: _complex_types_pb2.SummarizationFormat
    temperature: float
    model_alias: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., format: _Optional[_Union[_complex_types_pb2.SummarizationFormat, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[str] = ...) -> None: ...

class EmbeddingNormalizeExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class EmbeddingSimilarityExpr(_message.Message):
    __slots__ = ("expr", "other_expr", "query_vector", "metric")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_EXPR_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    other_expr: LogicalExpr
    query_vector: _complex_types_pb2.NumpyArray
    metric: _enums_pb2.SemanticSimilarityMetric
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., other_expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., query_vector: _Optional[_Union[_complex_types_pb2.NumpyArray, _Mapping]] = ..., metric: _Optional[_Union[_enums_pb2.SemanticSimilarityMetric, str]] = ...) -> None: ...

class TextractExpr(_message.Message):
    __slots__ = ("exprs", "patterns")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    PATTERNS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    patterns: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., patterns: _Optional[_Iterable[str]] = ...) -> None: ...

class TextChunkExpr(_message.Message):
    __slots__ = ("expr", "chunk_size", "overlap", "length_function", "character_set")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_SET_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    chunk_size: int
    overlap: int
    length_function: _enums_pb2.ChunkLengthFunction
    character_set: _enums_pb2.ChunkCharacterSet
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., chunk_size: _Optional[int] = ..., overlap: _Optional[int] = ..., length_function: _Optional[_Union[_enums_pb2.ChunkLengthFunction, str]] = ..., character_set: _Optional[_Union[_enums_pb2.ChunkCharacterSet, str]] = ...) -> None: ...

class RecursiveTextChunkExpr(_message.Message):
    __slots__ = ("expr", "chunk_size", "overlap", "length_function", "character_set")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_SET_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    chunk_size: int
    overlap: int
    length_function: _enums_pb2.ChunkLengthFunction
    character_set: _enums_pb2.ChunkCharacterSet
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., chunk_size: _Optional[int] = ..., overlap: _Optional[int] = ..., length_function: _Optional[_Union[_enums_pb2.ChunkLengthFunction, str]] = ..., character_set: _Optional[_Union[_enums_pb2.ChunkCharacterSet, str]] = ...) -> None: ...

class CountTokensExpr(_message.Message):
    __slots__ = ("input_expr", "model_alias")
    INPUT_EXPR_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    input_expr: LogicalExpr
    model_alias: str
    def __init__(self, input_expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., model_alias: _Optional[str] = ...) -> None: ...

class ConcatExpr(_message.Message):
    __slots__ = ("exprs", "separator")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    SEPARATOR_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    separator: str
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., separator: _Optional[str] = ...) -> None: ...

class ArrayJoinExpr(_message.Message):
    __slots__ = ("expr", "separator")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SEPARATOR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    separator: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., separator: _Optional[str] = ...) -> None: ...

class ContainsExpr(_message.Message):
    __slots__ = ("expr", "substring", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTRING_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    substring: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., substring: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class ContainsAnyExpr(_message.Message):
    __slots__ = ("expr", "substrings", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTRINGS_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    substrings: _containers.RepeatedScalarFieldContainer[str]
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., substrings: _Optional[_Iterable[str]] = ..., case_sensitive: bool = ...) -> None: ...

class RLikeExpr(_message.Message):
    __slots__ = ("expr", "pattern", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class LikeExpr(_message.Message):
    __slots__ = ("expr", "pattern", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class ILikeExpr(_message.Message):
    __slots__ = ("expr", "pattern")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[str] = ...) -> None: ...

class TsParseExpr(_message.Message):
    __slots__ = ("expr", "format")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    format: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., format: _Optional[str] = ...) -> None: ...

class StartsWithExpr(_message.Message):
    __slots__ = ("expr", "prefix", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    prefix: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., prefix: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class EndsWithExpr(_message.Message):
    __slots__ = ("expr", "suffix", "case_sensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_FIELD_NUMBER: _ClassVar[int]
    CASE_SENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    suffix: str
    case_sensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., suffix: _Optional[str] = ..., case_sensitive: bool = ...) -> None: ...

class RegexpSplitExpr(_message.Message):
    __slots__ = ("expr", "pattern", "limit")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: str
    limit: int
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class SplitPartExpr(_message.Message):
    __slots__ = ("expr", "delimiter", "index")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DELIMITER_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    delimiter: str
    index: int
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., delimiter: _Optional[str] = ..., index: _Optional[int] = ...) -> None: ...

class StringCasingExpr(_message.Message):
    __slots__ = ("expr", "case_type")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CASE_TYPE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    case_type: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., case_type: _Optional[str] = ...) -> None: ...

class StripCharsExpr(_message.Message):
    __slots__ = ("expr", "chars", "side")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CHARS_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    chars: str
    side: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., chars: _Optional[str] = ..., side: _Optional[str] = ...) -> None: ...

class ReplaceExpr(_message.Message):
    __slots__ = ("expr", "old_value", "new_value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OLD_VALUE_FIELD_NUMBER: _ClassVar[int]
    NEW_VALUE_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    old_value: str
    new_value: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., old_value: _Optional[str] = ..., new_value: _Optional[str] = ..., **kwargs) -> None: ...

class StrLengthExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class JqExpr(_message.Message):
    __slots__ = ("expr", "query")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    query: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., query: _Optional[str] = ...) -> None: ...

class JsonTypeExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class JsonContainsExpr(_message.Message):
    __slots__ = ("expr", "key")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    key: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., key: _Optional[str] = ...) -> None: ...

class MdToJsonExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class MdGetCodeBlocksExpr(_message.Message):
    __slots__ = ("expr", "language", "include_text")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_TEXT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    language: str
    include_text: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., language: _Optional[str] = ..., include_text: bool = ...) -> None: ...

class MdGenerateTocExpr(_message.Message):
    __slots__ = ("expr", "max_depth", "include_links")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_LINKS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    max_depth: int
    include_links: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., max_depth: _Optional[int] = ..., include_links: bool = ...) -> None: ...

class MdExtractHeaderChunks(_message.Message):
    __slots__ = ("expr", "max_chunk_size", "overlap")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    MAX_CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    max_chunk_size: int
    overlap: int
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., max_chunk_size: _Optional[int] = ..., overlap: _Optional[int] = ...) -> None: ...

class WhenExpr(_message.Message):
    __slots__ = ("expr", "condition", "value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    condition: LogicalExpr
    value: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., condition: _Optional[_Union[LogicalExpr, _Mapping]] = ..., value: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class OtherwiseExpr(_message.Message):
    __slots__ = ("expr", "value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    value: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., value: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...
