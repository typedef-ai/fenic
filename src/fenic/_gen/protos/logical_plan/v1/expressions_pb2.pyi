from fenic._gen.protos.logical_plan.v1 import complex_types_pb2 as _complex_types_pb2
from fenic._gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic._gen.protos.logical_plan.v1 import enums_pb2 as _enums_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogicalExpr(_message.Message):
    __slots__ = ("column", "literal", "alias", "sort", "index", "struct", "cast", "not_expr", "coalesce", "in_expr", "is_null", "unresolved_literal", "array", "array_length", "array_distinct", "array_max", "array_min", "array_sort", "array_reverse", "array_remove", "array_compact", "array_repeat", "flatten", "array_union", "array_intersect", "array_except", "arrays_overlap", "array_contains", "element_at", "array_slice", "greatest", "least", "arithmetic", "boolean", "equality_comparison", "numeric_comparison", "semantic_map", "semantic_extract", "semantic_pred", "semantic_reduce", "semantic_classify", "analyze_sentiment", "embeddings", "semantic_summarize", "semantic_parse_pdf", "embedding_normalize", "embedding_similarity", "textract", "text_chunk", "recursive_text_chunk", "count_tokens", "concat", "array_join", "contains", "contains_any", "rlike", "like", "ilike", "ts_parse", "starts_with", "ends_with", "regexp_split", "split_part", "string_casing", "strip_chars", "replace", "str_length", "byte_length", "jinja", "fuzzy_ratio", "fuzzy_token_sort_ratio", "fuzzy_token_set_ratio", "regexp_count", "regexp_extract", "regexp_extract_all", "regexp_instr", "regexp_substr", "jq", "json_type", "json_contains", "md_to_json", "md_get_code_blocks", "md_generate_toc", "md_extract_header_chunks", "when", "otherwise", "sum", "avg", "count", "max", "min", "first", "list", "std_dev", "count_distinct", "approx_count_distinct", "sum_distinct", "year", "month", "day", "hour", "minute", "second", "millisecond", "date_add", "timestamp_add", "date_trunc", "date_diff", "timestamp_diff", "now", "to_date", "to_timestamp", "date_format", "to_utc_timestamp", "from_utc_timestamp", "series_literal")
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    LITERAL_FIELD_NUMBER: _ClassVar[int]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    SORT_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    STRUCT_FIELD_NUMBER: _ClassVar[int]
    CAST_FIELD_NUMBER: _ClassVar[int]
    NOT_EXPR_FIELD_NUMBER: _ClassVar[int]
    COALESCE_FIELD_NUMBER: _ClassVar[int]
    IN_EXPR_FIELD_NUMBER: _ClassVar[int]
    IS_NULL_FIELD_NUMBER: _ClassVar[int]
    UNRESOLVED_LITERAL_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    ARRAY_LENGTH_FIELD_NUMBER: _ClassVar[int]
    ARRAY_DISTINCT_FIELD_NUMBER: _ClassVar[int]
    ARRAY_MAX_FIELD_NUMBER: _ClassVar[int]
    ARRAY_MIN_FIELD_NUMBER: _ClassVar[int]
    ARRAY_SORT_FIELD_NUMBER: _ClassVar[int]
    ARRAY_REVERSE_FIELD_NUMBER: _ClassVar[int]
    ARRAY_REMOVE_FIELD_NUMBER: _ClassVar[int]
    ARRAY_COMPACT_FIELD_NUMBER: _ClassVar[int]
    ARRAY_REPEAT_FIELD_NUMBER: _ClassVar[int]
    FLATTEN_FIELD_NUMBER: _ClassVar[int]
    ARRAY_UNION_FIELD_NUMBER: _ClassVar[int]
    ARRAY_INTERSECT_FIELD_NUMBER: _ClassVar[int]
    ARRAY_EXCEPT_FIELD_NUMBER: _ClassVar[int]
    ARRAYS_OVERLAP_FIELD_NUMBER: _ClassVar[int]
    ARRAY_CONTAINS_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_AT_FIELD_NUMBER: _ClassVar[int]
    ARRAY_SLICE_FIELD_NUMBER: _ClassVar[int]
    GREATEST_FIELD_NUMBER: _ClassVar[int]
    LEAST_FIELD_NUMBER: _ClassVar[int]
    ARITHMETIC_FIELD_NUMBER: _ClassVar[int]
    BOOLEAN_FIELD_NUMBER: _ClassVar[int]
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
    SEMANTIC_PARSE_PDF_FIELD_NUMBER: _ClassVar[int]
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
    BYTE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    JINJA_FIELD_NUMBER: _ClassVar[int]
    FUZZY_RATIO_FIELD_NUMBER: _ClassVar[int]
    FUZZY_TOKEN_SORT_RATIO_FIELD_NUMBER: _ClassVar[int]
    FUZZY_TOKEN_SET_RATIO_FIELD_NUMBER: _ClassVar[int]
    REGEXP_COUNT_FIELD_NUMBER: _ClassVar[int]
    REGEXP_EXTRACT_FIELD_NUMBER: _ClassVar[int]
    REGEXP_EXTRACT_ALL_FIELD_NUMBER: _ClassVar[int]
    REGEXP_INSTR_FIELD_NUMBER: _ClassVar[int]
    REGEXP_SUBSTR_FIELD_NUMBER: _ClassVar[int]
    JQ_FIELD_NUMBER: _ClassVar[int]
    JSON_TYPE_FIELD_NUMBER: _ClassVar[int]
    JSON_CONTAINS_FIELD_NUMBER: _ClassVar[int]
    MD_TO_JSON_FIELD_NUMBER: _ClassVar[int]
    MD_GET_CODE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    MD_GENERATE_TOC_FIELD_NUMBER: _ClassVar[int]
    MD_EXTRACT_HEADER_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    WHEN_FIELD_NUMBER: _ClassVar[int]
    OTHERWISE_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    AVG_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    FIRST_FIELD_NUMBER: _ClassVar[int]
    LIST_FIELD_NUMBER: _ClassVar[int]
    STD_DEV_FIELD_NUMBER: _ClassVar[int]
    COUNT_DISTINCT_FIELD_NUMBER: _ClassVar[int]
    APPROX_COUNT_DISTINCT_FIELD_NUMBER: _ClassVar[int]
    SUM_DISTINCT_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    MONTH_FIELD_NUMBER: _ClassVar[int]
    DAY_FIELD_NUMBER: _ClassVar[int]
    HOUR_FIELD_NUMBER: _ClassVar[int]
    MINUTE_FIELD_NUMBER: _ClassVar[int]
    SECOND_FIELD_NUMBER: _ClassVar[int]
    MILLISECOND_FIELD_NUMBER: _ClassVar[int]
    DATE_ADD_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_ADD_FIELD_NUMBER: _ClassVar[int]
    DATE_TRUNC_FIELD_NUMBER: _ClassVar[int]
    DATE_DIFF_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_DIFF_FIELD_NUMBER: _ClassVar[int]
    NOW_FIELD_NUMBER: _ClassVar[int]
    TO_DATE_FIELD_NUMBER: _ClassVar[int]
    TO_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DATE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    TO_UTC_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    FROM_UTC_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SERIES_LITERAL_FIELD_NUMBER: _ClassVar[int]
    column: ColumnExpr
    literal: LiteralExpr
    alias: AliasExpr
    sort: SortExpr
    index: IndexExpr
    struct: StructExpr
    cast: CastExpr
    not_expr: NotExpr
    coalesce: CoalesceExpr
    in_expr: InExpr
    is_null: IsNullExpr
    unresolved_literal: UnresolvedLiteralExpr
    array: ArrayExpr
    array_length: ArrayLengthExpr
    array_distinct: ArrayDistinctExpr
    array_max: ArrayMaxExpr
    array_min: ArrayMinExpr
    array_sort: ArraySortExpr
    array_reverse: ArrayReverseExpr
    array_remove: ArrayRemoveExpr
    array_compact: ArrayCompactExpr
    array_repeat: ArrayRepeatExpr
    flatten: FlattenExpr
    array_union: ArrayUnionExpr
    array_intersect: ArrayIntersectExpr
    array_except: ArrayExceptExpr
    arrays_overlap: ArraysOverlapExpr
    array_contains: ArrayContainsExpr
    element_at: ElementAtExpr
    array_slice: ArraySliceExpr
    greatest: GreatestExpr
    least: LeastExpr
    arithmetic: ArithmeticExpr
    boolean: BooleanExpr
    equality_comparison: EqualityComparisonExpr
    numeric_comparison: NumericComparisonExpr
    semantic_map: SemanticMapExpr
    semantic_extract: SemanticExtractExpr
    semantic_pred: SemanticPredExpr
    semantic_reduce: SemanticReduceExpr
    semantic_classify: SemanticClassifyExpr
    analyze_sentiment: AnalyzeSentimentExpr
    embeddings: EmbeddingsExpr
    semantic_summarize: SemanticSummarizeExpr
    semantic_parse_pdf: SemanticParsePDFExpr
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
    byte_length: ByteLengthExpr
    jinja: JinjaExpr
    fuzzy_ratio: FuzzyRatioExpr
    fuzzy_token_sort_ratio: FuzzyTokenSortRatioExpr
    fuzzy_token_set_ratio: FuzzyTokenSetRatioExpr
    regexp_count: RegexpCountExpr
    regexp_extract: RegexpExtractExpr
    regexp_extract_all: RegexpExtractAllExpr
    regexp_instr: RegexpInstrExpr
    regexp_substr: RegexpSubstrExpr
    jq: JqExpr
    json_type: JsonTypeExpr
    json_contains: JsonContainsExpr
    md_to_json: MdToJsonExpr
    md_get_code_blocks: MdGetCodeBlocksExpr
    md_generate_toc: MdGenerateTocExpr
    md_extract_header_chunks: MdExtractHeaderChunks
    when: WhenExpr
    otherwise: OtherwiseExpr
    sum: SumExpr
    avg: AvgExpr
    count: CountExpr
    max: MaxExpr
    min: MinExpr
    first: FirstExpr
    list: ListExpr
    std_dev: StdDevExpr
    count_distinct: CountDistinctExpr
    approx_count_distinct: ApproxCountDistinctExpr
    sum_distinct: SumDistinctExpr
    year: YearExpr
    month: MonthExpr
    day: DayExpr
    hour: HourExpr
    minute: MinuteExpr
    second: SecondExpr
    millisecond: MilliSecondExpr
    date_add: DateAddExpr
    timestamp_add: TimestampAddExpr
    date_trunc: DateTruncExpr
    date_diff: DateDiffExpr
    timestamp_diff: TimestampDiffExpr
    now: NowExpr
    to_date: ToDateExpr
    to_timestamp: ToTimestampExpr
    date_format: DateFormatExpr
    to_utc_timestamp: ToUTCTimestampExpr
    from_utc_timestamp: FromUTCTimestampExpr
    series_literal: SeriesLiteralExpr
    def __init__(self, column: _Optional[_Union[ColumnExpr, _Mapping]] = ..., literal: _Optional[_Union[LiteralExpr, _Mapping]] = ..., alias: _Optional[_Union[AliasExpr, _Mapping]] = ..., sort: _Optional[_Union[SortExpr, _Mapping]] = ..., index: _Optional[_Union[IndexExpr, _Mapping]] = ..., struct: _Optional[_Union[StructExpr, _Mapping]] = ..., cast: _Optional[_Union[CastExpr, _Mapping]] = ..., not_expr: _Optional[_Union[NotExpr, _Mapping]] = ..., coalesce: _Optional[_Union[CoalesceExpr, _Mapping]] = ..., in_expr: _Optional[_Union[InExpr, _Mapping]] = ..., is_null: _Optional[_Union[IsNullExpr, _Mapping]] = ..., unresolved_literal: _Optional[_Union[UnresolvedLiteralExpr, _Mapping]] = ..., array: _Optional[_Union[ArrayExpr, _Mapping]] = ..., array_length: _Optional[_Union[ArrayLengthExpr, _Mapping]] = ..., array_distinct: _Optional[_Union[ArrayDistinctExpr, _Mapping]] = ..., array_max: _Optional[_Union[ArrayMaxExpr, _Mapping]] = ..., array_min: _Optional[_Union[ArrayMinExpr, _Mapping]] = ..., array_sort: _Optional[_Union[ArraySortExpr, _Mapping]] = ..., array_reverse: _Optional[_Union[ArrayReverseExpr, _Mapping]] = ..., array_remove: _Optional[_Union[ArrayRemoveExpr, _Mapping]] = ..., array_compact: _Optional[_Union[ArrayCompactExpr, _Mapping]] = ..., array_repeat: _Optional[_Union[ArrayRepeatExpr, _Mapping]] = ..., flatten: _Optional[_Union[FlattenExpr, _Mapping]] = ..., array_union: _Optional[_Union[ArrayUnionExpr, _Mapping]] = ..., array_intersect: _Optional[_Union[ArrayIntersectExpr, _Mapping]] = ..., array_except: _Optional[_Union[ArrayExceptExpr, _Mapping]] = ..., arrays_overlap: _Optional[_Union[ArraysOverlapExpr, _Mapping]] = ..., array_contains: _Optional[_Union[ArrayContainsExpr, _Mapping]] = ..., element_at: _Optional[_Union[ElementAtExpr, _Mapping]] = ..., array_slice: _Optional[_Union[ArraySliceExpr, _Mapping]] = ..., greatest: _Optional[_Union[GreatestExpr, _Mapping]] = ..., least: _Optional[_Union[LeastExpr, _Mapping]] = ..., arithmetic: _Optional[_Union[ArithmeticExpr, _Mapping]] = ..., boolean: _Optional[_Union[BooleanExpr, _Mapping]] = ..., equality_comparison: _Optional[_Union[EqualityComparisonExpr, _Mapping]] = ..., numeric_comparison: _Optional[_Union[NumericComparisonExpr, _Mapping]] = ..., semantic_map: _Optional[_Union[SemanticMapExpr, _Mapping]] = ..., semantic_extract: _Optional[_Union[SemanticExtractExpr, _Mapping]] = ..., semantic_pred: _Optional[_Union[SemanticPredExpr, _Mapping]] = ..., semantic_reduce: _Optional[_Union[SemanticReduceExpr, _Mapping]] = ..., semantic_classify: _Optional[_Union[SemanticClassifyExpr, _Mapping]] = ..., analyze_sentiment: _Optional[_Union[AnalyzeSentimentExpr, _Mapping]] = ..., embeddings: _Optional[_Union[EmbeddingsExpr, _Mapping]] = ..., semantic_summarize: _Optional[_Union[SemanticSummarizeExpr, _Mapping]] = ..., semantic_parse_pdf: _Optional[_Union[SemanticParsePDFExpr, _Mapping]] = ..., embedding_normalize: _Optional[_Union[EmbeddingNormalizeExpr, _Mapping]] = ..., embedding_similarity: _Optional[_Union[EmbeddingSimilarityExpr, _Mapping]] = ..., textract: _Optional[_Union[TextractExpr, _Mapping]] = ..., text_chunk: _Optional[_Union[TextChunkExpr, _Mapping]] = ..., recursive_text_chunk: _Optional[_Union[RecursiveTextChunkExpr, _Mapping]] = ..., count_tokens: _Optional[_Union[CountTokensExpr, _Mapping]] = ..., concat: _Optional[_Union[ConcatExpr, _Mapping]] = ..., array_join: _Optional[_Union[ArrayJoinExpr, _Mapping]] = ..., contains: _Optional[_Union[ContainsExpr, _Mapping]] = ..., contains_any: _Optional[_Union[ContainsAnyExpr, _Mapping]] = ..., rlike: _Optional[_Union[RLikeExpr, _Mapping]] = ..., like: _Optional[_Union[LikeExpr, _Mapping]] = ..., ilike: _Optional[_Union[ILikeExpr, _Mapping]] = ..., ts_parse: _Optional[_Union[TsParseExpr, _Mapping]] = ..., starts_with: _Optional[_Union[StartsWithExpr, _Mapping]] = ..., ends_with: _Optional[_Union[EndsWithExpr, _Mapping]] = ..., regexp_split: _Optional[_Union[RegexpSplitExpr, _Mapping]] = ..., split_part: _Optional[_Union[SplitPartExpr, _Mapping]] = ..., string_casing: _Optional[_Union[StringCasingExpr, _Mapping]] = ..., strip_chars: _Optional[_Union[StripCharsExpr, _Mapping]] = ..., replace: _Optional[_Union[ReplaceExpr, _Mapping]] = ..., str_length: _Optional[_Union[StrLengthExpr, _Mapping]] = ..., byte_length: _Optional[_Union[ByteLengthExpr, _Mapping]] = ..., jinja: _Optional[_Union[JinjaExpr, _Mapping]] = ..., fuzzy_ratio: _Optional[_Union[FuzzyRatioExpr, _Mapping]] = ..., fuzzy_token_sort_ratio: _Optional[_Union[FuzzyTokenSortRatioExpr, _Mapping]] = ..., fuzzy_token_set_ratio: _Optional[_Union[FuzzyTokenSetRatioExpr, _Mapping]] = ..., regexp_count: _Optional[_Union[RegexpCountExpr, _Mapping]] = ..., regexp_extract: _Optional[_Union[RegexpExtractExpr, _Mapping]] = ..., regexp_extract_all: _Optional[_Union[RegexpExtractAllExpr, _Mapping]] = ..., regexp_instr: _Optional[_Union[RegexpInstrExpr, _Mapping]] = ..., regexp_substr: _Optional[_Union[RegexpSubstrExpr, _Mapping]] = ..., jq: _Optional[_Union[JqExpr, _Mapping]] = ..., json_type: _Optional[_Union[JsonTypeExpr, _Mapping]] = ..., json_contains: _Optional[_Union[JsonContainsExpr, _Mapping]] = ..., md_to_json: _Optional[_Union[MdToJsonExpr, _Mapping]] = ..., md_get_code_blocks: _Optional[_Union[MdGetCodeBlocksExpr, _Mapping]] = ..., md_generate_toc: _Optional[_Union[MdGenerateTocExpr, _Mapping]] = ..., md_extract_header_chunks: _Optional[_Union[MdExtractHeaderChunks, _Mapping]] = ..., when: _Optional[_Union[WhenExpr, _Mapping]] = ..., otherwise: _Optional[_Union[OtherwiseExpr, _Mapping]] = ..., sum: _Optional[_Union[SumExpr, _Mapping]] = ..., avg: _Optional[_Union[AvgExpr, _Mapping]] = ..., count: _Optional[_Union[CountExpr, _Mapping]] = ..., max: _Optional[_Union[MaxExpr, _Mapping]] = ..., min: _Optional[_Union[MinExpr, _Mapping]] = ..., first: _Optional[_Union[FirstExpr, _Mapping]] = ..., list: _Optional[_Union[ListExpr, _Mapping]] = ..., std_dev: _Optional[_Union[StdDevExpr, _Mapping]] = ..., count_distinct: _Optional[_Union[CountDistinctExpr, _Mapping]] = ..., approx_count_distinct: _Optional[_Union[ApproxCountDistinctExpr, _Mapping]] = ..., sum_distinct: _Optional[_Union[SumDistinctExpr, _Mapping]] = ..., year: _Optional[_Union[YearExpr, _Mapping]] = ..., month: _Optional[_Union[MonthExpr, _Mapping]] = ..., day: _Optional[_Union[DayExpr, _Mapping]] = ..., hour: _Optional[_Union[HourExpr, _Mapping]] = ..., minute: _Optional[_Union[MinuteExpr, _Mapping]] = ..., second: _Optional[_Union[SecondExpr, _Mapping]] = ..., millisecond: _Optional[_Union[MilliSecondExpr, _Mapping]] = ..., date_add: _Optional[_Union[DateAddExpr, _Mapping]] = ..., timestamp_add: _Optional[_Union[TimestampAddExpr, _Mapping]] = ..., date_trunc: _Optional[_Union[DateTruncExpr, _Mapping]] = ..., date_diff: _Optional[_Union[DateDiffExpr, _Mapping]] = ..., timestamp_diff: _Optional[_Union[TimestampDiffExpr, _Mapping]] = ..., now: _Optional[_Union[NowExpr, _Mapping]] = ..., to_date: _Optional[_Union[ToDateExpr, _Mapping]] = ..., to_timestamp: _Optional[_Union[ToTimestampExpr, _Mapping]] = ..., date_format: _Optional[_Union[DateFormatExpr, _Mapping]] = ..., to_utc_timestamp: _Optional[_Union[ToUTCTimestampExpr, _Mapping]] = ..., from_utc_timestamp: _Optional[_Union[FromUTCTimestampExpr, _Mapping]] = ..., series_literal: _Optional[_Union[SeriesLiteralExpr, _Mapping]] = ...) -> None: ...

class ColumnExpr(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class LiteralExpr(_message.Message):
    __slots__ = ("value", "data_type")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    value: _complex_types_pb2.ScalarValue
    data_type: _datatypes_pb2.DataType
    def __init__(self, value: _Optional[_Union[_complex_types_pb2.ScalarValue, _Mapping]] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ...) -> None: ...

class SeriesLiteralExpr(_message.Message):
    __slots__ = ("series_data", "data_type")
    SERIES_DATA_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    series_data: bytes
    data_type: _datatypes_pb2.DataType
    def __init__(self, series_data: _Optional[bytes] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ...) -> None: ...

class AliasExpr(_message.Message):
    __slots__ = ("expr", "name")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    name: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., name: _Optional[str] = ...) -> None: ...

class SortExpr(_message.Message):
    __slots__ = ("expr", "ascending", "nulls_last")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    ASCENDING_FIELD_NUMBER: _ClassVar[int]
    NULLS_LAST_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    ascending: bool
    nulls_last: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., ascending: bool = ..., nulls_last: bool = ...) -> None: ...

class IndexExpr(_message.Message):
    __slots__ = ("expr", "index")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    index: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., index: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

class StructExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("expr", "is_null")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    IS_NULL_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    is_null: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., is_null: bool = ...) -> None: ...

class ArrayLengthExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayDistinctExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayContainsExpr(_message.Message):
    __slots__ = ("expr", "other")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    other: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., other: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayMaxExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayMinExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArraySortExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayReverseExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayRemoveExpr(_message.Message):
    __slots__ = ("expr", "element")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    element: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., element: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayUnionExpr(_message.Message):
    __slots__ = ("left", "right")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayIntersectExpr(_message.Message):
    __slots__ = ("left", "right")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayExceptExpr(_message.Message):
    __slots__ = ("left", "right")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayCompactExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArrayRepeatExpr(_message.Message):
    __slots__ = ("element", "count")
    ELEMENT_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    element: LogicalExpr
    count: LogicalExpr
    def __init__(self, element: _Optional[_Union[LogicalExpr, _Mapping]] = ..., count: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class FlattenExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArraySliceExpr(_message.Message):
    __slots__ = ("expr", "start", "length")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    start: LogicalExpr
    length: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., start: _Optional[_Union[LogicalExpr, _Mapping]] = ..., length: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ElementAtExpr(_message.Message):
    __slots__ = ("expr", "index")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    index: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., index: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArraysOverlapExpr(_message.Message):
    __slots__ = ("left", "right")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ArithmeticExpr(_message.Message):
    __slots__ = ("left", "right", "operator")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    operator: _enums_pb2.Operator
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ..., operator: _Optional[_Union[_enums_pb2.Operator, str]] = ...) -> None: ...

class BooleanExpr(_message.Message):
    __slots__ = ("left", "right", "operator")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    operator: _enums_pb2.Operator
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ..., operator: _Optional[_Union[_enums_pb2.Operator, str]] = ...) -> None: ...

class NumericComparisonExpr(_message.Message):
    __slots__ = ("left", "right", "operator")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    operator: _enums_pb2.Operator
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ..., operator: _Optional[_Union[_enums_pb2.Operator, str]] = ...) -> None: ...

class EqualityComparisonExpr(_message.Message):
    __slots__ = ("left", "right", "operator")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    left: LogicalExpr
    right: LogicalExpr
    operator: _enums_pb2.Operator
    def __init__(self, left: _Optional[_Union[LogicalExpr, _Mapping]] = ..., right: _Optional[_Union[LogicalExpr, _Mapping]] = ..., operator: _Optional[_Union[_enums_pb2.Operator, str]] = ...) -> None: ...

class SemanticMapExpr(_message.Message):
    __slots__ = ("template", "strict", "exprs", "max_tokens", "temperature", "model_alias", "response_format", "examples")
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    template: str
    strict: bool
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    max_tokens: int
    temperature: float
    model_alias: _complex_types_pb2.ResolvedModelAlias
    response_format: _complex_types_pb2.ResolvedResponseFormat
    examples: _complex_types_pb2.MapExampleCollection
    def __init__(self, template: _Optional[str] = ..., strict: bool = ..., exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ..., response_format: _Optional[_Union[_complex_types_pb2.ResolvedResponseFormat, _Mapping]] = ..., examples: _Optional[_Union[_complex_types_pb2.MapExampleCollection, _Mapping]] = ...) -> None: ...

class SemanticExtractExpr(_message.Message):
    __slots__ = ("expr", "response_format", "max_tokens", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    response_format: _complex_types_pb2.ResolvedResponseFormat
    max_tokens: int
    temperature: float
    model_alias: _complex_types_pb2.ResolvedModelAlias
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., response_format: _Optional[_Union[_complex_types_pb2.ResolvedResponseFormat, _Mapping]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ...) -> None: ...

class SemanticPredExpr(_message.Message):
    __slots__ = ("template", "strict", "exprs", "temperature", "model_alias", "examples")
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    template: str
    strict: bool
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    temperature: float
    model_alias: _complex_types_pb2.ResolvedModelAlias
    examples: _complex_types_pb2.PredicateExampleCollection
    def __init__(self, template: _Optional[str] = ..., strict: bool = ..., exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ..., examples: _Optional[_Union[_complex_types_pb2.PredicateExampleCollection, _Mapping]] = ...) -> None: ...

class SemanticReduceExpr(_message.Message):
    __slots__ = ("instruction", "input_expr", "max_tokens", "temperature", "group_context_exprs", "order_by_exprs", "model_alias")
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    INPUT_EXPR_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    GROUP_CONTEXT_EXPRS_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_EXPRS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    instruction: str
    input_expr: LogicalExpr
    max_tokens: int
    temperature: float
    group_context_exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    order_by_exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    model_alias: _complex_types_pb2.ResolvedModelAlias
    def __init__(self, instruction: _Optional[str] = ..., input_expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., group_context_exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., order_by_exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ...) -> None: ...

class SemanticClassifyExpr(_message.Message):
    __slots__ = ("expr", "classes", "temperature", "model_alias", "examples")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    classes: _containers.RepeatedCompositeFieldContainer[_complex_types_pb2.ResolvedClassDefinition]
    temperature: float
    model_alias: _complex_types_pb2.ResolvedModelAlias
    examples: _complex_types_pb2.ClassifyExampleCollection
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., classes: _Optional[_Iterable[_Union[_complex_types_pb2.ResolvedClassDefinition, _Mapping]]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ..., examples: _Optional[_Union[_complex_types_pb2.ClassifyExampleCollection, _Mapping]] = ...) -> None: ...

class AnalyzeSentimentExpr(_message.Message):
    __slots__ = ("expr", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    temperature: float
    model_alias: _complex_types_pb2.ResolvedModelAlias
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ...) -> None: ...

class EmbeddingsExpr(_message.Message):
    __slots__ = ("expr", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    model_alias: _complex_types_pb2.ResolvedModelAlias
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ...) -> None: ...

class SemanticSummarizeExpr(_message.Message):
    __slots__ = ("expr", "format", "temperature", "model_alias")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    format: _complex_types_pb2.SummarizationFormat
    temperature: float
    model_alias: _complex_types_pb2.ResolvedModelAlias
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., format: _Optional[_Union[_complex_types_pb2.SummarizationFormat, _Mapping]] = ..., temperature: _Optional[float] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ...) -> None: ...

class SemanticParsePDFExpr(_message.Message):
    __slots__ = ("expr", "model_alias", "page_separator", "describe_images")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    PAGE_SEPARATOR_FIELD_NUMBER: _ClassVar[int]
    DESCRIBE_IMAGES_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    model_alias: _complex_types_pb2.ResolvedModelAlias
    page_separator: str
    describe_images: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ..., page_separator: _Optional[str] = ..., describe_images: bool = ...) -> None: ...

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
    metric: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., other_expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., query_vector: _Optional[_Union[_complex_types_pb2.NumpyArray, _Mapping]] = ..., metric: _Optional[str] = ...) -> None: ...

class TextractExpr(_message.Message):
    __slots__ = ("input_expr", "template")
    INPUT_EXPR_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    input_expr: LogicalExpr
    template: str
    def __init__(self, input_expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., template: _Optional[str] = ...) -> None: ...

class TextChunkExpr(_message.Message):
    __slots__ = ("expr", "configuration")
    class TextChunkExprConfiguration(_message.Message):
        __slots__ = ("desired_chunk_size", "chunk_overlap_percentage", "chunk_length_function_name")
        DESIRED_CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
        CHUNK_OVERLAP_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
        CHUNK_LENGTH_FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
        desired_chunk_size: int
        chunk_overlap_percentage: int
        chunk_length_function_name: _enums_pb2.ChunkLengthFunction
        def __init__(self, desired_chunk_size: _Optional[int] = ..., chunk_overlap_percentage: _Optional[int] = ..., chunk_length_function_name: _Optional[_Union[_enums_pb2.ChunkLengthFunction, str]] = ...) -> None: ...
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    configuration: TextChunkExpr.TextChunkExprConfiguration
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., configuration: _Optional[_Union[TextChunkExpr.TextChunkExprConfiguration, _Mapping]] = ...) -> None: ...

class RecursiveTextChunkExpr(_message.Message):
    __slots__ = ("input_expr", "configuration")
    class RecursiveTextChunkExprConfiguration(_message.Message):
        __slots__ = ("desired_chunk_size", "chunk_overlap_percentage", "chunk_length_function_name", "chunking_character_set_name", "chunking_character_set_custom_characters")
        DESIRED_CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
        CHUNK_OVERLAP_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
        CHUNK_LENGTH_FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
        CHUNKING_CHARACTER_SET_NAME_FIELD_NUMBER: _ClassVar[int]
        CHUNKING_CHARACTER_SET_CUSTOM_CHARACTERS_FIELD_NUMBER: _ClassVar[int]
        desired_chunk_size: int
        chunk_overlap_percentage: int
        chunk_length_function_name: _enums_pb2.ChunkLengthFunction
        chunking_character_set_name: _enums_pb2.ChunkCharacterSet
        chunking_character_set_custom_characters: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, desired_chunk_size: _Optional[int] = ..., chunk_overlap_percentage: _Optional[int] = ..., chunk_length_function_name: _Optional[_Union[_enums_pb2.ChunkLengthFunction, str]] = ..., chunking_character_set_name: _Optional[_Union[_enums_pb2.ChunkCharacterSet, str]] = ..., chunking_character_set_custom_characters: _Optional[_Iterable[str]] = ...) -> None: ...
    INPUT_EXPR_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    input_expr: LogicalExpr
    configuration: RecursiveTextChunkExpr.RecursiveTextChunkExprConfiguration
    def __init__(self, input_expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., configuration: _Optional[_Union[RecursiveTextChunkExpr.RecursiveTextChunkExprConfiguration, _Mapping]] = ...) -> None: ...

class CountTokensExpr(_message.Message):
    __slots__ = ("input_expr",)
    INPUT_EXPR_FIELD_NUMBER: _ClassVar[int]
    input_expr: LogicalExpr
    def __init__(self, input_expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ConcatExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

class ArrayJoinExpr(_message.Message):
    __slots__ = ("expr", "delimiter")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DELIMITER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    delimiter: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., delimiter: _Optional[str] = ...) -> None: ...

class ContainsExpr(_message.Message):
    __slots__ = ("expr", "substr")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    substr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., substr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ContainsAnyExpr(_message.Message):
    __slots__ = ("expr", "substrs", "case_insensitive")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTRS_FIELD_NUMBER: _ClassVar[int]
    CASE_INSENSITIVE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    substrs: _containers.RepeatedScalarFieldContainer[str]
    case_insensitive: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., substrs: _Optional[_Iterable[str]] = ..., case_insensitive: bool = ...) -> None: ...

class RLikeExpr(_message.Message):
    __slots__ = ("expr", "pattern")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class LikeExpr(_message.Message):
    __slots__ = ("expr", "pattern")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ILikeExpr(_message.Message):
    __slots__ = ("expr", "pattern")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class TsParseExpr(_message.Message):
    __slots__ = ("expr", "format")
    class TranscriptFormatType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SRT: _ClassVar[TsParseExpr.TranscriptFormatType]
        GENERIC: _ClassVar[TsParseExpr.TranscriptFormatType]
        WEBVTT: _ClassVar[TsParseExpr.TranscriptFormatType]
    SRT: TsParseExpr.TranscriptFormatType
    GENERIC: TsParseExpr.TranscriptFormatType
    WEBVTT: TsParseExpr.TranscriptFormatType
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    format: TsParseExpr.TranscriptFormatType
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., format: _Optional[_Union[TsParseExpr.TranscriptFormatType, str]] = ...) -> None: ...

class StartsWithExpr(_message.Message):
    __slots__ = ("expr", "substr")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    substr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., substr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class EndsWithExpr(_message.Message):
    __slots__ = ("expr", "substr")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SUBSTR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    substr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., substr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class RegexpSplitExpr(_message.Message):
    __slots__ = ("expr", "pattern", "limit")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: str
    limit: int
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class RegexpCountExpr(_message.Message):
    __slots__ = ("expr", "pattern")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class RegexpExtractExpr(_message.Message):
    __slots__ = ("expr", "pattern", "idx")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    idx: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ..., idx: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class RegexpExtractAllExpr(_message.Message):
    __slots__ = ("expr", "pattern", "idx")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    idx: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ..., idx: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class RegexpInstrExpr(_message.Message):
    __slots__ = ("expr", "pattern", "idx")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    IDX_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    idx: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ..., idx: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class RegexpSubstrExpr(_message.Message):
    __slots__ = ("expr", "pattern")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    pattern: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., pattern: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class SplitPartExpr(_message.Message):
    __slots__ = ("expr", "delimiter", "part_number")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DELIMITER_FIELD_NUMBER: _ClassVar[int]
    PART_NUMBER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    delimiter: LogicalExpr
    part_number: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., delimiter: _Optional[_Union[LogicalExpr, _Mapping]] = ..., part_number: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class StringCasingExpr(_message.Message):
    __slots__ = ("expr", "case")
    class StringCasingType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        LOWER: _ClassVar[StringCasingExpr.StringCasingType]
        UPPER: _ClassVar[StringCasingExpr.StringCasingType]
        TITLE: _ClassVar[StringCasingExpr.StringCasingType]
    LOWER: StringCasingExpr.StringCasingType
    UPPER: StringCasingExpr.StringCasingType
    TITLE: StringCasingExpr.StringCasingType
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CASE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    case: StringCasingExpr.StringCasingType
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., case: _Optional[_Union[StringCasingExpr.StringCasingType, str]] = ...) -> None: ...

class StripCharsExpr(_message.Message):
    __slots__ = ("expr", "chars", "side")
    class StripCharsSide(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        LEFT: _ClassVar[StripCharsExpr.StripCharsSide]
        RIGHT: _ClassVar[StripCharsExpr.StripCharsSide]
        BOTH: _ClassVar[StripCharsExpr.StripCharsSide]
    LEFT: StripCharsExpr.StripCharsSide
    RIGHT: StripCharsExpr.StripCharsSide
    BOTH: StripCharsExpr.StripCharsSide
    EXPR_FIELD_NUMBER: _ClassVar[int]
    CHARS_FIELD_NUMBER: _ClassVar[int]
    SIDE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    chars: LogicalExpr
    side: StripCharsExpr.StripCharsSide
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., chars: _Optional[_Union[LogicalExpr, _Mapping]] = ..., side: _Optional[_Union[StripCharsExpr.StripCharsSide, str]] = ...) -> None: ...

class ReplaceExpr(_message.Message):
    __slots__ = ("expr", "search", "replacement", "literal")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    SEARCH_FIELD_NUMBER: _ClassVar[int]
    REPLACEMENT_FIELD_NUMBER: _ClassVar[int]
    LITERAL_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    search: LogicalExpr
    replacement: LogicalExpr
    literal: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., search: _Optional[_Union[LogicalExpr, _Mapping]] = ..., replacement: _Optional[_Union[LogicalExpr, _Mapping]] = ..., literal: bool = ...) -> None: ...

class StrLengthExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ByteLengthExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class JinjaExpr(_message.Message):
    __slots__ = ("exprs", "template", "strict")
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    template: str
    strict: bool
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ..., template: _Optional[str] = ..., strict: bool = ...) -> None: ...

class FuzzyRatioExpr(_message.Message):
    __slots__ = ("expr", "other", "method")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    other: LogicalExpr
    method: _enums_pb2.FuzzySimilarityMethod
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., other: _Optional[_Union[LogicalExpr, _Mapping]] = ..., method: _Optional[_Union[_enums_pb2.FuzzySimilarityMethod, str]] = ...) -> None: ...

class FuzzyTokenSortRatioExpr(_message.Message):
    __slots__ = ("expr", "other", "method")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    other: LogicalExpr
    method: _enums_pb2.FuzzySimilarityMethod
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., other: _Optional[_Union[LogicalExpr, _Mapping]] = ..., method: _Optional[_Union[_enums_pb2.FuzzySimilarityMethod, str]] = ...) -> None: ...

class FuzzyTokenSetRatioExpr(_message.Message):
    __slots__ = ("expr", "other", "method")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    OTHER_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    other: LogicalExpr
    method: _enums_pb2.FuzzySimilarityMethod
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., other: _Optional[_Union[LogicalExpr, _Mapping]] = ..., method: _Optional[_Union[_enums_pb2.FuzzySimilarityMethod, str]] = ...) -> None: ...

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
    __slots__ = ("expr", "value")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    value: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., value: _Optional[str] = ...) -> None: ...

class MdToJsonExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class MdGetCodeBlocksExpr(_message.Message):
    __slots__ = ("expr", "language_filter")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FILTER_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    language_filter: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., language_filter: _Optional[str] = ...) -> None: ...

class MdGenerateTocExpr(_message.Message):
    __slots__ = ("expr", "max_level")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    MAX_LEVEL_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    max_level: int
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., max_level: _Optional[int] = ...) -> None: ...

class MdExtractHeaderChunks(_message.Message):
    __slots__ = ("expr", "header_level")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    HEADER_LEVEL_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    header_level: int
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., header_level: _Optional[int] = ...) -> None: ...

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

class SumExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class AvgExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class CountExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class MaxExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class MinExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class FirstExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class ListExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class StdDevExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class CountDistinctExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

class ApproxCountDistinctExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class SumDistinctExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class GreatestExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

class LeastExpr(_message.Message):
    __slots__ = ("exprs",)
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    exprs: _containers.RepeatedCompositeFieldContainer[LogicalExpr]
    def __init__(self, exprs: _Optional[_Iterable[_Union[LogicalExpr, _Mapping]]] = ...) -> None: ...

class UnresolvedLiteralExpr(_message.Message):
    __slots__ = ("data_type", "parameter_name")
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_NAME_FIELD_NUMBER: _ClassVar[int]
    data_type: _datatypes_pb2.DataType
    parameter_name: str
    def __init__(self, data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ..., parameter_name: _Optional[str] = ...) -> None: ...

class YearExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class MonthExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class DayExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class HourExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class MinuteExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class SecondExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class MilliSecondExpr(_message.Message):
    __slots__ = ("expr",)
    EXPR_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class DateAddExpr(_message.Message):
    __slots__ = ("expr", "days", "sub")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DAYS_FIELD_NUMBER: _ClassVar[int]
    SUB_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    days: LogicalExpr
    sub: bool
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., days: _Optional[_Union[LogicalExpr, _Mapping]] = ..., sub: bool = ...) -> None: ...

class TimestampAddExpr(_message.Message):
    __slots__ = ("expr", "quantity", "unit")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    QUANTITY_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    quantity: LogicalExpr
    unit: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., quantity: _Optional[_Union[LogicalExpr, _Mapping]] = ..., unit: _Optional[str] = ...) -> None: ...

class DateDiffExpr(_message.Message):
    __slots__ = ("end", "start")
    END_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    end: LogicalExpr
    start: LogicalExpr
    def __init__(self, end: _Optional[_Union[LogicalExpr, _Mapping]] = ..., start: _Optional[_Union[LogicalExpr, _Mapping]] = ...) -> None: ...

class TimestampDiffExpr(_message.Message):
    __slots__ = ("start", "end", "unit")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    start: LogicalExpr
    end: LogicalExpr
    unit: str
    def __init__(self, start: _Optional[_Union[LogicalExpr, _Mapping]] = ..., end: _Optional[_Union[LogicalExpr, _Mapping]] = ..., unit: _Optional[str] = ...) -> None: ...

class DateTruncExpr(_message.Message):
    __slots__ = ("expr", "unit")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    unit: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., unit: _Optional[str] = ...) -> None: ...

class NowExpr(_message.Message):
    __slots__ = ("as_date",)
    AS_DATE_FIELD_NUMBER: _ClassVar[int]
    as_date: bool
    def __init__(self, as_date: bool = ...) -> None: ...

class ToDateExpr(_message.Message):
    __slots__ = ("expr", "format")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    format: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., format: _Optional[str] = ...) -> None: ...

class ToTimestampExpr(_message.Message):
    __slots__ = ("expr", "format")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    format: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., format: _Optional[str] = ...) -> None: ...

class DateFormatExpr(_message.Message):
    __slots__ = ("expr", "format")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    format: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., format: _Optional[str] = ...) -> None: ...

class ToUTCTimestampExpr(_message.Message):
    __slots__ = ("expr", "timezone")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    timezone: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., timezone: _Optional[str] = ...) -> None: ...

class FromUTCTimestampExpr(_message.Message):
    __slots__ = ("expr", "timezone")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    expr: LogicalExpr
    timezone: str
    def __init__(self, expr: _Optional[_Union[LogicalExpr, _Mapping]] = ..., timezone: _Optional[str] = ...) -> None: ...
