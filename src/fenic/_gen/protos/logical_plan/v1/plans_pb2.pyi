from fenic._gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic._gen.protos.logical_plan.v1 import enums_pb2 as _enums_pb2
from fenic._gen.protos.logical_plan.v1 import expressions_pb2 as _expressions_pb2
from fenic._gen.protos.logical_plan.v1 import complex_types_pb2 as _complex_types_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FenicSchema(_message.Message):
    __slots__ = ("column_fields",)
    COLUMN_FIELDS_FIELD_NUMBER: _ClassVar[int]
    column_fields: _containers.RepeatedCompositeFieldContainer[ColumnField]
    def __init__(self, column_fields: _Optional[_Iterable[_Union[ColumnField, _Mapping]]] = ...) -> None: ...

class ColumnField(_message.Message):
    __slots__ = ("name", "data_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: _datatypes_pb2.DataType
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ...) -> None: ...

class CacheInfo(_message.Message):
    __slots__ = ("cache_key",)
    CACHE_KEY_FIELD_NUMBER: _ClassVar[int]
    cache_key: str
    def __init__(self, cache_key: _Optional[str] = ...) -> None: ...

class LogicalPlan(_message.Message):
    __slots__ = ("schema", "cache_info", "in_memory_source", "file_source", "table_source", "doc_source", "projection", "filter", "join", "aggregate", "union", "limit", "explode", "explode_with_index", "drop_duplicates", "sort", "unnest", "sql", "semantic_cluster", "semantic_join", "semantic_similarity_join", "file_sink", "table_sink")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    CACHE_INFO_FIELD_NUMBER: _ClassVar[int]
    IN_MEMORY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    FILE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    TABLE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    DOC_SOURCE_FIELD_NUMBER: _ClassVar[int]
    PROJECTION_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    JOIN_FIELD_NUMBER: _ClassVar[int]
    AGGREGATE_FIELD_NUMBER: _ClassVar[int]
    UNION_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    EXPLODE_FIELD_NUMBER: _ClassVar[int]
    EXPLODE_WITH_INDEX_FIELD_NUMBER: _ClassVar[int]
    DROP_DUPLICATES_FIELD_NUMBER: _ClassVar[int]
    SORT_FIELD_NUMBER: _ClassVar[int]
    UNNEST_FIELD_NUMBER: _ClassVar[int]
    SQL_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_CLUSTER_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_JOIN_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_SIMILARITY_JOIN_FIELD_NUMBER: _ClassVar[int]
    FILE_SINK_FIELD_NUMBER: _ClassVar[int]
    TABLE_SINK_FIELD_NUMBER: _ClassVar[int]
    schema: FenicSchema
    cache_info: CacheInfo
    in_memory_source: InMemorySource
    file_source: FileSource
    table_source: TableSource
    doc_source: DocSource
    projection: Projection
    filter: Filter
    join: Join
    aggregate: Aggregate
    union: Union
    limit: Limit
    explode: Explode
    explode_with_index: ExplodeWithIndex
    drop_duplicates: DropDuplicates
    sort: Sort
    unnest: Unnest
    sql: SQL
    semantic_cluster: SemanticCluster
    semantic_join: SemanticJoin
    semantic_similarity_join: SemanticSimilarityJoin
    file_sink: FileSink
    table_sink: TableSink
    def __init__(self, schema: _Optional[_Union[FenicSchema, _Mapping]] = ..., cache_info: _Optional[_Union[CacheInfo, _Mapping]] = ..., in_memory_source: _Optional[_Union[InMemorySource, _Mapping]] = ..., file_source: _Optional[_Union[FileSource, _Mapping]] = ..., table_source: _Optional[_Union[TableSource, _Mapping]] = ..., doc_source: _Optional[_Union[DocSource, _Mapping]] = ..., projection: _Optional[_Union[Projection, _Mapping]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., join: _Optional[_Union[Join, _Mapping]] = ..., aggregate: _Optional[_Union[Aggregate, _Mapping]] = ..., union: _Optional[_Union[Union, _Mapping]] = ..., limit: _Optional[_Union[Limit, _Mapping]] = ..., explode: _Optional[_Union[Explode, _Mapping]] = ..., explode_with_index: _Optional[_Union[ExplodeWithIndex, _Mapping]] = ..., drop_duplicates: _Optional[_Union[DropDuplicates, _Mapping]] = ..., sort: _Optional[_Union[Sort, _Mapping]] = ..., unnest: _Optional[_Union[Unnest, _Mapping]] = ..., sql: _Optional[_Union[SQL, _Mapping]] = ..., semantic_cluster: _Optional[_Union[SemanticCluster, _Mapping]] = ..., semantic_join: _Optional[_Union[SemanticJoin, _Mapping]] = ..., semantic_similarity_join: _Optional[_Union[SemanticSimilarityJoin, _Mapping]] = ..., file_sink: _Optional[_Union[FileSink, _Mapping]] = ..., table_sink: _Optional[_Union[TableSink, _Mapping]] = ...) -> None: ...

class InMemorySource(_message.Message):
    __slots__ = ("source",)
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    source: bytes
    def __init__(self, source: _Optional[bytes] = ...) -> None: ...

class FileSource(_message.Message):
    __slots__ = ("paths", "file_format", "options_merge_schema", "options_schema")
    PATHS_FIELD_NUMBER: _ClassVar[int]
    FILE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_MERGE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    paths: _containers.RepeatedScalarFieldContainer[str]
    file_format: str
    options_merge_schema: bool
    options_schema: FenicSchema
    def __init__(self, paths: _Optional[_Iterable[str]] = ..., file_format: _Optional[str] = ..., options_merge_schema: bool = ..., options_schema: _Optional[_Union[FenicSchema, _Mapping]] = ...) -> None: ...

class TableSource(_message.Message):
    __slots__ = ("table_name",)
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    table_name: str
    def __init__(self, table_name: _Optional[str] = ...) -> None: ...

class DocSource(_message.Message):
    __slots__ = ("paths", "content_type", "exclude", "recursive")
    PATHS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_FIELD_NUMBER: _ClassVar[int]
    RECURSIVE_FIELD_NUMBER: _ClassVar[int]
    paths: _containers.RepeatedScalarFieldContainer[str]
    content_type: _enums_pb2.DocContentType
    exclude: str
    recursive: bool
    def __init__(self, paths: _Optional[_Iterable[str]] = ..., content_type: _Optional[_Union[_enums_pb2.DocContentType, str]] = ..., exclude: _Optional[str] = ..., recursive: bool = ...) -> None: ...

class Projection(_message.Message):
    __slots__ = ("input", "exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class Filter(_message.Message):
    __slots__ = ("input", "predicate")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    predicate: _expressions_pb2.LogicalExpr
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., predicate: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ...) -> None: ...

class Join(_message.Message):
    __slots__ = ("left", "right", "join_type", "left_on", "right_on")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    JOIN_TYPE_FIELD_NUMBER: _ClassVar[int]
    LEFT_ON_FIELD_NUMBER: _ClassVar[int]
    RIGHT_ON_FIELD_NUMBER: _ClassVar[int]
    left: LogicalPlan
    right: LogicalPlan
    join_type: str
    left_on: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    right_on: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, left: _Optional[_Union[LogicalPlan, _Mapping]] = ..., right: _Optional[_Union[LogicalPlan, _Mapping]] = ..., join_type: _Optional[str] = ..., left_on: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ..., right_on: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class SemanticJoin(_message.Message):
    __slots__ = ("left", "right", "left_on", "right_on", "jinja_template", "strict", "temperature", "model_alias", "examples")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    LEFT_ON_FIELD_NUMBER: _ClassVar[int]
    RIGHT_ON_FIELD_NUMBER: _ClassVar[int]
    JINJA_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    left: LogicalPlan
    right: LogicalPlan
    left_on: _expressions_pb2.LogicalExpr
    right_on: _expressions_pb2.LogicalExpr
    jinja_template: str
    strict: bool
    temperature: float
    model_alias: _complex_types_pb2.ResolvedModelAlias
    examples: _complex_types_pb2.JoinExampleCollection
    def __init__(self, left: _Optional[_Union[LogicalPlan, _Mapping]] = ..., right: _Optional[_Union[LogicalPlan, _Mapping]] = ..., left_on: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., right_on: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., jinja_template: _Optional[str] = ..., strict: bool = ..., temperature: _Optional[float] = ..., model_alias: _Optional[_Union[_complex_types_pb2.ResolvedModelAlias, _Mapping]] = ..., examples: _Optional[_Union[_complex_types_pb2.JoinExampleCollection, _Mapping]] = ...) -> None: ...

class SemanticSimilarityJoin(_message.Message):
    __slots__ = ("left", "right", "left_on", "right_on", "k", "similarity_metric", "similarity_score_column")
    class SemanticSimilarityMetric(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        COSINE: _ClassVar[SemanticSimilarityJoin.SemanticSimilarityMetric]
        L2: _ClassVar[SemanticSimilarityJoin.SemanticSimilarityMetric]
        DOT: _ClassVar[SemanticSimilarityJoin.SemanticSimilarityMetric]
    COSINE: SemanticSimilarityJoin.SemanticSimilarityMetric
    L2: SemanticSimilarityJoin.SemanticSimilarityMetric
    DOT: SemanticSimilarityJoin.SemanticSimilarityMetric
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    LEFT_ON_FIELD_NUMBER: _ClassVar[int]
    RIGHT_ON_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_METRIC_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_SCORE_COLUMN_FIELD_NUMBER: _ClassVar[int]
    left: LogicalPlan
    right: LogicalPlan
    left_on: _expressions_pb2.LogicalExpr
    right_on: _expressions_pb2.LogicalExpr
    k: int
    similarity_metric: SemanticSimilarityJoin.SemanticSimilarityMetric
    similarity_score_column: str
    def __init__(self, left: _Optional[_Union[LogicalPlan, _Mapping]] = ..., right: _Optional[_Union[LogicalPlan, _Mapping]] = ..., left_on: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., right_on: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., k: _Optional[int] = ..., similarity_metric: _Optional[_Union[SemanticSimilarityJoin.SemanticSimilarityMetric, str]] = ..., similarity_score_column: _Optional[str] = ...) -> None: ...

class Aggregate(_message.Message):
    __slots__ = ("input", "group_exprs", "agg_exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    GROUP_EXPRS_FIELD_NUMBER: _ClassVar[int]
    AGG_EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    group_exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    agg_exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., group_exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ..., agg_exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class Union(_message.Message):
    __slots__ = ("inputs",)
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.RepeatedCompositeFieldContainer[LogicalPlan]
    def __init__(self, inputs: _Optional[_Iterable[_Union[LogicalPlan, _Mapping]]] = ...) -> None: ...

class Limit(_message.Message):
    __slots__ = ("input", "n")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    n: int
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., n: _Optional[int] = ...) -> None: ...

class Explode(_message.Message):
    __slots__ = ("input", "expr", "keep_null_and_empty")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    KEEP_NULL_AND_EMPTY_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    expr: _expressions_pb2.LogicalExpr
    keep_null_and_empty: bool
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., expr: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., keep_null_and_empty: bool = ...) -> None: ...

class ExplodeWithIndex(_message.Message):
    __slots__ = ("input", "expr", "index_name", "value_name", "keep_null_and_empty")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_NAME_FIELD_NUMBER: _ClassVar[int]
    KEEP_NULL_AND_EMPTY_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    expr: _expressions_pb2.LogicalExpr
    index_name: str
    value_name: str
    keep_null_and_empty: bool
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., expr: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., index_name: _Optional[str] = ..., value_name: _Optional[str] = ..., keep_null_and_empty: bool = ...) -> None: ...

class DropDuplicates(_message.Message):
    __slots__ = ("input", "subset")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    SUBSET_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    subset: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., subset: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class Sort(_message.Message):
    __slots__ = ("input", "sort_exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    SORT_EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    sort_exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., sort_exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class Unnest(_message.Message):
    __slots__ = ("input", "exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class SQL(_message.Message):
    __slots__ = ("inputs", "template_names", "templated_query")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_NAMES_FIELD_NUMBER: _ClassVar[int]
    TEMPLATED_QUERY_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.RepeatedCompositeFieldContainer[LogicalPlan]
    template_names: _containers.RepeatedScalarFieldContainer[str]
    templated_query: str
    def __init__(self, inputs: _Optional[_Iterable[_Union[LogicalPlan, _Mapping]]] = ..., template_names: _Optional[_Iterable[str]] = ..., templated_query: _Optional[str] = ...) -> None: ...

class SemanticCluster(_message.Message):
    __slots__ = ("input", "by_expr", "num_clusters", "max_iter", "num_init", "label_column", "centroid_column")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    BY_EXPR_FIELD_NUMBER: _ClassVar[int]
    NUM_CLUSTERS_FIELD_NUMBER: _ClassVar[int]
    MAX_ITER_FIELD_NUMBER: _ClassVar[int]
    NUM_INIT_FIELD_NUMBER: _ClassVar[int]
    LABEL_COLUMN_FIELD_NUMBER: _ClassVar[int]
    CENTROID_COLUMN_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    by_expr: _expressions_pb2.LogicalExpr
    num_clusters: int
    max_iter: int
    num_init: int
    label_column: str
    centroid_column: str
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., by_expr: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., num_clusters: _Optional[int] = ..., max_iter: _Optional[int] = ..., num_init: _Optional[int] = ..., label_column: _Optional[str] = ..., centroid_column: _Optional[str] = ...) -> None: ...

class FileSink(_message.Message):
    __slots__ = ("child", "sink_type", "path", "mode")
    CHILD_FIELD_NUMBER: _ClassVar[int]
    SINK_TYPE_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    child: LogicalPlan
    sink_type: str
    path: str
    mode: str
    def __init__(self, child: _Optional[_Union[LogicalPlan, _Mapping]] = ..., sink_type: _Optional[str] = ..., path: _Optional[str] = ..., mode: _Optional[str] = ...) -> None: ...

class TableSink(_message.Message):
    __slots__ = ("child", "table_name", "mode")
    CHILD_FIELD_NUMBER: _ClassVar[int]
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    child: LogicalPlan
    table_name: str
    mode: str
    def __init__(self, child: _Optional[_Union[LogicalPlan, _Mapping]] = ..., table_name: _Optional[str] = ..., mode: _Optional[str] = ...) -> None: ...
