from fenic.gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic.gen.protos.logical_plan.v1 import enums_pb2 as _enums_pb2
from fenic.gen.protos.logical_plan.v1 import expressions_pb2 as _expressions_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogicalPlan(_message.Message):
    __slots__ = ("in_memory_source", "file_source", "table_source", "projection", "filter", "join", "aggregate", "union", "limit", "explode", "drop_duplicates", "sort", "unnest", "sql", "semantic_cluster", "file_sink", "table_sink")
    IN_MEMORY_SOURCE_FIELD_NUMBER: _ClassVar[int]
    FILE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    TABLE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    PROJECTION_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    JOIN_FIELD_NUMBER: _ClassVar[int]
    AGGREGATE_FIELD_NUMBER: _ClassVar[int]
    UNION_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    EXPLODE_FIELD_NUMBER: _ClassVar[int]
    DROP_DUPLICATES_FIELD_NUMBER: _ClassVar[int]
    SORT_FIELD_NUMBER: _ClassVar[int]
    UNNEST_FIELD_NUMBER: _ClassVar[int]
    SQL_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_CLUSTER_FIELD_NUMBER: _ClassVar[int]
    FILE_SINK_FIELD_NUMBER: _ClassVar[int]
    TABLE_SINK_FIELD_NUMBER: _ClassVar[int]
    in_memory_source: InMemorySource
    file_source: FileSource
    table_source: TableSource
    projection: Projection
    filter: Filter
    join: Join
    aggregate: Aggregate
    union: Union
    limit: Limit
    explode: Explode
    drop_duplicates: DropDuplicates
    sort: Sort
    unnest: Unnest
    sql: SQL
    semantic_cluster: SemanticCluster
    file_sink: FileSink
    table_sink: TableSink
    def __init__(self, in_memory_source: _Optional[_Union[InMemorySource, _Mapping]] = ..., file_source: _Optional[_Union[FileSource, _Mapping]] = ..., table_source: _Optional[_Union[TableSource, _Mapping]] = ..., projection: _Optional[_Union[Projection, _Mapping]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., join: _Optional[_Union[Join, _Mapping]] = ..., aggregate: _Optional[_Union[Aggregate, _Mapping]] = ..., union: _Optional[_Union[Union, _Mapping]] = ..., limit: _Optional[_Union[Limit, _Mapping]] = ..., explode: _Optional[_Union[Explode, _Mapping]] = ..., drop_duplicates: _Optional[_Union[DropDuplicates, _Mapping]] = ..., sort: _Optional[_Union[Sort, _Mapping]] = ..., unnest: _Optional[_Union[Unnest, _Mapping]] = ..., sql: _Optional[_Union[SQL, _Mapping]] = ..., semantic_cluster: _Optional[_Union[SemanticCluster, _Mapping]] = ..., file_sink: _Optional[_Union[FileSink, _Mapping]] = ..., table_sink: _Optional[_Union[TableSink, _Mapping]] = ...) -> None: ...

class Schema(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[ColumnField]
    def __init__(self, fields: _Optional[_Iterable[_Union[ColumnField, _Mapping]]] = ...) -> None: ...

class ColumnField(_message.Message):
    __slots__ = ("name", "data_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: _datatypes_pb2.DataType
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ...) -> None: ...

class InMemorySource(_message.Message):
    __slots__ = ("dataframe_data", "schema")
    DATAFRAME_DATA_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    dataframe_data: bytes
    schema: Schema
    def __init__(self, dataframe_data: _Optional[bytes] = ..., schema: _Optional[_Union[Schema, _Mapping]] = ...) -> None: ...

class FileSource(_message.Message):
    __slots__ = ("paths", "format", "schema", "columns")
    PATHS_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    paths: _containers.RepeatedScalarFieldContainer[str]
    format: str
    schema: Schema
    columns: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, paths: _Optional[_Iterable[str]] = ..., format: _Optional[str] = ..., schema: _Optional[_Union[Schema, _Mapping]] = ..., columns: _Optional[_Iterable[str]] = ...) -> None: ...

class TableSource(_message.Message):
    __slots__ = ("table_name",)
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    table_name: str
    def __init__(self, table_name: _Optional[str] = ...) -> None: ...

class Projection(_message.Message):
    __slots__ = ("input", "expressions")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    expressions: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., expressions: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class Filter(_message.Message):
    __slots__ = ("input", "predicate")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    predicate: _expressions_pb2.LogicalExpr
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., predicate: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ...) -> None: ...

class Join(_message.Message):
    __slots__ = ("left", "right", "join_type", "left_keys", "right_keys", "filter")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    JOIN_TYPE_FIELD_NUMBER: _ClassVar[int]
    LEFT_KEYS_FIELD_NUMBER: _ClassVar[int]
    RIGHT_KEYS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    left: LogicalPlan
    right: LogicalPlan
    join_type: str
    left_keys: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    right_keys: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    filter: _expressions_pb2.LogicalExpr
    def __init__(self, left: _Optional[_Union[LogicalPlan, _Mapping]] = ..., right: _Optional[_Union[LogicalPlan, _Mapping]] = ..., join_type: _Optional[str] = ..., left_keys: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ..., right_keys: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ..., filter: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ...) -> None: ...

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
    __slots__ = ("input", "expr")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    expr: _expressions_pb2.LogicalExpr
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., expr: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ...) -> None: ...

class DropDuplicates(_message.Message):
    __slots__ = ("input", "exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

class Sort(_message.Message):
    __slots__ = ("input", "exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExpr]
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExpr, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("input", "expr", "n_clusters", "model_alias")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    N_CLUSTERS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    expr: _expressions_pb2.LogicalExpr
    n_clusters: int
    model_alias: str
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., expr: _Optional[_Union[_expressions_pb2.LogicalExpr, _Mapping]] = ..., n_clusters: _Optional[int] = ..., model_alias: _Optional[str] = ...) -> None: ...

class FileSink(_message.Message):
    __slots__ = ("input", "path", "format", "mode")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    path: str
    format: str
    mode: str
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., path: _Optional[str] = ..., format: _Optional[str] = ..., mode: _Optional[str] = ...) -> None: ...

class TableSink(_message.Message):
    __slots__ = ("input", "table_name", "mode")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlan
    table_name: str
    mode: str
    def __init__(self, input: _Optional[_Union[LogicalPlan, _Mapping]] = ..., table_name: _Optional[str] = ..., mode: _Optional[str] = ...) -> None: ...
