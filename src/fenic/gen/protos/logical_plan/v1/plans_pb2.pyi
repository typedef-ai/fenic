from fenic.gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic.gen.protos.logical_plan.v1 import enums_pb2 as _enums_pb2
from fenic.gen.protos.logical_plan.v1 import expressions_pb2 as _expressions_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogicalPlanProto(_message.Message):
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
    in_memory_source: InMemorySourceProto
    file_source: FileSourceProto
    table_source: TableSourceProto
    projection: ProjectionProto
    filter: FilterProto
    join: JoinProto
    aggregate: AggregateProto
    union: UnionProto
    limit: LimitProto
    explode: ExplodeProto
    drop_duplicates: DropDuplicatesProto
    sort: SortProto
    unnest: UnnestProto
    sql: SQLProto
    semantic_cluster: SemanticClusterProto
    file_sink: FileSinkProto
    table_sink: TableSinkProto
    def __init__(self, in_memory_source: _Optional[_Union[InMemorySourceProto, _Mapping]] = ..., file_source: _Optional[_Union[FileSourceProto, _Mapping]] = ..., table_source: _Optional[_Union[TableSourceProto, _Mapping]] = ..., projection: _Optional[_Union[ProjectionProto, _Mapping]] = ..., filter: _Optional[_Union[FilterProto, _Mapping]] = ..., join: _Optional[_Union[JoinProto, _Mapping]] = ..., aggregate: _Optional[_Union[AggregateProto, _Mapping]] = ..., union: _Optional[_Union[UnionProto, _Mapping]] = ..., limit: _Optional[_Union[LimitProto, _Mapping]] = ..., explode: _Optional[_Union[ExplodeProto, _Mapping]] = ..., drop_duplicates: _Optional[_Union[DropDuplicatesProto, _Mapping]] = ..., sort: _Optional[_Union[SortProto, _Mapping]] = ..., unnest: _Optional[_Union[UnnestProto, _Mapping]] = ..., sql: _Optional[_Union[SQLProto, _Mapping]] = ..., semantic_cluster: _Optional[_Union[SemanticClusterProto, _Mapping]] = ..., file_sink: _Optional[_Union[FileSinkProto, _Mapping]] = ..., table_sink: _Optional[_Union[TableSinkProto, _Mapping]] = ...) -> None: ...

class SchemaProto(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[ColumnFieldProto]
    def __init__(self, fields: _Optional[_Iterable[_Union[ColumnFieldProto, _Mapping]]] = ...) -> None: ...

class ColumnFieldProto(_message.Message):
    __slots__ = ("name", "data_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: _datatypes_pb2.DataTypeProto
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataTypeProto, _Mapping]] = ...) -> None: ...

class InMemorySourceProto(_message.Message):
    __slots__ = ("dataframe_data", "schema")
    DATAFRAME_DATA_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    dataframe_data: bytes
    schema: SchemaProto
    def __init__(self, dataframe_data: _Optional[bytes] = ..., schema: _Optional[_Union[SchemaProto, _Mapping]] = ...) -> None: ...

class FileSourceProto(_message.Message):
    __slots__ = ("paths", "format", "schema", "columns")
    PATHS_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    paths: _containers.RepeatedScalarFieldContainer[str]
    format: str
    schema: SchemaProto
    columns: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, paths: _Optional[_Iterable[str]] = ..., format: _Optional[str] = ..., schema: _Optional[_Union[SchemaProto, _Mapping]] = ..., columns: _Optional[_Iterable[str]] = ...) -> None: ...

class TableSourceProto(_message.Message):
    __slots__ = ("table_name",)
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    table_name: str
    def __init__(self, table_name: _Optional[str] = ...) -> None: ...

class ProjectionProto(_message.Message):
    __slots__ = ("input", "expressions")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    expressions: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., expressions: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ...) -> None: ...

class FilterProto(_message.Message):
    __slots__ = ("input", "predicate")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    predicate: _expressions_pb2.LogicalExprProto
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., predicate: _Optional[_Union[_expressions_pb2.LogicalExprProto, _Mapping]] = ...) -> None: ...

class JoinProto(_message.Message):
    __slots__ = ("left", "right", "join_type", "left_keys", "right_keys", "filter")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    JOIN_TYPE_FIELD_NUMBER: _ClassVar[int]
    LEFT_KEYS_FIELD_NUMBER: _ClassVar[int]
    RIGHT_KEYS_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    left: LogicalPlanProto
    right: LogicalPlanProto
    join_type: str
    left_keys: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    right_keys: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    filter: _expressions_pb2.LogicalExprProto
    def __init__(self, left: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., right: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., join_type: _Optional[str] = ..., left_keys: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ..., right_keys: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ..., filter: _Optional[_Union[_expressions_pb2.LogicalExprProto, _Mapping]] = ...) -> None: ...

class AggregateProto(_message.Message):
    __slots__ = ("input", "group_exprs", "agg_exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    GROUP_EXPRS_FIELD_NUMBER: _ClassVar[int]
    AGG_EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    group_exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    agg_exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., group_exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ..., agg_exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ...) -> None: ...

class UnionProto(_message.Message):
    __slots__ = ("inputs",)
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.RepeatedCompositeFieldContainer[LogicalPlanProto]
    def __init__(self, inputs: _Optional[_Iterable[_Union[LogicalPlanProto, _Mapping]]] = ...) -> None: ...

class LimitProto(_message.Message):
    __slots__ = ("input", "n")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    n: int
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., n: _Optional[int] = ...) -> None: ...

class ExplodeProto(_message.Message):
    __slots__ = ("input", "expr")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    expr: _expressions_pb2.LogicalExprProto
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., expr: _Optional[_Union[_expressions_pb2.LogicalExprProto, _Mapping]] = ...) -> None: ...

class DropDuplicatesProto(_message.Message):
    __slots__ = ("input", "exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ...) -> None: ...

class SortProto(_message.Message):
    __slots__ = ("input", "exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ...) -> None: ...

class UnnestProto(_message.Message):
    __slots__ = ("input", "exprs")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    exprs: _containers.RepeatedCompositeFieldContainer[_expressions_pb2.LogicalExprProto]
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., exprs: _Optional[_Iterable[_Union[_expressions_pb2.LogicalExprProto, _Mapping]]] = ...) -> None: ...

class SQLProto(_message.Message):
    __slots__ = ("inputs", "template_names", "templated_query")
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    TEMPLATE_NAMES_FIELD_NUMBER: _ClassVar[int]
    TEMPLATED_QUERY_FIELD_NUMBER: _ClassVar[int]
    inputs: _containers.RepeatedCompositeFieldContainer[LogicalPlanProto]
    template_names: _containers.RepeatedScalarFieldContainer[str]
    templated_query: str
    def __init__(self, inputs: _Optional[_Iterable[_Union[LogicalPlanProto, _Mapping]]] = ..., template_names: _Optional[_Iterable[str]] = ..., templated_query: _Optional[str] = ...) -> None: ...

class SemanticClusterProto(_message.Message):
    __slots__ = ("input", "expr", "n_clusters", "model_alias")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    N_CLUSTERS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ALIAS_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    expr: _expressions_pb2.LogicalExprProto
    n_clusters: int
    model_alias: str
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., expr: _Optional[_Union[_expressions_pb2.LogicalExprProto, _Mapping]] = ..., n_clusters: _Optional[int] = ..., model_alias: _Optional[str] = ...) -> None: ...

class FileSinkProto(_message.Message):
    __slots__ = ("input", "path", "format", "mode")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    path: str
    format: str
    mode: str
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., path: _Optional[str] = ..., format: _Optional[str] = ..., mode: _Optional[str] = ...) -> None: ...

class TableSinkProto(_message.Message):
    __slots__ = ("input", "table_name", "mode")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    input: LogicalPlanProto
    table_name: str
    mode: str
    def __init__(self, input: _Optional[_Union[LogicalPlanProto, _Mapping]] = ..., table_name: _Optional[str] = ..., mode: _Optional[str] = ...) -> None: ...
