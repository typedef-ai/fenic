from google.protobuf import any_pb2 as _any_pb2
from substrait.v1.extensions import extensions_pb2 as _extensions_pb2
from substrait.v1 import type_pb2 as _type_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AggregationPhase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AGGREGATION_PHASE_UNSPECIFIED: _ClassVar[AggregationPhase]
    AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE: _ClassVar[AggregationPhase]
    AGGREGATION_PHASE_INTERMEDIATE_TO_INTERMEDIATE: _ClassVar[AggregationPhase]
    AGGREGATION_PHASE_INITIAL_TO_RESULT: _ClassVar[AggregationPhase]
    AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT: _ClassVar[AggregationPhase]
AGGREGATION_PHASE_UNSPECIFIED: AggregationPhase
AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE: AggregationPhase
AGGREGATION_PHASE_INTERMEDIATE_TO_INTERMEDIATE: AggregationPhase
AGGREGATION_PHASE_INITIAL_TO_RESULT: AggregationPhase
AGGREGATION_PHASE_INTERMEDIATE_TO_RESULT: AggregationPhase

class RelCommon(_message.Message):
    __slots__ = ("direct", "emit", "hint", "advanced_extension")
    class Direct(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Emit(_message.Message):
        __slots__ = ("output_mapping",)
        OUTPUT_MAPPING_FIELD_NUMBER: _ClassVar[int]
        output_mapping: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, output_mapping: _Optional[_Iterable[int]] = ...) -> None: ...
    class Hint(_message.Message):
        __slots__ = ("stats", "constraint", "alias", "output_names", "advanced_extension", "saved_computations", "loaded_computations")
        class ComputationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            COMPUTATION_TYPE_UNSPECIFIED: _ClassVar[RelCommon.Hint.ComputationType]
            COMPUTATION_TYPE_HASHTABLE: _ClassVar[RelCommon.Hint.ComputationType]
            COMPUTATION_TYPE_BLOOM_FILTER: _ClassVar[RelCommon.Hint.ComputationType]
            COMPUTATION_TYPE_UNKNOWN: _ClassVar[RelCommon.Hint.ComputationType]
        COMPUTATION_TYPE_UNSPECIFIED: RelCommon.Hint.ComputationType
        COMPUTATION_TYPE_HASHTABLE: RelCommon.Hint.ComputationType
        COMPUTATION_TYPE_BLOOM_FILTER: RelCommon.Hint.ComputationType
        COMPUTATION_TYPE_UNKNOWN: RelCommon.Hint.ComputationType
        class Stats(_message.Message):
            __slots__ = ("row_count", "record_size", "advanced_extension")
            ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
            RECORD_SIZE_FIELD_NUMBER: _ClassVar[int]
            ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
            row_count: float
            record_size: float
            advanced_extension: _extensions_pb2.AdvancedExtension
            def __init__(self, row_count: _Optional[float] = ..., record_size: _Optional[float] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...
        class RuntimeConstraint(_message.Message):
            __slots__ = ("advanced_extension",)
            ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
            advanced_extension: _extensions_pb2.AdvancedExtension
            def __init__(self, advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...
        class SavedComputation(_message.Message):
            __slots__ = ("computation_id", "type")
            COMPUTATION_ID_FIELD_NUMBER: _ClassVar[int]
            TYPE_FIELD_NUMBER: _ClassVar[int]
            computation_id: int
            type: RelCommon.Hint.ComputationType
            def __init__(self, computation_id: _Optional[int] = ..., type: _Optional[_Union[RelCommon.Hint.ComputationType, str]] = ...) -> None: ...
        class LoadedComputation(_message.Message):
            __slots__ = ("computation_id_reference", "type")
            COMPUTATION_ID_REFERENCE_FIELD_NUMBER: _ClassVar[int]
            TYPE_FIELD_NUMBER: _ClassVar[int]
            computation_id_reference: int
            type: RelCommon.Hint.ComputationType
            def __init__(self, computation_id_reference: _Optional[int] = ..., type: _Optional[_Union[RelCommon.Hint.ComputationType, str]] = ...) -> None: ...
        STATS_FIELD_NUMBER: _ClassVar[int]
        CONSTRAINT_FIELD_NUMBER: _ClassVar[int]
        ALIAS_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_NAMES_FIELD_NUMBER: _ClassVar[int]
        ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
        SAVED_COMPUTATIONS_FIELD_NUMBER: _ClassVar[int]
        LOADED_COMPUTATIONS_FIELD_NUMBER: _ClassVar[int]
        stats: RelCommon.Hint.Stats
        constraint: RelCommon.Hint.RuntimeConstraint
        alias: str
        output_names: _containers.RepeatedScalarFieldContainer[str]
        advanced_extension: _extensions_pb2.AdvancedExtension
        saved_computations: _containers.RepeatedCompositeFieldContainer[RelCommon.Hint.SavedComputation]
        loaded_computations: _containers.RepeatedCompositeFieldContainer[RelCommon.Hint.LoadedComputation]
        def __init__(self, stats: _Optional[_Union[RelCommon.Hint.Stats, _Mapping]] = ..., constraint: _Optional[_Union[RelCommon.Hint.RuntimeConstraint, _Mapping]] = ..., alias: _Optional[str] = ..., output_names: _Optional[_Iterable[str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ..., saved_computations: _Optional[_Iterable[_Union[RelCommon.Hint.SavedComputation, _Mapping]]] = ..., loaded_computations: _Optional[_Iterable[_Union[RelCommon.Hint.LoadedComputation, _Mapping]]] = ...) -> None: ...
    DIRECT_FIELD_NUMBER: _ClassVar[int]
    EMIT_FIELD_NUMBER: _ClassVar[int]
    HINT_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    direct: RelCommon.Direct
    emit: RelCommon.Emit
    hint: RelCommon.Hint
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, direct: _Optional[_Union[RelCommon.Direct, _Mapping]] = ..., emit: _Optional[_Union[RelCommon.Emit, _Mapping]] = ..., hint: _Optional[_Union[RelCommon.Hint, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class ReadRel(_message.Message):
    __slots__ = ("common", "base_schema", "filter", "best_effort_filter", "projection", "advanced_extension", "virtual_table", "local_files", "named_table", "extension_table", "iceberg_table")
    class NamedTable(_message.Message):
        __slots__ = ("names", "advanced_extension")
        NAMES_FIELD_NUMBER: _ClassVar[int]
        ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
        names: _containers.RepeatedScalarFieldContainer[str]
        advanced_extension: _extensions_pb2.AdvancedExtension
        def __init__(self, names: _Optional[_Iterable[str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...
    class IcebergTable(_message.Message):
        __slots__ = ("direct",)
        class MetadataFileRead(_message.Message):
            __slots__ = ("metadata_uri", "snapshot_id", "snapshot_timestamp")
            METADATA_URI_FIELD_NUMBER: _ClassVar[int]
            SNAPSHOT_ID_FIELD_NUMBER: _ClassVar[int]
            SNAPSHOT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
            metadata_uri: str
            snapshot_id: str
            snapshot_timestamp: int
            def __init__(self, metadata_uri: _Optional[str] = ..., snapshot_id: _Optional[str] = ..., snapshot_timestamp: _Optional[int] = ...) -> None: ...
        DIRECT_FIELD_NUMBER: _ClassVar[int]
        direct: ReadRel.IcebergTable.MetadataFileRead
        def __init__(self, direct: _Optional[_Union[ReadRel.IcebergTable.MetadataFileRead, _Mapping]] = ...) -> None: ...
    class VirtualTable(_message.Message):
        __slots__ = ("values", "expressions")
        VALUES_FIELD_NUMBER: _ClassVar[int]
        EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
        values: _containers.RepeatedCompositeFieldContainer[Expression.Literal.Struct]
        expressions: _containers.RepeatedCompositeFieldContainer[Expression.Nested.Struct]
        def __init__(self, values: _Optional[_Iterable[_Union[Expression.Literal.Struct, _Mapping]]] = ..., expressions: _Optional[_Iterable[_Union[Expression.Nested.Struct, _Mapping]]] = ...) -> None: ...
    class ExtensionTable(_message.Message):
        __slots__ = ("detail",)
        DETAIL_FIELD_NUMBER: _ClassVar[int]
        detail: _any_pb2.Any
        def __init__(self, detail: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
    class LocalFiles(_message.Message):
        __slots__ = ("items", "advanced_extension")
        class FileOrFiles(_message.Message):
            __slots__ = ("uri_path", "uri_path_glob", "uri_file", "uri_folder", "partition_index", "start", "length", "parquet", "arrow", "orc", "extension", "dwrf", "text")
            class ParquetReadOptions(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            class ArrowReadOptions(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            class OrcReadOptions(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            class DwrfReadOptions(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            class DelimiterSeparatedTextReadOptions(_message.Message):
                __slots__ = ("field_delimiter", "max_line_size", "quote", "header_lines_to_skip", "escape", "value_treated_as_null")
                FIELD_DELIMITER_FIELD_NUMBER: _ClassVar[int]
                MAX_LINE_SIZE_FIELD_NUMBER: _ClassVar[int]
                QUOTE_FIELD_NUMBER: _ClassVar[int]
                HEADER_LINES_TO_SKIP_FIELD_NUMBER: _ClassVar[int]
                ESCAPE_FIELD_NUMBER: _ClassVar[int]
                VALUE_TREATED_AS_NULL_FIELD_NUMBER: _ClassVar[int]
                field_delimiter: str
                max_line_size: int
                quote: str
                header_lines_to_skip: int
                escape: str
                value_treated_as_null: str
                def __init__(self, field_delimiter: _Optional[str] = ..., max_line_size: _Optional[int] = ..., quote: _Optional[str] = ..., header_lines_to_skip: _Optional[int] = ..., escape: _Optional[str] = ..., value_treated_as_null: _Optional[str] = ...) -> None: ...
            URI_PATH_FIELD_NUMBER: _ClassVar[int]
            URI_PATH_GLOB_FIELD_NUMBER: _ClassVar[int]
            URI_FILE_FIELD_NUMBER: _ClassVar[int]
            URI_FOLDER_FIELD_NUMBER: _ClassVar[int]
            PARTITION_INDEX_FIELD_NUMBER: _ClassVar[int]
            START_FIELD_NUMBER: _ClassVar[int]
            LENGTH_FIELD_NUMBER: _ClassVar[int]
            PARQUET_FIELD_NUMBER: _ClassVar[int]
            ARROW_FIELD_NUMBER: _ClassVar[int]
            ORC_FIELD_NUMBER: _ClassVar[int]
            EXTENSION_FIELD_NUMBER: _ClassVar[int]
            DWRF_FIELD_NUMBER: _ClassVar[int]
            TEXT_FIELD_NUMBER: _ClassVar[int]
            uri_path: str
            uri_path_glob: str
            uri_file: str
            uri_folder: str
            partition_index: int
            start: int
            length: int
            parquet: ReadRel.LocalFiles.FileOrFiles.ParquetReadOptions
            arrow: ReadRel.LocalFiles.FileOrFiles.ArrowReadOptions
            orc: ReadRel.LocalFiles.FileOrFiles.OrcReadOptions
            extension: _any_pb2.Any
            dwrf: ReadRel.LocalFiles.FileOrFiles.DwrfReadOptions
            text: ReadRel.LocalFiles.FileOrFiles.DelimiterSeparatedTextReadOptions
            def __init__(self, uri_path: _Optional[str] = ..., uri_path_glob: _Optional[str] = ..., uri_file: _Optional[str] = ..., uri_folder: _Optional[str] = ..., partition_index: _Optional[int] = ..., start: _Optional[int] = ..., length: _Optional[int] = ..., parquet: _Optional[_Union[ReadRel.LocalFiles.FileOrFiles.ParquetReadOptions, _Mapping]] = ..., arrow: _Optional[_Union[ReadRel.LocalFiles.FileOrFiles.ArrowReadOptions, _Mapping]] = ..., orc: _Optional[_Union[ReadRel.LocalFiles.FileOrFiles.OrcReadOptions, _Mapping]] = ..., extension: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., dwrf: _Optional[_Union[ReadRel.LocalFiles.FileOrFiles.DwrfReadOptions, _Mapping]] = ..., text: _Optional[_Union[ReadRel.LocalFiles.FileOrFiles.DelimiterSeparatedTextReadOptions, _Mapping]] = ...) -> None: ...
        ITEMS_FIELD_NUMBER: _ClassVar[int]
        ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
        items: _containers.RepeatedCompositeFieldContainer[ReadRel.LocalFiles.FileOrFiles]
        advanced_extension: _extensions_pb2.AdvancedExtension
        def __init__(self, items: _Optional[_Iterable[_Union[ReadRel.LocalFiles.FileOrFiles, _Mapping]]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...
    COMMON_FIELD_NUMBER: _ClassVar[int]
    BASE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    BEST_EFFORT_FILTER_FIELD_NUMBER: _ClassVar[int]
    PROJECTION_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    VIRTUAL_TABLE_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FILES_FIELD_NUMBER: _ClassVar[int]
    NAMED_TABLE_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_TABLE_FIELD_NUMBER: _ClassVar[int]
    ICEBERG_TABLE_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    base_schema: _type_pb2.NamedStruct
    filter: Expression
    best_effort_filter: Expression
    projection: Expression.MaskExpression
    advanced_extension: _extensions_pb2.AdvancedExtension
    virtual_table: ReadRel.VirtualTable
    local_files: ReadRel.LocalFiles
    named_table: ReadRel.NamedTable
    extension_table: ReadRel.ExtensionTable
    iceberg_table: ReadRel.IcebergTable
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., base_schema: _Optional[_Union[_type_pb2.NamedStruct, _Mapping]] = ..., filter: _Optional[_Union[Expression, _Mapping]] = ..., best_effort_filter: _Optional[_Union[Expression, _Mapping]] = ..., projection: _Optional[_Union[Expression.MaskExpression, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ..., virtual_table: _Optional[_Union[ReadRel.VirtualTable, _Mapping]] = ..., local_files: _Optional[_Union[ReadRel.LocalFiles, _Mapping]] = ..., named_table: _Optional[_Union[ReadRel.NamedTable, _Mapping]] = ..., extension_table: _Optional[_Union[ReadRel.ExtensionTable, _Mapping]] = ..., iceberg_table: _Optional[_Union[ReadRel.IcebergTable, _Mapping]] = ...) -> None: ...

class ProjectRel(_message.Message):
    __slots__ = ("common", "input", "expressions", "advanced_extension")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    expressions: _containers.RepeatedCompositeFieldContainer[Expression]
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., expressions: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class JoinRel(_message.Message):
    __slots__ = ("common", "left", "right", "expression", "post_join_filter", "type", "advanced_extension")
    class JoinType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOIN_TYPE_UNSPECIFIED: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_INNER: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_OUTER: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_LEFT: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_RIGHT: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_LEFT_SEMI: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_LEFT_ANTI: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_LEFT_SINGLE: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_RIGHT_SEMI: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_RIGHT_ANTI: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_RIGHT_SINGLE: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_LEFT_MARK: _ClassVar[JoinRel.JoinType]
        JOIN_TYPE_RIGHT_MARK: _ClassVar[JoinRel.JoinType]
    JOIN_TYPE_UNSPECIFIED: JoinRel.JoinType
    JOIN_TYPE_INNER: JoinRel.JoinType
    JOIN_TYPE_OUTER: JoinRel.JoinType
    JOIN_TYPE_LEFT: JoinRel.JoinType
    JOIN_TYPE_RIGHT: JoinRel.JoinType
    JOIN_TYPE_LEFT_SEMI: JoinRel.JoinType
    JOIN_TYPE_LEFT_ANTI: JoinRel.JoinType
    JOIN_TYPE_LEFT_SINGLE: JoinRel.JoinType
    JOIN_TYPE_RIGHT_SEMI: JoinRel.JoinType
    JOIN_TYPE_RIGHT_ANTI: JoinRel.JoinType
    JOIN_TYPE_RIGHT_SINGLE: JoinRel.JoinType
    JOIN_TYPE_LEFT_MARK: JoinRel.JoinType
    JOIN_TYPE_RIGHT_MARK: JoinRel.JoinType
    COMMON_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    POST_JOIN_FILTER_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    left: Rel
    right: Rel
    expression: Expression
    post_join_filter: Expression
    type: JoinRel.JoinType
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., left: _Optional[_Union[Rel, _Mapping]] = ..., right: _Optional[_Union[Rel, _Mapping]] = ..., expression: _Optional[_Union[Expression, _Mapping]] = ..., post_join_filter: _Optional[_Union[Expression, _Mapping]] = ..., type: _Optional[_Union[JoinRel.JoinType, str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class CrossRel(_message.Message):
    __slots__ = ("common", "left", "right", "advanced_extension")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    left: Rel
    right: Rel
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., left: _Optional[_Union[Rel, _Mapping]] = ..., right: _Optional[_Union[Rel, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class FetchRel(_message.Message):
    __slots__ = ("common", "input", "offset", "offset_expr", "count", "count_expr", "advanced_extension")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    OFFSET_EXPR_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    COUNT_EXPR_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    offset: int
    offset_expr: Expression
    count: int
    count_expr: Expression
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., offset: _Optional[int] = ..., offset_expr: _Optional[_Union[Expression, _Mapping]] = ..., count: _Optional[int] = ..., count_expr: _Optional[_Union[Expression, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class AggregateRel(_message.Message):
    __slots__ = ("common", "input", "groupings", "measures", "grouping_expressions", "advanced_extension")
    class Grouping(_message.Message):
        __slots__ = ("grouping_expressions", "expression_references")
        GROUPING_EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
        EXPRESSION_REFERENCES_FIELD_NUMBER: _ClassVar[int]
        grouping_expressions: _containers.RepeatedCompositeFieldContainer[Expression]
        expression_references: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, grouping_expressions: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., expression_references: _Optional[_Iterable[int]] = ...) -> None: ...
    class Measure(_message.Message):
        __slots__ = ("measure", "filter")
        MEASURE_FIELD_NUMBER: _ClassVar[int]
        FILTER_FIELD_NUMBER: _ClassVar[int]
        measure: AggregateFunction
        filter: Expression
        def __init__(self, measure: _Optional[_Union[AggregateFunction, _Mapping]] = ..., filter: _Optional[_Union[Expression, _Mapping]] = ...) -> None: ...
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    GROUPINGS_FIELD_NUMBER: _ClassVar[int]
    MEASURES_FIELD_NUMBER: _ClassVar[int]
    GROUPING_EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    groupings: _containers.RepeatedCompositeFieldContainer[AggregateRel.Grouping]
    measures: _containers.RepeatedCompositeFieldContainer[AggregateRel.Measure]
    grouping_expressions: _containers.RepeatedCompositeFieldContainer[Expression]
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., groupings: _Optional[_Iterable[_Union[AggregateRel.Grouping, _Mapping]]] = ..., measures: _Optional[_Iterable[_Union[AggregateRel.Measure, _Mapping]]] = ..., grouping_expressions: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class ConsistentPartitionWindowRel(_message.Message):
    __slots__ = ("common", "input", "window_functions", "partition_expressions", "sorts", "advanced_extension")
    class WindowRelFunction(_message.Message):
        __slots__ = ("function_reference", "arguments", "options", "output_type", "phase", "invocation", "lower_bound", "upper_bound", "bounds_type")
        FUNCTION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
        OPTIONS_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
        PHASE_FIELD_NUMBER: _ClassVar[int]
        INVOCATION_FIELD_NUMBER: _ClassVar[int]
        LOWER_BOUND_FIELD_NUMBER: _ClassVar[int]
        UPPER_BOUND_FIELD_NUMBER: _ClassVar[int]
        BOUNDS_TYPE_FIELD_NUMBER: _ClassVar[int]
        function_reference: int
        arguments: _containers.RepeatedCompositeFieldContainer[FunctionArgument]
        options: _containers.RepeatedCompositeFieldContainer[FunctionOption]
        output_type: _type_pb2.Type
        phase: AggregationPhase
        invocation: AggregateFunction.AggregationInvocation
        lower_bound: Expression.WindowFunction.Bound
        upper_bound: Expression.WindowFunction.Bound
        bounds_type: Expression.WindowFunction.BoundsType
        def __init__(self, function_reference: _Optional[int] = ..., arguments: _Optional[_Iterable[_Union[FunctionArgument, _Mapping]]] = ..., options: _Optional[_Iterable[_Union[FunctionOption, _Mapping]]] = ..., output_type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., phase: _Optional[_Union[AggregationPhase, str]] = ..., invocation: _Optional[_Union[AggregateFunction.AggregationInvocation, str]] = ..., lower_bound: _Optional[_Union[Expression.WindowFunction.Bound, _Mapping]] = ..., upper_bound: _Optional[_Union[Expression.WindowFunction.Bound, _Mapping]] = ..., bounds_type: _Optional[_Union[Expression.WindowFunction.BoundsType, str]] = ...) -> None: ...
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_EXPRESSIONS_FIELD_NUMBER: _ClassVar[int]
    SORTS_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    window_functions: _containers.RepeatedCompositeFieldContainer[ConsistentPartitionWindowRel.WindowRelFunction]
    partition_expressions: _containers.RepeatedCompositeFieldContainer[Expression]
    sorts: _containers.RepeatedCompositeFieldContainer[SortField]
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., window_functions: _Optional[_Iterable[_Union[ConsistentPartitionWindowRel.WindowRelFunction, _Mapping]]] = ..., partition_expressions: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., sorts: _Optional[_Iterable[_Union[SortField, _Mapping]]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class SortRel(_message.Message):
    __slots__ = ("common", "input", "sorts", "advanced_extension")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    SORTS_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    sorts: _containers.RepeatedCompositeFieldContainer[SortField]
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., sorts: _Optional[_Iterable[_Union[SortField, _Mapping]]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class FilterRel(_message.Message):
    __slots__ = ("common", "input", "condition", "advanced_extension")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    condition: Expression
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., condition: _Optional[_Union[Expression, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class SetRel(_message.Message):
    __slots__ = ("common", "inputs", "op", "advanced_extension")
    class SetOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SET_OP_UNSPECIFIED: _ClassVar[SetRel.SetOp]
        SET_OP_MINUS_PRIMARY: _ClassVar[SetRel.SetOp]
        SET_OP_MINUS_PRIMARY_ALL: _ClassVar[SetRel.SetOp]
        SET_OP_MINUS_MULTISET: _ClassVar[SetRel.SetOp]
        SET_OP_INTERSECTION_PRIMARY: _ClassVar[SetRel.SetOp]
        SET_OP_INTERSECTION_MULTISET: _ClassVar[SetRel.SetOp]
        SET_OP_INTERSECTION_MULTISET_ALL: _ClassVar[SetRel.SetOp]
        SET_OP_UNION_DISTINCT: _ClassVar[SetRel.SetOp]
        SET_OP_UNION_ALL: _ClassVar[SetRel.SetOp]
    SET_OP_UNSPECIFIED: SetRel.SetOp
    SET_OP_MINUS_PRIMARY: SetRel.SetOp
    SET_OP_MINUS_PRIMARY_ALL: SetRel.SetOp
    SET_OP_MINUS_MULTISET: SetRel.SetOp
    SET_OP_INTERSECTION_PRIMARY: SetRel.SetOp
    SET_OP_INTERSECTION_MULTISET: SetRel.SetOp
    SET_OP_INTERSECTION_MULTISET_ALL: SetRel.SetOp
    SET_OP_UNION_DISTINCT: SetRel.SetOp
    SET_OP_UNION_ALL: SetRel.SetOp
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    inputs: _containers.RepeatedCompositeFieldContainer[Rel]
    op: SetRel.SetOp
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., inputs: _Optional[_Iterable[_Union[Rel, _Mapping]]] = ..., op: _Optional[_Union[SetRel.SetOp, str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class ExtensionSingleRel(_message.Message):
    __slots__ = ("common", "input", "detail")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    detail: _any_pb2.Any
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., detail: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class ExtensionLeafRel(_message.Message):
    __slots__ = ("common", "detail")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    detail: _any_pb2.Any
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., detail: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class ExtensionMultiRel(_message.Message):
    __slots__ = ("common", "inputs", "detail")
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    inputs: _containers.RepeatedCompositeFieldContainer[Rel]
    detail: _any_pb2.Any
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., inputs: _Optional[_Iterable[_Union[Rel, _Mapping]]] = ..., detail: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class ExchangeRel(_message.Message):
    __slots__ = ("common", "input", "partition_count", "targets", "scatter_by_fields", "single_target", "multi_target", "round_robin", "broadcast", "advanced_extension")
    class ScatterFields(_message.Message):
        __slots__ = ("fields",)
        FIELDS_FIELD_NUMBER: _ClassVar[int]
        fields: _containers.RepeatedCompositeFieldContainer[Expression.FieldReference]
        def __init__(self, fields: _Optional[_Iterable[_Union[Expression.FieldReference, _Mapping]]] = ...) -> None: ...
    class SingleBucketExpression(_message.Message):
        __slots__ = ("expression",)
        EXPRESSION_FIELD_NUMBER: _ClassVar[int]
        expression: Expression
        def __init__(self, expression: _Optional[_Union[Expression, _Mapping]] = ...) -> None: ...
    class MultiBucketExpression(_message.Message):
        __slots__ = ("expression", "constrained_to_count")
        EXPRESSION_FIELD_NUMBER: _ClassVar[int]
        CONSTRAINED_TO_COUNT_FIELD_NUMBER: _ClassVar[int]
        expression: Expression
        constrained_to_count: bool
        def __init__(self, expression: _Optional[_Union[Expression, _Mapping]] = ..., constrained_to_count: bool = ...) -> None: ...
    class Broadcast(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class RoundRobin(_message.Message):
        __slots__ = ("exact",)
        EXACT_FIELD_NUMBER: _ClassVar[int]
        exact: bool
        def __init__(self, exact: bool = ...) -> None: ...
    class ExchangeTarget(_message.Message):
        __slots__ = ("partition_id", "uri", "extended")
        PARTITION_ID_FIELD_NUMBER: _ClassVar[int]
        URI_FIELD_NUMBER: _ClassVar[int]
        EXTENDED_FIELD_NUMBER: _ClassVar[int]
        partition_id: _containers.RepeatedScalarFieldContainer[int]
        uri: str
        extended: _any_pb2.Any
        def __init__(self, partition_id: _Optional[_Iterable[int]] = ..., uri: _Optional[str] = ..., extended: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PARTITION_COUNT_FIELD_NUMBER: _ClassVar[int]
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    SCATTER_BY_FIELDS_FIELD_NUMBER: _ClassVar[int]
    SINGLE_TARGET_FIELD_NUMBER: _ClassVar[int]
    MULTI_TARGET_FIELD_NUMBER: _ClassVar[int]
    ROUND_ROBIN_FIELD_NUMBER: _ClassVar[int]
    BROADCAST_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    partition_count: int
    targets: _containers.RepeatedCompositeFieldContainer[ExchangeRel.ExchangeTarget]
    scatter_by_fields: ExchangeRel.ScatterFields
    single_target: ExchangeRel.SingleBucketExpression
    multi_target: ExchangeRel.MultiBucketExpression
    round_robin: ExchangeRel.RoundRobin
    broadcast: ExchangeRel.Broadcast
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., partition_count: _Optional[int] = ..., targets: _Optional[_Iterable[_Union[ExchangeRel.ExchangeTarget, _Mapping]]] = ..., scatter_by_fields: _Optional[_Union[ExchangeRel.ScatterFields, _Mapping]] = ..., single_target: _Optional[_Union[ExchangeRel.SingleBucketExpression, _Mapping]] = ..., multi_target: _Optional[_Union[ExchangeRel.MultiBucketExpression, _Mapping]] = ..., round_robin: _Optional[_Union[ExchangeRel.RoundRobin, _Mapping]] = ..., broadcast: _Optional[_Union[ExchangeRel.Broadcast, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class ExpandRel(_message.Message):
    __slots__ = ("common", "input", "fields")
    class ExpandField(_message.Message):
        __slots__ = ("switching_field", "consistent_field")
        SWITCHING_FIELD_FIELD_NUMBER: _ClassVar[int]
        CONSISTENT_FIELD_FIELD_NUMBER: _ClassVar[int]
        switching_field: ExpandRel.SwitchingField
        consistent_field: Expression
        def __init__(self, switching_field: _Optional[_Union[ExpandRel.SwitchingField, _Mapping]] = ..., consistent_field: _Optional[_Union[Expression, _Mapping]] = ...) -> None: ...
    class SwitchingField(_message.Message):
        __slots__ = ("duplicates",)
        DUPLICATES_FIELD_NUMBER: _ClassVar[int]
        duplicates: _containers.RepeatedCompositeFieldContainer[Expression]
        def __init__(self, duplicates: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...
    COMMON_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    input: Rel
    fields: _containers.RepeatedCompositeFieldContainer[ExpandRel.ExpandField]
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., fields: _Optional[_Iterable[_Union[ExpandRel.ExpandField, _Mapping]]] = ...) -> None: ...

class RelRoot(_message.Message):
    __slots__ = ("input", "names")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    NAMES_FIELD_NUMBER: _ClassVar[int]
    input: Rel
    names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, input: _Optional[_Union[Rel, _Mapping]] = ..., names: _Optional[_Iterable[str]] = ...) -> None: ...

class Rel(_message.Message):
    __slots__ = ("read", "filter", "fetch", "aggregate", "sort", "join", "project", "set", "extension_single", "extension_multi", "extension_leaf", "cross", "reference", "write", "ddl", "update", "hash_join", "merge_join", "nested_loop_join", "window", "exchange", "expand")
    READ_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    FETCH_FIELD_NUMBER: _ClassVar[int]
    AGGREGATE_FIELD_NUMBER: _ClassVar[int]
    SORT_FIELD_NUMBER: _ClassVar[int]
    JOIN_FIELD_NUMBER: _ClassVar[int]
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    SET_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_SINGLE_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_MULTI_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_LEAF_FIELD_NUMBER: _ClassVar[int]
    CROSS_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    WRITE_FIELD_NUMBER: _ClassVar[int]
    DDL_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    HASH_JOIN_FIELD_NUMBER: _ClassVar[int]
    MERGE_JOIN_FIELD_NUMBER: _ClassVar[int]
    NESTED_LOOP_JOIN_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    EXCHANGE_FIELD_NUMBER: _ClassVar[int]
    EXPAND_FIELD_NUMBER: _ClassVar[int]
    read: ReadRel
    filter: FilterRel
    fetch: FetchRel
    aggregate: AggregateRel
    sort: SortRel
    join: JoinRel
    project: ProjectRel
    set: SetRel
    extension_single: ExtensionSingleRel
    extension_multi: ExtensionMultiRel
    extension_leaf: ExtensionLeafRel
    cross: CrossRel
    reference: ReferenceRel
    write: WriteRel
    ddl: DdlRel
    update: UpdateRel
    hash_join: HashJoinRel
    merge_join: MergeJoinRel
    nested_loop_join: NestedLoopJoinRel
    window: ConsistentPartitionWindowRel
    exchange: ExchangeRel
    expand: ExpandRel
    def __init__(self, read: _Optional[_Union[ReadRel, _Mapping]] = ..., filter: _Optional[_Union[FilterRel, _Mapping]] = ..., fetch: _Optional[_Union[FetchRel, _Mapping]] = ..., aggregate: _Optional[_Union[AggregateRel, _Mapping]] = ..., sort: _Optional[_Union[SortRel, _Mapping]] = ..., join: _Optional[_Union[JoinRel, _Mapping]] = ..., project: _Optional[_Union[ProjectRel, _Mapping]] = ..., set: _Optional[_Union[SetRel, _Mapping]] = ..., extension_single: _Optional[_Union[ExtensionSingleRel, _Mapping]] = ..., extension_multi: _Optional[_Union[ExtensionMultiRel, _Mapping]] = ..., extension_leaf: _Optional[_Union[ExtensionLeafRel, _Mapping]] = ..., cross: _Optional[_Union[CrossRel, _Mapping]] = ..., reference: _Optional[_Union[ReferenceRel, _Mapping]] = ..., write: _Optional[_Union[WriteRel, _Mapping]] = ..., ddl: _Optional[_Union[DdlRel, _Mapping]] = ..., update: _Optional[_Union[UpdateRel, _Mapping]] = ..., hash_join: _Optional[_Union[HashJoinRel, _Mapping]] = ..., merge_join: _Optional[_Union[MergeJoinRel, _Mapping]] = ..., nested_loop_join: _Optional[_Union[NestedLoopJoinRel, _Mapping]] = ..., window: _Optional[_Union[ConsistentPartitionWindowRel, _Mapping]] = ..., exchange: _Optional[_Union[ExchangeRel, _Mapping]] = ..., expand: _Optional[_Union[ExpandRel, _Mapping]] = ...) -> None: ...

class NamedObjectWrite(_message.Message):
    __slots__ = ("names", "advanced_extension")
    NAMES_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    names: _containers.RepeatedScalarFieldContainer[str]
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, names: _Optional[_Iterable[str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class ExtensionObject(_message.Message):
    __slots__ = ("detail",)
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    detail: _any_pb2.Any
    def __init__(self, detail: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class DdlRel(_message.Message):
    __slots__ = ("named_object", "extension_object", "table_schema", "table_defaults", "object", "op", "view_definition", "common", "advanced_extension")
    class DdlObject(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DDL_OBJECT_UNSPECIFIED: _ClassVar[DdlRel.DdlObject]
        DDL_OBJECT_TABLE: _ClassVar[DdlRel.DdlObject]
        DDL_OBJECT_VIEW: _ClassVar[DdlRel.DdlObject]
    DDL_OBJECT_UNSPECIFIED: DdlRel.DdlObject
    DDL_OBJECT_TABLE: DdlRel.DdlObject
    DDL_OBJECT_VIEW: DdlRel.DdlObject
    class DdlOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DDL_OP_UNSPECIFIED: _ClassVar[DdlRel.DdlOp]
        DDL_OP_CREATE: _ClassVar[DdlRel.DdlOp]
        DDL_OP_CREATE_OR_REPLACE: _ClassVar[DdlRel.DdlOp]
        DDL_OP_ALTER: _ClassVar[DdlRel.DdlOp]
        DDL_OP_DROP: _ClassVar[DdlRel.DdlOp]
        DDL_OP_DROP_IF_EXIST: _ClassVar[DdlRel.DdlOp]
    DDL_OP_UNSPECIFIED: DdlRel.DdlOp
    DDL_OP_CREATE: DdlRel.DdlOp
    DDL_OP_CREATE_OR_REPLACE: DdlRel.DdlOp
    DDL_OP_ALTER: DdlRel.DdlOp
    DDL_OP_DROP: DdlRel.DdlOp
    DDL_OP_DROP_IF_EXIST: DdlRel.DdlOp
    NAMED_OBJECT_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_OBJECT_FIELD_NUMBER: _ClassVar[int]
    TABLE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    TABLE_DEFAULTS_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    VIEW_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    COMMON_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    named_object: NamedObjectWrite
    extension_object: ExtensionObject
    table_schema: _type_pb2.NamedStruct
    table_defaults: Expression.Literal.Struct
    object: DdlRel.DdlObject
    op: DdlRel.DdlOp
    view_definition: Rel
    common: RelCommon
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, named_object: _Optional[_Union[NamedObjectWrite, _Mapping]] = ..., extension_object: _Optional[_Union[ExtensionObject, _Mapping]] = ..., table_schema: _Optional[_Union[_type_pb2.NamedStruct, _Mapping]] = ..., table_defaults: _Optional[_Union[Expression.Literal.Struct, _Mapping]] = ..., object: _Optional[_Union[DdlRel.DdlObject, str]] = ..., op: _Optional[_Union[DdlRel.DdlOp, str]] = ..., view_definition: _Optional[_Union[Rel, _Mapping]] = ..., common: _Optional[_Union[RelCommon, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class WriteRel(_message.Message):
    __slots__ = ("named_table", "extension_table", "table_schema", "op", "input", "create_mode", "output", "common", "advanced_extension")
    class WriteOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        WRITE_OP_UNSPECIFIED: _ClassVar[WriteRel.WriteOp]
        WRITE_OP_INSERT: _ClassVar[WriteRel.WriteOp]
        WRITE_OP_DELETE: _ClassVar[WriteRel.WriteOp]
        WRITE_OP_UPDATE: _ClassVar[WriteRel.WriteOp]
        WRITE_OP_CTAS: _ClassVar[WriteRel.WriteOp]
    WRITE_OP_UNSPECIFIED: WriteRel.WriteOp
    WRITE_OP_INSERT: WriteRel.WriteOp
    WRITE_OP_DELETE: WriteRel.WriteOp
    WRITE_OP_UPDATE: WriteRel.WriteOp
    WRITE_OP_CTAS: WriteRel.WriteOp
    class CreateMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        CREATE_MODE_UNSPECIFIED: _ClassVar[WriteRel.CreateMode]
        CREATE_MODE_APPEND_IF_EXISTS: _ClassVar[WriteRel.CreateMode]
        CREATE_MODE_REPLACE_IF_EXISTS: _ClassVar[WriteRel.CreateMode]
        CREATE_MODE_IGNORE_IF_EXISTS: _ClassVar[WriteRel.CreateMode]
        CREATE_MODE_ERROR_IF_EXISTS: _ClassVar[WriteRel.CreateMode]
    CREATE_MODE_UNSPECIFIED: WriteRel.CreateMode
    CREATE_MODE_APPEND_IF_EXISTS: WriteRel.CreateMode
    CREATE_MODE_REPLACE_IF_EXISTS: WriteRel.CreateMode
    CREATE_MODE_IGNORE_IF_EXISTS: WriteRel.CreateMode
    CREATE_MODE_ERROR_IF_EXISTS: WriteRel.CreateMode
    class OutputMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        OUTPUT_MODE_UNSPECIFIED: _ClassVar[WriteRel.OutputMode]
        OUTPUT_MODE_NO_OUTPUT: _ClassVar[WriteRel.OutputMode]
        OUTPUT_MODE_MODIFIED_RECORDS: _ClassVar[WriteRel.OutputMode]
    OUTPUT_MODE_UNSPECIFIED: WriteRel.OutputMode
    OUTPUT_MODE_NO_OUTPUT: WriteRel.OutputMode
    OUTPUT_MODE_MODIFIED_RECORDS: WriteRel.OutputMode
    NAMED_TABLE_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_TABLE_FIELD_NUMBER: _ClassVar[int]
    TABLE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    CREATE_MODE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    COMMON_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    named_table: NamedObjectWrite
    extension_table: ExtensionObject
    table_schema: _type_pb2.NamedStruct
    op: WriteRel.WriteOp
    input: Rel
    create_mode: WriteRel.CreateMode
    output: WriteRel.OutputMode
    common: RelCommon
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, named_table: _Optional[_Union[NamedObjectWrite, _Mapping]] = ..., extension_table: _Optional[_Union[ExtensionObject, _Mapping]] = ..., table_schema: _Optional[_Union[_type_pb2.NamedStruct, _Mapping]] = ..., op: _Optional[_Union[WriteRel.WriteOp, str]] = ..., input: _Optional[_Union[Rel, _Mapping]] = ..., create_mode: _Optional[_Union[WriteRel.CreateMode, str]] = ..., output: _Optional[_Union[WriteRel.OutputMode, str]] = ..., common: _Optional[_Union[RelCommon, _Mapping]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class UpdateRel(_message.Message):
    __slots__ = ("named_table", "table_schema", "condition", "transformations", "advanced_extension")
    class TransformExpression(_message.Message):
        __slots__ = ("transformation", "column_target")
        TRANSFORMATION_FIELD_NUMBER: _ClassVar[int]
        COLUMN_TARGET_FIELD_NUMBER: _ClassVar[int]
        transformation: Expression
        column_target: int
        def __init__(self, transformation: _Optional[_Union[Expression, _Mapping]] = ..., column_target: _Optional[int] = ...) -> None: ...
    NAMED_TABLE_FIELD_NUMBER: _ClassVar[int]
    TABLE_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    TRANSFORMATIONS_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    named_table: NamedTable
    table_schema: _type_pb2.NamedStruct
    condition: Expression
    transformations: _containers.RepeatedCompositeFieldContainer[UpdateRel.TransformExpression]
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, named_table: _Optional[_Union[NamedTable, _Mapping]] = ..., table_schema: _Optional[_Union[_type_pb2.NamedStruct, _Mapping]] = ..., condition: _Optional[_Union[Expression, _Mapping]] = ..., transformations: _Optional[_Iterable[_Union[UpdateRel.TransformExpression, _Mapping]]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class NamedTable(_message.Message):
    __slots__ = ("names", "advanced_extension")
    NAMES_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    names: _containers.RepeatedScalarFieldContainer[str]
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, names: _Optional[_Iterable[str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class ComparisonJoinKey(_message.Message):
    __slots__ = ("left", "right", "comparison")
    class SimpleComparisonType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SIMPLE_COMPARISON_TYPE_UNSPECIFIED: _ClassVar[ComparisonJoinKey.SimpleComparisonType]
        SIMPLE_COMPARISON_TYPE_EQ: _ClassVar[ComparisonJoinKey.SimpleComparisonType]
        SIMPLE_COMPARISON_TYPE_IS_NOT_DISTINCT_FROM: _ClassVar[ComparisonJoinKey.SimpleComparisonType]
        SIMPLE_COMPARISON_TYPE_MIGHT_EQUAL: _ClassVar[ComparisonJoinKey.SimpleComparisonType]
    SIMPLE_COMPARISON_TYPE_UNSPECIFIED: ComparisonJoinKey.SimpleComparisonType
    SIMPLE_COMPARISON_TYPE_EQ: ComparisonJoinKey.SimpleComparisonType
    SIMPLE_COMPARISON_TYPE_IS_NOT_DISTINCT_FROM: ComparisonJoinKey.SimpleComparisonType
    SIMPLE_COMPARISON_TYPE_MIGHT_EQUAL: ComparisonJoinKey.SimpleComparisonType
    class ComparisonType(_message.Message):
        __slots__ = ("simple", "custom_function_reference")
        SIMPLE_FIELD_NUMBER: _ClassVar[int]
        CUSTOM_FUNCTION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        simple: ComparisonJoinKey.SimpleComparisonType
        custom_function_reference: int
        def __init__(self, simple: _Optional[_Union[ComparisonJoinKey.SimpleComparisonType, str]] = ..., custom_function_reference: _Optional[int] = ...) -> None: ...
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    COMPARISON_FIELD_NUMBER: _ClassVar[int]
    left: Expression.FieldReference
    right: Expression.FieldReference
    comparison: ComparisonJoinKey.ComparisonType
    def __init__(self, left: _Optional[_Union[Expression.FieldReference, _Mapping]] = ..., right: _Optional[_Union[Expression.FieldReference, _Mapping]] = ..., comparison: _Optional[_Union[ComparisonJoinKey.ComparisonType, _Mapping]] = ...) -> None: ...

class HashJoinRel(_message.Message):
    __slots__ = ("common", "left", "right", "left_keys", "right_keys", "keys", "post_join_filter", "type", "build_input", "advanced_extension")
    class JoinType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOIN_TYPE_UNSPECIFIED: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_INNER: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_OUTER: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_LEFT: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_RIGHT: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_LEFT_SEMI: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_RIGHT_SEMI: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_LEFT_ANTI: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_RIGHT_ANTI: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_LEFT_SINGLE: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_RIGHT_SINGLE: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_LEFT_MARK: _ClassVar[HashJoinRel.JoinType]
        JOIN_TYPE_RIGHT_MARK: _ClassVar[HashJoinRel.JoinType]
    JOIN_TYPE_UNSPECIFIED: HashJoinRel.JoinType
    JOIN_TYPE_INNER: HashJoinRel.JoinType
    JOIN_TYPE_OUTER: HashJoinRel.JoinType
    JOIN_TYPE_LEFT: HashJoinRel.JoinType
    JOIN_TYPE_RIGHT: HashJoinRel.JoinType
    JOIN_TYPE_LEFT_SEMI: HashJoinRel.JoinType
    JOIN_TYPE_RIGHT_SEMI: HashJoinRel.JoinType
    JOIN_TYPE_LEFT_ANTI: HashJoinRel.JoinType
    JOIN_TYPE_RIGHT_ANTI: HashJoinRel.JoinType
    JOIN_TYPE_LEFT_SINGLE: HashJoinRel.JoinType
    JOIN_TYPE_RIGHT_SINGLE: HashJoinRel.JoinType
    JOIN_TYPE_LEFT_MARK: HashJoinRel.JoinType
    JOIN_TYPE_RIGHT_MARK: HashJoinRel.JoinType
    class BuildInput(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        BUILD_INPUT_UNSPECIFIED: _ClassVar[HashJoinRel.BuildInput]
        BUILD_INPUT_LEFT: _ClassVar[HashJoinRel.BuildInput]
        BUILD_INPUT_RIGHT: _ClassVar[HashJoinRel.BuildInput]
    BUILD_INPUT_UNSPECIFIED: HashJoinRel.BuildInput
    BUILD_INPUT_LEFT: HashJoinRel.BuildInput
    BUILD_INPUT_RIGHT: HashJoinRel.BuildInput
    COMMON_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    LEFT_KEYS_FIELD_NUMBER: _ClassVar[int]
    RIGHT_KEYS_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    POST_JOIN_FILTER_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    BUILD_INPUT_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    left: Rel
    right: Rel
    left_keys: _containers.RepeatedCompositeFieldContainer[Expression.FieldReference]
    right_keys: _containers.RepeatedCompositeFieldContainer[Expression.FieldReference]
    keys: _containers.RepeatedCompositeFieldContainer[ComparisonJoinKey]
    post_join_filter: Expression
    type: HashJoinRel.JoinType
    build_input: HashJoinRel.BuildInput
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., left: _Optional[_Union[Rel, _Mapping]] = ..., right: _Optional[_Union[Rel, _Mapping]] = ..., left_keys: _Optional[_Iterable[_Union[Expression.FieldReference, _Mapping]]] = ..., right_keys: _Optional[_Iterable[_Union[Expression.FieldReference, _Mapping]]] = ..., keys: _Optional[_Iterable[_Union[ComparisonJoinKey, _Mapping]]] = ..., post_join_filter: _Optional[_Union[Expression, _Mapping]] = ..., type: _Optional[_Union[HashJoinRel.JoinType, str]] = ..., build_input: _Optional[_Union[HashJoinRel.BuildInput, str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class MergeJoinRel(_message.Message):
    __slots__ = ("common", "left", "right", "left_keys", "right_keys", "keys", "post_join_filter", "type", "advanced_extension")
    class JoinType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOIN_TYPE_UNSPECIFIED: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_INNER: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_OUTER: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_LEFT: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_RIGHT: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_LEFT_SEMI: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_RIGHT_SEMI: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_LEFT_ANTI: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_RIGHT_ANTI: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_LEFT_SINGLE: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_RIGHT_SINGLE: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_LEFT_MARK: _ClassVar[MergeJoinRel.JoinType]
        JOIN_TYPE_RIGHT_MARK: _ClassVar[MergeJoinRel.JoinType]
    JOIN_TYPE_UNSPECIFIED: MergeJoinRel.JoinType
    JOIN_TYPE_INNER: MergeJoinRel.JoinType
    JOIN_TYPE_OUTER: MergeJoinRel.JoinType
    JOIN_TYPE_LEFT: MergeJoinRel.JoinType
    JOIN_TYPE_RIGHT: MergeJoinRel.JoinType
    JOIN_TYPE_LEFT_SEMI: MergeJoinRel.JoinType
    JOIN_TYPE_RIGHT_SEMI: MergeJoinRel.JoinType
    JOIN_TYPE_LEFT_ANTI: MergeJoinRel.JoinType
    JOIN_TYPE_RIGHT_ANTI: MergeJoinRel.JoinType
    JOIN_TYPE_LEFT_SINGLE: MergeJoinRel.JoinType
    JOIN_TYPE_RIGHT_SINGLE: MergeJoinRel.JoinType
    JOIN_TYPE_LEFT_MARK: MergeJoinRel.JoinType
    JOIN_TYPE_RIGHT_MARK: MergeJoinRel.JoinType
    COMMON_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    LEFT_KEYS_FIELD_NUMBER: _ClassVar[int]
    RIGHT_KEYS_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    POST_JOIN_FILTER_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    left: Rel
    right: Rel
    left_keys: _containers.RepeatedCompositeFieldContainer[Expression.FieldReference]
    right_keys: _containers.RepeatedCompositeFieldContainer[Expression.FieldReference]
    keys: _containers.RepeatedCompositeFieldContainer[ComparisonJoinKey]
    post_join_filter: Expression
    type: MergeJoinRel.JoinType
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., left: _Optional[_Union[Rel, _Mapping]] = ..., right: _Optional[_Union[Rel, _Mapping]] = ..., left_keys: _Optional[_Iterable[_Union[Expression.FieldReference, _Mapping]]] = ..., right_keys: _Optional[_Iterable[_Union[Expression.FieldReference, _Mapping]]] = ..., keys: _Optional[_Iterable[_Union[ComparisonJoinKey, _Mapping]]] = ..., post_join_filter: _Optional[_Union[Expression, _Mapping]] = ..., type: _Optional[_Union[MergeJoinRel.JoinType, str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class NestedLoopJoinRel(_message.Message):
    __slots__ = ("common", "left", "right", "expression", "type", "advanced_extension")
    class JoinType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        JOIN_TYPE_UNSPECIFIED: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_INNER: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_OUTER: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_LEFT: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_RIGHT: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_LEFT_SEMI: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_RIGHT_SEMI: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_LEFT_ANTI: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_RIGHT_ANTI: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_LEFT_SINGLE: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_RIGHT_SINGLE: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_LEFT_MARK: _ClassVar[NestedLoopJoinRel.JoinType]
        JOIN_TYPE_RIGHT_MARK: _ClassVar[NestedLoopJoinRel.JoinType]
    JOIN_TYPE_UNSPECIFIED: NestedLoopJoinRel.JoinType
    JOIN_TYPE_INNER: NestedLoopJoinRel.JoinType
    JOIN_TYPE_OUTER: NestedLoopJoinRel.JoinType
    JOIN_TYPE_LEFT: NestedLoopJoinRel.JoinType
    JOIN_TYPE_RIGHT: NestedLoopJoinRel.JoinType
    JOIN_TYPE_LEFT_SEMI: NestedLoopJoinRel.JoinType
    JOIN_TYPE_RIGHT_SEMI: NestedLoopJoinRel.JoinType
    JOIN_TYPE_LEFT_ANTI: NestedLoopJoinRel.JoinType
    JOIN_TYPE_RIGHT_ANTI: NestedLoopJoinRel.JoinType
    JOIN_TYPE_LEFT_SINGLE: NestedLoopJoinRel.JoinType
    JOIN_TYPE_RIGHT_SINGLE: NestedLoopJoinRel.JoinType
    JOIN_TYPE_LEFT_MARK: NestedLoopJoinRel.JoinType
    JOIN_TYPE_RIGHT_MARK: NestedLoopJoinRel.JoinType
    COMMON_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_FIELD_NUMBER: _ClassVar[int]
    common: RelCommon
    left: Rel
    right: Rel
    expression: Expression
    type: NestedLoopJoinRel.JoinType
    advanced_extension: _extensions_pb2.AdvancedExtension
    def __init__(self, common: _Optional[_Union[RelCommon, _Mapping]] = ..., left: _Optional[_Union[Rel, _Mapping]] = ..., right: _Optional[_Union[Rel, _Mapping]] = ..., expression: _Optional[_Union[Expression, _Mapping]] = ..., type: _Optional[_Union[NestedLoopJoinRel.JoinType, str]] = ..., advanced_extension: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ...) -> None: ...

class FunctionArgument(_message.Message):
    __slots__ = ("enum", "type", "value")
    ENUM_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    enum: str
    type: _type_pb2.Type
    value: Expression
    def __init__(self, enum: _Optional[str] = ..., type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., value: _Optional[_Union[Expression, _Mapping]] = ...) -> None: ...

class FunctionOption(_message.Message):
    __slots__ = ("name", "preference")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PREFERENCE_FIELD_NUMBER: _ClassVar[int]
    name: str
    preference: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., preference: _Optional[_Iterable[str]] = ...) -> None: ...

class Expression(_message.Message):
    __slots__ = ("literal", "selection", "scalar_function", "window_function", "if_then", "switch_expression", "singular_or_list", "multi_or_list", "cast", "subquery", "nested", "dynamic_parameter", "enum")
    class Enum(_message.Message):
        __slots__ = ("specified", "unspecified")
        class Empty(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        SPECIFIED_FIELD_NUMBER: _ClassVar[int]
        UNSPECIFIED_FIELD_NUMBER: _ClassVar[int]
        specified: str
        unspecified: Expression.Enum.Empty
        def __init__(self, specified: _Optional[str] = ..., unspecified: _Optional[_Union[Expression.Enum.Empty, _Mapping]] = ...) -> None: ...
    class Literal(_message.Message):
        __slots__ = ("boolean", "i8", "i16", "i32", "i64", "fp32", "fp64", "string", "binary", "timestamp", "date", "time", "interval_year_to_month", "interval_day_to_second", "interval_compound", "fixed_char", "var_char", "fixed_binary", "decimal", "precision_time", "precision_timestamp", "precision_timestamp_tz", "struct", "map", "timestamp_tz", "uuid", "null", "list", "empty_list", "empty_map", "user_defined", "nullable", "type_variation_reference")
        class VarChar(_message.Message):
            __slots__ = ("value", "length")
            VALUE_FIELD_NUMBER: _ClassVar[int]
            LENGTH_FIELD_NUMBER: _ClassVar[int]
            value: str
            length: int
            def __init__(self, value: _Optional[str] = ..., length: _Optional[int] = ...) -> None: ...
        class Decimal(_message.Message):
            __slots__ = ("value", "precision", "scale")
            VALUE_FIELD_NUMBER: _ClassVar[int]
            PRECISION_FIELD_NUMBER: _ClassVar[int]
            SCALE_FIELD_NUMBER: _ClassVar[int]
            value: bytes
            precision: int
            scale: int
            def __init__(self, value: _Optional[bytes] = ..., precision: _Optional[int] = ..., scale: _Optional[int] = ...) -> None: ...
        class PrecisionTime(_message.Message):
            __slots__ = ("precision", "value")
            PRECISION_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            precision: int
            value: int
            def __init__(self, precision: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...
        class PrecisionTimestamp(_message.Message):
            __slots__ = ("precision", "value")
            PRECISION_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            precision: int
            value: int
            def __init__(self, precision: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...
        class Map(_message.Message):
            __slots__ = ("key_values",)
            class KeyValue(_message.Message):
                __slots__ = ("key", "value")
                KEY_FIELD_NUMBER: _ClassVar[int]
                VALUE_FIELD_NUMBER: _ClassVar[int]
                key: Expression.Literal
                value: Expression.Literal
                def __init__(self, key: _Optional[_Union[Expression.Literal, _Mapping]] = ..., value: _Optional[_Union[Expression.Literal, _Mapping]] = ...) -> None: ...
            KEY_VALUES_FIELD_NUMBER: _ClassVar[int]
            key_values: _containers.RepeatedCompositeFieldContainer[Expression.Literal.Map.KeyValue]
            def __init__(self, key_values: _Optional[_Iterable[_Union[Expression.Literal.Map.KeyValue, _Mapping]]] = ...) -> None: ...
        class IntervalYearToMonth(_message.Message):
            __slots__ = ("years", "months")
            YEARS_FIELD_NUMBER: _ClassVar[int]
            MONTHS_FIELD_NUMBER: _ClassVar[int]
            years: int
            months: int
            def __init__(self, years: _Optional[int] = ..., months: _Optional[int] = ...) -> None: ...
        class IntervalDayToSecond(_message.Message):
            __slots__ = ("days", "seconds", "microseconds", "precision", "subseconds")
            DAYS_FIELD_NUMBER: _ClassVar[int]
            SECONDS_FIELD_NUMBER: _ClassVar[int]
            MICROSECONDS_FIELD_NUMBER: _ClassVar[int]
            PRECISION_FIELD_NUMBER: _ClassVar[int]
            SUBSECONDS_FIELD_NUMBER: _ClassVar[int]
            days: int
            seconds: int
            microseconds: int
            precision: int
            subseconds: int
            def __init__(self, days: _Optional[int] = ..., seconds: _Optional[int] = ..., microseconds: _Optional[int] = ..., precision: _Optional[int] = ..., subseconds: _Optional[int] = ...) -> None: ...
        class IntervalCompound(_message.Message):
            __slots__ = ("interval_year_to_month", "interval_day_to_second")
            INTERVAL_YEAR_TO_MONTH_FIELD_NUMBER: _ClassVar[int]
            INTERVAL_DAY_TO_SECOND_FIELD_NUMBER: _ClassVar[int]
            interval_year_to_month: Expression.Literal.IntervalYearToMonth
            interval_day_to_second: Expression.Literal.IntervalDayToSecond
            def __init__(self, interval_year_to_month: _Optional[_Union[Expression.Literal.IntervalYearToMonth, _Mapping]] = ..., interval_day_to_second: _Optional[_Union[Expression.Literal.IntervalDayToSecond, _Mapping]] = ...) -> None: ...
        class Struct(_message.Message):
            __slots__ = ("fields",)
            FIELDS_FIELD_NUMBER: _ClassVar[int]
            fields: _containers.RepeatedCompositeFieldContainer[Expression.Literal]
            def __init__(self, fields: _Optional[_Iterable[_Union[Expression.Literal, _Mapping]]] = ...) -> None: ...
        class List(_message.Message):
            __slots__ = ("values",)
            VALUES_FIELD_NUMBER: _ClassVar[int]
            values: _containers.RepeatedCompositeFieldContainer[Expression.Literal]
            def __init__(self, values: _Optional[_Iterable[_Union[Expression.Literal, _Mapping]]] = ...) -> None: ...
        class UserDefined(_message.Message):
            __slots__ = ("type_reference", "type_parameters", "value", "struct")
            TYPE_REFERENCE_FIELD_NUMBER: _ClassVar[int]
            TYPE_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            STRUCT_FIELD_NUMBER: _ClassVar[int]
            type_reference: int
            type_parameters: _containers.RepeatedCompositeFieldContainer[_type_pb2.Type.Parameter]
            value: _any_pb2.Any
            struct: Expression.Literal.Struct
            def __init__(self, type_reference: _Optional[int] = ..., type_parameters: _Optional[_Iterable[_Union[_type_pb2.Type.Parameter, _Mapping]]] = ..., value: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., struct: _Optional[_Union[Expression.Literal.Struct, _Mapping]] = ...) -> None: ...
        BOOLEAN_FIELD_NUMBER: _ClassVar[int]
        I8_FIELD_NUMBER: _ClassVar[int]
        I16_FIELD_NUMBER: _ClassVar[int]
        I32_FIELD_NUMBER: _ClassVar[int]
        I64_FIELD_NUMBER: _ClassVar[int]
        FP32_FIELD_NUMBER: _ClassVar[int]
        FP64_FIELD_NUMBER: _ClassVar[int]
        STRING_FIELD_NUMBER: _ClassVar[int]
        BINARY_FIELD_NUMBER: _ClassVar[int]
        TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        DATE_FIELD_NUMBER: _ClassVar[int]
        TIME_FIELD_NUMBER: _ClassVar[int]
        INTERVAL_YEAR_TO_MONTH_FIELD_NUMBER: _ClassVar[int]
        INTERVAL_DAY_TO_SECOND_FIELD_NUMBER: _ClassVar[int]
        INTERVAL_COMPOUND_FIELD_NUMBER: _ClassVar[int]
        FIXED_CHAR_FIELD_NUMBER: _ClassVar[int]
        VAR_CHAR_FIELD_NUMBER: _ClassVar[int]
        FIXED_BINARY_FIELD_NUMBER: _ClassVar[int]
        DECIMAL_FIELD_NUMBER: _ClassVar[int]
        PRECISION_TIME_FIELD_NUMBER: _ClassVar[int]
        PRECISION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        PRECISION_TIMESTAMP_TZ_FIELD_NUMBER: _ClassVar[int]
        STRUCT_FIELD_NUMBER: _ClassVar[int]
        MAP_FIELD_NUMBER: _ClassVar[int]
        TIMESTAMP_TZ_FIELD_NUMBER: _ClassVar[int]
        UUID_FIELD_NUMBER: _ClassVar[int]
        NULL_FIELD_NUMBER: _ClassVar[int]
        LIST_FIELD_NUMBER: _ClassVar[int]
        EMPTY_LIST_FIELD_NUMBER: _ClassVar[int]
        EMPTY_MAP_FIELD_NUMBER: _ClassVar[int]
        USER_DEFINED_FIELD_NUMBER: _ClassVar[int]
        NULLABLE_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        boolean: bool
        i8: int
        i16: int
        i32: int
        i64: int
        fp32: float
        fp64: float
        string: str
        binary: bytes
        timestamp: int
        date: int
        time: int
        interval_year_to_month: Expression.Literal.IntervalYearToMonth
        interval_day_to_second: Expression.Literal.IntervalDayToSecond
        interval_compound: Expression.Literal.IntervalCompound
        fixed_char: str
        var_char: Expression.Literal.VarChar
        fixed_binary: bytes
        decimal: Expression.Literal.Decimal
        precision_time: Expression.Literal.PrecisionTime
        precision_timestamp: Expression.Literal.PrecisionTimestamp
        precision_timestamp_tz: Expression.Literal.PrecisionTimestamp
        struct: Expression.Literal.Struct
        map: Expression.Literal.Map
        timestamp_tz: int
        uuid: bytes
        null: _type_pb2.Type
        list: Expression.Literal.List
        empty_list: _type_pb2.Type.List
        empty_map: _type_pb2.Type.Map
        user_defined: Expression.Literal.UserDefined
        nullable: bool
        type_variation_reference: int
        def __init__(self, boolean: bool = ..., i8: _Optional[int] = ..., i16: _Optional[int] = ..., i32: _Optional[int] = ..., i64: _Optional[int] = ..., fp32: _Optional[float] = ..., fp64: _Optional[float] = ..., string: _Optional[str] = ..., binary: _Optional[bytes] = ..., timestamp: _Optional[int] = ..., date: _Optional[int] = ..., time: _Optional[int] = ..., interval_year_to_month: _Optional[_Union[Expression.Literal.IntervalYearToMonth, _Mapping]] = ..., interval_day_to_second: _Optional[_Union[Expression.Literal.IntervalDayToSecond, _Mapping]] = ..., interval_compound: _Optional[_Union[Expression.Literal.IntervalCompound, _Mapping]] = ..., fixed_char: _Optional[str] = ..., var_char: _Optional[_Union[Expression.Literal.VarChar, _Mapping]] = ..., fixed_binary: _Optional[bytes] = ..., decimal: _Optional[_Union[Expression.Literal.Decimal, _Mapping]] = ..., precision_time: _Optional[_Union[Expression.Literal.PrecisionTime, _Mapping]] = ..., precision_timestamp: _Optional[_Union[Expression.Literal.PrecisionTimestamp, _Mapping]] = ..., precision_timestamp_tz: _Optional[_Union[Expression.Literal.PrecisionTimestamp, _Mapping]] = ..., struct: _Optional[_Union[Expression.Literal.Struct, _Mapping]] = ..., map: _Optional[_Union[Expression.Literal.Map, _Mapping]] = ..., timestamp_tz: _Optional[int] = ..., uuid: _Optional[bytes] = ..., null: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., list: _Optional[_Union[Expression.Literal.List, _Mapping]] = ..., empty_list: _Optional[_Union[_type_pb2.Type.List, _Mapping]] = ..., empty_map: _Optional[_Union[_type_pb2.Type.Map, _Mapping]] = ..., user_defined: _Optional[_Union[Expression.Literal.UserDefined, _Mapping]] = ..., nullable: bool = ..., type_variation_reference: _Optional[int] = ...) -> None: ...
    class Nested(_message.Message):
        __slots__ = ("nullable", "type_variation_reference", "struct", "list", "map")
        class Map(_message.Message):
            __slots__ = ("key_values",)
            class KeyValue(_message.Message):
                __slots__ = ("key", "value")
                KEY_FIELD_NUMBER: _ClassVar[int]
                VALUE_FIELD_NUMBER: _ClassVar[int]
                key: Expression
                value: Expression
                def __init__(self, key: _Optional[_Union[Expression, _Mapping]] = ..., value: _Optional[_Union[Expression, _Mapping]] = ...) -> None: ...
            KEY_VALUES_FIELD_NUMBER: _ClassVar[int]
            key_values: _containers.RepeatedCompositeFieldContainer[Expression.Nested.Map.KeyValue]
            def __init__(self, key_values: _Optional[_Iterable[_Union[Expression.Nested.Map.KeyValue, _Mapping]]] = ...) -> None: ...
        class Struct(_message.Message):
            __slots__ = ("fields",)
            FIELDS_FIELD_NUMBER: _ClassVar[int]
            fields: _containers.RepeatedCompositeFieldContainer[Expression]
            def __init__(self, fields: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...
        class List(_message.Message):
            __slots__ = ("values",)
            VALUES_FIELD_NUMBER: _ClassVar[int]
            values: _containers.RepeatedCompositeFieldContainer[Expression]
            def __init__(self, values: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...
        NULLABLE_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        STRUCT_FIELD_NUMBER: _ClassVar[int]
        LIST_FIELD_NUMBER: _ClassVar[int]
        MAP_FIELD_NUMBER: _ClassVar[int]
        nullable: bool
        type_variation_reference: int
        struct: Expression.Nested.Struct
        list: Expression.Nested.List
        map: Expression.Nested.Map
        def __init__(self, nullable: bool = ..., type_variation_reference: _Optional[int] = ..., struct: _Optional[_Union[Expression.Nested.Struct, _Mapping]] = ..., list: _Optional[_Union[Expression.Nested.List, _Mapping]] = ..., map: _Optional[_Union[Expression.Nested.Map, _Mapping]] = ...) -> None: ...
    class ScalarFunction(_message.Message):
        __slots__ = ("function_reference", "arguments", "options", "output_type", "args")
        FUNCTION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
        OPTIONS_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
        ARGS_FIELD_NUMBER: _ClassVar[int]
        function_reference: int
        arguments: _containers.RepeatedCompositeFieldContainer[FunctionArgument]
        options: _containers.RepeatedCompositeFieldContainer[FunctionOption]
        output_type: _type_pb2.Type
        args: _containers.RepeatedCompositeFieldContainer[Expression]
        def __init__(self, function_reference: _Optional[int] = ..., arguments: _Optional[_Iterable[_Union[FunctionArgument, _Mapping]]] = ..., options: _Optional[_Iterable[_Union[FunctionOption, _Mapping]]] = ..., output_type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., args: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...
    class WindowFunction(_message.Message):
        __slots__ = ("function_reference", "arguments", "options", "output_type", "phase", "sorts", "invocation", "partitions", "bounds_type", "lower_bound", "upper_bound", "args")
        class BoundsType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            BOUNDS_TYPE_UNSPECIFIED: _ClassVar[Expression.WindowFunction.BoundsType]
            BOUNDS_TYPE_ROWS: _ClassVar[Expression.WindowFunction.BoundsType]
            BOUNDS_TYPE_RANGE: _ClassVar[Expression.WindowFunction.BoundsType]
        BOUNDS_TYPE_UNSPECIFIED: Expression.WindowFunction.BoundsType
        BOUNDS_TYPE_ROWS: Expression.WindowFunction.BoundsType
        BOUNDS_TYPE_RANGE: Expression.WindowFunction.BoundsType
        class Bound(_message.Message):
            __slots__ = ("preceding", "following", "current_row", "unbounded")
            class Preceding(_message.Message):
                __slots__ = ("offset",)
                OFFSET_FIELD_NUMBER: _ClassVar[int]
                offset: int
                def __init__(self, offset: _Optional[int] = ...) -> None: ...
            class Following(_message.Message):
                __slots__ = ("offset",)
                OFFSET_FIELD_NUMBER: _ClassVar[int]
                offset: int
                def __init__(self, offset: _Optional[int] = ...) -> None: ...
            class CurrentRow(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            class Unbounded(_message.Message):
                __slots__ = ()
                def __init__(self) -> None: ...
            PRECEDING_FIELD_NUMBER: _ClassVar[int]
            FOLLOWING_FIELD_NUMBER: _ClassVar[int]
            CURRENT_ROW_FIELD_NUMBER: _ClassVar[int]
            UNBOUNDED_FIELD_NUMBER: _ClassVar[int]
            preceding: Expression.WindowFunction.Bound.Preceding
            following: Expression.WindowFunction.Bound.Following
            current_row: Expression.WindowFunction.Bound.CurrentRow
            unbounded: Expression.WindowFunction.Bound.Unbounded
            def __init__(self, preceding: _Optional[_Union[Expression.WindowFunction.Bound.Preceding, _Mapping]] = ..., following: _Optional[_Union[Expression.WindowFunction.Bound.Following, _Mapping]] = ..., current_row: _Optional[_Union[Expression.WindowFunction.Bound.CurrentRow, _Mapping]] = ..., unbounded: _Optional[_Union[Expression.WindowFunction.Bound.Unbounded, _Mapping]] = ...) -> None: ...
        FUNCTION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
        OPTIONS_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
        PHASE_FIELD_NUMBER: _ClassVar[int]
        SORTS_FIELD_NUMBER: _ClassVar[int]
        INVOCATION_FIELD_NUMBER: _ClassVar[int]
        PARTITIONS_FIELD_NUMBER: _ClassVar[int]
        BOUNDS_TYPE_FIELD_NUMBER: _ClassVar[int]
        LOWER_BOUND_FIELD_NUMBER: _ClassVar[int]
        UPPER_BOUND_FIELD_NUMBER: _ClassVar[int]
        ARGS_FIELD_NUMBER: _ClassVar[int]
        function_reference: int
        arguments: _containers.RepeatedCompositeFieldContainer[FunctionArgument]
        options: _containers.RepeatedCompositeFieldContainer[FunctionOption]
        output_type: _type_pb2.Type
        phase: AggregationPhase
        sorts: _containers.RepeatedCompositeFieldContainer[SortField]
        invocation: AggregateFunction.AggregationInvocation
        partitions: _containers.RepeatedCompositeFieldContainer[Expression]
        bounds_type: Expression.WindowFunction.BoundsType
        lower_bound: Expression.WindowFunction.Bound
        upper_bound: Expression.WindowFunction.Bound
        args: _containers.RepeatedCompositeFieldContainer[Expression]
        def __init__(self, function_reference: _Optional[int] = ..., arguments: _Optional[_Iterable[_Union[FunctionArgument, _Mapping]]] = ..., options: _Optional[_Iterable[_Union[FunctionOption, _Mapping]]] = ..., output_type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., phase: _Optional[_Union[AggregationPhase, str]] = ..., sorts: _Optional[_Iterable[_Union[SortField, _Mapping]]] = ..., invocation: _Optional[_Union[AggregateFunction.AggregationInvocation, str]] = ..., partitions: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., bounds_type: _Optional[_Union[Expression.WindowFunction.BoundsType, str]] = ..., lower_bound: _Optional[_Union[Expression.WindowFunction.Bound, _Mapping]] = ..., upper_bound: _Optional[_Union[Expression.WindowFunction.Bound, _Mapping]] = ..., args: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...
    class IfThen(_message.Message):
        __slots__ = ("ifs",)
        class IfClause(_message.Message):
            __slots__ = ("then",)
            IF_FIELD_NUMBER: _ClassVar[int]
            THEN_FIELD_NUMBER: _ClassVar[int]
            then: Expression
            def __init__(self, then: _Optional[_Union[Expression, _Mapping]] = ..., **kwargs) -> None: ...
        IFS_FIELD_NUMBER: _ClassVar[int]
        ELSE_FIELD_NUMBER: _ClassVar[int]
        ifs: _containers.RepeatedCompositeFieldContainer[Expression.IfThen.IfClause]
        def __init__(self, ifs: _Optional[_Iterable[_Union[Expression.IfThen.IfClause, _Mapping]]] = ..., **kwargs) -> None: ...
    class Cast(_message.Message):
        __slots__ = ("type", "input", "failure_behavior")
        class FailureBehavior(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            FAILURE_BEHAVIOR_UNSPECIFIED: _ClassVar[Expression.Cast.FailureBehavior]
            FAILURE_BEHAVIOR_RETURN_NULL: _ClassVar[Expression.Cast.FailureBehavior]
            FAILURE_BEHAVIOR_THROW_EXCEPTION: _ClassVar[Expression.Cast.FailureBehavior]
        FAILURE_BEHAVIOR_UNSPECIFIED: Expression.Cast.FailureBehavior
        FAILURE_BEHAVIOR_RETURN_NULL: Expression.Cast.FailureBehavior
        FAILURE_BEHAVIOR_THROW_EXCEPTION: Expression.Cast.FailureBehavior
        TYPE_FIELD_NUMBER: _ClassVar[int]
        INPUT_FIELD_NUMBER: _ClassVar[int]
        FAILURE_BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
        type: _type_pb2.Type
        input: Expression
        failure_behavior: Expression.Cast.FailureBehavior
        def __init__(self, type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., input: _Optional[_Union[Expression, _Mapping]] = ..., failure_behavior: _Optional[_Union[Expression.Cast.FailureBehavior, str]] = ...) -> None: ...
    class SwitchExpression(_message.Message):
        __slots__ = ("match", "ifs")
        class IfValue(_message.Message):
            __slots__ = ("then",)
            IF_FIELD_NUMBER: _ClassVar[int]
            THEN_FIELD_NUMBER: _ClassVar[int]
            then: Expression
            def __init__(self, then: _Optional[_Union[Expression, _Mapping]] = ..., **kwargs) -> None: ...
        MATCH_FIELD_NUMBER: _ClassVar[int]
        IFS_FIELD_NUMBER: _ClassVar[int]
        ELSE_FIELD_NUMBER: _ClassVar[int]
        match: Expression
        ifs: _containers.RepeatedCompositeFieldContainer[Expression.SwitchExpression.IfValue]
        def __init__(self, match: _Optional[_Union[Expression, _Mapping]] = ..., ifs: _Optional[_Iterable[_Union[Expression.SwitchExpression.IfValue, _Mapping]]] = ..., **kwargs) -> None: ...
    class SingularOrList(_message.Message):
        __slots__ = ("value", "options")
        VALUE_FIELD_NUMBER: _ClassVar[int]
        OPTIONS_FIELD_NUMBER: _ClassVar[int]
        value: Expression
        options: _containers.RepeatedCompositeFieldContainer[Expression]
        def __init__(self, value: _Optional[_Union[Expression, _Mapping]] = ..., options: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...
    class MultiOrList(_message.Message):
        __slots__ = ("value", "options")
        class Record(_message.Message):
            __slots__ = ("fields",)
            FIELDS_FIELD_NUMBER: _ClassVar[int]
            fields: _containers.RepeatedCompositeFieldContainer[Expression]
            def __init__(self, fields: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...
        VALUE_FIELD_NUMBER: _ClassVar[int]
        OPTIONS_FIELD_NUMBER: _ClassVar[int]
        value: _containers.RepeatedCompositeFieldContainer[Expression]
        options: _containers.RepeatedCompositeFieldContainer[Expression.MultiOrList.Record]
        def __init__(self, value: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., options: _Optional[_Iterable[_Union[Expression.MultiOrList.Record, _Mapping]]] = ...) -> None: ...
    class EmbeddedFunction(_message.Message):
        __slots__ = ("arguments", "output_type", "python_pickle_function", "web_assembly_function")
        class PythonPickleFunction(_message.Message):
            __slots__ = ("function", "prerequisite")
            FUNCTION_FIELD_NUMBER: _ClassVar[int]
            PREREQUISITE_FIELD_NUMBER: _ClassVar[int]
            function: bytes
            prerequisite: _containers.RepeatedScalarFieldContainer[str]
            def __init__(self, function: _Optional[bytes] = ..., prerequisite: _Optional[_Iterable[str]] = ...) -> None: ...
        class WebAssemblyFunction(_message.Message):
            __slots__ = ("script", "prerequisite")
            SCRIPT_FIELD_NUMBER: _ClassVar[int]
            PREREQUISITE_FIELD_NUMBER: _ClassVar[int]
            script: bytes
            prerequisite: _containers.RepeatedScalarFieldContainer[str]
            def __init__(self, script: _Optional[bytes] = ..., prerequisite: _Optional[_Iterable[str]] = ...) -> None: ...
        ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
        PYTHON_PICKLE_FUNCTION_FIELD_NUMBER: _ClassVar[int]
        WEB_ASSEMBLY_FUNCTION_FIELD_NUMBER: _ClassVar[int]
        arguments: _containers.RepeatedCompositeFieldContainer[Expression]
        output_type: _type_pb2.Type
        python_pickle_function: Expression.EmbeddedFunction.PythonPickleFunction
        web_assembly_function: Expression.EmbeddedFunction.WebAssemblyFunction
        def __init__(self, arguments: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., output_type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., python_pickle_function: _Optional[_Union[Expression.EmbeddedFunction.PythonPickleFunction, _Mapping]] = ..., web_assembly_function: _Optional[_Union[Expression.EmbeddedFunction.WebAssemblyFunction, _Mapping]] = ...) -> None: ...
    class ReferenceSegment(_message.Message):
        __slots__ = ("map_key", "struct_field", "list_element")
        class MapKey(_message.Message):
            __slots__ = ("map_key", "child")
            MAP_KEY_FIELD_NUMBER: _ClassVar[int]
            CHILD_FIELD_NUMBER: _ClassVar[int]
            map_key: Expression.Literal
            child: Expression.ReferenceSegment
            def __init__(self, map_key: _Optional[_Union[Expression.Literal, _Mapping]] = ..., child: _Optional[_Union[Expression.ReferenceSegment, _Mapping]] = ...) -> None: ...
        class StructField(_message.Message):
            __slots__ = ("field", "child")
            FIELD_FIELD_NUMBER: _ClassVar[int]
            CHILD_FIELD_NUMBER: _ClassVar[int]
            field: int
            child: Expression.ReferenceSegment
            def __init__(self, field: _Optional[int] = ..., child: _Optional[_Union[Expression.ReferenceSegment, _Mapping]] = ...) -> None: ...
        class ListElement(_message.Message):
            __slots__ = ("offset", "child")
            OFFSET_FIELD_NUMBER: _ClassVar[int]
            CHILD_FIELD_NUMBER: _ClassVar[int]
            offset: int
            child: Expression.ReferenceSegment
            def __init__(self, offset: _Optional[int] = ..., child: _Optional[_Union[Expression.ReferenceSegment, _Mapping]] = ...) -> None: ...
        MAP_KEY_FIELD_NUMBER: _ClassVar[int]
        STRUCT_FIELD_FIELD_NUMBER: _ClassVar[int]
        LIST_ELEMENT_FIELD_NUMBER: _ClassVar[int]
        map_key: Expression.ReferenceSegment.MapKey
        struct_field: Expression.ReferenceSegment.StructField
        list_element: Expression.ReferenceSegment.ListElement
        def __init__(self, map_key: _Optional[_Union[Expression.ReferenceSegment.MapKey, _Mapping]] = ..., struct_field: _Optional[_Union[Expression.ReferenceSegment.StructField, _Mapping]] = ..., list_element: _Optional[_Union[Expression.ReferenceSegment.ListElement, _Mapping]] = ...) -> None: ...
    class MaskExpression(_message.Message):
        __slots__ = ("select", "maintain_singular_struct")
        class Select(_message.Message):
            __slots__ = ("struct", "list", "map")
            STRUCT_FIELD_NUMBER: _ClassVar[int]
            LIST_FIELD_NUMBER: _ClassVar[int]
            MAP_FIELD_NUMBER: _ClassVar[int]
            struct: Expression.MaskExpression.StructSelect
            list: Expression.MaskExpression.ListSelect
            map: Expression.MaskExpression.MapSelect
            def __init__(self, struct: _Optional[_Union[Expression.MaskExpression.StructSelect, _Mapping]] = ..., list: _Optional[_Union[Expression.MaskExpression.ListSelect, _Mapping]] = ..., map: _Optional[_Union[Expression.MaskExpression.MapSelect, _Mapping]] = ...) -> None: ...
        class StructSelect(_message.Message):
            __slots__ = ("struct_items",)
            STRUCT_ITEMS_FIELD_NUMBER: _ClassVar[int]
            struct_items: _containers.RepeatedCompositeFieldContainer[Expression.MaskExpression.StructItem]
            def __init__(self, struct_items: _Optional[_Iterable[_Union[Expression.MaskExpression.StructItem, _Mapping]]] = ...) -> None: ...
        class StructItem(_message.Message):
            __slots__ = ("field", "child")
            FIELD_FIELD_NUMBER: _ClassVar[int]
            CHILD_FIELD_NUMBER: _ClassVar[int]
            field: int
            child: Expression.MaskExpression.Select
            def __init__(self, field: _Optional[int] = ..., child: _Optional[_Union[Expression.MaskExpression.Select, _Mapping]] = ...) -> None: ...
        class ListSelect(_message.Message):
            __slots__ = ("selection", "child")
            class ListSelectItem(_message.Message):
                __slots__ = ("item", "slice")
                class ListElement(_message.Message):
                    __slots__ = ("field",)
                    FIELD_FIELD_NUMBER: _ClassVar[int]
                    field: int
                    def __init__(self, field: _Optional[int] = ...) -> None: ...
                class ListSlice(_message.Message):
                    __slots__ = ("start", "end")
                    START_FIELD_NUMBER: _ClassVar[int]
                    END_FIELD_NUMBER: _ClassVar[int]
                    start: int
                    end: int
                    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...
                ITEM_FIELD_NUMBER: _ClassVar[int]
                SLICE_FIELD_NUMBER: _ClassVar[int]
                item: Expression.MaskExpression.ListSelect.ListSelectItem.ListElement
                slice: Expression.MaskExpression.ListSelect.ListSelectItem.ListSlice
                def __init__(self, item: _Optional[_Union[Expression.MaskExpression.ListSelect.ListSelectItem.ListElement, _Mapping]] = ..., slice: _Optional[_Union[Expression.MaskExpression.ListSelect.ListSelectItem.ListSlice, _Mapping]] = ...) -> None: ...
            SELECTION_FIELD_NUMBER: _ClassVar[int]
            CHILD_FIELD_NUMBER: _ClassVar[int]
            selection: _containers.RepeatedCompositeFieldContainer[Expression.MaskExpression.ListSelect.ListSelectItem]
            child: Expression.MaskExpression.Select
            def __init__(self, selection: _Optional[_Iterable[_Union[Expression.MaskExpression.ListSelect.ListSelectItem, _Mapping]]] = ..., child: _Optional[_Union[Expression.MaskExpression.Select, _Mapping]] = ...) -> None: ...
        class MapSelect(_message.Message):
            __slots__ = ("key", "expression", "child")
            class MapKey(_message.Message):
                __slots__ = ("map_key",)
                MAP_KEY_FIELD_NUMBER: _ClassVar[int]
                map_key: str
                def __init__(self, map_key: _Optional[str] = ...) -> None: ...
            class MapKeyExpression(_message.Message):
                __slots__ = ("map_key_expression",)
                MAP_KEY_EXPRESSION_FIELD_NUMBER: _ClassVar[int]
                map_key_expression: str
                def __init__(self, map_key_expression: _Optional[str] = ...) -> None: ...
            KEY_FIELD_NUMBER: _ClassVar[int]
            EXPRESSION_FIELD_NUMBER: _ClassVar[int]
            CHILD_FIELD_NUMBER: _ClassVar[int]
            key: Expression.MaskExpression.MapSelect.MapKey
            expression: Expression.MaskExpression.MapSelect.MapKeyExpression
            child: Expression.MaskExpression.Select
            def __init__(self, key: _Optional[_Union[Expression.MaskExpression.MapSelect.MapKey, _Mapping]] = ..., expression: _Optional[_Union[Expression.MaskExpression.MapSelect.MapKeyExpression, _Mapping]] = ..., child: _Optional[_Union[Expression.MaskExpression.Select, _Mapping]] = ...) -> None: ...
        SELECT_FIELD_NUMBER: _ClassVar[int]
        MAINTAIN_SINGULAR_STRUCT_FIELD_NUMBER: _ClassVar[int]
        select: Expression.MaskExpression.StructSelect
        maintain_singular_struct: bool
        def __init__(self, select: _Optional[_Union[Expression.MaskExpression.StructSelect, _Mapping]] = ..., maintain_singular_struct: bool = ...) -> None: ...
    class FieldReference(_message.Message):
        __slots__ = ("direct_reference", "masked_reference", "expression", "root_reference", "outer_reference")
        class RootReference(_message.Message):
            __slots__ = ()
            def __init__(self) -> None: ...
        class OuterReference(_message.Message):
            __slots__ = ("steps_out",)
            STEPS_OUT_FIELD_NUMBER: _ClassVar[int]
            steps_out: int
            def __init__(self, steps_out: _Optional[int] = ...) -> None: ...
        DIRECT_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        MASKED_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        EXPRESSION_FIELD_NUMBER: _ClassVar[int]
        ROOT_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        OUTER_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        direct_reference: Expression.ReferenceSegment
        masked_reference: Expression.MaskExpression
        expression: Expression
        root_reference: Expression.FieldReference.RootReference
        outer_reference: Expression.FieldReference.OuterReference
        def __init__(self, direct_reference: _Optional[_Union[Expression.ReferenceSegment, _Mapping]] = ..., masked_reference: _Optional[_Union[Expression.MaskExpression, _Mapping]] = ..., expression: _Optional[_Union[Expression, _Mapping]] = ..., root_reference: _Optional[_Union[Expression.FieldReference.RootReference, _Mapping]] = ..., outer_reference: _Optional[_Union[Expression.FieldReference.OuterReference, _Mapping]] = ...) -> None: ...
    class Subquery(_message.Message):
        __slots__ = ("scalar", "in_predicate", "set_predicate", "set_comparison")
        class Scalar(_message.Message):
            __slots__ = ("input",)
            INPUT_FIELD_NUMBER: _ClassVar[int]
            input: Rel
            def __init__(self, input: _Optional[_Union[Rel, _Mapping]] = ...) -> None: ...
        class InPredicate(_message.Message):
            __slots__ = ("needles", "haystack")
            NEEDLES_FIELD_NUMBER: _ClassVar[int]
            HAYSTACK_FIELD_NUMBER: _ClassVar[int]
            needles: _containers.RepeatedCompositeFieldContainer[Expression]
            haystack: Rel
            def __init__(self, needles: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ..., haystack: _Optional[_Union[Rel, _Mapping]] = ...) -> None: ...
        class SetPredicate(_message.Message):
            __slots__ = ("predicate_op", "tuples")
            class PredicateOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = ()
                PREDICATE_OP_UNSPECIFIED: _ClassVar[Expression.Subquery.SetPredicate.PredicateOp]
                PREDICATE_OP_EXISTS: _ClassVar[Expression.Subquery.SetPredicate.PredicateOp]
                PREDICATE_OP_UNIQUE: _ClassVar[Expression.Subquery.SetPredicate.PredicateOp]
            PREDICATE_OP_UNSPECIFIED: Expression.Subquery.SetPredicate.PredicateOp
            PREDICATE_OP_EXISTS: Expression.Subquery.SetPredicate.PredicateOp
            PREDICATE_OP_UNIQUE: Expression.Subquery.SetPredicate.PredicateOp
            PREDICATE_OP_FIELD_NUMBER: _ClassVar[int]
            TUPLES_FIELD_NUMBER: _ClassVar[int]
            predicate_op: Expression.Subquery.SetPredicate.PredicateOp
            tuples: Rel
            def __init__(self, predicate_op: _Optional[_Union[Expression.Subquery.SetPredicate.PredicateOp, str]] = ..., tuples: _Optional[_Union[Rel, _Mapping]] = ...) -> None: ...
        class SetComparison(_message.Message):
            __slots__ = ("reduction_op", "comparison_op", "left", "right")
            class ComparisonOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = ()
                COMPARISON_OP_UNSPECIFIED: _ClassVar[Expression.Subquery.SetComparison.ComparisonOp]
                COMPARISON_OP_EQ: _ClassVar[Expression.Subquery.SetComparison.ComparisonOp]
                COMPARISON_OP_NE: _ClassVar[Expression.Subquery.SetComparison.ComparisonOp]
                COMPARISON_OP_LT: _ClassVar[Expression.Subquery.SetComparison.ComparisonOp]
                COMPARISON_OP_GT: _ClassVar[Expression.Subquery.SetComparison.ComparisonOp]
                COMPARISON_OP_LE: _ClassVar[Expression.Subquery.SetComparison.ComparisonOp]
                COMPARISON_OP_GE: _ClassVar[Expression.Subquery.SetComparison.ComparisonOp]
            COMPARISON_OP_UNSPECIFIED: Expression.Subquery.SetComparison.ComparisonOp
            COMPARISON_OP_EQ: Expression.Subquery.SetComparison.ComparisonOp
            COMPARISON_OP_NE: Expression.Subquery.SetComparison.ComparisonOp
            COMPARISON_OP_LT: Expression.Subquery.SetComparison.ComparisonOp
            COMPARISON_OP_GT: Expression.Subquery.SetComparison.ComparisonOp
            COMPARISON_OP_LE: Expression.Subquery.SetComparison.ComparisonOp
            COMPARISON_OP_GE: Expression.Subquery.SetComparison.ComparisonOp
            class ReductionOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = ()
                REDUCTION_OP_UNSPECIFIED: _ClassVar[Expression.Subquery.SetComparison.ReductionOp]
                REDUCTION_OP_ANY: _ClassVar[Expression.Subquery.SetComparison.ReductionOp]
                REDUCTION_OP_ALL: _ClassVar[Expression.Subquery.SetComparison.ReductionOp]
            REDUCTION_OP_UNSPECIFIED: Expression.Subquery.SetComparison.ReductionOp
            REDUCTION_OP_ANY: Expression.Subquery.SetComparison.ReductionOp
            REDUCTION_OP_ALL: Expression.Subquery.SetComparison.ReductionOp
            REDUCTION_OP_FIELD_NUMBER: _ClassVar[int]
            COMPARISON_OP_FIELD_NUMBER: _ClassVar[int]
            LEFT_FIELD_NUMBER: _ClassVar[int]
            RIGHT_FIELD_NUMBER: _ClassVar[int]
            reduction_op: Expression.Subquery.SetComparison.ReductionOp
            comparison_op: Expression.Subquery.SetComparison.ComparisonOp
            left: Expression
            right: Rel
            def __init__(self, reduction_op: _Optional[_Union[Expression.Subquery.SetComparison.ReductionOp, str]] = ..., comparison_op: _Optional[_Union[Expression.Subquery.SetComparison.ComparisonOp, str]] = ..., left: _Optional[_Union[Expression, _Mapping]] = ..., right: _Optional[_Union[Rel, _Mapping]] = ...) -> None: ...
        SCALAR_FIELD_NUMBER: _ClassVar[int]
        IN_PREDICATE_FIELD_NUMBER: _ClassVar[int]
        SET_PREDICATE_FIELD_NUMBER: _ClassVar[int]
        SET_COMPARISON_FIELD_NUMBER: _ClassVar[int]
        scalar: Expression.Subquery.Scalar
        in_predicate: Expression.Subquery.InPredicate
        set_predicate: Expression.Subquery.SetPredicate
        set_comparison: Expression.Subquery.SetComparison
        def __init__(self, scalar: _Optional[_Union[Expression.Subquery.Scalar, _Mapping]] = ..., in_predicate: _Optional[_Union[Expression.Subquery.InPredicate, _Mapping]] = ..., set_predicate: _Optional[_Union[Expression.Subquery.SetPredicate, _Mapping]] = ..., set_comparison: _Optional[_Union[Expression.Subquery.SetComparison, _Mapping]] = ...) -> None: ...
    LITERAL_FIELD_NUMBER: _ClassVar[int]
    SELECTION_FIELD_NUMBER: _ClassVar[int]
    SCALAR_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    IF_THEN_FIELD_NUMBER: _ClassVar[int]
    SWITCH_EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    SINGULAR_OR_LIST_FIELD_NUMBER: _ClassVar[int]
    MULTI_OR_LIST_FIELD_NUMBER: _ClassVar[int]
    CAST_FIELD_NUMBER: _ClassVar[int]
    SUBQUERY_FIELD_NUMBER: _ClassVar[int]
    NESTED_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_PARAMETER_FIELD_NUMBER: _ClassVar[int]
    ENUM_FIELD_NUMBER: _ClassVar[int]
    literal: Expression.Literal
    selection: Expression.FieldReference
    scalar_function: Expression.ScalarFunction
    window_function: Expression.WindowFunction
    if_then: Expression.IfThen
    switch_expression: Expression.SwitchExpression
    singular_or_list: Expression.SingularOrList
    multi_or_list: Expression.MultiOrList
    cast: Expression.Cast
    subquery: Expression.Subquery
    nested: Expression.Nested
    dynamic_parameter: DynamicParameter
    enum: Expression.Enum
    def __init__(self, literal: _Optional[_Union[Expression.Literal, _Mapping]] = ..., selection: _Optional[_Union[Expression.FieldReference, _Mapping]] = ..., scalar_function: _Optional[_Union[Expression.ScalarFunction, _Mapping]] = ..., window_function: _Optional[_Union[Expression.WindowFunction, _Mapping]] = ..., if_then: _Optional[_Union[Expression.IfThen, _Mapping]] = ..., switch_expression: _Optional[_Union[Expression.SwitchExpression, _Mapping]] = ..., singular_or_list: _Optional[_Union[Expression.SingularOrList, _Mapping]] = ..., multi_or_list: _Optional[_Union[Expression.MultiOrList, _Mapping]] = ..., cast: _Optional[_Union[Expression.Cast, _Mapping]] = ..., subquery: _Optional[_Union[Expression.Subquery, _Mapping]] = ..., nested: _Optional[_Union[Expression.Nested, _Mapping]] = ..., dynamic_parameter: _Optional[_Union[DynamicParameter, _Mapping]] = ..., enum: _Optional[_Union[Expression.Enum, _Mapping]] = ...) -> None: ...

class DynamicParameter(_message.Message):
    __slots__ = ("type", "parameter_reference")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    type: _type_pb2.Type
    parameter_reference: int
    def __init__(self, type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., parameter_reference: _Optional[int] = ...) -> None: ...

class SortField(_message.Message):
    __slots__ = ("expr", "direction", "comparison_function_reference")
    class SortDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SORT_DIRECTION_UNSPECIFIED: _ClassVar[SortField.SortDirection]
        SORT_DIRECTION_ASC_NULLS_FIRST: _ClassVar[SortField.SortDirection]
        SORT_DIRECTION_ASC_NULLS_LAST: _ClassVar[SortField.SortDirection]
        SORT_DIRECTION_DESC_NULLS_FIRST: _ClassVar[SortField.SortDirection]
        SORT_DIRECTION_DESC_NULLS_LAST: _ClassVar[SortField.SortDirection]
        SORT_DIRECTION_CLUSTERED: _ClassVar[SortField.SortDirection]
    SORT_DIRECTION_UNSPECIFIED: SortField.SortDirection
    SORT_DIRECTION_ASC_NULLS_FIRST: SortField.SortDirection
    SORT_DIRECTION_ASC_NULLS_LAST: SortField.SortDirection
    SORT_DIRECTION_DESC_NULLS_FIRST: SortField.SortDirection
    SORT_DIRECTION_DESC_NULLS_LAST: SortField.SortDirection
    SORT_DIRECTION_CLUSTERED: SortField.SortDirection
    EXPR_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    COMPARISON_FUNCTION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    expr: Expression
    direction: SortField.SortDirection
    comparison_function_reference: int
    def __init__(self, expr: _Optional[_Union[Expression, _Mapping]] = ..., direction: _Optional[_Union[SortField.SortDirection, str]] = ..., comparison_function_reference: _Optional[int] = ...) -> None: ...

class AggregateFunction(_message.Message):
    __slots__ = ("function_reference", "arguments", "options", "output_type", "phase", "sorts", "invocation", "args")
    class AggregationInvocation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        AGGREGATION_INVOCATION_UNSPECIFIED: _ClassVar[AggregateFunction.AggregationInvocation]
        AGGREGATION_INVOCATION_ALL: _ClassVar[AggregateFunction.AggregationInvocation]
        AGGREGATION_INVOCATION_DISTINCT: _ClassVar[AggregateFunction.AggregationInvocation]
    AGGREGATION_INVOCATION_UNSPECIFIED: AggregateFunction.AggregationInvocation
    AGGREGATION_INVOCATION_ALL: AggregateFunction.AggregationInvocation
    AGGREGATION_INVOCATION_DISTINCT: AggregateFunction.AggregationInvocation
    FUNCTION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    SORTS_FIELD_NUMBER: _ClassVar[int]
    INVOCATION_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    function_reference: int
    arguments: _containers.RepeatedCompositeFieldContainer[FunctionArgument]
    options: _containers.RepeatedCompositeFieldContainer[FunctionOption]
    output_type: _type_pb2.Type
    phase: AggregationPhase
    sorts: _containers.RepeatedCompositeFieldContainer[SortField]
    invocation: AggregateFunction.AggregationInvocation
    args: _containers.RepeatedCompositeFieldContainer[Expression]
    def __init__(self, function_reference: _Optional[int] = ..., arguments: _Optional[_Iterable[_Union[FunctionArgument, _Mapping]]] = ..., options: _Optional[_Iterable[_Union[FunctionOption, _Mapping]]] = ..., output_type: _Optional[_Union[_type_pb2.Type, _Mapping]] = ..., phase: _Optional[_Union[AggregationPhase, str]] = ..., sorts: _Optional[_Iterable[_Union[SortField, _Mapping]]] = ..., invocation: _Optional[_Union[AggregateFunction.AggregationInvocation, str]] = ..., args: _Optional[_Iterable[_Union[Expression, _Mapping]]] = ...) -> None: ...

class ReferenceRel(_message.Message):
    __slots__ = ("subtree_ordinal",)
    SUBTREE_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    subtree_ordinal: int
    def __init__(self, subtree_ordinal: _Optional[int] = ...) -> None: ...
