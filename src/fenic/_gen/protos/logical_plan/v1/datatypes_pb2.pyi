from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(_message.Message):
    __slots__ = ("string", "integer", "float", "double", "boolean", "array", "struct", "embedding", "transcript", "document_path", "markdown", "html", "json", "date", "timestamp")
    STRING_FIELD_NUMBER: _ClassVar[int]
    INTEGER_FIELD_NUMBER: _ClassVar[int]
    FLOAT_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_FIELD_NUMBER: _ClassVar[int]
    BOOLEAN_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    STRUCT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_PATH_FIELD_NUMBER: _ClassVar[int]
    MARKDOWN_FIELD_NUMBER: _ClassVar[int]
    HTML_FIELD_NUMBER: _ClassVar[int]
    JSON_FIELD_NUMBER: _ClassVar[int]
    DATE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    string: StringType
    integer: IntegerType
    float: FloatType
    double: DoubleType
    boolean: BooleanType
    array: ArrayType
    struct: StructType
    embedding: EmbeddingType
    transcript: TranscriptType
    document_path: DocumentPathType
    markdown: MarkdownType
    html: HTMLType
    json: JSONType
    date: DateType
    timestamp: TimestampType
    def __init__(self, string: _Optional[_Union[StringType, _Mapping]] = ..., integer: _Optional[_Union[IntegerType, _Mapping]] = ..., float: _Optional[_Union[FloatType, _Mapping]] = ..., double: _Optional[_Union[DoubleType, _Mapping]] = ..., boolean: _Optional[_Union[BooleanType, _Mapping]] = ..., array: _Optional[_Union[ArrayType, _Mapping]] = ..., struct: _Optional[_Union[StructType, _Mapping]] = ..., embedding: _Optional[_Union[EmbeddingType, _Mapping]] = ..., transcript: _Optional[_Union[TranscriptType, _Mapping]] = ..., document_path: _Optional[_Union[DocumentPathType, _Mapping]] = ..., markdown: _Optional[_Union[MarkdownType, _Mapping]] = ..., html: _Optional[_Union[HTMLType, _Mapping]] = ..., json: _Optional[_Union[JSONType, _Mapping]] = ..., date: _Optional[_Union[DateType, _Mapping]] = ..., timestamp: _Optional[_Union[TimestampType, _Mapping]] = ...) -> None: ...

class StringType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class IntegerType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FloatType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DoubleType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BooleanType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DateType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class TimestampType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ArrayType(_message.Message):
    __slots__ = ("element_type",)
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    element_type: DataType
    def __init__(self, element_type: _Optional[_Union[DataType, _Mapping]] = ...) -> None: ...

class StructType(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[StructField]
    def __init__(self, fields: _Optional[_Iterable[_Union[StructField, _Mapping]]] = ...) -> None: ...

class StructField(_message.Message):
    __slots__ = ("name", "data_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: DataType
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[DataType, _Mapping]] = ...) -> None: ...

class EmbeddingType(_message.Message):
    __slots__ = ("dimensions", "embedding_model")
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    dimensions: int
    embedding_model: str
    def __init__(self, dimensions: _Optional[int] = ..., embedding_model: _Optional[str] = ...) -> None: ...

class TranscriptType(_message.Message):
    __slots__ = ("format",)
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    format: str
    def __init__(self, format: _Optional[str] = ...) -> None: ...

class DocumentPathType(_message.Message):
    __slots__ = ("format",)
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    format: str
    def __init__(self, format: _Optional[str] = ...) -> None: ...

class MarkdownType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HTMLType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class JSONType(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
