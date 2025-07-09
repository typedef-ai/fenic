from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataTypeProto(_message.Message):
    __slots__ = ("string", "integer", "float", "double", "boolean", "array", "struct", "embedding", "transcript", "document_backed_path", "markdown", "html", "json")
    STRING_FIELD_NUMBER: _ClassVar[int]
    INTEGER_FIELD_NUMBER: _ClassVar[int]
    FLOAT_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_FIELD_NUMBER: _ClassVar[int]
    BOOLEAN_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FIELD_NUMBER: _ClassVar[int]
    STRUCT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_BACKED_PATH_FIELD_NUMBER: _ClassVar[int]
    MARKDOWN_FIELD_NUMBER: _ClassVar[int]
    HTML_FIELD_NUMBER: _ClassVar[int]
    JSON_FIELD_NUMBER: _ClassVar[int]
    string: StringTypeProto
    integer: IntegerTypeProto
    float: FloatTypeProto
    double: DoubleTypeProto
    boolean: BooleanTypeProto
    array: ArrayTypeProto
    struct: StructTypeProto
    embedding: EmbeddingTypeProto
    transcript: TranscriptTypeProto
    document_backed_path: DocumentBackedPathProto
    markdown: MarkdownTypeProto
    html: HTMLTypeProto
    json: JSONTypeProto
    def __init__(self, string: _Optional[_Union[StringTypeProto, _Mapping]] = ..., integer: _Optional[_Union[IntegerTypeProto, _Mapping]] = ..., float: _Optional[_Union[FloatTypeProto, _Mapping]] = ..., double: _Optional[_Union[DoubleTypeProto, _Mapping]] = ..., boolean: _Optional[_Union[BooleanTypeProto, _Mapping]] = ..., array: _Optional[_Union[ArrayTypeProto, _Mapping]] = ..., struct: _Optional[_Union[StructTypeProto, _Mapping]] = ..., embedding: _Optional[_Union[EmbeddingTypeProto, _Mapping]] = ..., transcript: _Optional[_Union[TranscriptTypeProto, _Mapping]] = ..., document_backed_path: _Optional[_Union[DocumentBackedPathProto, _Mapping]] = ..., markdown: _Optional[_Union[MarkdownTypeProto, _Mapping]] = ..., html: _Optional[_Union[HTMLTypeProto, _Mapping]] = ..., json: _Optional[_Union[JSONTypeProto, _Mapping]] = ...) -> None: ...

class StringTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class IntegerTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class FloatTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DoubleTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BooleanTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ArrayTypeProto(_message.Message):
    __slots__ = ("element_type",)
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    element_type: DataTypeProto
    def __init__(self, element_type: _Optional[_Union[DataTypeProto, _Mapping]] = ...) -> None: ...

class StructTypeProto(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[StructFieldProto]
    def __init__(self, fields: _Optional[_Iterable[_Union[StructFieldProto, _Mapping]]] = ...) -> None: ...

class StructFieldProto(_message.Message):
    __slots__ = ("name", "data_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: DataTypeProto
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[DataTypeProto, _Mapping]] = ...) -> None: ...

class EmbeddingTypeProto(_message.Message):
    __slots__ = ("dimensions", "embedding_model")
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    dimensions: int
    embedding_model: str
    def __init__(self, dimensions: _Optional[int] = ..., embedding_model: _Optional[str] = ...) -> None: ...

class TranscriptTypeProto(_message.Message):
    __slots__ = ("format",)
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    format: str
    def __init__(self, format: _Optional[str] = ...) -> None: ...

class DocumentBackedPathProto(_message.Message):
    __slots__ = ("format",)
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    format: str
    def __init__(self, format: _Optional[str] = ...) -> None: ...

class MarkdownTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HTMLTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class JSONTypeProto(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
