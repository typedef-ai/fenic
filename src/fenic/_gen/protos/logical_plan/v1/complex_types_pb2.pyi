from fenic._gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ScalarValue(_message.Message):
    __slots__ = ("string_value", "int_value", "double_value", "bool_value", "bytes_value", "array_value", "struct_value", "date_value", "timestamp_value")
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BYTES_VALUE_FIELD_NUMBER: _ClassVar[int]
    ARRAY_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRUCT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DATE_VALUE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_VALUE_FIELD_NUMBER: _ClassVar[int]
    string_value: str
    int_value: int
    double_value: float
    bool_value: bool
    bytes_value: bytes
    array_value: ScalarArray
    struct_value: ScalarStruct
    date_value: int
    timestamp_value: float
    def __init__(self, string_value: _Optional[str] = ..., int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., bool_value: bool = ..., bytes_value: _Optional[bytes] = ..., array_value: _Optional[_Union[ScalarArray, _Mapping]] = ..., struct_value: _Optional[_Union[ScalarStruct, _Mapping]] = ..., date_value: _Optional[int] = ..., timestamp_value: _Optional[float] = ...) -> None: ...

class ScalarArray(_message.Message):
    __slots__ = ("elements",)
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedCompositeFieldContainer[ScalarValue]
    def __init__(self, elements: _Optional[_Iterable[_Union[ScalarValue, _Mapping]]] = ...) -> None: ...

class ScalarStruct(_message.Message):
    __slots__ = ("fields",)
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[ScalarStructField]
    def __init__(self, fields: _Optional[_Iterable[_Union[ScalarStructField, _Mapping]]] = ...) -> None: ...

class ScalarStructField(_message.Message):
    __slots__ = ("name", "value")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: ScalarValue
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[ScalarValue, _Mapping]] = ...) -> None: ...

class ResolvedClassDefinition(_message.Message):
    __slots__ = ("label", "description")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    label: str
    description: str
    def __init__(self, label: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class ResolvedModelAlias(_message.Message):
    __slots__ = ("name", "profile")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROFILE_FIELD_NUMBER: _ClassVar[int]
    name: str
    profile: str
    def __init__(self, name: _Optional[str] = ..., profile: _Optional[str] = ...) -> None: ...

class ResolvedResponseFormat(_message.Message):
    __slots__ = ("schema", "struct_type", "prompt_schema_definition")
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    STRUCT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_SCHEMA_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    schema: str
    struct_type: _datatypes_pb2.DataType
    prompt_schema_definition: str
    def __init__(self, schema: _Optional[str] = ..., struct_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ..., prompt_schema_definition: _Optional[str] = ...) -> None: ...

class NumpyArray(_message.Message):
    __slots__ = ("data", "shape", "dtype")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...

class KeyPoints(_message.Message):
    __slots__ = ("max_points",)
    MAX_POINTS_FIELD_NUMBER: _ClassVar[int]
    max_points: int
    def __init__(self, max_points: _Optional[int] = ...) -> None: ...

class Paragraph(_message.Message):
    __slots__ = ("max_words",)
    MAX_WORDS_FIELD_NUMBER: _ClassVar[int]
    max_words: int
    def __init__(self, max_words: _Optional[int] = ...) -> None: ...

class SummarizationFormat(_message.Message):
    __slots__ = ("key_points", "paragraph")
    KEY_POINTS_FIELD_NUMBER: _ClassVar[int]
    PARAGRAPH_FIELD_NUMBER: _ClassVar[int]
    key_points: KeyPoints
    paragraph: Paragraph
    def __init__(self, key_points: _Optional[_Union[KeyPoints, _Mapping]] = ..., paragraph: _Optional[_Union[Paragraph, _Mapping]] = ...) -> None: ...

class MapExample(_message.Message):
    __slots__ = ("input", "output")
    class InputEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScalarValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ScalarValue, _Mapping]] = ...) -> None: ...
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    input: _containers.MessageMap[str, ScalarValue]
    output: str
    def __init__(self, input: _Optional[_Mapping[str, ScalarValue]] = ..., output: _Optional[str] = ...) -> None: ...

class MapExampleCollection(_message.Message):
    __slots__ = ("examples",)
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    examples: _containers.RepeatedCompositeFieldContainer[MapExample]
    def __init__(self, examples: _Optional[_Iterable[_Union[MapExample, _Mapping]]] = ...) -> None: ...

class ClassifyExample(_message.Message):
    __slots__ = ("input", "output")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    input: str
    output: str
    def __init__(self, input: _Optional[str] = ..., output: _Optional[str] = ...) -> None: ...

class ClassifyExampleCollection(_message.Message):
    __slots__ = ("examples",)
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    examples: _containers.RepeatedCompositeFieldContainer[ClassifyExample]
    def __init__(self, examples: _Optional[_Iterable[_Union[ClassifyExample, _Mapping]]] = ...) -> None: ...

class PredicateExample(_message.Message):
    __slots__ = ("input", "output")
    class InputEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ScalarValue
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ScalarValue, _Mapping]] = ...) -> None: ...
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    input: _containers.MessageMap[str, ScalarValue]
    output: bool
    def __init__(self, input: _Optional[_Mapping[str, ScalarValue]] = ..., output: bool = ...) -> None: ...

class PredicateExampleCollection(_message.Message):
    __slots__ = ("examples",)
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    examples: _containers.RepeatedCompositeFieldContainer[PredicateExample]
    def __init__(self, examples: _Optional[_Iterable[_Union[PredicateExample, _Mapping]]] = ...) -> None: ...

class JoinExample(_message.Message):
    __slots__ = ("left", "right", "output")
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    left: ScalarValue
    right: ScalarValue
    output: bool
    def __init__(self, left: _Optional[_Union[ScalarValue, _Mapping]] = ..., right: _Optional[_Union[ScalarValue, _Mapping]] = ..., output: bool = ...) -> None: ...

class JoinExampleCollection(_message.Message):
    __slots__ = ("examples",)
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    examples: _containers.RepeatedCompositeFieldContainer[JoinExample]
    def __init__(self, examples: _Optional[_Iterable[_Union[JoinExample, _Mapping]]] = ...) -> None: ...
