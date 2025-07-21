from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NumpyArray(_message.Message):
    __slots__ = ("data", "shape", "dtype")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...

class PydanticModelType(_message.Message):
    __slots__ = ("json_schema",)
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    json_schema: str
    def __init__(self, json_schema: _Optional[str] = ...) -> None: ...

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
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    input: _containers.ScalarMap[str, str]
    output: str
    def __init__(self, input: _Optional[_Mapping[str, str]] = ..., output: _Optional[str] = ...) -> None: ...

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
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    input: _containers.ScalarMap[str, str]
    output: bool
    def __init__(self, input: _Optional[_Mapping[str, str]] = ..., output: bool = ...) -> None: ...

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
    left: str
    right: str
    output: bool
    def __init__(self, left: _Optional[str] = ..., right: _Optional[str] = ..., output: bool = ...) -> None: ...

class JoinExampleCollection(_message.Message):
    __slots__ = ("examples",)
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    examples: _containers.RepeatedCompositeFieldContainer[JoinExample]
    def __init__(self, examples: _Optional[_Iterable[_Union[JoinExample, _Mapping]]] = ...) -> None: ...
