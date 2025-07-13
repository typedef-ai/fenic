from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Capabilities(_message.Message):
    __slots__ = ("substrait_versions", "advanced_extension_type_urls", "simple_extensions")
    class SimpleExtension(_message.Message):
        __slots__ = ("uri", "function_keys", "type_keys", "type_variation_keys")
        URI_FIELD_NUMBER: _ClassVar[int]
        FUNCTION_KEYS_FIELD_NUMBER: _ClassVar[int]
        TYPE_KEYS_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_KEYS_FIELD_NUMBER: _ClassVar[int]
        uri: str
        function_keys: _containers.RepeatedScalarFieldContainer[str]
        type_keys: _containers.RepeatedScalarFieldContainer[str]
        type_variation_keys: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, uri: _Optional[str] = ..., function_keys: _Optional[_Iterable[str]] = ..., type_keys: _Optional[_Iterable[str]] = ..., type_variation_keys: _Optional[_Iterable[str]] = ...) -> None: ...
    SUBSTRAIT_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSION_TYPE_URLS_FIELD_NUMBER: _ClassVar[int]
    SIMPLE_EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    substrait_versions: _containers.RepeatedScalarFieldContainer[str]
    advanced_extension_type_urls: _containers.RepeatedScalarFieldContainer[str]
    simple_extensions: _containers.RepeatedCompositeFieldContainer[Capabilities.SimpleExtension]
    def __init__(self, substrait_versions: _Optional[_Iterable[str]] = ..., advanced_extension_type_urls: _Optional[_Iterable[str]] = ..., simple_extensions: _Optional[_Iterable[_Union[Capabilities.SimpleExtension, _Mapping]]] = ...) -> None: ...
