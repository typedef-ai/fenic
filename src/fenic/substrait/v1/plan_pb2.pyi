from substrait.v1 import algebra_pb2 as _algebra_pb2
from substrait.v1.extensions import extensions_pb2 as _extensions_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PlanRel(_message.Message):
    __slots__ = ("rel", "root")
    REL_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    rel: _algebra_pb2.Rel
    root: _algebra_pb2.RelRoot
    def __init__(self, rel: _Optional[_Union[_algebra_pb2.Rel, _Mapping]] = ..., root: _Optional[_Union[_algebra_pb2.RelRoot, _Mapping]] = ...) -> None: ...

class Plan(_message.Message):
    __slots__ = ("version", "extension_uris", "extensions", "relations", "advanced_extensions", "expected_type_urls", "parameter_bindings")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    EXTENSION_URIS_FIELD_NUMBER: _ClassVar[int]
    EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    ADVANCED_EXTENSIONS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_TYPE_URLS_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_BINDINGS_FIELD_NUMBER: _ClassVar[int]
    version: Version
    extension_uris: _containers.RepeatedCompositeFieldContainer[_extensions_pb2.SimpleExtensionURI]
    extensions: _containers.RepeatedCompositeFieldContainer[_extensions_pb2.SimpleExtensionDeclaration]
    relations: _containers.RepeatedCompositeFieldContainer[PlanRel]
    advanced_extensions: _extensions_pb2.AdvancedExtension
    expected_type_urls: _containers.RepeatedScalarFieldContainer[str]
    parameter_bindings: _containers.RepeatedCompositeFieldContainer[DynamicParameterBinding]
    def __init__(self, version: _Optional[_Union[Version, _Mapping]] = ..., extension_uris: _Optional[_Iterable[_Union[_extensions_pb2.SimpleExtensionURI, _Mapping]]] = ..., extensions: _Optional[_Iterable[_Union[_extensions_pb2.SimpleExtensionDeclaration, _Mapping]]] = ..., relations: _Optional[_Iterable[_Union[PlanRel, _Mapping]]] = ..., advanced_extensions: _Optional[_Union[_extensions_pb2.AdvancedExtension, _Mapping]] = ..., expected_type_urls: _Optional[_Iterable[str]] = ..., parameter_bindings: _Optional[_Iterable[_Union[DynamicParameterBinding, _Mapping]]] = ...) -> None: ...

class PlanVersion(_message.Message):
    __slots__ = ("version",)
    VERSION_FIELD_NUMBER: _ClassVar[int]
    version: Version
    def __init__(self, version: _Optional[_Union[Version, _Mapping]] = ...) -> None: ...

class Version(_message.Message):
    __slots__ = ("major_number", "minor_number", "patch_number", "git_hash", "producer")
    MAJOR_NUMBER_FIELD_NUMBER: _ClassVar[int]
    MINOR_NUMBER_FIELD_NUMBER: _ClassVar[int]
    PATCH_NUMBER_FIELD_NUMBER: _ClassVar[int]
    GIT_HASH_FIELD_NUMBER: _ClassVar[int]
    PRODUCER_FIELD_NUMBER: _ClassVar[int]
    major_number: int
    minor_number: int
    patch_number: int
    git_hash: str
    producer: str
    def __init__(self, major_number: _Optional[int] = ..., minor_number: _Optional[int] = ..., patch_number: _Optional[int] = ..., git_hash: _Optional[str] = ..., producer: _Optional[str] = ...) -> None: ...

class DynamicParameterBinding(_message.Message):
    __slots__ = ("parameter_anchor", "value")
    PARAMETER_ANCHOR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    parameter_anchor: int
    value: _algebra_pb2.Expression.Literal
    def __init__(self, parameter_anchor: _Optional[int] = ..., value: _Optional[_Union[_algebra_pb2.Expression.Literal, _Mapping]] = ...) -> None: ...
