from fenic._gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic._gen.protos.logical_plan.v1 import complex_types_pb2 as _complex_types_pb2
from fenic._gen.protos.logical_plan.v1 import plans_pb2 as _plans_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ToolParameter(_message.Message):
    __slots__ = ("name", "description", "data_type", "required", "has_default", "default_value", "allowed_values")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    HAS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_VALUES_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    data_type: _datatypes_pb2.DataType
    required: bool
    has_default: bool
    default_value: _complex_types_pb2.ScalarValue
    allowed_values: _containers.RepeatedCompositeFieldContainer[_complex_types_pb2.ScalarValue]
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ..., required: bool = ..., has_default: bool = ..., default_value: _Optional[_Union[_complex_types_pb2.ScalarValue, _Mapping]] = ..., allowed_values: _Optional[_Iterable[_Union[_complex_types_pb2.ScalarValue, _Mapping]]] = ...) -> None: ...

class ToolDefinition(_message.Message):
    __slots__ = ("name", "description", "params", "parameterized_view", "result_limit")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    PARAMETERIZED_VIEW_FIELD_NUMBER: _ClassVar[int]
    RESULT_LIMIT_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    params: _containers.RepeatedCompositeFieldContainer[ToolParameter]
    parameterized_view: _plans_pb2.LogicalPlan
    result_limit: int
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., params: _Optional[_Iterable[_Union[ToolParameter, _Mapping]]] = ..., parameterized_view: _Optional[_Union[_plans_pb2.LogicalPlan, _Mapping]] = ..., result_limit: _Optional[int] = ...) -> None: ...
