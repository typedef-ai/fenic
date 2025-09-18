from fenic._gen.protos.logical_plan.v1 import datatypes_pb2 as _datatypes_pb2
from fenic._gen.protos.logical_plan.v1 import complex_types_pb2 as _complex_types_pb2
from fenic._gen.protos.logical_plan.v1 import plans_pb2 as _plans_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NumericConstraint(_message.Message):
    __slots__ = ("int_value", "float_value")
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    int_value: int
    float_value: float
    def __init__(self, int_value: _Optional[int] = ..., float_value: _Optional[float] = ...) -> None: ...

class ToolParameterConstraints(_message.Message):
    __slots__ = ("gt", "ge", "lt", "le", "multiple_of", "min_length", "max_length", "pattern")
    GT_FIELD_NUMBER: _ClassVar[int]
    GE_FIELD_NUMBER: _ClassVar[int]
    LT_FIELD_NUMBER: _ClassVar[int]
    LE_FIELD_NUMBER: _ClassVar[int]
    MULTIPLE_OF_FIELD_NUMBER: _ClassVar[int]
    MIN_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_LENGTH_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    gt: NumericConstraint
    ge: NumericConstraint
    lt: NumericConstraint
    le: NumericConstraint
    multiple_of: NumericConstraint
    min_length: int
    max_length: int
    pattern: str
    def __init__(self, gt: _Optional[_Union[NumericConstraint, _Mapping]] = ..., ge: _Optional[_Union[NumericConstraint, _Mapping]] = ..., lt: _Optional[_Union[NumericConstraint, _Mapping]] = ..., le: _Optional[_Union[NumericConstraint, _Mapping]] = ..., multiple_of: _Optional[_Union[NumericConstraint, _Mapping]] = ..., min_length: _Optional[int] = ..., max_length: _Optional[int] = ..., pattern: _Optional[str] = ...) -> None: ...

class ToolParameter(_message.Message):
    __slots__ = ("name", "description", "data_type", "required", "has_default", "default_value", "allowed_values", "constraints", "validator_names")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    HAS_DEFAULT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_VALUES_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    VALIDATOR_NAMES_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    data_type: _datatypes_pb2.DataType
    required: bool
    has_default: bool
    default_value: _complex_types_pb2.ScalarValue
    allowed_values: _containers.RepeatedCompositeFieldContainer[_complex_types_pb2.ScalarValue]
    constraints: ToolParameterConstraints
    validator_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., data_type: _Optional[_Union[_datatypes_pb2.DataType, _Mapping]] = ..., required: bool = ..., has_default: bool = ..., default_value: _Optional[_Union[_complex_types_pb2.ScalarValue, _Mapping]] = ..., allowed_values: _Optional[_Iterable[_Union[_complex_types_pb2.ScalarValue, _Mapping]]] = ..., constraints: _Optional[_Union[ToolParameterConstraints, _Mapping]] = ..., validator_names: _Optional[_Iterable[str]] = ...) -> None: ...

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
