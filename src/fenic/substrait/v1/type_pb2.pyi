from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Type(_message.Message):
    __slots__ = ("bool", "i8", "i16", "i32", "i64", "fp32", "fp64", "string", "binary", "timestamp", "date", "time", "interval_year", "interval_day", "interval_compound", "timestamp_tz", "uuid", "fixed_char", "varchar", "fixed_binary", "decimal", "precision_time", "precision_timestamp", "precision_timestamp_tz", "struct", "list", "map", "user_defined", "user_defined_type_reference")
    class Nullability(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        NULLABILITY_UNSPECIFIED: _ClassVar[Type.Nullability]
        NULLABILITY_NULLABLE: _ClassVar[Type.Nullability]
        NULLABILITY_REQUIRED: _ClassVar[Type.Nullability]
    NULLABILITY_UNSPECIFIED: Type.Nullability
    NULLABILITY_NULLABLE: Type.Nullability
    NULLABILITY_REQUIRED: Type.Nullability
    class Boolean(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class I8(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class I16(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class I32(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class I64(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class FP32(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class FP64(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class String(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class Binary(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class Timestamp(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class Date(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class Time(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class TimestampTZ(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class IntervalYear(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class IntervalDay(_message.Message):
        __slots__ = ("type_variation_reference", "nullability", "precision")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        precision: int
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ..., precision: _Optional[int] = ...) -> None: ...
    class IntervalCompound(_message.Message):
        __slots__ = ("type_variation_reference", "nullability", "precision")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        precision: int
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ..., precision: _Optional[int] = ...) -> None: ...
    class UUID(_message.Message):
        __slots__ = ("type_variation_reference", "nullability")
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class FixedChar(_message.Message):
        __slots__ = ("length", "type_variation_reference", "nullability")
        LENGTH_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        length: int
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, length: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class VarChar(_message.Message):
        __slots__ = ("length", "type_variation_reference", "nullability")
        LENGTH_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        length: int
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, length: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class FixedBinary(_message.Message):
        __slots__ = ("length", "type_variation_reference", "nullability")
        LENGTH_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        length: int
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, length: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class Decimal(_message.Message):
        __slots__ = ("scale", "precision", "type_variation_reference", "nullability")
        SCALE_FIELD_NUMBER: _ClassVar[int]
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        scale: int
        precision: int
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, scale: _Optional[int] = ..., precision: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class PrecisionTime(_message.Message):
        __slots__ = ("precision", "type_variation_reference", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: int
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, precision: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class PrecisionTimestamp(_message.Message):
        __slots__ = ("precision", "type_variation_reference", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: int
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, precision: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class PrecisionTimestampTZ(_message.Message):
        __slots__ = ("precision", "type_variation_reference", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: int
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, precision: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class Struct(_message.Message):
        __slots__ = ("types", "type_variation_reference", "nullability")
        TYPES_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        types: _containers.RepeatedCompositeFieldContainer[Type]
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, types: _Optional[_Iterable[_Union[Type, _Mapping]]] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class List(_message.Message):
        __slots__ = ("type", "type_variation_reference", "nullability")
        TYPE_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type: Type
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, type: _Optional[_Union[Type, _Mapping]] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class Map(_message.Message):
        __slots__ = ("key", "value", "type_variation_reference", "nullability")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        key: Type
        value: Type
        type_variation_reference: int
        nullability: Type.Nullability
        def __init__(self, key: _Optional[_Union[Type, _Mapping]] = ..., value: _Optional[_Union[Type, _Mapping]] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ...) -> None: ...
    class UserDefined(_message.Message):
        __slots__ = ("type_reference", "type_variation_reference", "nullability", "type_parameters")
        TYPE_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        TYPE_VARIATION_REFERENCE_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        TYPE_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
        type_reference: int
        type_variation_reference: int
        nullability: Type.Nullability
        type_parameters: _containers.RepeatedCompositeFieldContainer[Type.Parameter]
        def __init__(self, type_reference: _Optional[int] = ..., type_variation_reference: _Optional[int] = ..., nullability: _Optional[_Union[Type.Nullability, str]] = ..., type_parameters: _Optional[_Iterable[_Union[Type.Parameter, _Mapping]]] = ...) -> None: ...
    class Parameter(_message.Message):
        __slots__ = ("null", "data_type", "boolean", "integer", "enum", "string")
        NULL_FIELD_NUMBER: _ClassVar[int]
        DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
        BOOLEAN_FIELD_NUMBER: _ClassVar[int]
        INTEGER_FIELD_NUMBER: _ClassVar[int]
        ENUM_FIELD_NUMBER: _ClassVar[int]
        STRING_FIELD_NUMBER: _ClassVar[int]
        null: _empty_pb2.Empty
        data_type: Type
        boolean: bool
        integer: int
        enum: str
        string: str
        def __init__(self, null: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., data_type: _Optional[_Union[Type, _Mapping]] = ..., boolean: bool = ..., integer: _Optional[int] = ..., enum: _Optional[str] = ..., string: _Optional[str] = ...) -> None: ...
    BOOL_FIELD_NUMBER: _ClassVar[int]
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
    INTERVAL_YEAR_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_DAY_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_COMPOUND_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_TZ_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    FIXED_CHAR_FIELD_NUMBER: _ClassVar[int]
    VARCHAR_FIELD_NUMBER: _ClassVar[int]
    FIXED_BINARY_FIELD_NUMBER: _ClassVar[int]
    DECIMAL_FIELD_NUMBER: _ClassVar[int]
    PRECISION_TIME_FIELD_NUMBER: _ClassVar[int]
    PRECISION_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    PRECISION_TIMESTAMP_TZ_FIELD_NUMBER: _ClassVar[int]
    STRUCT_FIELD_NUMBER: _ClassVar[int]
    LIST_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    USER_DEFINED_FIELD_NUMBER: _ClassVar[int]
    USER_DEFINED_TYPE_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    bool: Type.Boolean
    i8: Type.I8
    i16: Type.I16
    i32: Type.I32
    i64: Type.I64
    fp32: Type.FP32
    fp64: Type.FP64
    string: Type.String
    binary: Type.Binary
    timestamp: Type.Timestamp
    date: Type.Date
    time: Type.Time
    interval_year: Type.IntervalYear
    interval_day: Type.IntervalDay
    interval_compound: Type.IntervalCompound
    timestamp_tz: Type.TimestampTZ
    uuid: Type.UUID
    fixed_char: Type.FixedChar
    varchar: Type.VarChar
    fixed_binary: Type.FixedBinary
    decimal: Type.Decimal
    precision_time: Type.PrecisionTime
    precision_timestamp: Type.PrecisionTimestamp
    precision_timestamp_tz: Type.PrecisionTimestampTZ
    struct: Type.Struct
    list: Type.List
    map: Type.Map
    user_defined: Type.UserDefined
    user_defined_type_reference: int
    def __init__(self, bool: _Optional[_Union[Type.Boolean, _Mapping]] = ..., i8: _Optional[_Union[Type.I8, _Mapping]] = ..., i16: _Optional[_Union[Type.I16, _Mapping]] = ..., i32: _Optional[_Union[Type.I32, _Mapping]] = ..., i64: _Optional[_Union[Type.I64, _Mapping]] = ..., fp32: _Optional[_Union[Type.FP32, _Mapping]] = ..., fp64: _Optional[_Union[Type.FP64, _Mapping]] = ..., string: _Optional[_Union[Type.String, _Mapping]] = ..., binary: _Optional[_Union[Type.Binary, _Mapping]] = ..., timestamp: _Optional[_Union[Type.Timestamp, _Mapping]] = ..., date: _Optional[_Union[Type.Date, _Mapping]] = ..., time: _Optional[_Union[Type.Time, _Mapping]] = ..., interval_year: _Optional[_Union[Type.IntervalYear, _Mapping]] = ..., interval_day: _Optional[_Union[Type.IntervalDay, _Mapping]] = ..., interval_compound: _Optional[_Union[Type.IntervalCompound, _Mapping]] = ..., timestamp_tz: _Optional[_Union[Type.TimestampTZ, _Mapping]] = ..., uuid: _Optional[_Union[Type.UUID, _Mapping]] = ..., fixed_char: _Optional[_Union[Type.FixedChar, _Mapping]] = ..., varchar: _Optional[_Union[Type.VarChar, _Mapping]] = ..., fixed_binary: _Optional[_Union[Type.FixedBinary, _Mapping]] = ..., decimal: _Optional[_Union[Type.Decimal, _Mapping]] = ..., precision_time: _Optional[_Union[Type.PrecisionTime, _Mapping]] = ..., precision_timestamp: _Optional[_Union[Type.PrecisionTimestamp, _Mapping]] = ..., precision_timestamp_tz: _Optional[_Union[Type.PrecisionTimestampTZ, _Mapping]] = ..., struct: _Optional[_Union[Type.Struct, _Mapping]] = ..., list: _Optional[_Union[Type.List, _Mapping]] = ..., map: _Optional[_Union[Type.Map, _Mapping]] = ..., user_defined: _Optional[_Union[Type.UserDefined, _Mapping]] = ..., user_defined_type_reference: _Optional[int] = ...) -> None: ...

class NamedStruct(_message.Message):
    __slots__ = ("names", "struct")
    NAMES_FIELD_NUMBER: _ClassVar[int]
    STRUCT_FIELD_NUMBER: _ClassVar[int]
    names: _containers.RepeatedScalarFieldContainer[str]
    struct: Type.Struct
    def __init__(self, names: _Optional[_Iterable[str]] = ..., struct: _Optional[_Union[Type.Struct, _Mapping]] = ...) -> None: ...
