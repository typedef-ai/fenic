from substrait.v1 import type_pb2 as _type_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DerivationExpression(_message.Message):
    __slots__ = ("bool", "i8", "i16", "i32", "i64", "fp32", "fp64", "string", "binary", "timestamp", "date", "time", "interval_year", "timestamp_tz", "uuid", "interval_day", "interval_compound", "fixed_char", "varchar", "fixed_binary", "decimal", "precision_time", "precision_timestamp", "precision_timestamp_tz", "struct", "list", "map", "user_defined", "user_defined_pointer", "type_parameter_name", "integer_parameter_name", "integer_literal", "unary_op", "binary_op", "if_else", "return_program")
    class ExpressionFixedChar(_message.Message):
        __slots__ = ("length", "variation_pointer", "nullability")
        LENGTH_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        length: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, length: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionVarChar(_message.Message):
        __slots__ = ("length", "variation_pointer", "nullability")
        LENGTH_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        length: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, length: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionFixedBinary(_message.Message):
        __slots__ = ("length", "variation_pointer", "nullability")
        LENGTH_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        length: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, length: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionDecimal(_message.Message):
        __slots__ = ("scale", "precision", "variation_pointer", "nullability")
        SCALE_FIELD_NUMBER: _ClassVar[int]
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        scale: DerivationExpression
        precision: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, scale: _Optional[_Union[DerivationExpression, _Mapping]] = ..., precision: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionPrecisionTime(_message.Message):
        __slots__ = ("precision", "variation_pointer", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, precision: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionPrecisionTimestamp(_message.Message):
        __slots__ = ("precision", "variation_pointer", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, precision: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionIntervalDay(_message.Message):
        __slots__ = ("precision", "variation_pointer", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, precision: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionIntervalCompound(_message.Message):
        __slots__ = ("precision", "variation_pointer", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, precision: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionPrecisionTimestampTZ(_message.Message):
        __slots__ = ("precision", "variation_pointer", "nullability")
        PRECISION_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        precision: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, precision: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionStruct(_message.Message):
        __slots__ = ("types", "variation_pointer", "nullability")
        TYPES_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        types: _containers.RepeatedCompositeFieldContainer[DerivationExpression]
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, types: _Optional[_Iterable[_Union[DerivationExpression, _Mapping]]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionNamedStruct(_message.Message):
        __slots__ = ("names", "struct")
        NAMES_FIELD_NUMBER: _ClassVar[int]
        STRUCT_FIELD_NUMBER: _ClassVar[int]
        names: _containers.RepeatedScalarFieldContainer[str]
        struct: DerivationExpression.ExpressionStruct
        def __init__(self, names: _Optional[_Iterable[str]] = ..., struct: _Optional[_Union[DerivationExpression.ExpressionStruct, _Mapping]] = ...) -> None: ...
    class ExpressionList(_message.Message):
        __slots__ = ("type", "variation_pointer", "nullability")
        TYPE_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, type: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionMap(_message.Message):
        __slots__ = ("key", "value", "variation_pointer", "nullability")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        key: DerivationExpression
        value: DerivationExpression
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, key: _Optional[_Union[DerivationExpression, _Mapping]] = ..., value: _Optional[_Union[DerivationExpression, _Mapping]] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class ExpressionUserDefined(_message.Message):
        __slots__ = ("type_pointer", "variation_pointer", "nullability")
        TYPE_POINTER_FIELD_NUMBER: _ClassVar[int]
        VARIATION_POINTER_FIELD_NUMBER: _ClassVar[int]
        NULLABILITY_FIELD_NUMBER: _ClassVar[int]
        type_pointer: int
        variation_pointer: int
        nullability: _type_pb2.Type.Nullability
        def __init__(self, type_pointer: _Optional[int] = ..., variation_pointer: _Optional[int] = ..., nullability: _Optional[_Union[_type_pb2.Type.Nullability, str]] = ...) -> None: ...
    class IfElse(_message.Message):
        __slots__ = ("if_condition", "if_return", "else_return")
        IF_CONDITION_FIELD_NUMBER: _ClassVar[int]
        IF_RETURN_FIELD_NUMBER: _ClassVar[int]
        ELSE_RETURN_FIELD_NUMBER: _ClassVar[int]
        if_condition: DerivationExpression
        if_return: DerivationExpression
        else_return: DerivationExpression
        def __init__(self, if_condition: _Optional[_Union[DerivationExpression, _Mapping]] = ..., if_return: _Optional[_Union[DerivationExpression, _Mapping]] = ..., else_return: _Optional[_Union[DerivationExpression, _Mapping]] = ...) -> None: ...
    class UnaryOp(_message.Message):
        __slots__ = ("op_type", "arg")
        class UnaryOpType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            UNARY_OP_TYPE_UNSPECIFIED: _ClassVar[DerivationExpression.UnaryOp.UnaryOpType]
            UNARY_OP_TYPE_BOOLEAN_NOT: _ClassVar[DerivationExpression.UnaryOp.UnaryOpType]
        UNARY_OP_TYPE_UNSPECIFIED: DerivationExpression.UnaryOp.UnaryOpType
        UNARY_OP_TYPE_BOOLEAN_NOT: DerivationExpression.UnaryOp.UnaryOpType
        OP_TYPE_FIELD_NUMBER: _ClassVar[int]
        ARG_FIELD_NUMBER: _ClassVar[int]
        op_type: DerivationExpression.UnaryOp.UnaryOpType
        arg: DerivationExpression
        def __init__(self, op_type: _Optional[_Union[DerivationExpression.UnaryOp.UnaryOpType, str]] = ..., arg: _Optional[_Union[DerivationExpression, _Mapping]] = ...) -> None: ...
    class BinaryOp(_message.Message):
        __slots__ = ("op_type", "arg1", "arg2")
        class BinaryOpType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = ()
            BINARY_OP_TYPE_UNSPECIFIED: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_PLUS: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_MINUS: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_MULTIPLY: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_DIVIDE: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_MIN: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_MAX: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_GREATER_THAN: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_LESS_THAN: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_AND: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_OR: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_EQUALS: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
            BINARY_OP_TYPE_COVERS: _ClassVar[DerivationExpression.BinaryOp.BinaryOpType]
        BINARY_OP_TYPE_UNSPECIFIED: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_PLUS: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_MINUS: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_MULTIPLY: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_DIVIDE: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_MIN: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_MAX: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_GREATER_THAN: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_LESS_THAN: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_AND: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_OR: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_EQUALS: DerivationExpression.BinaryOp.BinaryOpType
        BINARY_OP_TYPE_COVERS: DerivationExpression.BinaryOp.BinaryOpType
        OP_TYPE_FIELD_NUMBER: _ClassVar[int]
        ARG1_FIELD_NUMBER: _ClassVar[int]
        ARG2_FIELD_NUMBER: _ClassVar[int]
        op_type: DerivationExpression.BinaryOp.BinaryOpType
        arg1: DerivationExpression
        arg2: DerivationExpression
        def __init__(self, op_type: _Optional[_Union[DerivationExpression.BinaryOp.BinaryOpType, str]] = ..., arg1: _Optional[_Union[DerivationExpression, _Mapping]] = ..., arg2: _Optional[_Union[DerivationExpression, _Mapping]] = ...) -> None: ...
    class ReturnProgram(_message.Message):
        __slots__ = ("assignments", "final_expression")
        class Assignment(_message.Message):
            __slots__ = ("name", "expression")
            NAME_FIELD_NUMBER: _ClassVar[int]
            EXPRESSION_FIELD_NUMBER: _ClassVar[int]
            name: str
            expression: DerivationExpression
            def __init__(self, name: _Optional[str] = ..., expression: _Optional[_Union[DerivationExpression, _Mapping]] = ...) -> None: ...
        ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
        FINAL_EXPRESSION_FIELD_NUMBER: _ClassVar[int]
        assignments: _containers.RepeatedCompositeFieldContainer[DerivationExpression.ReturnProgram.Assignment]
        final_expression: DerivationExpression
        def __init__(self, assignments: _Optional[_Iterable[_Union[DerivationExpression.ReturnProgram.Assignment, _Mapping]]] = ..., final_expression: _Optional[_Union[DerivationExpression, _Mapping]] = ...) -> None: ...
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
    TIMESTAMP_TZ_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_DAY_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_COMPOUND_FIELD_NUMBER: _ClassVar[int]
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
    USER_DEFINED_POINTER_FIELD_NUMBER: _ClassVar[int]
    TYPE_PARAMETER_NAME_FIELD_NUMBER: _ClassVar[int]
    INTEGER_PARAMETER_NAME_FIELD_NUMBER: _ClassVar[int]
    INTEGER_LITERAL_FIELD_NUMBER: _ClassVar[int]
    UNARY_OP_FIELD_NUMBER: _ClassVar[int]
    BINARY_OP_FIELD_NUMBER: _ClassVar[int]
    IF_ELSE_FIELD_NUMBER: _ClassVar[int]
    RETURN_PROGRAM_FIELD_NUMBER: _ClassVar[int]
    bool: _type_pb2.Type.Boolean
    i8: _type_pb2.Type.I8
    i16: _type_pb2.Type.I16
    i32: _type_pb2.Type.I32
    i64: _type_pb2.Type.I64
    fp32: _type_pb2.Type.FP32
    fp64: _type_pb2.Type.FP64
    string: _type_pb2.Type.String
    binary: _type_pb2.Type.Binary
    timestamp: _type_pb2.Type.Timestamp
    date: _type_pb2.Type.Date
    time: _type_pb2.Type.Time
    interval_year: _type_pb2.Type.IntervalYear
    timestamp_tz: _type_pb2.Type.TimestampTZ
    uuid: _type_pb2.Type.UUID
    interval_day: DerivationExpression.ExpressionIntervalDay
    interval_compound: DerivationExpression.ExpressionIntervalCompound
    fixed_char: DerivationExpression.ExpressionFixedChar
    varchar: DerivationExpression.ExpressionVarChar
    fixed_binary: DerivationExpression.ExpressionFixedBinary
    decimal: DerivationExpression.ExpressionDecimal
    precision_time: DerivationExpression.ExpressionPrecisionTime
    precision_timestamp: DerivationExpression.ExpressionPrecisionTimestamp
    precision_timestamp_tz: DerivationExpression.ExpressionPrecisionTimestampTZ
    struct: DerivationExpression.ExpressionStruct
    list: DerivationExpression.ExpressionList
    map: DerivationExpression.ExpressionMap
    user_defined: DerivationExpression.ExpressionUserDefined
    user_defined_pointer: int
    type_parameter_name: str
    integer_parameter_name: str
    integer_literal: int
    unary_op: DerivationExpression.UnaryOp
    binary_op: DerivationExpression.BinaryOp
    if_else: DerivationExpression.IfElse
    return_program: DerivationExpression.ReturnProgram
    def __init__(self, bool: _Optional[_Union[_type_pb2.Type.Boolean, _Mapping]] = ..., i8: _Optional[_Union[_type_pb2.Type.I8, _Mapping]] = ..., i16: _Optional[_Union[_type_pb2.Type.I16, _Mapping]] = ..., i32: _Optional[_Union[_type_pb2.Type.I32, _Mapping]] = ..., i64: _Optional[_Union[_type_pb2.Type.I64, _Mapping]] = ..., fp32: _Optional[_Union[_type_pb2.Type.FP32, _Mapping]] = ..., fp64: _Optional[_Union[_type_pb2.Type.FP64, _Mapping]] = ..., string: _Optional[_Union[_type_pb2.Type.String, _Mapping]] = ..., binary: _Optional[_Union[_type_pb2.Type.Binary, _Mapping]] = ..., timestamp: _Optional[_Union[_type_pb2.Type.Timestamp, _Mapping]] = ..., date: _Optional[_Union[_type_pb2.Type.Date, _Mapping]] = ..., time: _Optional[_Union[_type_pb2.Type.Time, _Mapping]] = ..., interval_year: _Optional[_Union[_type_pb2.Type.IntervalYear, _Mapping]] = ..., timestamp_tz: _Optional[_Union[_type_pb2.Type.TimestampTZ, _Mapping]] = ..., uuid: _Optional[_Union[_type_pb2.Type.UUID, _Mapping]] = ..., interval_day: _Optional[_Union[DerivationExpression.ExpressionIntervalDay, _Mapping]] = ..., interval_compound: _Optional[_Union[DerivationExpression.ExpressionIntervalCompound, _Mapping]] = ..., fixed_char: _Optional[_Union[DerivationExpression.ExpressionFixedChar, _Mapping]] = ..., varchar: _Optional[_Union[DerivationExpression.ExpressionVarChar, _Mapping]] = ..., fixed_binary: _Optional[_Union[DerivationExpression.ExpressionFixedBinary, _Mapping]] = ..., decimal: _Optional[_Union[DerivationExpression.ExpressionDecimal, _Mapping]] = ..., precision_time: _Optional[_Union[DerivationExpression.ExpressionPrecisionTime, _Mapping]] = ..., precision_timestamp: _Optional[_Union[DerivationExpression.ExpressionPrecisionTimestamp, _Mapping]] = ..., precision_timestamp_tz: _Optional[_Union[DerivationExpression.ExpressionPrecisionTimestampTZ, _Mapping]] = ..., struct: _Optional[_Union[DerivationExpression.ExpressionStruct, _Mapping]] = ..., list: _Optional[_Union[DerivationExpression.ExpressionList, _Mapping]] = ..., map: _Optional[_Union[DerivationExpression.ExpressionMap, _Mapping]] = ..., user_defined: _Optional[_Union[DerivationExpression.ExpressionUserDefined, _Mapping]] = ..., user_defined_pointer: _Optional[int] = ..., type_parameter_name: _Optional[str] = ..., integer_parameter_name: _Optional[str] = ..., integer_literal: _Optional[int] = ..., unary_op: _Optional[_Union[DerivationExpression.UnaryOp, _Mapping]] = ..., binary_op: _Optional[_Union[DerivationExpression.BinaryOp, _Mapping]] = ..., if_else: _Optional[_Union[DerivationExpression.IfElse, _Mapping]] = ..., return_program: _Optional[_Union[DerivationExpression.ReturnProgram, _Mapping]] = ...) -> None: ...
