"""Enum serialization/deserialization. Handles the serialization of enums to and from protobuf ints."""
from enum import Enum
from functools import singledispatch
from typing import Optional, Type

from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from fenic.core._serde.proto.errors import DeserializationError, SerializationError
from fenic.core._serde.proto.serde_context import EnumType, SerdeContext


@singledispatch
def serialize_enum_value(value: Enum, target_proto: EnumTypeWrapper, context: SerdeContext) -> int:
    """Serialize an enum value to the protobuf int representation.

    If the enum names are 1:1 matches to the proto enum names, we can use this auto-serde.
    Otherwise, define a `_serialize_<type>` function below to add custom enum mappings.
    """
    if value.name in target_proto.keys():
        return target_proto.Value(value.name)
    else:
        raise context.create_serde_error(
            SerializationError,
            f"Enum value {value} for enum type {value.__class__} "
            f"does not have a corresponding protobuf value."
            f"Available protobuf values are: {target_proto.keys()}",
            value.__class__,
        )

@singledispatch
def deserialize_enum_value(
    target_type: Type[EnumType],
    proto_enum_type: EnumTypeWrapper,
    _serialized_value: int,
    context: SerdeContext,
) -> Optional[EnumType]:
    """Deserialize an enum value.

    If the enum names are 1:1 matches to the proto enum names, we can use this auto-serde.
    Otherwise, define a `_deserialize_<type>` function below to add custom enum mappings.
    """
    enum_name = proto_enum_type.Name(_serialized_value)
    if enum_name in target_type.__members__:
        return target_type[enum_name]
    else:
        raise context.create_serde_error(
            DeserializationError,
            f"Protobuf enum name {enum_name} for enum type {proto_enum_type} "
            f"is not present in {target_type}. "
            f"The Protobuf spec includes keys: {proto_enum_type.keys()}"
            f"The target enum type has keys: {target_type._member_names_}",
            target_type,
        )
