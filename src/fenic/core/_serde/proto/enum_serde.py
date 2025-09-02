"""Enum serialization/deserialization. Handles the serialization of enums to and from protobuf ints."""

from enum import Enum
from functools import singledispatch
from typing import Optional, Type

from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from fenic.core._serde.proto.errors import DeserializationError, SerializationError
from fenic.core._serde.proto.serde_context import SerdeContext


@singledispatch
def serialize_enum_value(
    value: Enum, target_proto: EnumTypeWrapper, context: SerdeContext
) -> int:
    """Serialize an enum value to the protobuf int representation.

    This function uses singledispatch to handle different enum types. If the enum names
    are 1:1 matches to the proto enum names, this auto-serde function can be used.
    Otherwise, define a `_serialize_<type>` function below to add custom enum mappings.

    Args:
        value: The enum value to serialize.
        target_proto: The protobuf enum type wrapper containing the target enum values.
        context: The serde context for error reporting and path tracking.

    Returns:
        int: The protobuf int representation of the enum value.

    Raises:
        SerializationError: If the enum value does not have a corresponding protobuf value.
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
    target_type: Type[Enum],
    proto_enum_type: EnumTypeWrapper,
    _serialized_value: int,
    context: SerdeContext,
) -> Optional[Enum]:
    """Deserialize an enum value from its protobuf int representation.

    This function uses singledispatch to handle different enum types. If the enum names
    are 1:1 matches to the proto enum names, this auto-serde function can be used.
    Otherwise, define a `_deserialize_<type>` function below to add custom enum mappings.

    Args:
        target_type: The target enum type to deserialize to.
        proto_enum_type: The protobuf enum type wrapper containing the source enum values.
        _serialized_value: The protobuf int representation of the enum value.
        context: The serde context for error reporting and path tracking.

    Returns:
        Optional[EnumType]: The deserialized enum value

    Raises:
        DeserializationError: If the protobuf enum name is not present in the target enum type.
    """
    if _serialized_value not in proto_enum_type.values():
        raise context.create_serde_error(
            DeserializationError,
            f"Protobuf enum value {_serialized_value} for enum {proto_enum_type} "
            f"is not present in {proto_enum_type}. "
            f"The Protobuf spec includes keys: {proto_enum_type.DESCRIPTOR.values_by_number}",
            proto_enum_type,
        )

    enum_name = proto_enum_type.Name(_serialized_value)
    if enum_name not in target_type._member_names_:
        raise context.create_serde_error(
            DeserializationError,
            f"Protobuf enum name {enum_name} for enum {proto_enum_type} "
            f"is not present in the target enum {target_type}. "
            f"The Protobuf spec includes keys: {proto_enum_type.keys()}"
            f"The target enum has keys: {target_type._member_names_}",
            target_type,
        )
    return target_type[enum_name]
