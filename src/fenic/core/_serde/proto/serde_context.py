"""Helper utilities for path tracking in serde operations."""

from __future__ import annotations

import datetime
import json
from contextlib import contextmanager
from enum import Enum
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import numpy as np
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper

from fenic.core._logical_plan import LogicalExpr
from fenic.core._logical_plan.expressions.semantic import ResolvedClassDefinition
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.resolved_types import (
    ResolvedModelAlias,
    ResolvedResponseFormat,
)
from fenic.core._serde.proto.errors import (
    DeserializationError,
    SerdeError,
    SerializationError,
)
from fenic.core._serde.proto.types import (
    ColumnFieldProto,
    DataTypeProto,
    FenicSchemaProto,
    LogicalExprProto,
    LogicalPlanProto,
    NumpyArrayProto,
    ResolvedClassDefinitionProto,
    ResolvedModelAliasProto,
    ResolvedResponseFormatProto,
    ScalarArrayProto,
    ScalarStructFieldProto,
    ScalarStructProto,
    ScalarValueProto,
    ToolDefinitionProto,
    ToolParameterProto,
)
from fenic.core._utils.structured_outputs import (
    check_if_model_uses_unserializable_features,
)
from fenic.core.mcp.types import BoundToolParam, UserDefinedTool
from fenic.core.types.datatypes import DataType
from fenic.core.types.schema import ColumnField, Schema

EnumType = TypeVar("EnumType", bound=Enum)


class SerdeContext:
    """Context for managing serialization/deserialization state and path tracking.

    Provides centralized error handling, path tracking, and field-level serde operations
    for protobuf serialization/deserialization. All serde operations should use this
    context to ensure consistent error reporting and path information.
    """

    # Common field name constants for improved usability
    EXPR = "expr"
    EXPRS = "exprs"
    OTHER = "other"
    INPUT = "input"
    INPUTS = "inputs"
    LEFT = "left"
    RIGHT = "right"
    VALUE = "value"
    VALUES = "values"
    DATA_TYPE = "data_type"
    SUBSTR = "substr"
    SCHEMA = "schema"
    FORMAT = "format"
    CONDITION = "condition"
    THEN = "then"
    OPERATOR = "operator"
    CHILD = "child"
    PATTERN = "pattern"

    def __init__(self):
        """Initialize a SerdeContext with an empty path tracker."""
        self._path_tracker = PathTracker()

    @property
    def current_path(self) -> str:
        """Get the current serde path for error reporting."""
        return self._path_tracker.current_path

    def clear_path(self) -> None:
        """Clear the current serde path."""
        self._path_tracker.clear()

    @contextmanager
    def path_context(self, field_name: str):
        """Context manager for tracking field paths during serde operations."""
        self._path_tracker.push(field_name)
        try:
            yield
        finally:
            self._path_tracker.pop()

    def create_serde_error(
        self,
        error_class: Type[SerdeError],
        message: str,
        object_type: Optional[Type] = None,
    ) -> SerdeError:
        """Create a serde error with the current path automatically included.

        Args:
            error_class: The type of serde error to create.
            message: The error message.
            object_type: Optional type information for the error.

        Returns:
            A serde error with path information included.
        """
        current_path = self.current_path
        return error_class(message, object_type, current_path if current_path else None)

    def _handle_serde_error(self, e: Exception) -> None:
        # If it's already a serde error with a path, re-raise as-is
        if hasattr(e, "field_path") and e.field_path:
            raise

        # Otherwise, wrap it with path information
        current_path = self.current_path
        if current_path:
            # Create a new error with path information
            if isinstance(e, SerdeError):
                # Re-create the error with path information
                new_error = type(e)(
                    str(e), getattr(e, "object_type", None), current_path
                )
                raise new_error from e
            else:
                # Wrap non-serde errors
                raise RuntimeError(f"{str(e)} at {current_path}") from e
        else:
            # No path context, re-raise as-is
            raise

    # =============================================================================
    # Core Serde Function Wrappers to preserve field path tracking
    # =============================================================================

    def serialize_logical_expr(
        self, field_name: str, expr: LogicalExpr
    ) -> LogicalExprProto:
        """Serialize a logical expression with field path tracking.

        Args:
            field_name: The name of the field being serialized.
            expr: The logical expression to serialize.

        Returns:
            The serialized protobuf representation.
        """
        from fenic.core._serde.proto.expression_serde import serialize_logical_expr

        with self.path_context(field_name):
            try:
                return serialize_logical_expr(expr, self)
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_logical_expr(
        self, field_name: str, expr_proto: LogicalExprProto
    ) -> Optional[LogicalExpr]:
        """Deserialize a logical expression with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            expr_proto: The protobuf representation to deserialize.

        Returns:
            The deserialized logical expression.
        """
        from fenic.core._serde.proto.expression_serde import deserialize_logical_expr

        with self.path_context(field_name):
            try:
                return deserialize_logical_expr(expr_proto, self)
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_logical_expr_list(
        self, field_name: str, expr_list: List[LogicalExpr]
    ) -> List[LogicalExprProto]:
        """Serialize a list of logical expressions with field path tracking.

        Args:
            field_name: The name of the field being serialized.
            expr_list: The list of logical expressions to serialize.

        Returns:
            A list of serialized protobuf representations.
        """
        result = []
        with self.path_context(field_name):
            for i, expr in enumerate(expr_list):
                with self.path_context(f"[{i}]"):
                    try:
                        result.append(self.serialize_logical_expr("expr", expr))
                    except Exception as e:
                        self._handle_serde_error(e)
        return result

    def deserialize_logical_expr_list(
        self, field_name: str, expr_proto_list: Iterable[LogicalExprProto]
    ) -> List[LogicalExpr]:
        """Deserialize a list of logical expressions with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            expr_proto_list: The list of protobuf representations to deserialize.

        Returns:
            A list of deserialized logical expressions.
        """
        result = []
        if not expr_proto_list:
            return result
        with self.path_context(field_name):
            for i, expr_proto in enumerate(expr_proto_list):
                with self.path_context(f"[{i}]"):
                    try:
                        result.append(self.deserialize_logical_expr("expr", expr_proto))
                    except Exception as e:
                        self._handle_serde_error(e)
        return result

    def serialize_python_literal(
        self, field_name: str, value: str, target_proto: EnumTypeWrapper
    ) -> int:
        """Serialize a python literal with field path tracking.

        Args:
            field_name: The name of the field being serialized.
            value: The string value from the Python Literal type.
            target_proto: The protobuf enum type wrapper to convert to.

        Returns:
            The protobuf enum integer value.

        Raises:
            SerializationError: If the value cannot be safely serialized.
        """
        with self.path_context(field_name):
            try:
                value_upper = value.upper()
                if value_upper in target_proto.keys():
                    return target_proto.Value(value_upper)
                else:
                    raise SerializationError(
                        f"Could not serialize python literal '{value}/{value_upper}' to protobuf enum {target_proto.DESCRIPTOR.name}. "
                        f"Value '{value}/{value_upper}' is not available in the protobuf enum. "
                        f"Available protobuf enum values are: {list(target_proto.keys())}. "
                        f"This suggests a mismatch between the Python Literal type and the protobuf enum definition."
                    )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_python_literal(
        self,
        field_name: str,
        value: int,
        target_type: type,
        proto_enum: EnumTypeWrapper,
    ) -> Any:
        """Deserialize a python literal with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            value: The protobuf enum integer value.
            target_type: The target Python Literal type.
            proto_enum: The protobuf enum type wrapper to convert int to string.

        Returns:
            The string value from the literal type.

        Raises:
            DeserializationError: If the value cannot be safely deserialized.
        """
        with self.path_context(field_name):
            try:
                # First, validate that target_type is actually a Literal type
                if get_origin(target_type) is not Literal:
                    raise DeserializationError(
                        f"target_type {target_type} is not a typing.Literal type. "
                        f"Expected typing.Literal[...] but got {target_type} with origin {get_origin(target_type)}."
                    )

                literal_values = get_args(target_type)
                if not literal_values:
                    raise DeserializationError(
                        f"target_type {target_type} is not a valid Literal type. "
                        f"Expected a typing.Literal[...] type with literal values, but get_args() returned empty tuple."
                    )

                # Validate the protobuf enum value is valid
                if value not in proto_enum.values():
                    raise DeserializationError(
                        f"Protobuf enum value {value} is not valid for {proto_enum}. "
                        f"Valid values are: {list(proto_enum.values())}"
                    )

                # Convert protobuf int to string name
                enum_name = proto_enum.Name(value).lower()

                # Check if the enum name is in the target literal type
                if enum_name not in literal_values:
                    raise DeserializationError(
                        f"Protobuf enum name '{enum_name}' is not available in the python literal type {target_type}. "
                        f"Available literal values are: {literal_values}. "
                        f"This suggests a mismatch between the protobuf enum definition and the Python Literal type."
                    )

                return cast(target_type, enum_name)
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_enum_value(
        self, field_name: str, enum_value: type[Enum], target_proto: EnumTypeWrapper
    ) -> int:
        """Serialize an enum value with field path tracking.

        Args:
            field_name: The name of the field being serialized.
            enum_value: The enum value to serialize.
            target_proto: The protobuf enum type wrapper.

        Returns:
            The protobuf int representation of the enum value.
        """
        from fenic.core._serde.proto.enum_serde import serialize_enum_value

        with self.path_context(field_name):
            try:
                return serialize_enum_value(enum_value, target_proto, self)
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_enum_value(
        self,
        field_name: str,
        target_type: Type[EnumType],
        proto_enum_type: EnumTypeWrapper,
        serialized_value: int,
    ) -> Optional[EnumType]:
        """Deserialize an enum value with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            target_type: The target enum type to deserialize to.
            proto_enum_type: The protobuf enum type wrapper.
            serialized_value: The protobuf int representation of the enum value.

        Returns:
            The deserialized enum value.
        """
        from fenic.core._serde.proto.enum_serde import deserialize_enum_value

        with self.path_context(field_name):
            try:
                return deserialize_enum_value(
                    target_type, proto_enum_type, serialized_value, self
                )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_logical_plan(
        self, field_name: str, plan: LogicalPlan
    ) -> LogicalPlanProto:
        """Serialize a logical plan with field path tracking.

        Args:
            field_name: The name of the field being serialized.
            plan: The logical plan to serialize.

        Returns:
            The serialized protobuf representation of the logical plan.
        """
        from fenic.core._serde.proto.plan_serde import serialize_logical_plan

        with self.path_context(field_name):
            try:
                return serialize_logical_plan(plan, self)
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_logical_plan_list(
        self, field_name: str, plan_list: List[LogicalPlan]
    ) -> List[LogicalPlanProto]:
        """Serialize a list of logical plans with field path tracking.

        Args:
            field_name: The name of the field being serialized.
            plan_list: The list of logical plans to serialize.

        Returns:
            A list of serialized protobuf representations of logical plans.
        """
        result = []
        with self.path_context(field_name):
            for i, plan in enumerate(plan_list):
                with self.path_context(f"[{i}]"):
                    try:
                        result.append(self.serialize_logical_plan("plan", plan))
                    except Exception as e:
                        self._handle_serde_error(e)
        return result

    def deserialize_logical_plan(
        self,
        field_name: str,
        plan_proto: LogicalPlanProto,
    ) -> LogicalPlan:
        """Deserialize a logical plan with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            plan_proto: The protobuf representation to deserialize.

        Returns:
            The deserialized logical plan.
        """
        from fenic.core._serde.proto.plan_serde import deserialize_logical_plan

        with self.path_context(field_name):
            try:
                return deserialize_logical_plan(plan_proto, self)
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_logical_plan_list(
        self,
        field_name: str,
        plan_proto_list: List[LogicalPlanProto],
    ) -> List[LogicalPlan]:
        """Deserialize a list of logical plans with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            plan_proto_list: The list of protobuf representations to deserialize.

        Returns:
            A list of deserialized logical plans.
        """
        result = []
        with self.path_context(field_name):
            for i, plan_proto in enumerate(plan_proto_list):
                with self.path_context(f"[{i}]"):
                    try:
                        result.append(self.deserialize_logical_plan("plan", plan_proto))
                    except Exception as e:
                        self._handle_serde_error(e)
        return result

    def serialize_data_type(
        self, field_name: str, data_type: DataType
    ) -> DataTypeProto:
        """Serialize a data type with field path tracking.

        Args:
            field_name: The name of the field being serialized.
            data_type: The data type to serialize.

        Returns:
            The serialized protobuf representation of the data type.
        """
        from fenic.core._serde.proto.datatype_serde import serialize_data_type

        with self.path_context(field_name):
            try:
                return serialize_data_type(data_type, self)
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_data_type(
        self, field_name: str, data_type_proto: DataTypeProto
    ) -> Optional[DataType]:
        """Deserialize a data type with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            data_type_proto: The protobuf representation to deserialize.

        Returns:
            The deserialized data type.
        """
        from fenic.core._serde.proto.datatype_serde import deserialize_data_type

        with self.path_context(field_name):
            try:
                return deserialize_data_type(data_type_proto, self)
            except Exception as e:
                self._handle_serde_error(e)

    # =============================================================================
    # Common Utility Serde Functions
    # =============================================================================
    def serialize_resolved_model_alias(
        self,
        field_name: str,
        model_alias: Optional[ResolvedModelAlias],
    ) -> ResolvedModelAliasProto:
        """Serialize a resolved model alias."""
        if not model_alias:
            return None
        with self.path_context(field_name):
            try:
                return ResolvedModelAliasProto(
                    name=model_alias.name,
                    profile=model_alias.profile,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_resolved_model_alias(
        self,
        field_name: str,
        model_alias_proto: ResolvedModelAliasProto,
    ) -> ResolvedModelAlias:
        """Deserialize a resolved model alias."""
        with self.path_context(field_name):
            try:
                # Optional field will be populated as "" if not present, which is false-y.
                if not model_alias_proto.profile:
                    return ResolvedModelAlias(name=model_alias_proto.name)
                else:
                    return ResolvedModelAlias(
                        name=model_alias_proto.name,
                        profile=model_alias_proto.profile,
                    )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_resolved_class_definition(
        self,
        field_name: str,
        class_definition: ResolvedClassDefinition,
    ) -> ResolvedClassDefinitionProto:
        """Serialize a resolved class definition.

        Args:
            field_name: The name of the field being serialized.
            class_definition: The resolved class definition to serialize.

        Returns:
            The serialized protobuf representation of the class definition.
        """
        with self.path_context(field_name):
            try:
                return ResolvedClassDefinitionProto(
                    label=class_definition.label,
                    description=class_definition.description,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_resolved_class_definition(
        self,
        field_name: str,
        class_definition_proto: ResolvedClassDefinitionProto,
    ) -> Optional[ResolvedClassDefinition]:
        """Deserialize a resolved class definition.

        Args:
            field_name: The name of the field being deserialized.
            class_definition_proto: The protobuf representation to deserialize.

        Returns:
            The deserialized resolved class definition.
        """
        if not class_definition_proto:
            return None
        with self.path_context(field_name):
            try:
                return ResolvedClassDefinition(
                    label=class_definition_proto.label,
                    description=class_definition_proto.description
                    if class_definition_proto.description
                    else None,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_resolved_response_format(
        self,
        field_name: str,
        resolved_response_format: ResolvedResponseFormat,
    ) -> ResolvedResponseFormatProto:
        """Serialize a ResolvedResponseFormat to a protobuf model.

        Args:
            field_name: The name of the field being serialized.
            resolved_response_format: The ResolvedResponseFormat to serialize.

        Returns:
            The serialized protobuf representation containing the JSON schema.
        """
        with self.path_context(field_name):
            try:
                check_if_model_uses_unserializable_features(resolved_response_format.pydantic_model)
                return ResolvedResponseFormatProto(
                    schema=json.dumps(resolved_response_format.json_schema),
                    struct_type=self.serialize_data_type(
                        "struct_type", resolved_response_format.struct_type
                    ) if resolved_response_format.struct_type is not None else None,
                    prompt_schema_definition=resolved_response_format.prompt_schema_definition,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_resolved_response_format(
        self,
        field_name: str,
        resolved_response_format_proto: ResolvedResponseFormatProto,
    ) -> Optional[ResolvedResponseFormat]:
        """Deserialize a ResolvedResponseFormat from a protobuf model.

        Args:
            field_name: The name of the field being deserialized.
            resolved_response_format_proto: The protobuf representation to deserialize.

        Returns:
            The deserialized Pydantic model type, or None if empty.
        """
        with self.path_context(field_name):
            try:
                return ResolvedResponseFormat.from_json_schema(
                    raw_schema=json.loads(resolved_response_format_proto.schema),
                    struct_type=self.deserialize_data_type(
                        "struct_type", resolved_response_format_proto.struct_type
                    ),
                    prompt_schema_definition=resolved_response_format_proto.prompt_schema_definition,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_numpy_array(
        self, field_name: str, array: np.ndarray
    ) -> NumpyArrayProto:
        """Serialize a numpy array.

        Args:
            field_name: The name of the field being serialized.
            array: The numpy array to serialize.

        Returns:
            The serialized protobuf representation of the numpy array.
        """
        with self.path_context(field_name):
            try:
                return NumpyArrayProto(
                    data=array.tobytes(),
                    shape=array.shape,
                    dtype=array.dtype,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_numpy_array(
        self, field_name: str, serialized_array: NumpyArrayProto
    ) -> np.ndarray:
        """Deserialize a numpy array.

        Args:
            field_name: The name of the field being deserialized.
            serialized_array: The protobuf representation to deserialize.

        Returns:
            The deserialized numpy array.
        """
        with self.path_context(field_name):
            try:
                np_array = np.frombuffer(
                    serialized_array.data, dtype=serialized_array.dtype
                )
                return np_array.reshape(serialized_array.shape)
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_scalar_value(self, field_name: str, value: Any) -> ScalarValueProto:
        """Serialize a Python value to ScalarValue oneof.

        Supports primitive types (str, int, float, bool, bytes), arrays, and structs.
        Recursively serializes nested structures.

        Args:
            field_name: The name of the field being serialized.
            value: The Python value to serialize.

        Returns:
            The serialized protobuf representation of the scalar value.

        Raises:
            SerializationError: If the value type is not supported or is None.
        """
        if value is None:
            return ScalarValueProto()
        with self.path_context(field_name):
            try:
                if isinstance(value, str):
                    return ScalarValueProto(string_value=value)
                elif isinstance(value, bool):
                    return ScalarValueProto(bool_value=value)
                elif isinstance(value, int):
                    return ScalarValueProto(int_value=value)
                elif isinstance(value, float):
                    return ScalarValueProto(double_value=value)
                elif isinstance(value, bytes):
                    return ScalarValueProto(bytes_value=value)
                elif isinstance(value, datetime.datetime):
                    # since datetime is a subclass of date, check it first so it doesn't
                    # match to a date.
                    return ScalarValueProto(timestamp_value=value.timestamp())
                elif isinstance(value, datetime.date):
                    return ScalarValueProto(date_value=value.toordinal())
                elif isinstance(value, list):
                    # Serialize arrays recursively
                    elements = [
                        self.serialize_scalar_value("element", element)
                        for element in value
                    ]
                    return ScalarValueProto(
                        array_value=ScalarArrayProto(elements=elements)
                    )
                elif isinstance(value, dict):
                    # Serialize structs recursively, ensuring sorted field order for consistency
                    fields = []
                    for key in sorted(value.keys()):
                        field = ScalarStructFieldProto(
                            name=key,
                            value=self.serialize_scalar_value("value", value[key]),
                        )
                        fields.append(field)
                    return ScalarValueProto(
                        struct_value=ScalarStructProto(fields=fields)
                    )
                else:
                    raise self.create_serde_error(
                        SerializationError,
                        f"Cannot serialize scalar value of type: {type(value)}",
                    )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_scalar_value(
        self, field_name: str, scalar_value: ScalarValueProto
    ) -> Any:
        """Deserialize a ScalarValue oneof to Python value.

        Supports primitive types (str, int, float, bool, bytes), arrays, and structs.
        Recursively deserializes nested structures.

        Args:
            field_name: The name of the field being deserialized.
            scalar_value: The protobuf representation to deserialize.

        Returns:
            The deserialized Python value.

        Raises:
            DeserializationError: If the value type is unknown or unsupported.
        """
        with self.path_context(field_name):
            try:
                which_oneof = scalar_value.WhichOneof("value_type")
                if which_oneof is None:
                    return None
                if which_oneof == "string_value":
                    return scalar_value.string_value
                elif which_oneof == "int_value":
                    return scalar_value.int_value
                elif which_oneof == "double_value":
                    return scalar_value.double_value
                elif which_oneof == "bool_value":
                    return scalar_value.bool_value
                elif which_oneof == "bytes_value":
                    return scalar_value.bytes_value
                elif which_oneof == "date_value":
                    return datetime.date.fromordinal(scalar_value.date_value)
                elif which_oneof == "timestamp_value":
                    return datetime.datetime.fromtimestamp(scalar_value.timestamp_value)
                elif which_oneof == "array_value":
                    # Deserialize arrays recursively
                    return [
                        self.deserialize_scalar_value("element", element)
                        for element in scalar_value.array_value.elements
                    ]
                elif which_oneof == "struct_value":
                    # Deserialize structs recursively
                    result: dict[str, Any] = {}
                    for field in scalar_value.struct_value.fields:
                        result[field.name] = self.deserialize_scalar_value(
                            field.name, field.value
                        )
                    return result
                else:
                    raise self.create_serde_error(
                        DeserializationError,
                        f"Unknown scalar value type: {which_oneof}",
                    )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_fenic_schema(
        self,
        schema: Schema,
        field_name: str = "schema",
    ) -> FenicSchemaProto:
        """Serialize a Fenic schema.

        Args:
            schema: The Fenic schema to serialize.
            field_name: The name of the field being serialized.

        Returns:
            The serialized protobuf representation of the schema.
        """
        with self.path_context(field_name):
            try:
                return FenicSchemaProto(
                    column_fields=[
                        ColumnFieldProto(
                            name=field.name,
                            data_type=self.serialize_data_type(
                                "data_type", field.data_type
                            ),
                        )
                        for field in schema.column_fields
                    ]
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_fenic_schema(
        self, schema_proto: FenicSchemaProto, field_name: str = "schema"
    ) -> Schema:
        """Deserialize a Fenic schema.

        Args:
            schema_proto: The protobuf representation to deserialize.
            field_name: The name of the field being deserialized.

        Returns:
            The deserialized Fenic schema.
        """
        with self.path_context(field_name):
            try:
                return Schema(
                    column_fields=[
                        ColumnField(
                            name=field.name,
                            data_type=self.deserialize_data_type(
                                "data_type", field.data_type
                            ),
                        )
                        for field in schema_proto.column_fields
                    ]
                )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_tool_parameter(
        self,
        tool_param: BoundToolParam,
        field_name: str = "params",
    ) -> ToolParameterProto:
        """Serialize a ToolParameter."""
        with self.path_context(field_name):
            try:
                allowed_values = None
                if tool_param.allowed_values:
                    allowed_values = [
                        self.serialize_scalar_value("allowed_values", allowed_value) for allowed_value in
                        tool_param.allowed_values
                    ]
                return ToolParameterProto(
                    name=tool_param.name,
                    description=tool_param.description,
                    data_type=self.serialize_data_type(self.DATA_TYPE, tool_param.data_type),
                    required=tool_param.required,
                    has_default=tool_param.has_default,
                    default_value=self.serialize_scalar_value("default_value", tool_param.default_value),
                    allowed_values=allowed_values,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_tool_parameter(
        self,
        tool_param_proto: ToolParameterProto,
        field_name: str = "params"
    ) -> BoundToolParam:
        """Deserialize a ToolParameter."""
        with self.path_context(field_name):
            try:
                allowed_values = None
                if tool_param_proto.allowed_values:
                    allowed_values = [
                        self.deserialize_scalar_value("allowed_values", allowed_value) for allowed_value in
                        tool_param_proto.allowed_values]
                return BoundToolParam(
                    name=tool_param_proto.name,
                    description=tool_param_proto.description,
                    data_type=self.deserialize_data_type(self.DATA_TYPE, tool_param_proto.data_type),
                    required=tool_param_proto.required,
                    has_default=tool_param_proto.has_default,
                    default_value=self.deserialize_scalar_value("default_value", tool_param_proto.default_value),
                    allowed_values=allowed_values,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_tool_definition(
        self,
        tool_definition: UserDefinedTool,
        field_name: str = "tool_definition"
    ) -> ToolDefinitionProto:
        with self.path_context(field_name):
            serialized_params = [
                self.serialize_tool_parameter(tool_param) for tool_param in tool_definition.params
            ]
            try:
                return ToolDefinitionProto(
                    name=tool_definition.name,
                    description=tool_definition.description,
                    params=serialized_params,
                    parameterized_view=self.serialize_logical_plan("parameterized_view",
                                                                   tool_definition._parameterized_view),
                    result_limit=tool_definition.max_result_limit,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_tool_definition(
        self,
        tool_definition_proto: ToolDefinitionProto,
        field_name: str = "tool_definition"
    ) -> UserDefinedTool:
        """Deserialize a ToolDefinition."""
        with self.path_context(field_name):
            try:
                return UserDefinedTool(
                    name=tool_definition_proto.name,
                    description=tool_definition_proto.description,
                    params=[self.deserialize_tool_parameter(tool_param) for tool_param in tool_definition_proto.params],
                    _parameterized_view=self.deserialize_logical_plan("parameterized_view",
                                                                      tool_definition_proto.parameterized_view),
                    max_result_limit=tool_definition_proto.result_limit,
                )
            except Exception as e:
                self._handle_serde_error(e)


def create_serde_context() -> SerdeContext:
    """Create a new SerdeContext instance.

    This is the preferred way to get a context for serde operations.
    Each context is independent and can be used concurrently.

    Returns:
        A new SerdeContext instance ready for serde operations.
    """
    return SerdeContext()


class PathTracker:
    """Path tracker for serde operations."""

    def __init__(self):
        """Initialize a PathTracker."""
        self._path_stack = []

    @property
    def current_path(self) -> str:
        """Get the current field path as a string."""
        return ".".join(self._path_stack) if self._path_stack else ""

    def push(self, field_name: str) -> None:
        """Push a field name onto the path stack."""
        self._path_stack.append(field_name)

    def pop(self) -> None:
        """Pop the last field name from the path stack."""
        if self._path_stack:
            self._path_stack.pop()

    def clear(self) -> None:
        """Clear the entire path stack."""
        self._path_stack.clear()
