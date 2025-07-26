"""Helper utilities for path tracking in serde operations."""

from __future__ import annotations

import json
from contextlib import contextmanager
from enum import Enum
from typing import Any, List, Optional, Type, TypeVar

import numpy as np
from google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from jambo import SchemaConverter
from pydantic import BaseModel

from fenic import DataType
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan import LogicalExpr
from fenic.core._logical_plan.expressions.semantic import ResolvedClassDefinition
from fenic.core._logical_plan.expressions.text import (
    ChunkCharacterSet,
    ChunkLengthFunction,
    RecursiveTextChunkExprConfiguration,
    TextChunkExprConfiguration,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde.proto.errors import (
    DeserializationError,
    SerdeError,
    SerializationError,
)
from fenic.core._serde.proto.types import (
    ChunkCharacterSetProto,
    ChunkLengthFunctionProto,
    ColumnFieldProto,
    DataTypeProto,
    FenicSchemaProto,
    LogicalExprProto,
    LogicalPlanProto,
    NumpyArrayProto,
    PydanticModelTypeProto,
    RecursiveTextChunkExprConfigurationProto,
    ResolvedClassDefinitionProto,
    ScalarArrayProto,
    ScalarStructFieldProto,
    ScalarStructProto,
    ScalarValueProto,
    TextChunkExprConfigurationProto,
)
from fenic.core.types.schema import ColumnField, Schema

# Used for type hinting support in serialize_enum_value and deserialize_enum_value
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
    ) -> LogicalExpr:
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
        self, field_name: str, expr_proto_list: List[LogicalExprProto]
    ) -> List[LogicalExpr]:
        """Deserialize a list of logical expressions with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            expr_proto_list: The list of protobuf representations to deserialize.

        Returns:
            A list of deserialized logical expressions.
        """
        result = []
        with self.path_context(field_name):
            for i, expr_proto in enumerate(expr_proto_list):
                with self.path_context(f"[{i}]"):
                    try:
                        result.append(self.deserialize_logical_expr("expr", expr_proto))
                    except Exception as e:
                        self._handle_serde_error(e)
        return result

    def serialize_enum_value(
        self, field_name: str, enum_value: EnumType, target_proto: EnumTypeWrapper
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
    ) -> EnumType:
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
        session_state: Optional[BaseSessionState] = None,
    ) -> LogicalPlan:
        """Deserialize a logical plan with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            plan_proto: The protobuf representation to deserialize.
            session_state: Optional session state to include in the plan.

        Returns:
            The deserialized logical plan.
        """
        from fenic.core._serde.proto.plan_serde import deserialize_logical_plan

        with self.path_context(field_name):
            try:
                return deserialize_logical_plan(plan_proto, self, session_state)
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_logical_plan_list(
        self,
        field_name: str,
        plan_proto_list: List[LogicalPlanProto],
        session_state: Optional[BaseSessionState] = None,
    ) -> List[LogicalPlan]:
        """Deserialize a list of logical plans with field path tracking.

        Args:
            field_name: The name of the field being deserialized.
            plan_proto_list: The list of protobuf representations to deserialize.
            session_state: Optional session state to include in the plans.

        Returns:
            A list of deserialized logical plans.
        """
        result = []
        with self.path_context(field_name):
            for i, plan_proto in enumerate(plan_proto_list):
                with self.path_context(f"[{i}]"):
                    try:
                        result.append(
                            self.deserialize_logical_plan(
                                "plan", plan_proto, session_state
                            )
                        )
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
                    description=class_definition_proto.description,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_pydantic_model_type(
        self,
        field_name: str,
        pydantic_model: Type[BaseModel],
    ) -> PydanticModelTypeProto:
        """Serialize a Pydantic model to a protobuf model by dumping its json schema.

        Args:
            field_name: The name of the field being serialized.
            pydantic_model: The Pydantic model type to serialize.

        Returns:
            The serialized protobuf representation containing the JSON schema.
        """
        with self.path_context(field_name):
            try:
                json_schema = pydantic_model.model_json_schema()
                serialized_json_schema = json.dumps(json_schema)
                return PydanticModelTypeProto(json_schema=serialized_json_schema)
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_pydantic_model_type(
        self,
        field_name: str,
        pydantic_proto: PydanticModelTypeProto,
    ) -> Optional[Type[BaseModel]]:
        """Deserialize a Pydantic model from a protobuf model by loading its json schema.

        While ideally, we would like to work directly with the json schema, we need the actual pydantic model type
        so that we can use it to validate structured examples. For now, we can use `jambo` to deserialize the json schema.
        The limitations below detail the known issues with this approach, but they should not impact the functionality of the system,
        as long as the json that is generated from instances of the two are always the same.

        Known Limitations:
        - If a type is defined as a `Literal` in the original pydantic model, it will be deserialized as a dynamic enum type.
        - If a type is defined as a `list` in the original pydantic model, in the deserialized pydantic model, the list will be omitted from pydantic's `required` fields.

        Args:
            field_name: The name of the field being deserialized.
            pydantic_proto: The protobuf representation to deserialize.

        Returns:
            The deserialized Pydantic model type, or None if empty.
        """
        with self.path_context(field_name):
            try:
                if (
                    not pydantic_proto.json_schema
                ):  # Optional field is not populated in parent message (proto3)
                    return None
                json_schema = json.loads(pydantic_proto.json_schema)
                return SchemaConverter.build(json_schema)
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

    # =============================================================================
    # Text Chunking Configuration Serde Functions
    # =============================================================================

    def serialize_recursive_text_chunk_expr_configuration(
        self,
        field_name: str,
        config: RecursiveTextChunkExprConfiguration,
    ) -> RecursiveTextChunkExprConfigurationProto:
        """Serialize a recursive text chunk expression configuration.

        Args:
            field_name: The name of the field being serialized.
            config: The recursive text chunk configuration to serialize.

        Returns:
            The serialized protobuf representation of the configuration.
        """
        with self.path_context(field_name):
            try:
                return RecursiveTextChunkExprConfigurationProto(
                    desired_chunk_size=config.desired_chunk_size,
                    chunk_overlap_percentage=config.chunk_overlap_percentage,
                    chunk_length_function_name=self.serialize_enum_value(
                        "chunk_length_function_name",
                        config.chunk_length_function_name,
                        ChunkLengthFunctionProto,
                    ),
                    chunking_character_set_name=self.serialize_enum_value(
                        "chunking_character_set_name",
                        config.chunking_character_set_name,
                        ChunkCharacterSetProto,
                    ),
                    chunking_character_set_custom_characters=config.chunking_character_set_custom_characters,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_recursive_text_chunk_expr_configuration(
        self,
        field_name: str,
        config: RecursiveTextChunkExprConfigurationProto,
    ) -> RecursiveTextChunkExprConfiguration:
        """Deserialize a recursive text chunk expression configuration.

        Args:
            field_name: The name of the field being deserialized.
            config: The protobuf representation to deserialize.

        Returns:
            The deserialized recursive text chunk configuration.
        """
        with self.path_context(field_name):
            try:
                return RecursiveTextChunkExprConfiguration(
                    desired_chunk_size=config.desired_chunk_size,
                    chunk_overlap_percentage=config.chunk_overlap_percentage,
                    chunk_length_function_name=self.deserialize_enum_value(
                        "chunk_length_function_name",
                        ChunkLengthFunction,
                        ChunkLengthFunctionProto,
                        config.chunk_length_function_name,
                    ),
                    chunking_character_set_name=self.deserialize_enum_value(
                        "chunking_character_set_name",
                        ChunkCharacterSet,
                        ChunkCharacterSetProto,
                        config.chunking_character_set_name,
                    ),
                    chunking_character_set_custom_characters=config.chunking_character_set_custom_characters,
                )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_text_chunk_expr_configuration(
        self,
        field_name: str,
        config: TextChunkExprConfiguration,
    ) -> TextChunkExprConfigurationProto:
        """Serialize a text chunk expression configuration.

        Args:
            field_name: The name of the field being serialized.
            config: The text chunk configuration to serialize.

        Returns:
            The serialized protobuf representation of the configuration.
        """
        with self.path_context(field_name):
            try:
                return TextChunkExprConfigurationProto(
                    desired_chunk_size=config.desired_chunk_size,
                    chunk_overlap_percentage=config.chunk_overlap_percentage,
                    chunk_length_function_name=self.serialize_enum_value(
                        "chunk_length_function_name",
                        config.chunk_length_function_name,
                        ChunkLengthFunctionProto,
                    ),
                )
            except Exception as e:
                self._handle_serde_error(e)

    def deserialize_text_chunk_expr_configuration(
        self,
        field_name: str,
        config: TextChunkExprConfigurationProto,
    ) -> TextChunkExprConfiguration:
        """Deserialize a text chunk expression configuration.

        Args:
            field_name: The name of the field being deserialized.
            config: The protobuf representation to deserialize.

        Returns:
            The deserialized text chunk configuration.
        """
        with self.path_context(field_name):
            try:
                return TextChunkExprConfiguration(
                    desired_chunk_size=config.desired_chunk_size,
                    chunk_overlap_percentage=config.chunk_overlap_percentage,
                    chunk_length_function_name=self.deserialize_enum_value(
                        "chunk_length_function_name",
                        ChunkLengthFunction,
                        ChunkLengthFunctionProto,
                        config.chunk_length_function_name,
                    ),
                )
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
                elif which_oneof == "array_value":
                    # Deserialize arrays recursively
                    return [
                        self.deserialize_scalar_value(element)
                        for element in scalar_value.array_value.elements
                    ]
                elif which_oneof == "struct_value":
                    # Deserialize structs recursively
                    result = {}
                    for field in scalar_value.struct_value.fields:
                        result[field.name] = self.deserialize_scalar_value(field.value)
                    return result
                else:
                    raise self.create_serde_error(
                        DeserializationError,
                        f"Unknown scalar value type: {which_oneof}",
                    )
            except Exception as e:
                self._handle_serde_error(e)

    def serialize_fenic_schema(
        self, field_name: str, schema: Schema
    ) -> FenicSchemaProto:
        """Serialize a Fenic schema.

        Args:
            field_name: The name of the field being serialized.
            schema: The Fenic schema to serialize.

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
        self, field_name: str, schema_proto: FenicSchemaProto
    ) -> Schema:
        """Deserialize a Fenic schema.

        Args:
            field_name: The name of the field being deserialized.
            schema_proto: The protobuf representation to deserialize.

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
