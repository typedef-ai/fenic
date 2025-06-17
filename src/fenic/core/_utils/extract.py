"""Utility functions for working with structured extraction schemas."""
import typing
from typing import Annotated, Union

from pydantic import BaseModel, Field, create_model

from fenic.core.types.datatypes import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    _PrimitiveType,
)
from fenic.core.types.extract_schema import (
    ExtractSchema,
    ExtractSchemaList,
)


def convert_extract_schema_to_pydantic_type(
    schema: ExtractSchema,
) -> type[BaseModel]:
    """Create a Pydantic model type from an ExtractSchema schema.

    Args:
        schema: The ExtractSchema defining the structure

    Returns:
        A Pydantic model type that matches the schema structure
    """
    return _create_nested_model(schema=schema)


def validate_extract_schema_structure(
    model: Union[type[BaseModel], ExtractSchema],
) -> None:
    """Check a ExtractSchema or Pydantic model to ensure it is valid schema for a semantic extract."""
    if not (
        (isinstance(model, type) and issubclass(model, BaseModel))
        or isinstance(model, ExtractSchema)
    ):
        raise TypeError(
            f"Output schema must be a pydantic model object or ExtractSchema class, got type:{type(model).__name__}"
        )

    if isinstance(model, ExtractSchema):
        if len(model.struct_fields) == 0:
            raise ValueError(
                "Output schema cannot be empty. "
                "Please specify at least one output field."
            )

        # ExtractSchema class construction enforces the field structure
        return

    # Check the field structure and the types for the pydantic model
    if len(model.model_fields.items()) == 0:
        raise ValueError(
            "Output schema cannot be empty. "
            "Please specify at least one output field."
        )
    for field_name, field_info in model.model_fields.items():
        if field_info.description is None:
            raise ValueError(
                f"Extract schema field {field_name} has no description.  Please specify a description for each field."
            )
        if isinstance(field_info.annotation, type) and issubclass(
            field_info.annotation, BaseModel
        ):
            validate_extract_schema_structure(field_info.annotation)

        elif isinstance(field_info.annotation, list):
            element_type = field_info.annotation.__args__[0]
            if isinstance(element_type, type) and issubclass(element_type, BaseModel):
                validate_extract_schema_structure(element_type)
            else:
                if not isinstance(element_type, Union[int, float, str, bool]):
                    raise ValueError(
                        f"Unsupported data type in extract schema: {type(element_type).__name__}"
                    )
        elif (
            field_info.annotation is not bool
            and field_info.annotation is not int
            and field_info.annotation is not float
            and field_info.annotation is not str
            and not (hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ is typing.Literal)
            ):
            raise TypeError(
                f"Unsupported data type in extract schema: {type(field_info.annotation).__name__}"
            )


def _convert_primitive_dtype_to_pytype(dtype: _PrimitiveType) -> type:
    """Convert a PrimitiveType to a Python type.

    Args:
        dtype: The PrimitiveType to convert

    Returns:
        The corresponding Python type

    Raises:
        ValueError: If the data type cannot be converted

    """
    if isinstance(dtype, _PrimitiveType):
        if dtype == StringType:
            return str
        elif dtype == IntegerType:
            return int
        elif dtype == FloatType:
            return float
        elif dtype == DoubleType:
            return float
        elif dtype == BooleanType:
            return bool
        elif hasattr(dtype, '__origin__') and dtype.__origin__ is typing.Literal:
            return str
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
    else:
        raise ValueError(f"Unsupported data type: {type(dtype).__name__}")


def _create_nested_model(
    schema: ExtractSchema, model_name: str = "NestedModel"
) -> type[BaseModel]:
    """Create a Pydantic model type from an ExtractSchema schema, handling nested structures.

    Args:
        schema: The ExtractSchema to convert to a Pydantic model
        model_name: The name to use for the generated model class

    Returns:
        A Pydantic model type that matches the schema structure

    """
    annotated_fields = {}
    for field in schema.struct_fields:
        if isinstance(field.data_type, ExtractSchema):
            # For nested structs, create a new model recursively
            nested_model = _create_nested_model(
                field.data_type, f"{model_name}_{field.name}"
            )
            annotated_fields[field.name] = Annotated[
                nested_model, Field(description=field.description)
            ]
        elif isinstance(field.data_type, ExtractSchemaList):
            # For lists, handle the element type
            if isinstance(field.data_type.element_type, ExtractSchema):
                nested_model = _create_nested_model(
                    field.data_type.element_type, f"{model_name}_{field.name}"
                )
                annotated_fields[field.name] = Annotated[
                    list[nested_model], Field(description=field.description)
                ]
            else:
                annotated_fields[field.name] = Annotated[
                    list[
                        _convert_primitive_dtype_to_pytype(field.data_type.element_type)
                    ],
                    Field(description=field.description),
                ]
        else:
            annotated_fields[field.name] = Annotated[
                _convert_primitive_dtype_to_pytype(field.data_type),
                Field(description=field.description),
            ]
    return create_model(model_name, **annotated_fields)
