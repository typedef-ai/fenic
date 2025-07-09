"""Simple DataType serialization utilities using asdict and json.dumps."""

import json
from dataclasses import asdict
from typing import Any, Dict

from pydantic import TypeAdapter
from pydantic.dataclasses import is_pydantic_dataclass

from fenic.core.types.datatypes import DataType


def serialize_datatype(dtype: DataType) -> Dict[str, Any]:
    """Serialize a DataType to a dictionary using asdict.
    
    Args:
        dtype: The DataType to serialize
        
    Returns:
        A dictionary representation of the DataType
    """
    return asdict(dtype)


def serialize_datatype_to_json(dtype: DataType) -> str:
    """Serialize a DataType to a JSON string.
    
    Args:
        dtype: The DataType to serialize
        
    Returns:
        A JSON string representation of the DataType
    """
    return json.dumps(serialize_datatype(dtype))


def deserialize_datatype_from_json(json_str: str, dtype_class: type) -> DataType:
    """Deserialize a DataType from a JSON string.
    
    Args:
        json_str: The JSON string to deserialize
        dtype_class: The DataType class to construct
        
    Returns:
        The reconstructed DataType
    """
    if is_pydantic_dataclass(dtype_class):
        return TypeAdapter(dtype_class).validate_json(json_str)
    data = json.loads(json_str)
    return dtype_class(**data)