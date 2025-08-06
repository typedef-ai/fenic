#!/usr/bin/env python3
"""Test script to verify that the Literal type validation works correctly."""

from typing import Union

from fenic._gen.protos.logical_plan.v1.enums_pb2 import (
    FuzzySimilarityMethod as FuzzySimilarityMethodProto,
)
from fenic.core._serde.proto.errors import DeserializationError
from fenic.core._serde.proto.serde_context import create_serde_context
from fenic.core.types.enums import FuzzySimilarityMethod


def test_literal_validation():
    """Test that the Literal type validation works correctly."""
    
    context = create_serde_context()
    
    print("Testing valid Literal type...")
    try:
        # This should work - FuzzySimilarityMethod is a valid Literal type
        result = context.deserialize_python_literal(
            "method", 
            FuzzySimilarityMethodProto.INDEL,  # 0
            FuzzySimilarityMethod, 
            FuzzySimilarityMethodProto
        )
        assert result == "indel"
        print(f"✅ Valid Literal type worked: {result}")
    except Exception as e:
        print(f"❌ Valid Literal type failed: {e}")

    print("Testing Literal type with hyphen")
    try:
        # This should work - FuzzySimilarityMethod is a valid Literal type
        result = context.deserialize_python_literal(
            "method",
            FuzzySimilarityMethodProto.JARO_WINKLER,
            FuzzySimilarityMethod,
            FuzzySimilarityMethodProto
        )
        assert result == "jaro_winkler"
        print(f"✅ Valid Literal type worked: {result}")
    except Exception as e:
        print(f"❌ Valid Literal type failed: {e}")
    
    print("\nTesting invalid type (not a Literal)...")
    try:
        # This should fail - str is not a Literal type
        result = context.deserialize_python_literal(
            "method", 
            FuzzySimilarityMethodProto.INDEL,  # 0
            str,  # Not a Literal type!
            FuzzySimilarityMethodProto
        )
        print(f"❌ Invalid type should have failed but got: {result}")
    except DeserializationError as e:
        print(f"✅ Invalid type correctly failed: {e}")
    except Exception as e:
        print(f"❌ Invalid type failed with unexpected error: {e}")
    
    print("\nTesting invalid type (Union instead of Literal)...")
    try:
        # This should fail - Union is not a Literal type
        result = context.deserialize_python_literal(
            "method", 
            FuzzySimilarityMethodProto.indel,  # 0
            Union[str, int],  # Not a Literal type!
            FuzzySimilarityMethodProto
        )
        print(f"❌ Union type should have failed but got: {result}")
    except DeserializationError as e:
        print(f"✅ Union type correctly failed: {e}")
    except Exception as e:
        print(f"❌ Union type failed with unexpected error: {e}")

