#!/usr/bin/env python3
"""Test script to verify that the Literal type validation works correctly."""

from typing import Union

import pytest

from fenic._gen.protos.logical_plan.v1.enums_pb2 import (
    FuzzySimilarityMethod as FuzzySimilarityMethodProto,
)
from fenic.core._serde.proto.errors import DeserializationError
from fenic.core._serde.proto.serde_context import SerdeContext, create_serde_context
from fenic.core.types.enums import FuzzySimilarityMethod


def test_valid_literal_serde():
    context = create_serde_context()
    initial = "indel"
    serialized = context.serialize_python_literal("method", initial, FuzzySimilarityMethodProto)
    # This should work - FuzzySimilarityMethod is a valid Literal type
    result = context.deserialize_python_literal(
        "method",
        serialized,  # 0
        FuzzySimilarityMethod,
        FuzzySimilarityMethodProto
    )
    assert result == initial

    initial_jaro = "jaro_winkler"
    serialized_jaro = context.serialize_python_literal("method", initial_jaro, FuzzySimilarityMethodProto)
    result = context.deserialize_python_literal(
        "method",
        serialized_jaro,
        FuzzySimilarityMethod,
        FuzzySimilarityMethodProto
    )
    assert result == "jaro_winkler"

def test_invalid_literal_str():
    """Test that the Literal type validation works correctly."""
    # This should fail - str is not a Literal type
    with pytest.raises(DeserializationError):
        _ = SerdeContext().deserialize_python_literal(
            "method",
            FuzzySimilarityMethodProto.INDEL,  # 0
            str,  # Not a Literal type!
            FuzzySimilarityMethodProto
        )

def test_invalid_union_type():
    with pytest.raises(DeserializationError):
        _ = SerdeContext().deserialize_python_literal(
            "method",
            FuzzySimilarityMethodProto.INDEL,  # 0
            Union[str, int],  # Not a Literal type!
            FuzzySimilarityMethodProto
        )

