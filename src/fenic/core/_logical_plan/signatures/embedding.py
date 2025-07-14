"""Embedding function signatures for the fenic signature system.

This module registers function signatures for embedding operations,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.function_signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.type_signature import (
    EqualTypes,
    InstanceOf,
    OneOf,
)
from fenic.core.types.datatypes import EmbeddingType, FloatType


def register_embedding_signatures():
    """Register all embedding function signatures."""
    # Normalize embedding vectors to unit length
    FunctionRegistry.register("embedding.normalize", FunctionSignature(
        function_name="embedding.normalize",
        type_signature=InstanceOf([EmbeddingType]),  # Any EmbeddingType instance
        return_type=ReturnTypeStrategy.SAME_AS_INPUT  # Returns same EmbeddingType
    ))

    # Compute similarity between embedding vectors
    FunctionRegistry.register("embedding.compute_similarity", FunctionSignature(
        function_name="embedding.compute_similarity",
        type_signature=OneOf([
            InstanceOf([EmbeddingType]),  # embedding input only (other is query vector parameter)
            EqualTypes(EmbeddingType),  # embedding input + embedding column (must have matching types)
        ]),
        return_type=FloatType
    ))


# Register all signatures when module is imported
register_embedding_signatures()
