"""Semantic function signatures for the fenic signature system.

This module registers function signatures for semantic AI functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.types import Exact, VariadicUniform
from fenic.core.types.datatypes import BooleanType, StringType


def register_semantic_signatures():
    """Register all semantic function signatures for ScalarFunctions."""
    # Semantic map - instruction-based text transformation
    FunctionRegistry.register("semantic.map", FunctionSignature(
        function_name="semantic.map",
        type_signature=VariadicUniform(expected_min_args=1, required_type=StringType),  # Variable string inputs based on instruction
        return_type=StringType
    ))
    
    # Semantic extract - schema-based information extraction
    FunctionRegistry.register("semantic.extract", FunctionSignature(
        function_name="semantic.extract",
        type_signature=Exact([StringType]),  # String input (schema/template are parameters)
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns StructType based on schema
    ))
    
    # Semantic predicate - instruction-based boolean predicate
    FunctionRegistry.register("semantic.predicate", FunctionSignature(
        function_name="semantic.predicate",
        type_signature=VariadicUniform(expected_min_args=1, required_type=StringType),  # Variable string inputs based on instruction
        return_type=BooleanType
    ))
    
    # Semantic reduce - instruction-based aggregation  
    FunctionRegistry.register("semantic.reduce", FunctionSignature(
        function_name="semantic.reduce",
        type_signature=VariadicUniform(expected_min_args=1, required_type=StringType),  # Variable string inputs based on instruction
        return_type=StringType
    ))
    
    # Semantic classify - classification into labels/enum
    FunctionRegistry.register("semantic.classify", FunctionSignature(
        function_name="semantic.classify",
        type_signature=Exact([StringType]),  # String input (labels are parameters)
        return_type=StringType
    ))
    
    # Sentiment analysis - analyze sentiment of text
    FunctionRegistry.register("semantic.analyze_sentiment", FunctionSignature(
        function_name="semantic.analyze_sentiment",
        type_signature=Exact([StringType]),  # String input
        return_type=StringType
    ))
    
    # Embeddings - generate embeddings for text
    FunctionRegistry.register("semantic.embed", FunctionSignature(
        function_name="semantic.embed",
        type_signature=Exact([StringType]),  # String input (model_alias is parameter)
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns EmbeddingType with specific dimensions
    ))

    FunctionRegistry.register("semantic.summarize", FunctionSignature(
        function_name="semantic.summarize",
        type_signature=Exact([StringType]),  # String input (model_alias is parameter)
        return_type=StringType
    ))


# Register all signatures when module is imported
register_semantic_signatures()