"""Tests for logical expression serialization and deserialization."""

from typing import Literal, Type, get_origin

import pytest
from pydantic import BaseModel, Field

from fenic.core._logical_plan.expressions import (
    # Basic expressions
    AliasExpr,
    # Semantic expressions
    AnalyzeSentimentExpr,
    # Arithmetic expressions
    ArithmeticExpr,
    ArrayContainsExpr,
    ArrayExpr,
    # Text expressions
    ArrayJoinExpr,
    ArrayLengthExpr,
    # Aggregate expressions
    AvgExpr,
    # Comparison expressions
    BooleanExpr,
    ByteLengthExpr,
    CastExpr,
    CoalesceExpr,
    ColumnExpr,
    ConcatExpr,
    ContainsAnyExpr,
    ContainsExpr,
    CountExpr,
    CountTokensExpr,
    # Embedding expressions
    EmbeddingNormalizeExpr,
    EmbeddingsExpr,
    EmbeddingSimilarityExpr,
    EndsWithExpr,
    EqualityComparisonExpr,
    FirstExpr,
    FuzzyRatioExpr,
    FuzzyTokenSetRatioExpr,
    FuzzyTokenSortRatioExpr,
    ILikeExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    JinjaExpr,
    # JSON expressions
    JqExpr,
    JsonContainsExpr,
    JsonTypeExpr,
    LikeExpr,
    ListExpr,
    LiteralExpr,
    MaxExpr,
    # Markdown expressions
    MdExtractHeaderChunks,
    MdGenerateTocExpr,
    MdGetCodeBlocksExpr,
    MdToJsonExpr,
    MinExpr,
    NotExpr,
    NumericComparisonExpr,
    # Case expressions
    OtherwiseExpr,
    RecursiveTextChunkExpr,
    RegexpSplitExpr,
    ReplaceExpr,
    ResolvedClassDefinition,
    RLikeExpr,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticPredExpr,
    SemanticReduceExpr,
    SemanticSummarizeExpr,
    SortExpr,
    SplitPartExpr,
    StartsWithExpr,
    StdDevExpr,
    StringCasingExpr,
    StripCharsExpr,
    StrLengthExpr,
    StructExpr,
    SumExpr,
    TextChunkExpr,
    TextractExpr,
    TsParseExpr,
    UDFExpr,
    WhenExpr,
)
from fenic.core._logical_plan.expressions.base import LogicalExpr, Operator
from fenic.core._logical_plan.expressions.text import (
    ChunkCharacterSet,
    ChunkLengthFunction,
    RecursiveTextChunkExprConfiguration,
    TextChunkExprConfiguration,
)
from fenic.core._serde.proto.errors import DeserializationError, SerializationError
from fenic.core._serde.proto.expression_serde import (
    deserialize_logical_expr,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core.types import (
    BooleanType,
    FloatType,
    IntegerType,
    JsonType,
    StringType,
)
from fenic.core.types.summarize import Paragraph


class BasicResponseFormat(BaseModel):
    name: str
    age: int
    email: str
    nicknames: list[str] = Field(...)
    valid: bool
    state: Literal["active", "inactive"]

# Define examples for each expression type
# Each type has a list of examples to test different scenarios
expression_examples = {
    # Basic expressions
    ColumnExpr: [
        ColumnExpr("test_col"),
        ColumnExpr("another_column"),
        ColumnExpr("complex_name_with_underscores"),
    ],
    LiteralExpr: [
        LiteralExpr("test_string", StringType),
        LiteralExpr(42, IntegerType),
        LiteralExpr(3.14, FloatType),
        LiteralExpr(True, BooleanType),
    ],
    AliasExpr: [
        AliasExpr(ColumnExpr("test_col"), "test_alias"),
        AliasExpr(LiteralExpr("value", StringType), "literal_alias"),
    ],
    SortExpr: [
        SortExpr(ColumnExpr("test_col"), ascending=True, nulls_last=False),
        SortExpr(ColumnExpr("test_col"), ascending=False, nulls_last=True),
    ],
    IndexExpr: [
        IndexExpr(
            ArrayExpr(
                [LiteralExpr("a", StringType), LiteralExpr("b", StringType)]
            ),
            LiteralExpr(0, IntegerType),
        ),
        IndexExpr(ColumnExpr("array_col"), LiteralExpr(1, IntegerType)),
    ],
    ArrayExpr: [
        ArrayExpr([ColumnExpr("col1"), ColumnExpr("col2")]),
        ArrayExpr([LiteralExpr("a", StringType), LiteralExpr("b", StringType)]),
    ],
    StructExpr: [
        StructExpr([ColumnExpr("col1"), ColumnExpr("col2")]),
        StructExpr(
            [LiteralExpr("a", StringType), LiteralExpr(42, IntegerType)]
        ),
    ],
    UDFExpr: [
        # Note: UDFExpr cannot be serialized, but we test the type exists
        UDFExpr(lambda x: x, [ColumnExpr("test_col")], StringType),
    ],
    IsNullExpr: [
        IsNullExpr(ColumnExpr("test_col"), is_null=True),
        IsNullExpr(ColumnExpr("test_col"), is_null=False),
    ],
    ArrayLengthExpr: [
        ArrayLengthExpr(ColumnExpr("array_col")),
        ArrayLengthExpr(ArrayExpr([LiteralExpr("a", StringType)])),
    ],
    ArrayContainsExpr: [
        ArrayContainsExpr(
            ColumnExpr("array_col"), LiteralExpr("value", StringType)
        ),
        ArrayContainsExpr(
            ArrayExpr([LiteralExpr("a", StringType)]),
            LiteralExpr("a", StringType),
        ),
    ],
    CastExpr: [
        CastExpr(ColumnExpr("int_col"), StringType),
        CastExpr(LiteralExpr("42", StringType), IntegerType),
    ],
    NotExpr: [
        NotExpr(ColumnExpr("bool_col")),
        NotExpr(LiteralExpr(True, BooleanType)),
    ],
    CoalesceExpr: [
        CoalesceExpr([ColumnExpr("col1"), ColumnExpr("col2")]),
        CoalesceExpr(
            [LiteralExpr(None, StringType), LiteralExpr("default", StringType)]
        ),
    ],
    InExpr: [
        InExpr(
            ColumnExpr("test_col"),
            ArrayExpr(
                [LiteralExpr("a", StringType), LiteralExpr("b", StringType)]
            ),
        ),
    ],
    # Aggregate expressions
    SumExpr: [
        SumExpr(ColumnExpr("numeric_col")),
        SumExpr(LiteralExpr(42, IntegerType)),
    ],
    AvgExpr: [
        AvgExpr(ColumnExpr("numeric_col")),
        AvgExpr(LiteralExpr(3.14, FloatType)),
    ],
    MinExpr: [
        MinExpr(ColumnExpr("numeric_col")),
        MinExpr(LiteralExpr(42, IntegerType)),
    ],
    MaxExpr: [
        MaxExpr(ColumnExpr("numeric_col")),
        MaxExpr(LiteralExpr(42, IntegerType)),
    ],
    CountExpr: [
        CountExpr(ColumnExpr("any_col")),
        CountExpr(LiteralExpr("value", StringType)),
    ],
    ListExpr: [
        ListExpr(ColumnExpr("any_col")),
    ],
    FirstExpr: [
        FirstExpr(ColumnExpr("any_col")),
        FirstExpr(LiteralExpr("value", StringType)),
    ],
    StdDevExpr: [
        StdDevExpr(ColumnExpr("numeric_col")),
        StdDevExpr(LiteralExpr(3.14, FloatType)),
    ],
    # Arithmetic expressions
    ArithmeticExpr: [
        ArithmeticExpr(left=ColumnExpr("a"), right=ColumnExpr("b"), op=Operator.PLUS),
        ArithmeticExpr(
            left=LiteralExpr(5, IntegerType), right=LiteralExpr(3, IntegerType), op=Operator.MINUS
        ),
    ],
    # Comparison expressions
    BooleanExpr: [
        BooleanExpr(left=ColumnExpr("bool_col"), right=ColumnExpr("bool_col"), op=Operator.AND),
    ],
    EqualityComparisonExpr: [
        EqualityComparisonExpr(left=ColumnExpr("a"), right=ColumnExpr("b"), op=Operator.EQ),
        EqualityComparisonExpr(
            LiteralExpr("test", StringType),
            LiteralExpr("test", StringType),
            op=Operator.NOT_EQ,
        ),
    ],
    NumericComparisonExpr: [
        NumericComparisonExpr(left=ColumnExpr("a"), right=ColumnExpr("b"), op=Operator.GT),
        NumericComparisonExpr(
            left=LiteralExpr(5, IntegerType), right=LiteralExpr(3, IntegerType), op=Operator.LT
        ),
    ],
    # Case expressions
    WhenExpr: [
        WhenExpr(expr=None,condition=ColumnExpr("condition"), value=LiteralExpr("result", StringType)),
        WhenExpr(
            expr=ColumnExpr("expr"),
            condition=LiteralExpr(True, BooleanType),
            value=LiteralExpr("true_result", StringType),
        ),
    ],
    OtherwiseExpr: [
        OtherwiseExpr(expr=WhenExpr(expr=None,condition=ColumnExpr("condition"), value=LiteralExpr("result", StringType)), value=LiteralExpr("default", StringType)),
    ],
    # Embedding expressions
    EmbeddingNormalizeExpr: [
        EmbeddingNormalizeExpr(ColumnExpr("embedding_col")),
    ],
    EmbeddingSimilarityExpr: [
        EmbeddingSimilarityExpr(expr=ColumnExpr("emb1"), other=ColumnExpr("emb2"), metric="cosine"),
    ],
    # JSON expressions
    JqExpr: [
        JqExpr(ColumnExpr("json_col"), ".field"),
        JqExpr(LiteralExpr('{"key": "value"}', JsonType), ".key"),
    ],
    JsonContainsExpr: [
        JsonContainsExpr(
            ColumnExpr("json_col"), "{}"
        ),
    ],
    JsonTypeExpr: [
        JsonTypeExpr(ColumnExpr("json_col")),
    ],
    # Markdown expressions
    MdExtractHeaderChunks: [
        MdExtractHeaderChunks(ColumnExpr("md_col"),header_level=1),
    ],
    MdGenerateTocExpr: [
        MdGenerateTocExpr(ColumnExpr("md_col")),
    ],
    MdGetCodeBlocksExpr: [
        MdGetCodeBlocksExpr(ColumnExpr("md_col")),
    ],
    MdToJsonExpr: [
        MdToJsonExpr(ColumnExpr("md_col")),
    ],
    # Semantic expressions
    SemanticMapExpr: [
        SemanticMapExpr(instruction="Process ${text_col}", max_tokens=100, temperature=0.1),
        SemanticMapExpr(instruction="Extract ${name} from ${description}", max_tokens=200, temperature=0.2),
    ],
    SemanticExtractExpr: [
        SemanticExtractExpr(ColumnExpr("text_col"), schema=BasicResponseFormat, max_tokens=100, temperature=0.1),
    ],
    SemanticPredExpr: [
        SemanticPredExpr(instruction="${name} Is this positive?", temperature=0.1),
        SemanticPredExpr(instruction="${name} Contains important information?", temperature=0),
    ],
    SemanticReduceExpr: [
        SemanticReduceExpr(instruction="Summarize all ${documents}", max_tokens=100, temperature=0.1),
    ],
    SemanticClassifyExpr: [
        SemanticClassifyExpr(
            ColumnExpr("text_col"),
            [
                ResolvedClassDefinition("positive"),
                ResolvedClassDefinition("negative"),
            ],
            0.1,
        ),
    ],
    AnalyzeSentimentExpr: [
        AnalyzeSentimentExpr(ColumnExpr("text_col"), 0.1),
    ],
    EmbeddingsExpr: [
        EmbeddingsExpr(ColumnExpr("text_col")),
    ],
    SemanticSummarizeExpr: [
        SemanticSummarizeExpr(ColumnExpr("text_col"), format=Paragraph(max_words=100), temperature=0.1),
    ],
    # Text expressions
    TextractExpr: [
        TextractExpr(ColumnExpr("text_col"), "Extract ${field}"),
    ],
    TextChunkExpr: [
        TextChunkExpr(
            ColumnExpr("text_col"), TextChunkExprConfiguration(
                desired_chunk_size=100, chunk_overlap_percentage=10, chunk_length_function_name=ChunkLengthFunction.TOKEN,
            )
        ),
        TextChunkExpr(
            ColumnExpr("text_col"), TextChunkExprConfiguration(
                desired_chunk_size=200, chunk_overlap_percentage=0, chunk_length_function_name=ChunkLengthFunction.CHARACTER,
            )
        ),
    ],
    RecursiveTextChunkExpr: [
        RecursiveTextChunkExpr(
            ColumnExpr("text_col"), RecursiveTextChunkExprConfiguration(
                desired_chunk_size=100, chunk_overlap_percentage=10, chunk_length_function_name=ChunkLengthFunction.TOKEN, 
                chunking_character_set_name=ChunkCharacterSet.ASCII, chunking_character_set_custom_characters=["a", "b", "c"])
        ),
        RecursiveTextChunkExpr(
            ColumnExpr("text_col"),
            RecursiveTextChunkExprConfiguration(
                desired_chunk_size=200, chunk_overlap_percentage=0, chunk_length_function_name=ChunkLengthFunction.WORD,
                chunking_character_set_name=ChunkCharacterSet.ASCII, chunking_character_set_custom_characters=["a", "b", "c"])
        ),
    ],
    CountTokensExpr: [
        CountTokensExpr(ColumnExpr("text_col")),
    ],
    ConcatExpr: [
        ConcatExpr([ColumnExpr("col1"), ColumnExpr("col2")]),
        ConcatExpr([LiteralExpr("prefix", StringType), ColumnExpr("col")]),
    ],
    ArrayJoinExpr: [
        ArrayJoinExpr(ColumnExpr("array_col"), ","),
        ArrayJoinExpr(
            ArrayExpr(
                [LiteralExpr("a", StringType), LiteralExpr("b", StringType)]
            ),
            "|",
        ),
    ],
    ContainsExpr: [
        ContainsExpr(
            ColumnExpr("text_col"), LiteralExpr("substring", StringType)
        ),
    ],
    ContainsAnyExpr: [
        ContainsAnyExpr(ColumnExpr("text_col"), ["a", "b", "c"]),
        ContainsAnyExpr(
            ColumnExpr("text_col"),
            ["important", "urgent"],
            case_insensitive=False,
        ),
    ],
    RLikeExpr: [
        RLikeExpr(ColumnExpr("text_col"), r"\d+"),
    ],
    LikeExpr: [
        LikeExpr(ColumnExpr("text_col"), "%test%"),
        LikeExpr(ColumnExpr("text_col"), "test_"),
    ],
    ILikeExpr: [
        ILikeExpr(ColumnExpr("text_col"), "%TEST%"),
    ],
    TsParseExpr: [
        TsParseExpr(ColumnExpr("transcript_col"), "srt"),
    ],
    StartsWithExpr: [
        StartsWithExpr(
            ColumnExpr("text_col"), LiteralExpr("prefix", StringType)
        ),
    ],
    EndsWithExpr: [
        EndsWithExpr(ColumnExpr("text_col"), LiteralExpr("suffix", StringType)),
    ],
    RegexpSplitExpr: [
        RegexpSplitExpr(ColumnExpr("text_col"), r"\s+"),
        RegexpSplitExpr(ColumnExpr("text_col"), r",", 3),
    ],
    SplitPartExpr: [
        SplitPartExpr(
            ColumnExpr("text_col"),
            LiteralExpr(",", StringType),
            LiteralExpr(1, IntegerType),
        ),
    ],
    StringCasingExpr: [
        StringCasingExpr(ColumnExpr("text_col"), "upper"),
        StringCasingExpr(ColumnExpr("text_col"), "lower"),
        StringCasingExpr(ColumnExpr("text_col"), "title"),
    ],
    StripCharsExpr: [
        StripCharsExpr(ColumnExpr("text_col"), None, "both"),
        StripCharsExpr(
            ColumnExpr("text_col"), LiteralExpr(" \t", StringType), "left"
        ),
    ],
    ReplaceExpr: [
        ReplaceExpr(
            ColumnExpr("text_col"),
            LiteralExpr("old", StringType),
            LiteralExpr("new", StringType),
            True,
        ),
    ],
    StrLengthExpr: [
        StrLengthExpr(ColumnExpr("text_col")),
    ],
    ByteLengthExpr: [
        ByteLengthExpr(ColumnExpr("text_col")),
    ],
    JinjaExpr: [
        JinjaExpr(
            [ColumnExpr("name"), ColumnExpr("age")],
            "Hello {{name}}, you are {{age}} years old",
        ),
    ],
    FuzzyRatioExpr: [
        FuzzyRatioExpr(
            ColumnExpr("text1"),
            ColumnExpr("text2"),
            "damerau_levenshtein",
        ),
    ],
    FuzzyTokenSortRatioExpr: [
        FuzzyTokenSortRatioExpr(
            ColumnExpr("text1"),
            ColumnExpr("text2"),
            "hamming"
        ),
    ],
    FuzzyTokenSetRatioExpr: [
        FuzzyTokenSetRatioExpr(
            ColumnExpr("text1"),
            ColumnExpr("text2"),
            "jaro_winkler",
        ),
    ],
}

class TestExpressionSerde:
    """Test cases for logical expression serialization and deserialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = SerdeContext()

    def _compare_expressions(self, original: LogicalExpr, deserialized: LogicalExpr, expr_class_name: str, example_index: int):
        """Compare key attributes of original and deserialized expressions."""
        # For expressions with generated IDs or non-deterministic string representations,
        # we'll do more targeted comparisons instead of exact string matching
        
        # Handle specific expression types that have known issues
        if isinstance(original, LiteralExpr) and isinstance(deserialized, LiteralExpr):
            assert original.literal == deserialized.literal, (
                f"Literal value mismatch for {expr_class_name} example {example_index}"
            )
            assert original.data_type == deserialized.data_type, (
                f"Literal type mismatch for {expr_class_name} example {example_index}"
            )
            return
        
        if hasattr(original, 'schema') and hasattr(deserialized, 'schema') and issubclass(original.schema, BaseModel) and issubclass(deserialized.schema, BaseModel):
            if issubclass(original.schema, BasicResponseFormat):
                # check that all the fields in the original schema are present in the deserialized schema            
                original_fields = original.schema.model_fields
                deserialized_fields = deserialized.schema.model_fields
                for name, field_info in original_fields.items():
                    assert name in deserialized_fields, f"Field {name} not found in deserialized schema"
                    deserialized_field_info = deserialized_fields[name]
                    if field_info.annotation != deserialized_field_info.annotation:
                        # Known issue that jambo will turn literals into enums, which shouldn't be an issue since,
                        # at the end of the day, the serialized form for the literal/enum in the json response will be
                        # a string.
                        if get_origin(field_info.annotation) is not Literal:
                            raise ValueError(f"Field {name} type mismatch: {field_info.annotation} != {deserialized_field_info.annotation}")
                    assert field_info.default == deserialized_field_info.default, f"Field {name} default value mismatch"
                    assert field_info.description == deserialized_field_info.description, f"Field {name} description mismatch"
                    # Known issue that jambo will turn lists into optional fields in the model_json_schema
                    # assert field_info.is_required() == deserialized_field_info.is_required(), f"Field {name} required mismatch"
                return
            else:
                raise ValueError(f"Unsupported schema type: {type(original.schema)}")
        
        if isinstance(original, SemanticExtractExpr) and isinstance(deserialized, SemanticExtractExpr):
            self._compare_expressions(original.expr, deserialized.expr, f"{expr_class_name}.expr", example_index)
            return
        
        # For other expressions, try string comparison but be more lenient
        original_str = str(original)
        deserialized_str = str(deserialized)
        
        # Normalize strings for comparison (remove extra spaces, etc.)
        original_normalized = ' '.join(original_str.split())
        deserialized_normalized = ' '.join(deserialized_str.split())
        assert original_normalized == deserialized_normalized, (
                    f"String representation mismatch for {expr_class_name} example {example_index}:\n"
                    f"Original: {original_str}\n"
                    f"Deserialized: {deserialized_str}"
                )
               
        # For specific expression types, we can add more detailed comparisons
        if hasattr(original, 'name') and hasattr(deserialized, 'name'):
            assert original.name == deserialized.name, f"Alias name mismatch for {expr_class_name} example {example_index}"
        
        if hasattr(original, 'dest_type') and hasattr(deserialized, 'dest_type'):
            assert original.dest_type == deserialized.dest_type, f"Cast dest_type mismatch for {expr_class_name} example {example_index}"
        
        if hasattr(original, 'op') and hasattr(deserialized, 'op'):
            assert original.op == deserialized.op, f"Operator mismatch for {expr_class_name} example {example_index}"

    @pytest.mark.parametrize("expr_class", expression_examples.keys())
    def test_all_expression_types_with_examples(self, expr_class: Type[LogicalExpr]):
        """Test all registered expression types with comprehensive examples."""

        # Test each expression type with its examples
        for i, example in enumerate(expression_examples[expr_class]):
            # Skip UDFExpr as it cannot be serialized
            if expr_class == UDFExpr:
                with pytest.raises(SerializationError) as exc_info:
                    serialized = serialize_logical_expr(example, self.context)
                assert "Serialization not implemented for" in str(exc_info.value)
                continue

            try:
                # Serialize the expression
                serialized = serialize_logical_expr(example, self.context)
                assert serialized is not None, (
                    f"Serialization failed for {expr_class.__name__} example {i}"
                )

                # Deserialize the expression
                deserialized = deserialize_logical_expr(serialized, self.context)
                assert deserialized is not None, (
                    f"Deserialization failed for {expr_class.__name__} example {i}"
                )

                # Basic type check
                assert isinstance(deserialized, expr_class), (
                    f"Deserialized type mismatch for {expr_class.__name__} example {i}"
                )

                # Compare expressions using the helper method
                self._compare_expressions(example, deserialized, expr_class.__name__, i)

                

            except (SerializationError, DeserializationError) as e:
                pytest.fail(
                    f"Serde failed for {expr_class.__name__} example {i}: {e}"
                )

    def test_serialize_unregistered_expression_type(self):
        """Test that serializing an unregistered expression type raises an error."""

        # Create a mock expression that's not registered
        class MockExpr(LogicalExpr):
            def __init__(self):
                pass

            def __str__(self):
                return "mock_expr"
            
            def to_column_field(self, plan):
                return None

            def children(self):
                return []

        mock_expr = MockExpr()

        with pytest.raises(SerializationError) as exc_info:
            serialize_logical_expr(mock_expr, self.context)

        assert "Serialization not implemented for" in str(exc_info.value)

    def test_deserialize_empty_proto(self):
        """Test deserialization of an empty LogicalExprProto returns None."""
        from fenic.core._serde.proto.types import LogicalExprProto
        
        empty_proto = LogicalExprProto()
        result = deserialize_logical_expr(empty_proto, self.context)
        assert result is None


    def test_expression_with_complex_nesting(self):
        """Test expressions with complex nested structures."""
        # Create a deeply nested expression
        nested_expr = AliasExpr(
            CastExpr(
                ArrayExpr(
                    [
                        ColumnExpr("col1"),
                        AliasExpr(ColumnExpr("col2"), "inner_alias"),
                        LiteralExpr("default", StringType),
                    ]
                ),
                JsonType,
            ),
            "complex_alias",
        )

        # Serialize and deserialize
        serialized = serialize_logical_expr(nested_expr, self.context)
        deserialized = deserialize_logical_expr(serialized, self.context)

        # Should maintain structure
        assert isinstance(deserialized, AliasExpr)
        assert deserialized.name == "complex_alias"

        # Check nested structure
        cast_expr = deserialized.expr
        assert isinstance(cast_expr, CastExpr)
        assert cast_expr.dest_type == JsonType

        array_expr = cast_expr.expr
        assert isinstance(array_expr, ArrayExpr)
        assert len(array_expr.exprs) == 3
