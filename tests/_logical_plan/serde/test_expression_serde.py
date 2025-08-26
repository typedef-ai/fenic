"""Tests for logical expression serialization and deserialization."""

import asyncio
from typing import List, Literal, Optional, Type

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
    AsyncUDFExpr,
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
from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    Operator,
    UnparameterizedExpr,
)
from fenic.core._logical_plan.expressions.basic import GreatestExpr, LeastExpr
from fenic.core._logical_plan.expressions.text import (
    ChunkCharacterSet,
    ChunkLengthFunction,
    RecursiveTextChunkExprConfiguration,
    TextChunkExprConfiguration,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._serde.proto.errors import SerializationError, UnsupportedTypeError
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
from fenic.core.types.semantic_examples import (
    MapExample,
    MapExampleCollection,
    PredicateExample,
    PredicateExampleCollection,
)
from fenic.core.types.summarize import Paragraph


class NestedResponseFormat(BaseModel):
    string: str
    inner_list: List[str]

class BasicResponseFormat(BaseModel):
    name: str
    age: int
    email: str
    nicknames: list[str] = Field(...)
    valid: bool
    state: Literal["active", "inactive"]
    explanation: Optional[str] = None
    nested: NestedResponseFormat


async def async_udf_fn(x: str) -> str:
    await asyncio.sleep(1)
    return x + " async"

# Define examples for each expression type
# Each type has a list of examples to test different scenarios. In general when adding new expressions:
# 1. Add a new entry to the expression_examples dictionary with the expression class as the key and a list of examples as the value.
# 2. If the expression has optional parameters, ensure that examples exist for all permutations of optional parameters provided/not provided.
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
    AsyncUDFExpr: [
        # Note: AsyncUDFExpr cannot be serialized, but we test the type exists
        AsyncUDFExpr(async_udf_fn, [ColumnExpr("test_col")], StringType),
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
        MdGenerateTocExpr(ColumnExpr("md_col"), max_level=2),
    ],
    MdGetCodeBlocksExpr: [
        MdGetCodeBlocksExpr(ColumnExpr("md_col")),
        MdGetCodeBlocksExpr(ColumnExpr("md_col"), language_filter="python"),
    ],
    MdToJsonExpr: [
        MdToJsonExpr(ColumnExpr("md_col")),
    ],
    # Semantic expressions
    SemanticMapExpr: [
        SemanticMapExpr(jinja_template="Process {{text_col}}", strict=True, max_tokens=100, temperature=0.1,  exprs=[ColumnExpr("text_col")]),
        SemanticMapExpr(jinja_template="Process {{text_col}}", strict=False, temperature=0, max_tokens=100, examples=MapExampleCollection(
            [
                MapExample(input={"text_col": "text1"}, output="A result"),
                MapExample(input={"text_col": "text2"}, output="Another result"),
            ]
        ), exprs=[ColumnExpr("text_col")]),
        SemanticMapExpr(jinja_template="Process {{struct_col}}", strict=False, temperature=0, max_tokens=100, examples=MapExampleCollection(
            [
                MapExample(input={"struct_col": {"name": "John", "age": 30}}, output="A result"),
                MapExample(input={"struct_col": {"name": "Jane", "age": 25}}, output="Another result"),
            ]
        ), exprs=[ColumnExpr("struct_col")]),
        SemanticMapExpr(jinja_template="Process {{text_col}}", strict=False, temperature=0, max_tokens=100, response_format=ResolvedResponseFormat.from_pydantic_model(BasicResponseFormat), exprs=[ColumnExpr("text_col")]),
    ],
    SemanticExtractExpr: [
        SemanticExtractExpr(ColumnExpr("text_col"), response_format=ResolvedResponseFormat.from_pydantic_model(BasicResponseFormat), max_tokens=100, temperature=0.1),
    ],
    SemanticPredExpr: [
        SemanticPredExpr(jinja_template="{{name}} Is this positive?", strict=True, exprs=[ColumnExpr("name")],  temperature=0.1),
        SemanticPredExpr(jinja_template="{{name}} Contains important information?", strict=True, exprs=[ColumnExpr("name")], temperature=0),
        SemanticPredExpr(jinja_template="{{name}} Is traditionally male?", strict=True, exprs=[ColumnExpr("name")], temperature=0, examples=PredicateExampleCollection(
            [
                PredicateExample(input={"name": "John"}, output=True),
                PredicateExample(input={"name": "Jane"}, output=False),
            ]
        )),
        SemanticPredExpr(jinja_template="{{struct_col}} Are the two names the same?", strict=True, exprs=[ColumnExpr("struct_col")], temperature=0, examples=PredicateExampleCollection(
            [
                PredicateExample(input={"struct_col": {"name": "John", "name_copy": "John"}}, output=True),
                PredicateExample(input={"struct_col": {"name": "Jane", "name_copy": "John"}}, output=False),
            ]
        )),
    ],
    SemanticReduceExpr: [
        SemanticReduceExpr(instruction="Summarize all documents in group", group_context_exprs=[], order_by_exprs=[], input_expr=ColumnExpr("document"), max_tokens=100, temperature=0.1),
        SemanticReduceExpr(instruction="Summarize all documents in group in {{date}}", group_context_exprs=[ColumnExpr("date")], order_by_exprs=[ColumnExpr("doc_index")], input_expr=ColumnExpr("document"), max_tokens=100, temperature=0.1),

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
        SemanticClassifyExpr(
            ColumnExpr("text_col"),
            [
                ResolvedClassDefinition("positive", description="A positive class"),
                ResolvedClassDefinition("negative", description="A negative class"),
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
                chunking_character_set_name=ChunkCharacterSet.ASCII)
        ),
        RecursiveTextChunkExpr(
            ColumnExpr("text_col"),
            RecursiveTextChunkExprConfiguration(
                desired_chunk_size=200, chunk_overlap_percentage=0, chunk_length_function_name=ChunkLengthFunction.WORD,
                chunking_character_set_name=ChunkCharacterSet.CUSTOM, chunking_character_set_custom_characters=["a", "b", "c"])
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
            strict=True
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
    GreatestExpr: [
        GreatestExpr([ColumnExpr("a"), ColumnExpr("b"), LiteralExpr(1, IntegerType)]),
    ],
    LeastExpr: [
        LeastExpr([ColumnExpr("a"), ColumnExpr("b"), LiteralExpr(1, IntegerType)]),
    ],
}

class TestExpressionSerde:
    """Test cases for logical expression serialization and deserialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = SerdeContext()

    def _compare_expressions(self, original: LogicalExpr, deserialized: LogicalExpr, expr_class_name: str, example_index: int):
        """Compare key attributes of original and deserialized expressions."""
        if not original == deserialized:
            raise ValueError(f"Original {original} does not match deserialized {deserialized}. Class Name: {expr_class_name}, Example Index: {example_index}")

    @pytest.mark.parametrize("expr_class", expression_examples.keys())
    def test_all_expression_types_with_examples(self, expr_class: Type[LogicalExpr]):
        """Test all registered expression types with comprehensive examples."""

        # Test each expression type with its examples
        for i, example in enumerate(expression_examples[expr_class]):
            # Expect serialization errors for UDFExpr
            if expr_class == UDFExpr or expr_class == AsyncUDFExpr:
                with pytest.raises(UnsupportedTypeError, match="UDFExpr cannot be serialized"):
                    serialized = serialize_logical_expr(example, self.context)
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

            except Exception as e:
                pytest.fail(
                    f"Serde failed for {expr_class.__name__} example {i}: {e}"
                )

    def test_serialize_unregistered_expression_type(self):
        """Test that serializing an unregistered expression type raises an error."""

        # Create a mock expression that's not registered
        class MockExpr(UnparameterizedExpr, LogicalExpr):
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
        assert nested_expr == deserialized

    def test_all_logical_expr_subclasses_covered(self):
        """Test that all concrete LogicalExpr subclasses are covered in the test file."""
        import importlib
        import inspect

        from fenic.core._logical_plan.expressions.base import LogicalExpr
        # Import all expression modules
        expression_modules = [
            "fenic.core._logical_plan.expressions.basic",
            "fenic.core._logical_plan.expressions.aggregate",
            "fenic.core._logical_plan.expressions.arithmetic",
            "fenic.core._logical_plan.expressions.comparison",
            "fenic.core._logical_plan.expressions.case",
            "fenic.core._logical_plan.expressions.embedding",
            "fenic.core._logical_plan.expressions.json",
            "fenic.core._logical_plan.expressions.markdown",
            "fenic.core._logical_plan.expressions.semantic",
            "fenic.core._logical_plan.expressions.text",
        ]

        # Find all concrete LogicalExpr subclasses
        concrete_subclasses = set()
        for module_name in expression_modules:
            try:
                module = importlib.import_module(module_name)
                for _name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, LogicalExpr) and
                        obj != LogicalExpr and
                        not inspect.isabstract(obj)):
                        concrete_subclasses.add(obj.__name__)
            except ImportError:
                continue

        # Get all tested expression classes from the expression_examples dictionary
        tested_classes = set(cls.__name__ for cls in expression_examples.keys())

        # Find missing classes
        missing = concrete_subclasses - tested_classes

        if missing:
            pytest.fail(
                f"Missing {len(missing)} concrete LogicalExpr subclasses from tests: {sorted(missing)}. "
                f"Add them to the expression_examples dictionary in this test file."
            )

        # Optional: Check for extra classes (not LogicalExpr subclasses)
        extra = tested_classes - concrete_subclasses
        if extra:
            print(f"Warning: {len(extra)} tested classes are not concrete LogicalExpr subclasses: {sorted(extra)}")

        # Verify coverage
        coverage = len(concrete_subclasses - missing) / len(concrete_subclasses) * 100
        assert coverage == 100.0, f"Expression coverage is {coverage:.1f}%, expected 100%"
