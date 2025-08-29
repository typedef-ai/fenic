"""Tests for LogicalExpr equality implementation.

This test suite focuses on:
1. One comprehensive test for recursive __eq__ logic
2. Minimal tests for each expression's unique equality attributes
"""


from fenic.core._logical_plan.expressions.aggregate import (
    AvgExpr,
    CountExpr,
    FirstExpr,
    ListExpr,
    MaxExpr,
    MinExpr,
    StdDevExpr,
    SumExpr,
)
from fenic.core._logical_plan.expressions.arithmetic import ArithmeticExpr
from fenic.core._logical_plan.expressions.base import Operator
from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ArrayContainsExpr,
    ArrayExpr,
    ArrayLengthExpr,
    AsyncUDFExpr,
    CastExpr,
    CoalesceExpr,
    ColumnExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    LiteralExpr,
    NotExpr,
    SortExpr,
    StructExpr,
    UDFExpr,
)
from fenic.core._logical_plan.expressions.case import (
    OtherwiseExpr,
    WhenExpr,
)
from fenic.core._logical_plan.expressions.comparison import (
    EqualityComparisonExpr,
    NumericComparisonExpr,
)
from fenic.core._logical_plan.expressions.embedding import (
    EmbeddingNormalizeExpr,
    EmbeddingSimilarityExpr,
)
from fenic.core._logical_plan.expressions.json import (
    JqExpr,
    JsonContainsExpr,
    JsonTypeExpr,
)
from fenic.core._logical_plan.expressions.markdown import (
    MdExtractHeaderChunks,
    MdGenerateTocExpr,
    MdGetCodeBlocksExpr,
    MdToJsonExpr,
)
from fenic.core._logical_plan.expressions.semantic import (
    AnalyzeSentimentExpr,
    EmbeddingsExpr,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticPredExpr,
    SemanticReduceExpr,
    SemanticSummarizeExpr,
)
from fenic.core._logical_plan.expressions.text import (
    ArrayJoinExpr,
    ByteLengthExpr,
    ChunkCharacterSet,
    ChunkLengthFunction,
    ConcatExpr,
    ContainsAnyExpr,
    ContainsExpr,
    CountTokensExpr,
    EndsWithExpr,
    FuzzyRatioExpr,
    FuzzyTokenSetRatioExpr,
    FuzzyTokenSortRatioExpr,
    ILikeExpr,
    JinjaExpr,
    LikeExpr,
    RecursiveTextChunkExpr,
    RecursiveTextChunkExprConfiguration,
    RegexpSplitExpr,
    ReplaceExpr,
    RLikeExpr,
    SplitPartExpr,
    StartsWithExpr,
    StringCasingExpr,
    StripCharsExpr,
    StrLengthExpr,
    TextChunkExpr,
    TextChunkExprConfiguration,
    TextractExpr,
    TsParseExpr,
)
from fenic.core.types.datatypes import FloatType, IntegerType, StringType


class TestRecursiveEquality:
    """Test the base __eq__ recursive logic once."""

    def test_recursive_equality_comprehensive(self):
        """Test that __eq__ correctly handles recursion, type checking, and edge cases."""
        # Test recursive equality with nested expressions
        col1 = ColumnExpr("x")
        col2 = ColumnExpr("x")
        col3 = ColumnExpr("y")

        lit1 = LiteralExpr(10, IntegerType)
        lit2 = LiteralExpr(10, IntegerType)

        # Create: alias(alias(col + lit))
        inner_expr1 = ArithmeticExpr(col1, lit1, Operator.PLUS)
        inner_expr2 = ArithmeticExpr(col2, lit2, Operator.PLUS)
        inner_expr3 = ArithmeticExpr(col3, lit1, Operator.PLUS)  # Different column

        inner_alias1 = AliasExpr(inner_expr1, "inner")
        inner_alias2 = AliasExpr(inner_expr2, "inner")
        inner_alias3 = AliasExpr(inner_expr3, "inner")

        outer_alias1 = AliasExpr(inner_alias1, "outer")
        outer_alias2 = AliasExpr(inner_alias2, "outer")
        outer_alias3 = AliasExpr(inner_alias3, "outer")

        # Identical nested structures should be equal
        assert outer_alias1 == outer_alias2

        # Different deep expressions should not be equal
        assert outer_alias1 != outer_alias3

        # Test type checking
        assert col1 != lit1  # Different types
        assert col1 != "string"  # Non-LogicalExpr
        assert col1 is not None  # None

        # Test self equality
        assert outer_alias1 == outer_alias1

    def test_children_count_mismatch(self):
        """Test that expressions with different child counts are not equal."""
        col = ColumnExpr("x")

        # UnaryExpr (1 child) vs BinaryExpr (2 children)
        alias = AliasExpr(col, "name")  # 1 child
        arith = ArithmeticExpr(col, col, Operator.PLUS)  # 2 children

        assert alias != arith


class TestBasicExpressions:
    """Test equality for basic expressions."""

    def test_column_expr(self):
        """Test ColumnExpr compares name."""
        col1 = ColumnExpr("name")
        col2 = ColumnExpr("name")
        col3 = ColumnExpr("age")

        assert col1 == col2  # Same name
        assert col1 != col3  # Different name

    def test_literal_expr(self):
        """Test LiteralExpr compares literal and data_type."""
        # Same value and type
        lit1 = LiteralExpr(42, IntegerType)
        lit2 = LiteralExpr(42, IntegerType)
        assert lit1 == lit2

        # Different value
        lit3 = LiteralExpr(43, IntegerType)
        assert lit1 != lit3

        # Different type
        lit4 = LiteralExpr(42, FloatType)
        assert lit1 != lit4

    def test_alias_expr(self):
        """Test AliasExpr compares name only."""
        col1 = ColumnExpr("age")
        col2 = ColumnExpr("years")  # Different expression

        alias1 = AliasExpr(col1, "user_age")
        alias2 = AliasExpr(col2, "user_age")  # Same name, different expr
        alias3 = AliasExpr(col1, "person_age")  # Different name, same expr

        # Note: expressions are compared via children, name via _eq_specific
        assert alias1 != alias2  # Different child expression
        assert alias1 != alias3  # Different name

    def test_sort_expr(self):
        """Test SortExpr compares ascending and nulls_last."""
        col = ColumnExpr("name")

        sort1 = SortExpr(col, ascending=True, nulls_last=False)
        sort2 = SortExpr(col, ascending=True, nulls_last=False)
        sort3 = SortExpr(col, ascending=False, nulls_last=False)  # Different ascending
        sort4 = SortExpr(col, ascending=True, nulls_last=True)   # Different nulls_last

        assert sort1 == sort2
        assert sort1 != sort3
        assert sort1 != sort4

    def test_index_expr(self):
        """Test IndexExpr compares key."""
        arr = ColumnExpr("array_col")

        # Same key
        index1 = IndexExpr(arr, "field1")
        index2 = IndexExpr(arr, "field1")
        assert index1 == index2

        # Different key
        index3 = IndexExpr(arr, "field2")
        assert index1 != index3

        # Different index
        index = ColumnExpr("blah")
        index4 = IndexExpr(arr, index)
        assert index1 != index4

    def test_array_expr(self):
        """Test ArrayExpr has no specific attributes."""
        col1 = ColumnExpr("x")
        col2 = ColumnExpr("y")

        array1 = ArrayExpr([col1, col2])
        array2 = ArrayExpr([col2, col1])  # Different order

        # Different children order, should not be equal
        assert array1 != array2

    def test_struct_expr(self):
        """Test StructExpr has no specific attributes."""
        col1 = ColumnExpr("name")
        col2 = ColumnExpr("age")

        struct1 = StructExpr([col1, col2])
        struct2 = StructExpr([col2, col1])  # Different order

        # Different children order, should not be equal
        assert struct1 != struct2

    def test_udf_expr(self):
        """Test UDFExpr compares func identity and return_type."""
        def func1(x): return x * 2
        def func2(x): return x * 3

        col = ColumnExpr("x")

        # Same function and return type
        udf1 = UDFExpr(func1, [col], IntegerType)
        udf2 = UDFExpr(func1, [col], IntegerType)
        assert udf1 == udf2

        # Different function
        udf3 = UDFExpr(func2, [col], IntegerType)
        assert udf1 != udf3

        # Different return type
        udf4 = UDFExpr(func1, [col], StringType)
        assert udf1 != udf4

    def test_async_udf_expr(self):
        """Test AsyncUDFExpr compares func identity, return_type, and async-specific attributes."""
        async def func1(x): return x * 2
        async def func2(x): return x * 3

        col = ColumnExpr("x")

        # Same function and all attributes
        audf1 = AsyncUDFExpr(func1, [col], IntegerType, max_concurrency=10, timeout_seconds=30, num_retries=2)
        audf2 = AsyncUDFExpr(func1, [col], IntegerType, max_concurrency=10, timeout_seconds=30, num_retries=2)
        assert audf1 == audf2

        # Different function
        audf3 = AsyncUDFExpr(func2, [col], IntegerType, max_concurrency=10, timeout_seconds=30, num_retries=2)
        assert audf1 != audf3

        # Different return type
        audf4 = AsyncUDFExpr(func1, [col], StringType, max_concurrency=10, timeout_seconds=30, num_retries=2)
        assert audf1 != audf4

        # Different max_concurrency
        audf5 = AsyncUDFExpr(func1, [col], IntegerType, max_concurrency=20, timeout_seconds=30, num_retries=2)
        assert audf1 != audf5

        # Different timeout_seconds
        audf6 = AsyncUDFExpr(func1, [col], IntegerType, max_concurrency=10, timeout_seconds=60, num_retries=2)
        assert audf1 != audf6

        # Different num_retries
        audf7 = AsyncUDFExpr(func1, [col], IntegerType, max_concurrency=10, timeout_seconds=30, num_retries=3)
        assert audf1 != audf7

    def test_is_null_expr(self):
        """Test IsNullExpr compares is_null flag."""
        col = ColumnExpr("x")

        # Same is_null flag
        null1 = IsNullExpr(col, True)
        null2 = IsNullExpr(col, True)
        assert null1 == null2

        # Different is_null flag
        null3 = IsNullExpr(col, False)
        assert null1 != null3

    def test_array_length_expr(self):
        """Test ArrayLengthExpr has no specific attributes."""
        arr1 = ColumnExpr("array1")
        arr2 = ColumnExpr("array2")

        len1 = ArrayLengthExpr(arr1)
        len2 = ArrayLengthExpr(arr2)

        # Different children, should not be equal
        assert len1 != len2

    def test_array_contains_expr(self):
        """Test ArrayContainsExpr has no specific attributes."""
        arr = ColumnExpr("array_col")
        val1 = LiteralExpr("item1", StringType)
        val2 = LiteralExpr("item2", StringType)

        contains1 = ArrayContainsExpr(arr, val1)
        contains2 = ArrayContainsExpr(arr, val2)

        # Different children, should not be equal
        assert contains1 != contains2

    def test_cast_expr(self):
        """Test CastExpr compares target_type."""
        col = ColumnExpr("x")

        # Same target type
        cast1 = CastExpr(col, StringType)
        cast2 = CastExpr(col, StringType)
        assert cast1 == cast2

        # Different target type
        cast3 = CastExpr(col, IntegerType)
        assert cast1 != cast3

    def test_not_expr(self):
        """Test NotExpr has no specific attributes."""
        col1 = ColumnExpr("bool1")
        col2 = ColumnExpr("bool2")

        not1 = NotExpr(col1)
        not2 = NotExpr(col2)

        # Different children, should not be equal
        assert not1 != not2

    def test_coalesce_expr(self):
        """Test CoalesceExpr has no specific attributes."""
        col1 = ColumnExpr("x")
        col2 = ColumnExpr("y")
        col3 = ColumnExpr("z")

        coalesce1 = CoalesceExpr([col1, col2])
        coalesce2 = CoalesceExpr([col2, col3])  # Different expressions

        # Different children, should not be equal
        assert coalesce1 != coalesce2

    def test_in_expr(self):
        """Test InExpr has no specific attributes."""
        col = ColumnExpr("x")
        val1 = LiteralExpr(1, IntegerType)
        val2 = LiteralExpr(2, IntegerType)
        val3 = LiteralExpr(3, IntegerType)

        in1 = InExpr(col, [val1, val2])
        in2 = InExpr(col, [val2, val3])  # Different values

        # Different children, should not be equal
        assert in1 != in2


class TestBinaryExpressions:
    """Test equality for binary expressions."""

    def test_binary_expr_operator(self):
        """Test BinaryExpr compares operator."""
        col = ColumnExpr("x")
        lit = LiteralExpr(10, IntegerType)

        add = ArithmeticExpr(col, lit, Operator.PLUS)
        sub = ArithmeticExpr(col, lit, Operator.MINUS)
        add2 = ArithmeticExpr(col, lit, Operator.PLUS)

        assert add == add2    # Same operator
        assert add != sub     # Different operator

    def test_comparison_expr_types(self):
        """Test different comparison expression types are not equal."""
        col = ColumnExpr("x")
        lit = LiteralExpr(10, IntegerType)

        eq_comp = EqualityComparisonExpr(col, lit, Operator.EQ)
        num_comp = NumericComparisonExpr(col, lit, Operator.EQ)

        # Even with same operator, different types should not be equal
        assert eq_comp != num_comp


class TestJsonExpressions:
    """Test equality for JSON expressions."""

    def test_jq_expr(self):
        """Test JqExpr compares query."""
        col = ColumnExpr("data")

        jq1 = JqExpr(col, ".foo.bar")
        jq2 = JqExpr(col, ".foo.bar")
        jq3 = JqExpr(col, ".baz")

        assert jq1 == jq2  # Same query
        assert jq1 != jq3  # Different query

    def test_json_type_expr(self):
        """Test JsonTypeExpr has no specific attributes."""
        col1 = ColumnExpr("data1")
        col2 = ColumnExpr("data2")

        type1 = JsonTypeExpr(col1)
        type2 = JsonTypeExpr(col2)

        # Different children, should not be equal
        assert type1 != type2

    def test_json_contains_expr(self):
        """Test JsonContainsExpr compares value."""
        col = ColumnExpr("data")

        contains1 = JsonContainsExpr(col, '{"key": "value"}')
        contains2 = JsonContainsExpr(col, '{"key": "value"}')
        contains3 = JsonContainsExpr(col, '{"other": "data"}')

        assert contains1 == contains2  # Same value
        assert contains1 != contains3  # Different value


class TestMarkdownExpressions:
    """Test equality for Markdown expressions."""

    def test_md_to_json_expr(self):
        """Test MdToJsonExpr has no specific attributes."""
        col1 = ColumnExpr("md1")
        col2 = ColumnExpr("md2")

        md1 = MdToJsonExpr(col1)
        md2 = MdToJsonExpr(col2)

        # Different children, should not be equal
        assert md1 != md2

    def test_md_get_code_blocks_expr(self):
        """Test MdGetCodeBlocksExpr compares language_filter."""
        col = ColumnExpr("md")

        blocks1 = MdGetCodeBlocksExpr(col, language_filter="python")
        blocks2 = MdGetCodeBlocksExpr(col, language_filter="python")
        blocks3 = MdGetCodeBlocksExpr(col, language_filter="javascript")
        blocks4 = MdGetCodeBlocksExpr(col, language_filter=None)

        assert blocks1 == blocks2  # Same filter
        assert blocks1 != blocks3  # Different filter
        assert blocks1 != blocks4  # Filter vs None

    def test_md_generate_toc_expr(self):
        """Test MdGenerateTocExpr compares max_level."""
        col = ColumnExpr("md")

        toc1 = MdGenerateTocExpr(col, max_level=3)
        toc2 = MdGenerateTocExpr(col, max_level=3)
        toc3 = MdGenerateTocExpr(col, max_level=2)

        assert toc1 == toc2  # Same max_level
        assert toc1 != toc3  # Different max_level

    def test_md_extract_header_chunks_expr(self):
        """Test MdExtractHeaderChunks compares header_level."""
        col = ColumnExpr("md")

        chunks1 = MdExtractHeaderChunks(col, header_level=2)
        chunks2 = MdExtractHeaderChunks(col, header_level=2)
        chunks3 = MdExtractHeaderChunks(col, header_level=3)

        assert chunks1 == chunks2  # Same header_level
        assert chunks1 != chunks3  # Different header_level


class TestCaseExpressions:
    """Test equality for case expressions."""

    def test_when_expr(self):
        """Test WhenExpr compares whether expr is None."""
        col = ColumnExpr("x")
        cond = EqualityComparisonExpr(col, LiteralExpr(1, IntegerType), Operator.EQ)
        val = LiteralExpr("result", StringType)

        # Both with None expr
        when1 = WhenExpr(None, cond, val)
        when2 = WhenExpr(None, cond, val)
        assert when1 == when2

        # Both with non-None expr
        when3 = WhenExpr(col, cond, val)
        when4 = WhenExpr(col, cond, val)
        assert when3 == when4

        # One None, one not None
        assert when1 != when3

    def test_otherwise_expr(self):
        """Test OtherwiseExpr has no specific attributes."""
        col = ColumnExpr("x")
        cond = EqualityComparisonExpr(col, LiteralExpr(1, IntegerType), Operator.EQ)
        val = LiteralExpr("result", StringType)
        when = WhenExpr(None, cond, val)

        else_val1 = LiteralExpr("default1", StringType)
        else_val2 = LiteralExpr("default2", StringType)

        otherwise1 = OtherwiseExpr(when, else_val1)
        otherwise2 = OtherwiseExpr(when, else_val2)

        # Different else values, should not be equal due to children
        assert otherwise1 != otherwise2


class TestAggregateExpressions:
    """Test equality for aggregate expressions."""

    def test_sum_expr(self):
        """Test SumExpr has no specific attributes."""
        col1 = ColumnExpr("amount1")
        col2 = ColumnExpr("amount2")

        sum1 = SumExpr(col1)
        sum2 = SumExpr(col2)

        # Different children, should not be equal
        assert sum1 != sum2

    def test_avg_expr(self):
        """Test AvgExpr has no specific attributes."""
        col1 = ColumnExpr("score1")
        col2 = ColumnExpr("score2")

        avg1 = AvgExpr(col1)
        avg2 = AvgExpr(col2)

        # Different children, should not be equal
        assert avg1 != avg2

    def test_min_expr(self):
        """Test MinExpr has no specific attributes."""
        col1 = ColumnExpr("value1")
        col2 = ColumnExpr("value2")

        min1 = MinExpr(col1)
        min2 = MinExpr(col2)

        # Different children, should not be equal
        assert min1 != min2

    def test_max_expr(self):
        """Test MaxExpr has no specific attributes."""
        col1 = ColumnExpr("value1")
        col2 = ColumnExpr("value2")

        max1 = MaxExpr(col1)
        max2 = MaxExpr(col2)

        # Different children, should not be equal
        assert max1 != max2

    def test_count_expr(self):
        """Test CountExpr has no specific attributes."""
        col1 = ColumnExpr("id1")
        col2 = ColumnExpr("id2")

        count1 = CountExpr(col1)
        count2 = CountExpr(col2)

        # Different children, should not be equal
        assert count1 != count2

    def test_list_expr(self):
        """Test ListExpr has no specific attributes."""
        col1 = ColumnExpr("item1")
        col2 = ColumnExpr("item2")

        list1 = ListExpr(col1)
        list2 = ListExpr(col2)

        # Different children, should not be equal
        assert list1 != list2

    def test_first_expr(self):
        """Test FirstExpr has no specific attributes."""
        col1 = ColumnExpr("data1")
        col2 = ColumnExpr("data2")

        first1 = FirstExpr(col1)
        first2 = FirstExpr(col2)

        # Different children, should not be equal
        assert first1 != first2

    def test_stddev_expr(self):
        """Test StdDevExpr has no specific attributes."""
        col1 = ColumnExpr("values1")
        col2 = ColumnExpr("values2")

        std1 = StdDevExpr(col1)
        std2 = StdDevExpr(col2)

        # Different children, should not be equal
        assert std1 != std2

    def test_aggregate_types_never_equal(self):
        """Test different aggregate expression types are never equal."""
        col = ColumnExpr("x")

        sum_expr = SumExpr(col)
        avg_expr = AvgExpr(col)
        min_expr = MinExpr(col)
        max_expr = MaxExpr(col)
        count_expr = CountExpr(col)
        list_expr = ListExpr(col)
        first_expr = FirstExpr(col)
        std_expr = StdDevExpr(col)

        agg_exprs = [sum_expr, avg_expr, min_expr, max_expr, count_expr, list_expr, first_expr, std_expr]

        for i, expr1 in enumerate(agg_exprs):
            for j, expr2 in enumerate(agg_exprs):
                if i != j:
                    assert expr1 != expr2


class TestEmbeddingExpressions:
    """Test equality for embedding expressions."""

    def test_embedding_normalize_expr(self):
        """Test EmbeddingNormalizeExpr has no specific attributes."""
        col1 = ColumnExpr("embedding1")
        col2 = ColumnExpr("embedding2")

        norm1 = EmbeddingNormalizeExpr(col1)
        norm2 = EmbeddingNormalizeExpr(col2)

        # Different children, should not be equal
        assert norm1 != norm2

    def test_embedding_similarity_expr(self):
        """Test EmbeddingSimilarityExpr compares metric and other type."""
        import numpy as np


        col1 = ColumnExpr("embedding1")
        col2 = ColumnExpr("embedding2")
        query_vec1 = np.array([1.0, 2.0, 3.0])
        query_vec2 = np.array([1.0, 2.0, 3.0])
        query_vec3 = np.array([4.0, 5.0, 6.0])

        # Same metric, same query vector
        sim1 = EmbeddingSimilarityExpr(col1, query_vec1, "cosine")
        sim2 = EmbeddingSimilarityExpr(col1, query_vec2, "cosine")
        assert sim1 == sim2

        # Different metric
        sim3 = EmbeddingSimilarityExpr(col1, query_vec1, "euclidean")
        assert sim1 != sim3

        # Different query vector
        sim4 = EmbeddingSimilarityExpr(col1, query_vec3, "cosine")
        assert sim1 != sim4

        # LogicalExpr vs numpy array
        sim5 = EmbeddingSimilarityExpr(col1, col2, "cosine")
        assert sim1 != sim5

        # Both LogicalExpr, same metric
        sim6 = EmbeddingSimilarityExpr(col1, col2, "cosine")
        assert sim5 == sim6


class TestTextExpressions:
    """Test equality for text expressions."""

    def test_textract_expr(self):
        """Test TextractExpr compares template."""
        col = ColumnExpr("text")

        # Same template
        textract1 = TextractExpr(col, "Name: ${name}")
        textract2 = TextractExpr(col, "Name: ${name}")
        assert textract1 == textract2

        # Different template
        textract3 = TextractExpr(col, "Age: ${age}")
        assert textract1 != textract3

    def test_text_chunk_expr(self):
        """Test TextChunkExpr compares chunk_configuration."""
        col = ColumnExpr("text")

        # Same configuration
        chunk1 = TextChunkExpr(col, TextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
        ))
        chunk2 = TextChunkExpr(col, TextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
        ))
        assert chunk1 == chunk2

        # Different desired_chunk_size
        chunk3 = TextChunkExpr(col, TextChunkExprConfiguration(
            desired_chunk_size=200,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
        ))
        assert chunk1 != chunk3

        # Different chunk_overlap_percentage
        chunk4 = TextChunkExpr(col, TextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=20,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
        ))
        assert chunk1 != chunk4

        # Different chunk_length_function_name
        chunk5 = TextChunkExpr(col, TextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.WORD,
        ))
        assert chunk1 != chunk5

    def test_recursive_text_chunk_expr(self):
        """Test RecursiveTextChunkExpr compares chunking_configuration."""
        col = ColumnExpr("text")

        # Same configuration
        chunk1 = RecursiveTextChunkExpr(col, RecursiveTextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
            chunking_character_set_name=ChunkCharacterSet.ASCII,
        ))
        chunk2 = RecursiveTextChunkExpr(col, RecursiveTextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
            chunking_character_set_name=ChunkCharacterSet.ASCII,
        ))
        assert chunk1 == chunk2

        # Different chunking_character_set_name
        chunk3 = RecursiveTextChunkExpr(col, RecursiveTextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
            chunking_character_set_name=ChunkCharacterSet.UNICODE,
        ))
        assert chunk1 != chunk3

        # Different chunk_character_set_custom_characters
        chunk4 = RecursiveTextChunkExpr(col, RecursiveTextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
            chunking_character_set_name=ChunkCharacterSet.CUSTOM,
            chunking_character_set_custom_characters=["a", "b", "c"],
        ))

        chunk5 = RecursiveTextChunkExpr(col, RecursiveTextChunkExprConfiguration(
            desired_chunk_size=100,
            chunk_overlap_percentage=10,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
            chunking_character_set_name=ChunkCharacterSet.CUSTOM,
            chunking_character_set_custom_characters=["a", "b", "c", "d"],
        ))
        assert chunk4 != chunk5

    def test_count_tokens_expr(self):
        """Test CountTokensExpr has no specific attributes."""
        col1 = ColumnExpr("text1")
        col2 = ColumnExpr("text2")

        count1 = CountTokensExpr(col1)
        count2 = CountTokensExpr(col2)

        # Different children, should not be equal
        assert count1 != count2

    def test_concat_expr(self):
        """Test ConcatExpr has no specific attributes."""
        col1 = ColumnExpr("text1")
        col2 = ColumnExpr("text2")

        concat1 = ConcatExpr([col1, col2])
        concat2 = ConcatExpr([col2, col1])  # Different order

        # Different children order, should not be equal
        assert concat1 != concat2

    def test_array_join_expr(self):
        """Test ArrayJoinExpr compares delimiter."""
        col = ColumnExpr("array_col")

        # Same delimiter
        join1 = ArrayJoinExpr(col, ",")
        join2 = ArrayJoinExpr(col, ",")
        assert join1 == join2

        # Different delimiter
        join3 = ArrayJoinExpr(col, ";")
        assert join1 != join3

    def test_contains_expr(self):
        """Test ContainsExpr has no specific attributes."""
        col = ColumnExpr("text")
        substr1 = LiteralExpr("hello", StringType)
        substr2 = LiteralExpr("world", StringType)

        contains1 = ContainsExpr(col, substr1)
        contains2 = ContainsExpr(col, substr2)

        # Different children, should not be equal
        assert contains1 != contains2

    def test_contains_any_expr(self):
        """Test ContainsAnyExpr compares substrs and case_insensitive."""
        col = ColumnExpr("text")

        # Same substrs and case_insensitive
        contains1 = ContainsAnyExpr(col, ["hello", "world"], True)
        contains2 = ContainsAnyExpr(col, ["hello", "world"], True)
        assert contains1 == contains2

        # Different substrs
        contains3 = ContainsAnyExpr(col, ["foo", "bar"], True)
        assert contains1 != contains3

        # Different case_insensitive
        contains4 = ContainsAnyExpr(col, ["hello", "world"], False)
        assert contains1 != contains4

    def test_rlike_expr(self):
        """Test RLikeExpr compares pattern."""
        col = ColumnExpr("text")

        # Same pattern
        rlike1 = RLikeExpr(col, LiteralExpr(r"\d+", StringType))
        rlike2 = RLikeExpr(col, LiteralExpr(r"\d+", StringType))
        assert rlike1 == rlike2

        # Different pattern
        rlike3 = RLikeExpr(col, LiteralExpr(r"[a-z]+", StringType))
        assert rlike1 != rlike3

    def test_like_expr(self):
        """Test LikeExpr compares raw_pattern."""
        col = ColumnExpr("text")

        # Same pattern
        like1 = LikeExpr(col, LiteralExpr("hello%", StringType))
        like2 = LikeExpr(col, LiteralExpr("hello%", StringType))
        assert like1 == like2

        # Different pattern
        like3 = LikeExpr(col, LiteralExpr("world%", StringType))
        assert like1 != like3

    def test_ilike_expr(self):
        """Test ILikeExpr compares raw_pattern."""
        col = ColumnExpr("text")

        # Same pattern
        ilike1 = ILikeExpr(col, LiteralExpr("HELLO%", StringType))
        ilike2 = ILikeExpr(col, LiteralExpr("HELLO%", StringType))
        assert ilike1 == ilike2

        # Different pattern
        ilike3 = ILikeExpr(col, LiteralExpr("WORLD%", StringType))
        assert ilike1 != ilike3

    def test_ts_parse_expr(self):
        """Test TsParseExpr compares format."""
        col = ColumnExpr("transcript")

        # Same format
        ts1 = TsParseExpr(col, "srt")
        ts2 = TsParseExpr(col, "srt")
        assert ts1 == ts2

        # Different format
        ts3 = TsParseExpr(col, "vtt")
        assert ts1 != ts3

    def test_starts_with_expr(self):
        """Test StartsWithExpr has no specific attributes."""
        col = ColumnExpr("text")
        substr1 = LiteralExpr("hello", StringType)
        substr2 = LiteralExpr("world", StringType)

        starts1 = StartsWithExpr(col, substr1)
        starts2 = StartsWithExpr(col, substr2)

        # Different children, should not be equal
        assert starts1 != starts2

    def test_ends_with_expr(self):
        """Test EndsWithExpr has no specific attributes."""
        col = ColumnExpr("text")
        substr1 = LiteralExpr("hello", StringType)
        substr2 = LiteralExpr("world", StringType)

        ends1 = EndsWithExpr(col, substr1)
        ends2 = EndsWithExpr(col, substr2)

        # Different children, should not be equal
        assert ends1 != ends2

    def test_regexp_split_expr(self):
        """Test RegexpSplitExpr compares pattern and limit."""
        col = ColumnExpr("text")

        # Same pattern and limit
        split1 = RegexpSplitExpr(col, r"\s+", 5)
        split2 = RegexpSplitExpr(col, r"\s+", 5)
        assert split1 == split2

        # Different pattern
        split3 = RegexpSplitExpr(col, r"\d+", 5)
        assert split1 != split3

        # Different limit
        split4 = RegexpSplitExpr(col, r"\s+", 10)
        assert split1 != split4

    def test_split_part_expr(self):
        """Test SplitPartExpr has no specific attributes."""
        col = ColumnExpr("text")
        delim = LiteralExpr(",", StringType)
        part1 = LiteralExpr(1, IntegerType)
        part2 = LiteralExpr(2, IntegerType)

        split1 = SplitPartExpr(col, delim, part1)
        split2 = SplitPartExpr(col, delim, part2)

        # Different children, should not be equal
        assert split1 != split2

    def test_string_casing_expr(self):
        """Test StringCasingExpr compares case."""
        col = ColumnExpr("text")

        # Same case
        upper1 = StringCasingExpr(col, "upper")
        upper2 = StringCasingExpr(col, "upper")
        assert upper1 == upper2

        # Different case
        lower1 = StringCasingExpr(col, "lower")
        assert upper1 != lower1

    def test_strip_chars_expr(self):
        """Test StripCharsExpr compares side."""
        col = ColumnExpr("text")
        chars = LiteralExpr(" ", StringType)

        # Same side
        strip1 = StripCharsExpr(col, chars, "both")
        strip2 = StripCharsExpr(col, chars, "both")
        assert strip1 == strip2

        # Different side
        strip3 = StripCharsExpr(col, chars, "left")
        assert strip1 != strip3

    def test_replace_expr(self):
        """Test ReplaceExpr compares literal."""
        col = ColumnExpr("text")
        search = LiteralExpr("old", StringType)
        replacement = LiteralExpr("new", StringType)

        # Same literal
        replace1 = ReplaceExpr(col, search, replacement, True)
        replace2 = ReplaceExpr(col, search, replacement, True)
        assert replace1 == replace2

        # Different literal
        replace3 = ReplaceExpr(col, search, replacement, False)
        assert replace1 != replace3

    def test_str_length_expr(self):
        """Test StrLengthExpr has no specific attributes."""
        col1 = ColumnExpr("text1")
        col2 = ColumnExpr("text2")

        len1 = StrLengthExpr(col1)
        len2 = StrLengthExpr(col2)

        # Different children, should not be equal
        assert len1 != len2

    def test_byte_length_expr(self):
        """Test ByteLengthExpr has no specific attributes."""
        col1 = ColumnExpr("text1")
        col2 = ColumnExpr("text2")

        len1 = ByteLengthExpr(col1)
        len2 = ByteLengthExpr(col2)

        # Different children, should not be equal
        assert len1 != len2

    def test_jinja_expr(self):
        """Test JinjaExpr compares template."""
        col1 = ColumnExpr("name")
        col2 = ColumnExpr("age")

        # Same template
        jinja1 = JinjaExpr([col1, col2], "Hello {{name}}, age {{age}}", True)
        jinja2 = JinjaExpr([col1, col2], "Hello {{name}}, age {{age}}", True)
        assert jinja1 == jinja2

        # Different template
        jinja3 = JinjaExpr([col1, col2], "Hi {{name}}, you are {{age}}", True)
        assert jinja1 != jinja3

        # Different strict
        jinja4 = JinjaExpr([col1, col2], "Hello {{name}}, age {{age}}", False)
        assert jinja1 != jinja4

    def test_fuzzy_ratio_expr(self):
        """Test FuzzyRatioExpr compares method."""
        col1 = ColumnExpr("text1")
        col2 = ColumnExpr("text2")

        # Same method
        fuzzy1 = FuzzyRatioExpr(col1, col2, "levenshtein")
        fuzzy2 = FuzzyRatioExpr(col1, col2, "levenshtein")
        assert fuzzy1 == fuzzy2

        # Different method
        fuzzy3 = FuzzyRatioExpr(col1, col2, "jaro_winkler")
        assert fuzzy1 != fuzzy3

    def test_fuzzy_token_sort_ratio_expr(self):
        """Test FuzzyTokenSortRatioExpr compares method."""
        col1 = ColumnExpr("text1")
        col2 = ColumnExpr("text2")

        # Same method
        fuzzy1 = FuzzyTokenSortRatioExpr(col1, col2, "levenshtein")
        fuzzy2 = FuzzyTokenSortRatioExpr(col1, col2, "levenshtein")
        assert fuzzy1 == fuzzy2

        # Different method
        fuzzy3 = FuzzyTokenSortRatioExpr(col1, col2, "jaro_winkler")
        assert fuzzy1 != fuzzy3

    def test_fuzzy_token_set_ratio_expr(self):
        """Test FuzzyTokenSetRatioExpr compares method."""
        col1 = ColumnExpr("text1")
        col2 = ColumnExpr("text2")

        # Same method
        fuzzy1 = FuzzyTokenSetRatioExpr(col1, col2, "levenshtein")
        fuzzy2 = FuzzyTokenSetRatioExpr(col1, col2, "levenshtein")
        assert fuzzy1 == fuzzy2

        # Different method
        fuzzy3 = FuzzyTokenSetRatioExpr(col1, col2, "jaro_winkler")
        assert fuzzy1 != fuzzy3


class TestSemanticExpressions:
    """Test equality for semantic expressions."""

    def test_semantic_map_expr(self):
        """Test SemanticMapExpr compares all attributes including examples."""
        from fenic.core.types import MapExample, MapExampleCollection

        col1 = ColumnExpr("text1")
        col1_alias = AliasExpr(ColumnExpr("blah"), "text1")

        # Create example collections
        examples1 = MapExampleCollection(examples=[
            MapExample(input={"text1": "Hello world"}, output="Processed: Hello world"),
        ])
        examples2 = MapExampleCollection(examples=[
            MapExample(input={"text1": "Different input"}, output="Processed: Different input"),
        ])

        # Same attributes
        map1 = SemanticMapExpr("Process {{text1}}", True, [col1], 100, 0.5)
        map2 = SemanticMapExpr("Process {{text1}}", True, [col1], 100, 0.5)
        assert map1 == map2

        # Different template
        map3 = SemanticMapExpr("Analyze {{text1}}", True, [col1], 100, 0.5)
        assert map1 != map3

        # Different max_tokens
        map4 = SemanticMapExpr("Process {{text1}}", True, [col1], 200, 0.5)
        assert map1 != map4

        # Different temperature
        map5 = SemanticMapExpr("Process {{text1}}", True, [col1], 100, 0.7)
        assert map1 != map5

        # Same examples
        map6 = SemanticMapExpr("Process {{text1}}", True, [col1], 100, 0.5, examples=examples1)
        map7 = SemanticMapExpr("Process {{text1}}", True, [col1], 100, 0.5, examples=examples1)
        assert map6 == map7

        # Different examples
        map8 = SemanticMapExpr("Process {{text1}}", True, [col1], 100, 0.5, examples=examples2)
        assert map6 != map8

        # One with examples, one without
        assert map1 != map6

        # Different expressions
        map9 = SemanticMapExpr("Process {{text1}}", True, [col1_alias], 100, 0.5, examples=examples1)
        assert map1 != map9

        # Different strict
        map10 = SemanticMapExpr("Process {{text1}}", False, [col1], 100, 0.5, examples=examples1)
        assert map1 != map10

    def test_semantic_extract_expr(self):
        """Test SemanticExtractExpr compares schema, max_tokens, temperature, and model_alias."""
        from pydantic import BaseModel

        class Schema1(BaseModel):
            name: str

        class Schema2(BaseModel):
            age: int

        col = ColumnExpr("text")

        # Same attributes
        extract1 = SemanticExtractExpr(col, Schema1, 150, 0.3)
        extract2 = SemanticExtractExpr(col, Schema1, 150, 0.3)
        assert extract1 == extract2

        # Different schema
        extract3 = SemanticExtractExpr(col, Schema2, 150, 0.3)
        assert extract1 != extract3

        # Different max_tokens
        extract4 = SemanticExtractExpr(col, Schema1, 200, 0.3)
        assert extract1 != extract4

        # Different temperature
        extract5 = SemanticExtractExpr(col, Schema1, 150, 0.5)
        assert extract1 != extract5

    def test_semantic_pred_expr(self):
        """Test SemanticPredExpr compares template, temperature, model_alias, and examples."""
        from fenic.core.types import PredicateExample, PredicateExampleCollection

        col1 = ColumnExpr("text1")
        col1_alias = AliasExpr(ColumnExpr("blah"), "text1")

        # Create example collections
        examples1 = PredicateExampleCollection(examples=[
            PredicateExample(input={"text1": "Great job!"}, output=True),
        ])
        examples2 = PredicateExampleCollection(examples=[
            PredicateExample(input={"text1": "Terrible work"}, output=False),
        ])

        # Same attributes
        pred1 = SemanticPredExpr("Is {{text1}} positive?", True, [col1], 0.3)
        pred2 = SemanticPredExpr("Is {{text1}} positive?", True, [col1], 0.3)
        assert pred1 == pred2

        # Different template
        pred3 = SemanticPredExpr("Is {{text1}} negative?", True, [col1], 0.3)
        assert pred1 != pred3

        # Different temperature
        pred4 = SemanticPredExpr("Is {{text1}} positive?", True, [col1], 0.5)
        assert pred1 != pred4

        # Same examples
        pred5 = SemanticPredExpr("Is {{text1}} positive?", True, [col1], 0.3, examples=examples1)
        pred6 = SemanticPredExpr("Is {{text1}} positive?", True, [col1], 0.3, examples=examples1)
        assert pred5 == pred6

        # Different examples
        pred7 = SemanticPredExpr("Is {{text1}} positive?", True, [col1], 0.3, examples=examples2)
        assert pred5 != pred7

        # One with examples, one without
        assert pred1 != pred5

        # Different expressions
        pred8 = SemanticPredExpr("Is {{text1}} positive?", True, [col1_alias], 0.3, examples=examples1)
        assert pred1 != pred8

        # Different strict
        pred9 = SemanticPredExpr("Is {{text1}} positive?", False, [col1], 0.3, examples=examples1)
        assert pred1 != pred9

    def test_semantic_reduce_expr(self):
        """Test SemanticReduceExpr compares all attributes including group_context and order_by."""
        col1 = ColumnExpr("content")
        col1_alias = AliasExpr(ColumnExpr("blah"), "content")
        col2 = ColumnExpr("category")
        col3 = ColumnExpr("priority")
        col4 = ColumnExpr("timestamp")

        # Same attributes (no group context or order by)
        reduce1 = SemanticReduceExpr("Summarize content", col1, 150, 0.4, [], [])
        reduce2 = SemanticReduceExpr("Summarize content", col1, 150, 0.4, [], [])
        assert reduce1 == reduce2

        # Different instruction
        reduce3 = SemanticReduceExpr("Extract key points", col1, 150, 0.4, [], [])
        assert reduce1 != reduce3

        # Different max_tokens
        reduce4 = SemanticReduceExpr("Summarize content", col1, 200, 0.4, [], [])
        assert reduce1 != reduce4

        # Different temperature
        reduce5 = SemanticReduceExpr("Summarize content", col1, 150, 0.6, [], [])
        assert reduce1 != reduce5

        # Same with group context
        reduce6 = SemanticReduceExpr("Summarize {{category}} content", col1, 150, 0.4, [col2], [])
        reduce7 = SemanticReduceExpr("Summarize {{category}} content", col1, 150, 0.4, [col2], [])
        assert reduce6 == reduce7

        # Different group context expressions
        reduce8 = SemanticReduceExpr("Summarize {{priority}} content", col1, 150, 0.4, [col3], [])
        assert reduce6 != reduce8

        # One with group context, one without
        assert reduce1 != reduce6

        # Same with order by
        reduce9 = SemanticReduceExpr("Summarize content", col1, 150, 0.4, [], [col4])
        reduce10 = SemanticReduceExpr("Summarize content", col1, 150, 0.4, [], [col4])
        assert reduce9 == reduce10

        # Different order by expressions
        reduce11 = SemanticReduceExpr("Summarize content", col1, 150, 0.4, [], [col2])
        assert reduce9 != reduce11

        # One with order by, one without
        assert reduce1 != reduce9

        # Different order by with SortExpr
        sort_asc = SortExpr(col4, ascending=True)
        sort_desc = SortExpr(col4, ascending=False)
        reduce12 = SemanticReduceExpr("Summarize content", col1, 150, 0.4, [], [sort_asc])
        reduce13 = SemanticReduceExpr("Summarize content", col1, 150, 0.4, [], [sort_desc])
        assert reduce12 != reduce13

        # Different expressions
        reduce14 = SemanticReduceExpr("Summarize content", col1_alias, 150, 0.4, [], [sort_asc])
        assert reduce1 != reduce14

    def test_semantic_classify_expr(self):
        """Test SemanticClassifyExpr compares temperature, model_alias, classes, and examples."""
        from fenic.core._logical_plan.resolved_types import ResolvedClassDefinition
        from fenic.core.types import ClassifyExample, ClassifyExampleCollection

        col = ColumnExpr("text")
        classes1 = [ResolvedClassDefinition(label="positive", description="Positive sentiment"), ResolvedClassDefinition(label="negative", description="Negative sentiment")]
        classes2 = [ResolvedClassDefinition(label="foo", description="foo"), ResolvedClassDefinition(label="bar", description="bar")]

        # Create example collections
        examples1 = ClassifyExampleCollection(examples=[
            ClassifyExample(input="Great product!", output="positive"),
        ])
        examples2 = ClassifyExampleCollection(examples=[
            ClassifyExample(input="Awful experience", output="negative"),
        ])

        # Same attributes
        classify1 = SemanticClassifyExpr(col, classes1, 0.2)
        classify2 = SemanticClassifyExpr(col, classes1, 0.2)
        assert classify1 == classify2

        # Different classes
        classify3 = SemanticClassifyExpr(col, classes2, 0.2)
        assert classify1 != classify3

        # Different temperature
        classify4 = SemanticClassifyExpr(col, classes1, 0.5)
        assert classify1 != classify4

        # Same examples
        classify5 = SemanticClassifyExpr(col, classes1, 0.2, examples=examples1)
        classify6 = SemanticClassifyExpr(col, classes1, 0.2, examples=examples1)
        assert classify5 == classify6

        # Different examples
        classify7 = SemanticClassifyExpr(col, classes1, 0.2, examples=examples2)
        assert classify5 != classify7

        # One with examples, one without
        assert classify1 != classify5

    def test_analyze_sentiment_expr(self):
        """Test AnalyzeSentimentExpr compares temperature and model_alias."""
        col = ColumnExpr("text")

        # Same attributes
        sentiment1 = AnalyzeSentimentExpr(col, 0.3)
        sentiment2 = AnalyzeSentimentExpr(col, 0.3)
        assert sentiment1 == sentiment2

        # Different temperature
        sentiment3 = AnalyzeSentimentExpr(col, 0.7)
        assert sentiment1 != sentiment3

    def test_embeddings_expr(self):
        """Test EmbeddingsExpr compares model_alias."""
        col = ColumnExpr("text")

        # Same model_alias (None)
        embed1 = EmbeddingsExpr(col)
        embed2 = EmbeddingsExpr(col)
        assert embed1 == embed2

        # Note: Can't easily test different model_alias without mock objects

    def test_semantic_summarize_expr(self):
        """Test SemanticSummarizeExpr compares temperature, model_alias, and format."""
        from fenic.core.types import KeyPoints, Paragraph

        col = ColumnExpr("text")

        # Same attributes
        summarize1 = SemanticSummarizeExpr(col, KeyPoints(), 0.4)
        summarize2 = SemanticSummarizeExpr(col, KeyPoints(), 0.4)
        assert summarize1 == summarize2

        # Different format
        summarize3 = SemanticSummarizeExpr(col, Paragraph(), 0.4)
        assert summarize1 != summarize3

        # Different temperature
        summarize4 = SemanticSummarizeExpr(col, KeyPoints(), 0.6)
        assert summarize1 != summarize4


class TestCrossTypeInequality:
    """Test that different expression types are never equal."""

    def test_basic_types_never_equal(self):
        """Test basic expression types are never equal to each other."""
        col = ColumnExpr("x")
        lit = LiteralExpr(10, IntegerType)
        alias = AliasExpr(col, "name")

        assert col != lit
        assert col != alias
        assert lit != alias

    def test_complex_types_never_equal(self):
        """Test complex expression types are never equal to each other."""
        col = ColumnExpr("x")

        async def async_func(x): return x

        # Different categories of expressions
        arith = ArithmeticExpr(col, col, Operator.PLUS)
        jq = JqExpr(col, ".foo")
        md = MdToJsonExpr(col)
        when = WhenExpr(None, EqualityComparisonExpr(col, LiteralExpr(1, IntegerType), Operator.EQ), col)
        sum_expr = SumExpr(col)
        embed = EmbeddingNormalizeExpr(col)
        async_udf = AsyncUDFExpr(async_func, [col], IntegerType)

        exprs = [arith, jq, md, when, sum_expr, embed, async_udf]
        for i, expr1 in enumerate(exprs):
            for j, expr2 in enumerate(exprs):
                if i != j:
                    assert expr1 != expr2
