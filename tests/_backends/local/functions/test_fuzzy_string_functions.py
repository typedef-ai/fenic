import pytest

from fenic import col, text


class TestFuzzySimilarity:
    """Test suite for fuzzy similarity functions using pytest parametrize."""

    @pytest.mark.parametrize("method,data,expected_col_to_col,expected_col_to_literal,literal_value", [
        # Levenshtein tests
        (
            "levenshtein",
            {
                "text1": ["kitten", None, "test", None],
                "text2": ["sitting", "test", None, None],
            },
            [57, None, None, None],
            [57, None, 14, None],
            "sitting"
        ),
        # Damerau-Levenshtein tests
        (
            "damerau_levenshtein",
            {
                "text1": ["form", "abc", None, "test", None],
                "text2": ["from", "acb", "test", None, None],
            },
            [75, 67, None, None, None],
            [75, 0, None, 0, None],
            "from"
        ),
        # Jaro tests
        (
            "jaro",
            {
                "text1": ["MARTHA", "abc", None, "test", None],
                "text2": ["MARHTA", "def", "test", None, None],
            },
            [94, 0, None, None, None],
            [82, 0, None, 0, None],
            "MATCH"
        ),
        # Jaro-Winkler tests
        (
            "jaro_winkler",
            {
                "text1": ["MARTHA", "DWAYNE", None, "test", None],
                "text2": ["MARHTA", "DUANE", "test", None, None],
            },
            [96, 84, None, None, None],
            [86, 46, None, 0, None],
            "MATCH"
        ),
        # Hamming tests
        (
            "hamming",
            {
                "text1": ["hobo", "abc", "saturday", None, "intention"],
                "text2": ["hobby", "def", None, "execution", None],
            },
            [60, 0, None, None, None],
            [60, 0, 0, None, 0],
            "hobby"
        ),
    ])
    def test_compute_fuzzy_similarity(self, local_session, method, data, expected_col_to_col,
                                     expected_col_to_literal, literal_value):
        """Test fuzzy similarity computation for various methods."""
        source_df = local_session.create_dataframe(data)

        # Test column to column comparison
        df_col = source_df.select(
            text.compute_fuzzy_ratio(col("text1"), col("text2"), method=method).alias("similarity")
        )
        result_col = df_col.to_polars()["similarity"].to_list()
        assert result_col == expected_col_to_col

        # Test column to literal comparison
        df_literal = source_df.select(
            text.compute_fuzzy_ratio(col("text1"), literal_value, method=method).alias("similarity")
        )
        result_literal = df_literal.to_polars()["similarity"].to_list()
        assert result_literal == expected_col_to_literal

    @pytest.mark.parametrize("column_value, other_value, expected", [
        ("new  york    city", "city new york", 100),
        ("apple orange banana", "banana   apple", 63), # 5 (apple), 6 (banana), 6 (orange), 2 (spaces) -  12 (apple, banana, one space)/19 =  63%
    ])
    def test_compute_fuzzy_token_sort_ratio(self, local_session, column_value, other_value, expected):
        df = local_session.create_dataframe({"text": [column_value]})

        result_df = df.select(
            text.compute_fuzzy_token_sort_ratio(col("text"), other_value, method="levenshtein").alias("similarity")
        )
        similarity_scores = result_df.to_polars()["similarity"].to_list()

        assert similarity_scores == [expected]
