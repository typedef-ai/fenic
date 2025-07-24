import pytest

from fenic import col, text
from fenic.core.types.datatypes import DoubleType
from fenic.core.types.schema import ColumnField


class TestFuzzySimilarity:
    """Test suite for fuzzy similarity functions using pytest parametrize."""

    @pytest.mark.parametrize("method,data,expected_col_to_col,expected_col_to_literal,literal_value", [
        # Indel tests
        (
            "indel",
            {
                "text1": ["lewenstein", None, "levenshtein", None],
                "text2": ["levenshtein", "levenshtein", None, None],
            },
            [85.71428571428571, None, None, None],
            [85.71428571428571, None, 100, None],
            "levenshtein"
        ),
        # Levenshtein tests
        (
            "levenshtein",
            {
                "text1": ["lewenstein", None, "levenshtein", None],
                "text2": ["levenshtein", "levenshtein", None, None],
            },
            [81.81818181818181, None, None, None],
            [81.81818181818181, None, 100, None],
            "levenshtein"
        ),
        # Damerau-Levenshtein tests
        (
            "damerau_levenshtein",
            {
                "text1": ["form", "abc", None, "from", None],
                "text2": ["from", "acb", "from", None, None],
            },
            [75.0, 66.66666666666666, None, None, None],
            [75.0, 0, None, 100, None],
            "from"
        ),
        # Jaro tests
        (
            "jaro",
            {
                "text1": ["martha", "opq", None, "marhta", None],
                "text2": ["marhta", "zyz", "marhta", None, None],
            },
            [94.44444444444444, 0, None, None, None],
            [94.44444444444444, 0, None, 100, None],
            "marhta"
        ),
        # Jaro-Winkler tests
        (
            "jaro_winkler",
            {
                "text1": ["martha", "dwayne", None, "marhta", None],
                "text2": ["marhta", "duane", "marhta", None, None],
            },
            [96.11111111111111, 84.0, None, None, None],
            [96.11111111111111, 44.44444444444444, None, 100, None],
            "marhta"
        ),
        # Hamming tests
        (
            "hamming",
            {
                "text1": ["hobo", "abc", None, "hobby", None],
                "text2": ["hobby", "def", "hobby", None, None],
            },
            [60, 0, None, None, None],
            [60, 0, None, 100, None],
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
        assert df_col.schema.column_fields == [
            ColumnField(name="similarity", data_type=DoubleType)
        ]
        result_col = df_col.to_polars()["similarity"].to_list()
        assert result_col == pytest.approx(expected_col_to_col, abs=1e-6)

        # Test column to literal comparison
        df_literal = source_df.select(
            text.compute_fuzzy_ratio(col("text1"), literal_value, method=method).alias("similarity")
        )
        result_literal = df_literal.to_polars()["similarity"].to_list()
        assert result_literal == pytest.approx(expected_col_to_literal, abs=1e-6)


    @pytest.mark.parametrize("data, expected_col_to_col, expected_col_to_literal, literal_value", [
        (
            {
                "text1": ["fuzzy  wuzzy was  a bear", "fuzzy was a bear  ", None, "fuzzy was a bear wuzzy", None],
                "text2": ["wuzzy  fuzzy was a  bear", "fuzzy fuzzy was a bear", "who cares", None, None],
            },
            [100, 84.21052631578947, None, None, None],
            [100, 84.21052631578947, None, 100, None],
            "wuzzy fuzzy was a bear"
        ),
    ])
    def test_compute_fuzzy_token_sort_ratio(self, local_session, data, expected_col_to_col, expected_col_to_literal, literal_value):
        df = local_session.create_dataframe(data)

        # Column-to-column
        df_col = df.select(
            text.compute_fuzzy_token_sort_ratio(col("text1"), col("text2")).alias("similarity")
        )
        assert df_col.schema.column_fields == [
            ColumnField(name="similarity", data_type=DoubleType)
        ]
        result_col = df_col.to_polars()["similarity"].to_list()
        assert result_col == pytest.approx(expected_col_to_col, abs=1e-6)

        # Column-to-literal
        df_literal = df.select(
            text.compute_fuzzy_token_sort_ratio(col("text1"), literal_value).alias("similarity")
        )
        result_literal = df_literal.to_polars()["similarity"].to_list()
        assert result_literal == pytest.approx(expected_col_to_literal, abs=1e-6)

    @pytest.mark.parametrize("data, expected_col_to_col, expected_col_to_literal, literal_value", [
        (
            {
                "text1": [
                    "fuzzy was a bear but not a dog",
                    "fuzzy was a bear but not a dog",
                    "apple banana",
                    "new york city",
                    None,
                    "fuzzy  was a bear",
                    None,
                ],
                "text2": [
                    "fuzzy fuzzy was a bear",
                    "fuzzy was a bear but not a cat",
                    "cherry date",
                    "paris london new york",
                    "other text",
                    None,
                    None,
                ],
            },
            [100, 92.3076923076923, 26.08695652173913, 76.19047619047619, None, None, None],
            [100, 100, 35.714285714285715, 27.586206896551724, None, 100, None],
            "fuzzy fuzzy was a bear"
        ),
    ])
    def test_compute_fuzzy_token_set_ratio(self, local_session, data, expected_col_to_col, expected_col_to_literal, literal_value):
        df = local_session.create_dataframe(data)

        # Column-to-column
        df_col = df.select(
            text.compute_fuzzy_token_set_ratio(col("text1"), col("text2")).alias("similarity")
        )
        assert df_col.schema.column_fields == [
            ColumnField(name="similarity", data_type=DoubleType)
        ]
        result_col = df_col.to_polars()["similarity"].to_list()
        assert result_col == pytest.approx(expected_col_to_col, abs=1e-6, nan_ok=True)

        # Column-to-literal
        df_literal = df.select(
            text.compute_fuzzy_token_set_ratio(col("text1"), literal_value).alias("similarity")
        )
        result_literal = df_literal.to_polars()["similarity"].to_list()
        assert result_literal == pytest.approx(expected_col_to_literal, abs=1e-6, nan_ok=True)
