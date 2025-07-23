import pytest

from fenic import col, text
from fenic.core.types.datatypes import DoubleType
from fenic.core.types.schema import ColumnField


@pytest.fixture
def fuzzy_similarity_test_df(local_session):
    data = {
        "text1": ["hello", "world", "hello", "world"],
        "text2": ["world", "hello", "world", "hello"],
    }
    return local_session.create_dataframe(data)


def test_fuzzy_similarity_levenshtein(fuzzy_similarity_test_df):
    df = fuzzy_similarity_test_df.select(
        text.fuzzy_similarity(col("text1"), col("text2"), method="levenshtein").alias("levenshtein_similarity")
    )
    assert df.schema.column_fields == [
        ColumnField(name="levenshtein_similarity", data_type=DoubleType),
    ]
    assert df.to_polars()["levenshtein_similarity"].to_list() == pytest.approx([0.2, 0.2, 0.2, 0.2], abs=1e-6)

    df = fuzzy_similarity_test_df.select(
        text.fuzzy_similarity(col("text1"), "world", method="levenshtein").alias("levenshtein_similarity")
    )
    assert df.schema.column_fields == [
        ColumnField(name="levenshtein_similarity", data_type=DoubleType),
    ]
    assert df.to_polars()["levenshtein_similarity"].to_list() == pytest.approx([0.2, 1.0, 0.2, 1.0], abs=1e-6)
