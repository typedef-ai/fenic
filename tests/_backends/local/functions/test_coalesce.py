import polars as pl

from fenic import coalesce, col, lit


def test_coalesce(local_session):
    df = local_session.create_dataframe(
        {"a": ["2", None, None], "b": [None, "2d", None], "c": ["a", "b", None]}
    )
    df = df.with_column("coalesced", coalesce(col("a"), col("b"), col("c"), lit("10")))
    result = df.to_polars()
    assert result.schema["coalesced"] == pl.String
    assert result["coalesced"][0] == "2"
    assert result["coalesced"][1] == "2d"
    assert result["coalesced"][2] == "10"
