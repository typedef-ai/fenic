import polars as pl

from fenic import col, collect_list


def test_collect_list_aggregation(sample_df):
    result = sample_df.group_by("city").agg(collect_list(col("age"))).to_polars()
    assert len(result) == 2
    assert "collect_list(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert set(sf_row[1]) == {25, 30}
    assert set(seattle_row[1]) == {35}
