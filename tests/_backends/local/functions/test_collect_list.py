import polars as pl
from _utils.serde_utils import _test_df_serialization

from fenic import col, collect_list


def test_collect_list_aggregation(sample_df, local_session):
    result = sample_df.group_by("city").agg(collect_list(col("age")))
    deserialized_df = _test_df_serialization(result, local_session._session_state)
    assert deserialized_df

    result = result.to_polars()
    assert len(result) == 2
    assert "collect_list(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert set(sf_row[1]) == {25, 30}
    assert set(seattle_row[1]) == {35}
