from datetime import datetime

import polars as pl
import pytest

from fenic.core.types.semantic_examples import (
    JoinExample,
    JoinExampleCollection,
    MapExample,
    MapExampleCollection,
    PredicateExample,
    PredicateExampleCollection,
)


@pytest.mark.parametrize("kind", ["map", "predicate", "join"])
def test_example_exports_preserve_float_precision_and_naive_datetimes(kind):
    observed = datetime(2026, 10, 5, 12, 30)
    if kind == "join":
        collection = JoinExampleCollection(
            [JoinExample(left_on=0.1, right_on=observed, output=True)]
        )
        float_column, datetime_column = "left_on", "right_on"
    else:
        values = {"amount": 0.1, "observed": observed}
        collection = (
            MapExampleCollection([MapExample(input=values, output="example")])
            if kind == "map"
            else PredicateExampleCollection(
                [PredicateExample(input=values, output=True)]
            )
        )
        float_column, datetime_column = "amount", "observed"

    result = collection.to_polars()

    assert result[float_column].dtype == pl.Float64
    assert result[datetime_column].dtype == pl.Datetime("us")
    assert result[float_column].to_list() == [0.1]
    assert result[datetime_column].to_list() == [observed]
    assert (
        result[float_column].to_list() == collection.to_pandas()[float_column].to_list()
    )
