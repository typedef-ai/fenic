from enum import Enum
from typing import List

import pytest
from pydantic import ConfigDict, ValidationError, validate_call

from fenic import (
    PredicateExample,
    PredicateExampleCollection,
)
from fenic.api.column import ColumnOrName
from fenic.api.functions import array, avg, col, semantic, when


def test_catch_various_as_column():
    with pytest.raises(ValidationError):
        avg(10)
    with pytest.raises(ValidationError):
        avg(True)
    with pytest.raises(ValidationError):
        avg(None)
    with pytest.raises(ValidationError):
        col(10)
    with pytest.raises(ValidationError):
        col(True)
    with pytest.raises(ValidationError):
        col(10.0)
    with pytest.raises(ValidationError):
        col(None)


def test_catch_list_of_columns_as_column():
    with pytest.raises(ValidationError):
        avg([col("age")])
    with pytest.raises(ValidationError):
        avg([col("age"), col("city")])
    with pytest.raises(ValidationError):
        col([col("age"), 10])
    with pytest.raises(ValidationError):
        when(col("age") > 10, [col("age")])


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def _arbitrary_call_with_list_of_columns(cols: List[ColumnOrName]):
    for c in cols:
        print(c)


def test_catch_various_as_list_of_columns():
    _arbitrary_call_with_list_of_columns([col("name")])
    with pytest.raises(ValidationError):
        _arbitrary_call_with_list_of_columns(col("name"))
    with pytest.raises(ValidationError):
        _arbitrary_call_with_list_of_columns(10)
    with pytest.raises(ValidationError):
        _arbitrary_call_with_list_of_columns([col("name"), 10])
    with pytest.raises(ValidationError):
        _arbitrary_call_with_list_of_columns([10, 20])


def test_catch_iterables_as_list_of_columns():
    with pytest.raises(ValidationError):
        _arbitrary_call_with_list_of_columns({"col": col("name")})
    with pytest.raises(ValidationError):
        _arbitrary_call_with_list_of_columns(set(["name", "city"]))
    with pytest.raises(ValidationError):
        _arbitrary_call_with_list_of_columns(Enum("Test", ["FOO", "BAR"]))


def test_catch_iterables_as_column():
    with pytest.raises(ValidationError):
        avg({"col": col("age")})
    with pytest.raises(ValidationError):
        avg(set(["age", "city"]))
    with pytest.raises(ValidationError):
        when(col("age") > 10, Enum("Test", ["FOO", "BAR"]))
    with pytest.raises(ValidationError):
        col(["age", "city"])
    with pytest.raises(ValidationError):
        col(set(["age", "city"]))


def test_catch_various_as_union_col_or_list():
    with pytest.raises(ValidationError):
        array(10)
    with pytest.raises(ValidationError):
        array({"col": col("name")})
    with pytest.raises(ValidationError):
        array(set(["name", 10]))
    with pytest.raises(ValidationError):
        array(Enum("Test", ["FOO", "BAR"]))
    with pytest.raises(ValidationError):
        array(Enum("Test", [("FOO", 2), ("BAR", 3)]))


def test_catch_various_as_examples(local_session):
    df = local_session.create_dataframe(
        {"name": ["Alice", "Bob"], "city": ["New York", "Los Angeles"]}
    )
    single_example = PredicateExample(
        input={"name": "Alice", "city": "New York"},
        output=True,
    )
    examples = PredicateExampleCollection().create_example(single_example)

    with pytest.raises(ValidationError):
        df.filter(semantic.predicate("Is this {name} from {city}?", single_example))
    with pytest.raises(ValidationError):
        df.filter(semantic.predicate("Is this {name} from {city}?", [examples]))
    with pytest.raises(ValidationError):
        df.filter(
            semantic.predicate(
                "Is this {name} from {city}?", {"name": "Alice", "city": "New York"}
            )
        )
    with pytest.raises(ValidationError):
        df.filter(semantic.predicate("Is this {name} from {city}?", col("name")))
    with pytest.raises(ValidationError):
        df.filter(
            semantic.predicate(
                "Is this {name} from {city}?", [col("name"), col("city")]
            )
        )
