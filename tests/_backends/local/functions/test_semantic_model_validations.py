import pytest

from fenic import col, semantic
from fenic.core.error import ValidationError
from fenic.core.types.semantic import ModelAlias


def test_invalid_temperature(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {name} live in given that they live in {city}?"
    with pytest.raises(ValidationError):
        df_select = source.select(
            semantic.map(state_prompt).alias("state"),
            col("name"),
            semantic.map(instruction="What is the typical weather in {city} in summer?", temperature=4).alias("weather"),
        )
        df_select.to_polars()

def test_invalid_alias(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {name} live in given that they live in {city}?"
    with pytest.raises(ValidationError):
        df_select = source.select(
            semantic.map(state_prompt).alias("state"),
            col("name"),
            semantic.map(instruction="What is the typical weather in {city} in summer?", model_alias=ModelAlias(name="not_in_configuration", profile="unknown")).alias("weather"),
        )
        df_select.to_polars()

def test_invalid_max_tokens(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {name} live in given that they live in {city}?"
    with pytest.raises(ValidationError):
        df_select = source.select(
            semantic.map(state_prompt).alias("state"),
            col("name"),
            semantic.map(instruction="What is the typical weather in {city} in summer?", max_output_tokens=250_000).alias("weather"),
        )
        df_select.to_polars()
