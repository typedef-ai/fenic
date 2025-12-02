import pytest

from fenic import col, semantic
from fenic.core.error import ValidationError
from fenic.core.types.semantic import ModelAlias

source_data = {"name": ["Alice"], "city": ["New York"]}
state_prompt = "What is the typical weather in {{city}} in summer?"

def test_invalid_temperature(local_session):
    source = local_session.create_dataframe(source_data)
    with pytest.raises(ValidationError, match='temperature must be between'):
        df_select = source.select(
            col("name"),
            # high temperature is invalid
            semantic.map(state_prompt, city=col("city"), temperature=4).alias("weather"),
        )
        df_select.to_polars()

def test_invalid_alias(local_session):
    source = local_session.create_dataframe(source_data)
    with pytest.raises(ValidationError, match="Language model alias 'not_in_configuration' not found in SessionConfig. Available models:"):
        df_select = source.select(
            col("name"),
            semantic.map(state_prompt, city=col("city"), model_alias=ModelAlias(name="not_in_configuration", profile="unknown")).alias("weather"),
        )
        df_select.to_polars()

def test_invalid_max_tokens(local_session):
    source = local_session.create_dataframe(source_data)
    state_prompt = "What is the typical weather in {{city}} in summer?"
    with pytest.raises(ValidationError, match="max_output_tokens must be a positive integer less than or equal to"):
        df_select = source.select(
            col("name"),
            # high max output tokens is invalid
            semantic.map(state_prompt, city=col("city"), max_output_tokens=250_000).alias("weather"),
        )
        df_select.to_polars()
