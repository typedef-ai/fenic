import polars as pl

from fenic import ColumnField, Schema, StringType, col
from fenic.api.functions import semantic


def test_semantic_map_multiple_models(multi_model_local_session):
    source = multi_model_local_session.create_dataframe(
        {"name": ["Alice", "Bob", "Charlie"], "state": ["New York", "California", "Michigan"], })
    state_prompt = "What is the capital of {state}? Only return the name of the capital city."
    population_prompt = "What is the population of {state}? Only return the number of people."
    df_select = source.select(
        col("name"),
        semantic.map(instruction=state_prompt, model_alias="model_1", max_output_tokens=64).alias("state_capital"),
        semantic.map(instruction=population_prompt, model_alias="model_2", max_output_tokens=64).alias(
            "state_population"),
    )

    pre_collect_schema = df_select.schema
    assert pre_collect_schema == Schema(
        column_fields=[
            ColumnField(name='name', data_type=StringType),
            ColumnField(name='state_capital', data_type=StringType),
            ColumnField(name='state_population', data_type=StringType)])
    result = df_select.to_polars()
    assert result.schema == {
        "name": pl.String,
        "state_capital": pl.String,
        "state_population": pl.String,
    }
