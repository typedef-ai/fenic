import polars as pl

from fenic import MapExample, MapExampleCollection, col, semantic


def test_semantic_map(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {name} live in given that they live in {city}?"
    df_select = source.select(
        semantic.map(state_prompt).alias("state"),
        col("name"),
        semantic.map(instruction="What is the typical weather in {city} in summer?").alias("weather"),
    )
    result = df_select.to_polars()
    assert result.schema == {
        "state": pl.String,
        "name": pl.String,
        "weather": pl.String,
    }

    weather_prompt = "What is the typical weather in {city} in summer?"
    df_with_column = source.with_column(
        "weather",
        semantic.map(instruction=weather_prompt),
    )
    result = df_with_column.to_polars()
    assert result.schema == {
        "name": pl.String,
        "city": pl.String,
        "weather": pl.String,
    }


def test_semantic_map_with_examples(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    weather_prompt = "What is the weather in {city}?"
    weather_collection = MapExampleCollection()
    weather_collection.create_example(
        MapExample(
            input={"city": "Seattle"},
            output="It is rainy and 60 degrees",
        )
    ).create_example(
        MapExample(
            input={"city": "Los Angeles"},
            output="It is sunny and 70 degrees",
        )
    )
    df_with_column = source.with_column(
        "weather",
        semantic.map(
            instruction=weather_prompt,
            examples=weather_collection,
        ),
    )
    result = df_with_column.to_polars()
    assert result.schema == {
        "name": pl.String,
        "city": pl.String,
        "weather": pl.String,
    }


def test_semantic_map_with_nulls(local_session):
    # have a data source with some nulls.
    source = local_session.create_dataframe(
        {"name": ["Alice", "Bob"], "city": ["New York", None]}
    )
    state_prompt = "What state does {name} live in given that they live in {city}?"
    df_select = source.select(
        col("name"),
        semantic.map(state_prompt).alias("state"),
    )
    result = df_select.to_polars()
    assert result.schema == {
        "name": pl.String,
        "state": pl.String,
    }
    result_list = result["state"].to_list()
    assert len(result_list) == 2
    # Make sure that Bob's state is None.
    assert result_list[1] is None
