from textwrap import dedent

import polars as pl

from fenic.core.types import IntegerType


def test_polars_int_types_conversion(local_session):
    # create a polars dataframe with all int types
    # with a schema definition to enforce the types
    df = pl.DataFrame(
        {
            "int8": [1],
            "int16": [1],
            "int32": [1],
            "int64": [1],
            "int128": [1],
            "uint8": [1],
            "uint16": [1],
            "uint32": [1],
            "uint64": [1],
        },
        schema={
            "int8": pl.Int8,
            "int16": pl.Int16,
            "int32": pl.Int32,
            "int64": pl.Int64,
            "int128": pl.Int128,
            "uint8": pl.UInt8,
            "uint16": pl.UInt16,
            "uint32": pl.UInt32,
            "uint64": pl.UInt64,
        },
    )
    # create an Omni dataframe from the polars dataframe
    omni_df = local_session.create_dataframe(df)
    # check that the schema is correct
    expected_schema = {
        "int8": IntegerType,
        "int16": IntegerType,
        "int32": IntegerType,
        "int64": IntegerType,
        "int128": IntegerType,
        "uint8": IntegerType,
        "uint16": IntegerType,
        "uint32": IntegerType,
        "uint64": IntegerType,
    }
    # Convert StructType to dict for comparison
    actual_schema = {
        field.name: field.data_type for field in omni_df.schema.column_fields
    }
    assert actual_schema == expected_schema


def test_schema_str_representation(local_session):
    df = local_session.create_dataframe([
        {
            "a": 1,
            "b": True,
            "c": "hello",
            "d": [1, 2, 3],
            "e": {"x": 1, "y": 2},
            "f": [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}],
            "g": {
                "location": {"lat": 37.7749, "lng": -122.4194},
                "tags": ["city", "usa"]
            }
        }
    ])

    expected = dedent("""\
        Schema(
          ColumnField(name='a', data_type=IntegerType)
          ColumnField(name='b', data_type=BooleanType)
          ColumnField(name='c', data_type=StringType)
          ColumnField(name='d', data_type=ArrayType(
            element_type=IntegerType
          ))
          ColumnField(name='e', data_type=StructType(
            StructField(name='x', data_type=IntegerType)
            StructField(name='y', data_type=IntegerType)
          ))
          ColumnField(name='f', data_type=ArrayType(
            element_type=StructType(
              StructField(name='name', data_type=StringType)
              StructField(name='age', data_type=IntegerType)
            )
          ))
          ColumnField(name='g', data_type=StructType(
            StructField(name='location', data_type=StructType(
              StructField(name='lat', data_type=DoubleType)
              StructField(name='lng', data_type=DoubleType)
            ))
            StructField(name='tags', data_type=ArrayType(
              element_type=StringType
            ))
          ))
        )""")

    assert str(df.schema) == expected
