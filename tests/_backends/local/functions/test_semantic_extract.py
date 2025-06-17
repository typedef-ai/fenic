import polars as pl
import pytest
from pydantic import BaseModel, Field, ValidationError

from fenic import (
    BooleanType,
    ExtractSchema,
    ExtractSchemaField,
    ExtractSchemaList,
    IntegerType,
    StringType,
    col,
    semantic,
)
from fenic.core._utils.extract import (
    convert_extract_schema_to_pydantic_type,
)
from fenic.core._utils.schema import (
    convert_custom_dtype_to_polars,
    convert_pydantic_type_to_custom_struct_type,
)


def test_semantic_extract(extract_data_df):
    # Test with basic extract schema
    output_schema = ExtractSchema(
        [
            ExtractSchemaField(
                name="product_name",
                data_type=StringType,
                description="The name of the product mentioned in the review or support ticket",
            ),
            ExtractSchemaField(
                name="phone_version",
                data_type=IntegerType,
                description="specific product number",
            ),
            ExtractSchemaField(
                name="contains_negative_feedback",
                data_type=BooleanType,
                description="the review contains some negative feedback",
            ),
        ]
    )

    df = extract_data_df.select(
        semantic.extract(col("review"), output_schema).alias("review")
    )
    result = df.to_polars()
    pl_model = convert_custom_dtype_to_polars(
        convert_pydantic_type_to_custom_struct_type(
            convert_extract_schema_to_pydantic_type(output_schema)
        )
    )
    assert result.schema == pl.Schema({"review": pl_model})

    # Test with basic pydantic model
    class BasicReviewModel(BaseModel):
        positive_feature: str = Field(
            ..., description="Positive feature described in the review"
        )
        negative_feature: str = Field(
            ..., description="Negative feature described in the review"
        )
        product_number: int = Field(..., description="Specific product number")

    df = extract_data_df.select(
        semantic.extract(col("review"), BasicReviewModel).alias("review_out")
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "review_out": pl.Struct(
                {
                    "positive_feature": pl.String,
                    "negative_feature": pl.String,
                    "product_number": pl.Int64,
                }
            )
        }
    )

    # Test extract as derived column expressions
    df = extract_data_df.select(
        semantic.extract(col("review"), schema=output_schema).alias("review")
    )
    df = df.select(
        col("review").get_item("product_name"),
        col("review").get_item("phone_version"),
        col("review").get_item("contains_negative_feedback"),
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "review[product_name]": pl.String,
            "review[phone_version]": pl.Int64,
            "review[contains_negative_feedback]": pl.Boolean,
        }
    )

    df = extract_data_df.with_column(
        "contains_negative_feedback",
        semantic.extract(col("review"), schema=output_schema).get_item(
            "contains_negative_feedback"
        ),
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "review": pl.String,
            "contains_negative_feedback": pl.Boolean,
        }
    )


def test_semantic_extract_list(extract_data_df):
    # Test basic list

    list_output_schema = ExtractSchema(
        [
            ExtractSchemaField(
                name="issues_reported",
                data_type=ExtractSchemaList(element_type=StringType),
                description="All issues reported about the product",
            ),
            ExtractSchemaField(
                name="phone_version",
                data_type=IntegerType,
                description="specific product number",
            ),
        ]
    )
    df = extract_data_df.select(
        semantic.extract(col("review"), list_output_schema).alias("review_issues")
    )
    result = df.to_polars()
    pl_model = convert_custom_dtype_to_polars(
        convert_pydantic_type_to_custom_struct_type(
            convert_extract_schema_to_pydantic_type(list_output_schema)
        )
    )
    assert result.schema == pl.Schema({"review_issues": pl_model})

    # Test list of structs

    list_output_schema = ExtractSchema(
        [
            ExtractSchemaField(
                name="issues_reported",
                data_type=ExtractSchemaList(
                    element_type=ExtractSchema(
                        [
                            ExtractSchemaField(
                                name="sentiment_positive",
                                data_type=BooleanType,
                                description="this sentiment is positive",
                            ),
                            ExtractSchemaField(
                                name="sentiment_negative",
                                data_type=BooleanType,
                                description="this sentiment is negative",
                            ),
                            ExtractSchemaField(
                                name="sentiment_summary",
                                data_type=StringType,
                                description="a summary of the sentiment",
                            ),
                        ]
                    ),
                ),
                description="A list of sentiments about this product",
            ),
            ExtractSchemaField(
                name="phone_version",
                data_type=IntegerType,
                description="specific product number",
            ),
        ]
    )

    df = extract_data_df.select(
        semantic.extract(col("review"), list_output_schema).alias("review_issues")
    )
    result = df.to_polars()
    pl_model = convert_custom_dtype_to_polars(
        convert_pydantic_type_to_custom_struct_type(
            convert_extract_schema_to_pydantic_type(list_output_schema)
        )
    )
    assert result.schema == pl.Schema({"review_issues": pl_model})


def test_semantic_extract_bad_schema(extract_data_df):
    # extract schema with no description
    with pytest.raises(ValidationError):
        ExtractSchema(
            [
                ExtractSchemaField(
                    name="product_name",
                    data_type=StringType,
                ),
            ]
        )
    # extract schema with no type
    with pytest.raises(ValidationError):
        ExtractSchema(
            [
                ExtractSchemaField(
                    name="product_name",
                    description="The name of the product mentioned in the review or support ticket",
                ),
            ]
        )
    # pydantic model with no description
    with pytest.raises(ValueError):

        class BadModel(BaseModel):
            positive_feature: str = Field(
                ...,
            )

        extract_data_df.select(
            semantic.extract(col("review"), BadModel).alias("review_out")
        ).to_polars()


def test_semantic_extract_nested(extract_data_df):
    # test nested struct, and list inside struct
    nested_output_schema = ExtractSchema(
        [
            ExtractSchemaField(
                name="product_data",
                data_type=ExtractSchema(
                    [
                        ExtractSchemaField(
                            name="manufacturer_data",
                            data_type=ExtractSchema(
                                [
                                    ExtractSchemaField(
                                        name="phone_version",
                                        data_type=IntegerType,
                                        description="specific product number",
                                    ),
                                    ExtractSchemaField(
                                        name="manufacturer_name",
                                        data_type=StringType,
                                        description="The name of the manufacturer of the product",
                                    ),
                                ]
                            ),
                            description="Make and model information",
                        ),
                        ExtractSchemaField(
                            name="issues_reported",
                            data_type=ExtractSchemaList(element_type=StringType),
                            description="All issues reported about the product",
                        ),
                    ]
                ),
                description="Pulling data about the product",
            ),
            ExtractSchemaField(
                name="review_data",
                data_type=ExtractSchema(
                    [
                        ExtractSchemaField(
                            name="contains_negative_feedback",
                            data_type=BooleanType,
                            description="the review contains some negative feedback",
                        ),
                        ExtractSchemaField(
                            name="positive_feature",
                            data_type=StringType,
                            description="A positive feature described in the review",
                        ),
                    ]
                ),
                description="Pulling data about the review",
            ),
        ]
    )

    df = extract_data_df.select(
        semantic.extract(col("review"), nested_output_schema).alias("review")
    )
    result = df.to_polars()
    pl_model = convert_custom_dtype_to_polars(
        convert_pydantic_type_to_custom_struct_type(
            convert_extract_schema_to_pydantic_type(nested_output_schema)
        )
    )
    assert result.schema == pl.Schema({"review": pl_model})

    # Test field selection in nested struct
    df = df.select(
        col("review")
        .get_item("product_data")
        .get_item("manufacturer_data")
        .get_item("phone_version"),
        col("review").get_item("review_data").get_item("contains_negative_feedback"),
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "review[product_data][manufacturer_data][phone_version]": pl.Int64,
            "review[review_data][contains_negative_feedback]": pl.Boolean,
        }
    )


def test_semantic_extract_with_none(local_session):
    # test with none values

    df = local_session.create_dataframe(
        {
            "review": [
                "The iPhone 13 has a great camera but average battery life.",
                "I love my Samsung Galaxy S21! It performs well and the screen is amazing.",
                "The Google Pixel 6 heats up during gaming but has a solid build.",
                None,
            ],
        }
    )
    # Test with basic extract schema
    output_schema = ExtractSchema(
        [
            ExtractSchemaField(
                name="product_name",
                data_type=StringType,
                description="The name of the product mentioned in the review or support ticket",
            ),
            ExtractSchemaField(
                name="phone_version",
                data_type=IntegerType,
                description="specific product number",
            ),
            ExtractSchemaField(
                name="contains_negative_feedback",
                data_type=BooleanType,
                description="the review contains some negative feedback",
            ),
        ]
    )

    # Use the extended dataframe with None values.
    df = df.select(semantic.extract(col("review"), output_schema).alias("review"))
    result = df.to_polars()
    assert result["review"].to_list()[3] is None
