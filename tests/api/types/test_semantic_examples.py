import pandas as pd
import polars as pl
import pytest
from pydantic import BaseModel, Field

from fenic import FloatType, IntegerType, StringType
from fenic.core._utils.type_inference import TypeInferenceError
from fenic.core.types.semantic_examples import (
    ClassifyExample,
    ClassifyExampleCollection,
    InvalidExampleCollectionError,
    JoinExample,
    JoinExampleCollection,
    MapExample,
    MapExampleCollection,
    PredicateExample,
    PredicateExampleCollection,
)


class ProductSummary(BaseModel):
    """Test BaseModel for structured output."""
    name: str = Field(description="Product name")
    description: str = Field(description="One-line description")
    category: str = Field(description="Product category")


class PersonInfo(BaseModel):
    """Another test BaseModel for structured output."""
    first_name: str = Field(description="First name")
    last_name: str = Field(description="Last name")
    age: int = Field(description="Age in years")


class TestMapExampleCollection:
    """Test cases for MapExampleCollection."""

    def test_create_example_basic(self):
        """Test adding examples to the collection."""
        collection = MapExampleCollection()
        assert len(collection.examples) == 0

        # First example establishes the schema
        example = MapExample(input={"text": "Hello", "count": 5}, output="greeting")
        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].input == {"text": "Hello", "count": 5}
        assert collection.examples[0].output == "greeting"

        # Wrong example type should raise error
        with pytest.raises(InvalidExampleCollectionError):
            collection.create_example(ClassifyExample(input="Hello", output="greeting"))

    def test_first_example_cannot_have_none(self):
        """Test that first example cannot have None values."""
        collection = MapExampleCollection()

        example = MapExample(input={"text": "Hello", "count": None}, output="greeting")
        with pytest.raises(InvalidExampleCollectionError, match="Example #1: None values are not allowed."):
            collection.create_example(example)

    def test_subsequent_examples_cannot_have_none(self):
        """Test that subsequent examples can have None values."""
        collection = MapExampleCollection()

        # First example establishes schema
        example1 = MapExample(input={"text": "Hello", "count": 5}, output="greeting")
        collection.create_example(example1)

        # Second example cannot have None
        example2 = MapExample(input={"text": "Hi", "count": None}, output="greeting")
        with pytest.raises(InvalidExampleCollectionError, match="Example #2: None values are not allowed."):
            collection.create_example(example2)

    def test_type_consistency_enforcement(self):
        """Test that field types must be consistent across examples."""
        collection = MapExampleCollection()

        # First example with integer count
        example1 = MapExample(input={"text": "Hello", "count": 5}, output="greeting")
        collection.create_example(example1)

        # Second example with string count should fail
        example2 = MapExample(input={"text": "Hi", "count": "five"}, output="greeting")
        with pytest.raises(InvalidExampleCollectionError, match="Field 'count' type mismatch"):
            collection.create_example(example2)

    def test_field_consistency_enforcement(self):
        """Test that all examples must have the same fields."""
        collection = MapExampleCollection()

        # First example
        example1 = MapExample(input={"text": "Hello", "lang": "en"}, output="greeting")
        collection.create_example(example1)

        # Second example with different fields should fail
        example2 = MapExample(input={"text": "Hi", "language": "en"}, output="greeting")
        with pytest.raises(InvalidExampleCollectionError, match="inconsistent fields"):
            collection.create_example(example2)

        # Missing field should fail
        example3 = MapExample(input={"text": "Hola"}, output="greeting")
        with pytest.raises(InvalidExampleCollectionError, match="inconsistent fields"):
            collection.create_example(example3)


    def test_validate_against_column_types(self):
        """Test validation against actual column types."""
        collection = MapExampleCollection()

        # Add example
        example = MapExample(
            input={"name": "Alice", "age": 30},
            output="Person"
        )
        collection.create_example(example)

        # Valid column types
        column_types = {
            "name": StringType,
            "age": IntegerType
        }
        collection._validate_against_column_types(column_types)  # Should not raise

        # Type mismatch
        wrong_types = {
            "name": StringType,
            "age": StringType  # Wrong type
        }
        with pytest.raises(InvalidExampleCollectionError, match="type mismatch"):
            collection._validate_against_column_types(wrong_types)

        # Missing field in examples
        extra_columns = {
            "name": StringType,
            "age": IntegerType,
            "email": StringType  # Not in examples
        }
        with pytest.raises(InvalidExampleCollectionError, match="missing from examples"):
            collection._validate_against_column_types(extra_columns)

    def test_basemodel_output_consistency(self):
        """Test BaseModel output type consistency."""
        collection = MapExampleCollection()

        # First example with BaseModel
        product1 = ProductSummary(
            name="Lamp",
            description="A nice lamp",
            category="lighting"
        )
        example1 = MapExample(
            input={"name": "Lamp"},
            output=product1
        )
        collection.create_example(example1)

        # Second example with same BaseModel type - should work
        product2 = ProductSummary(
            name="Chair",
            description="A comfy chair",
            category="furniture"
        )
        example2 = MapExample(
            input={"name": "Chair"},
            output=product2
        )
        collection.create_example(example2)

        # String output should fail
        example3 = MapExample(
            input={"name": "Table"},
            output="A table"
        )
        with pytest.raises(InvalidExampleCollectionError, match="consistent output types"):
            collection.create_example(example3)

        person = PersonInfo(
            first_name="John",
            last_name="Doe",
            age=30
        )
        example4 = MapExample(
            input={"name": "John", "count": 1},
            output=person
        )
        with pytest.raises(InvalidExampleCollectionError, match="All BaseModel examples must be of the same type."):
            collection.create_example(example4)

    def test_from_polars(self):
        """Test creating collection from a Polars DataFrame."""
        # Valid DataFrame
        df = pl.DataFrame(
            {
                "text": ["Hello", "Bonjour", "Hola"],
                "count": [1, 2, 3],
                "output": [
                    "greeting in English",
                    "greeting in French",
                    "greeting in Spanish",
                ],
            }
        )

        collection = MapExampleCollection.from_polars(df)
        assert len(collection.examples) == 3
        assert collection.examples[0].input["text"] == "Hello"
        assert collection.examples[0].input["count"] == 1
        assert collection.examples[0].output == "greeting in English"

        # Missing output column
        df_missing_output = pl.DataFrame(
            {"text": ["Hello", "Goodbye"], "count": [1, 2]}
        )

        with pytest.raises(ValueError, match="missing required 'output' column"):
            MapExampleCollection.from_polars(df_missing_output)

        # No input columns
        df_no_inputs = pl.DataFrame({"output": ["result1", "result2"]})

        with pytest.raises(ValueError, match="must have at least one input column"):
            MapExampleCollection.from_polars(df_no_inputs)


    def test_from_pandas(self):
        """Test creating collection from a Pandas DataFrame."""
        # Valid DataFrame
        df = pd.DataFrame(
            {
                "text": ["Hello", "Bonjour", "Hola"],
                "count": [1, 2, 3],
                "output": [
                    "greeting in English",
                    "greeting in French",
                    "greeting in Spanish",
                ],
            }
        )

        collection = MapExampleCollection.from_pandas(df)
        assert len(collection.examples) == 3
        assert collection.examples[0].input["text"] == "Hello"
        assert collection.examples[0].input["count"] == 1
        assert collection.examples[0].output == "greeting in English"

        # Missing output column
        df_missing_output = pd.DataFrame(
            {"text": ["Hello", "Goodbye"], "count": [1, 2]}
        )

        with pytest.raises(ValueError, match="missing required 'output' column"):
            MapExampleCollection.from_pandas(df_missing_output)

        # No input columns
        df_no_inputs = pd.DataFrame({"output": ["result1", "result2"]})

        with pytest.raises(ValueError, match="must have at least one input column"):
            MapExampleCollection.from_pandas(df_no_inputs)

    def test_to_polars(self):
        """Test converting collection to a Polars DataFrame."""
        # Empty collection
        empty_collection = MapExampleCollection()
        empty_df = empty_collection.to_polars()
        assert len(empty_df) == 0

        # Collection with examples
        collection = MapExampleCollection()
        collection.create_example(
            MapExample(input={"text": "Hello", "count": 1}, output="greeting")
        )
        collection.create_example(
            MapExample(input={"text": "Goodbye", "count": 2}, output="farewell")
        )

        df = collection.to_polars()
        assert len(df) == 2
        assert df.columns == ["text", "count", "output"]
        assert df["text"].to_list() == ["Hello", "Goodbye"]
        assert df["count"].to_list() == [1, 2]
        assert df["output"].to_list() == ["greeting", "farewell"]


    def test_to_pandas(self):
        """Test converting collection to a Pandas DataFrame."""
        collection = MapExampleCollection()
        collection.create_example(
            MapExample(input={"text": "Hello", "count": 1}, output="greeting")
        )
        collection.create_example(
            MapExample(input={"text": "Goodbye", "count": 2}, output="farewell")
        )

        df = collection.to_pandas()
        assert len(df) == 2
        assert list(df.columns) == ["text", "count", "output"]
        assert df["text"].tolist() == ["Hello", "Goodbye"]
        assert df["count"].tolist() == [1, 2]
        assert df["output"].tolist() == ["greeting", "farewell"]

    def test_create_example_with_basemodel_output(self):
        """Test creating MapExample with BaseModel output."""
        collection = MapExampleCollection()

        # Create BaseModel instance
        product = ProductSummary(
            name="GlowMate",
            description="Modern touch-controlled lamp for better sleep",
            category="lighting"
        )

        # Create example with BaseModel output
        example = MapExample(
            input={"name": "GlowMate", "details": "A rechargeable bedside lamp"},
            output=product
        )

        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].input["name"] == "GlowMate"
        assert isinstance(collection.examples[0].output, ProductSummary)
        assert collection.examples[0].output.name == "GlowMate"
        assert collection.examples[0].output.category == "lighting"

    def test_basemodel_output_dataframe_serialization(self):
        """Test that BaseModel outputs are properly serialized in DataFrame conversion."""
        collection = MapExampleCollection()

        # Add BaseModel output example
        product = ProductSummary(
            name="SmartLamp",
            description="WiFi-enabled smart lighting solution",
            category="smart_home"
        )
        example = MapExample(
            input={"name": "SmartLamp", "details": "WiFi-enabled lamp"},
            output=product
        )
        collection.create_example(example)

        # Convert to DataFrame
        df = collection.to_polars()

        # Check that output is serialized as JSON string
        output_value = df['output'].to_list()[0]
        assert isinstance(output_value, str)
        assert '"name":"SmartLamp"' in output_value
        assert '"category":"smart_home"' in output_value

        # Test pandas conversion too
        df_pandas = collection.to_pandas()
        output_value_pandas = df_pandas['output'].tolist()[0]
        assert isinstance(output_value_pandas, str)
        assert '"name":"SmartLamp"' in output_value_pandas

    def test_basemodel_only_dataframe_conversion(self):
        """Test DataFrame conversion with only BaseModel outputs."""
        collection = MapExampleCollection()

        # Add first BaseModel output
        product1 = ProductSummary(
            name="SmartLamp",
            description="WiFi-enabled smart lighting solution",
            category="smart_home"
        )
        collection.create_example(MapExample(
            input={"name": "SmartLamp", "details": "WiFi-enabled lamp"},
            output=product1
        ))

        # Add second BaseModel output
        product2 = ProductSummary(
            name="DeskLamp",
            description="Adjustable desk lamp for work",
            category="office"
        )
        collection.create_example(MapExample(
            input={"name": "DeskLamp", "details": "Adjustable desk lamp"},
            output=product2
        ))

        # Convert to DataFrame
        df = collection.to_polars()

        output_values = df['output'].to_list()
        assert len(output_values) == 2

        # Both outputs should be serialized JSON
        assert isinstance(output_values[0], str)
        assert '"name":"SmartLamp"' in output_values[0]
        assert '"category":"smart_home"' in output_values[0]

        assert isinstance(output_values[1], str)
        assert '"name":"DeskLamp"' in output_values[1]
        assert '"category":"office"' in output_values[1]


class TestClassifyExampleCollection:
    """Test cases for ClassifyExampleCollection."""

    def test_create_example(self):
        """Test adding examples to the collection."""
        collection = ClassifyExampleCollection()
        assert len(collection.examples) == 0

        example = ClassifyExample(input="This is positive", output="positive")
        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].input == "This is positive"
        assert collection.examples[0].output == "positive"

        # Wrong example type should raise error
        with pytest.raises(InvalidExampleCollectionError):
            collection.create_example(
                MapExample(input={"text": "Hello"}, output="greeting")
            )

    def test_from_polars(self):
        """Test creating collection from a Polars DataFrame."""
        # Valid DataFrame
        df = pl.DataFrame(
            {
                "input": ["Great product", "Terrible experience", "Okay service"],
                "output": ["positive", "negative", "neutral"],
            }
        )

        collection = ClassifyExampleCollection.from_polars(df)
        assert len(collection.examples) == 3
        assert collection.examples[0].input == "Great product"
        assert collection.examples[0].output == "positive"

        # Missing input column
        df_missing_input = pl.DataFrame(
            {
                "text": ["This is good", "This is bad"],
                "output": ["positive", "negative"],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'input' column"
        ):
            ClassifyExampleCollection.from_polars(df_missing_input)

        # Missing output column
        df_missing_output = pl.DataFrame({"input": ["This is good", "This is bad"]})

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'output' column"
        ):
            ClassifyExampleCollection.from_polars(df_missing_output)

        # Null values in input
        df_null_input = pl.DataFrame(
            {"input": ["This is good", None], "output": ["positive", "negative"]}
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'input' column"
        ):
            ClassifyExampleCollection.from_polars(df_null_input)

        # Null values in output
        df_null_output = pl.DataFrame(
            {"input": ["This is good", "This is bad"], "output": ["positive", None]}
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'output' column"
        ):
            ClassifyExampleCollection.from_polars(df_null_output)

    def test_to_polars(self):
        """Test converting collection to a Polars DataFrame."""
        collection = ClassifyExampleCollection()
        collection.create_example(
            ClassifyExample(input="This is positive", output="positive")
        )
        collection.create_example(
            ClassifyExample(input="This is negative", output="negative")
        )

        df = collection.to_polars()
        assert len(df) == 2
        assert df.columns == ["input", "output"]
        assert df["input"].to_list() == ["This is positive", "This is negative"]
        assert df["output"].to_list() == ["positive", "negative"]

    def test_to_pandas(self):
        """Test converting collection to a Pandas DataFrame."""
        collection = ClassifyExampleCollection()
        collection.create_example(
            ClassifyExample(input="This is positive", output="positive")
        )
        collection.create_example(
            ClassifyExample(input="This is negative", output="negative")
        )

        df = collection.to_pandas()
        assert len(df) == 2
        assert list(df.columns) == ["input", "output"]
        assert df["input"].tolist() == ["This is positive", "This is negative"]
        assert df["output"].tolist() == ["positive", "negative"]


class TestPredicateExampleCollection:
    """Test cases for PredicateExampleCollection."""

    def test_create_example_basic(self):
        """Test adding examples to the collection."""
        collection = PredicateExampleCollection()
        assert len(collection.examples) == 0

        # First example establishes the schema
        example = PredicateExample(input={"text": "Hello", "count": 5}, output=True)
        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].input == {"text": "Hello", "count": 5}
        assert collection.examples[0].output

        # Wrong example type should raise error
        with pytest.raises(InvalidExampleCollectionError):
            collection.create_example(MapExample(input={"text": "Hello", "count": 5}, output="greeting"))

    def test_from_polars(self):
        """Test creating collection from a Polars DataFrame."""
        # Valid DataFrame
        df = pl.DataFrame(
            {
                "age": ["19", "17", "21"],
                "country": ["US", "UK", "CA"],
                "output": [True, False, True],
            }
        )

        collection = PredicateExampleCollection.from_polars(df)
        assert len(collection.examples) == 3
        assert collection.examples[0].input["age"] == "19"
        assert collection.examples[0].input["country"] == "US"
        assert collection.examples[0].output is True

        # Missing output column
        df_missing_output = pl.DataFrame({"age": ["19", "17"], "country": ["US", "UK"]})

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'output' column"
        ):
            PredicateExampleCollection.from_polars(df_missing_output)

        # No input columns
        df_no_inputs = pl.DataFrame({"output": [True, False]})

        with pytest.raises(
            InvalidExampleCollectionError, match="must have at least one input column"
        ):
            PredicateExampleCollection.from_polars(df_no_inputs)

        # Null values in output
        df_null_output = pl.DataFrame({"age": ["19", "17"], "output": [True, None]})

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'output' column"
        ):
            PredicateExampleCollection.from_polars(df_null_output)

    def test_from_pandas(self):
        """Test creating collection from a Pandas DataFrame."""
        # Valid DataFrame
        df = pd.DataFrame(
            {
                "age": ["19", "17", "21"],
                "country": ["US", "UK", "CA"],
                "output": [True, False, True],
            }
        )

        collection = PredicateExampleCollection.from_pandas(df)
        assert len(collection.examples) == 3
        assert collection.examples[0].input["age"] == "19"
        assert collection.examples[0].input["country"] == "US"
        assert collection.examples[0].output is True

        # Missing output column
        df_missing_output = pd.DataFrame({"age": ["19", "17"], "country": ["US", "UK"]})

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'output' column"
        ):
            PredicateExampleCollection.from_pandas(df_missing_output)

        # No input columns
        df_no_inputs = pd.DataFrame({"output": [True, False]})

        with pytest.raises(
            InvalidExampleCollectionError, match="must have at least one input column"
        ):
            PredicateExampleCollection.from_pandas(df_no_inputs)

        # Null values in output
        df_null_output = pd.DataFrame({"age": ["19", "17"], "output": [True, None]})

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'output' column"
        ):
            PredicateExampleCollection.from_pandas(df_null_output)

    def test_to_polars(self):
        """Test converting collection to a Polars DataFrame."""
        collection = PredicateExampleCollection()
        collection.create_example(PredicateExample(input={"age": "19"}, output=True))
        collection.create_example(PredicateExample(input={"age": "17"}, output=False))

        df = collection.to_polars()
        assert len(df) == 2
        assert df.columns == ["age", "output"]
        # This assertion would need to be adjusted if the implementation is fixed to expand input dict
        assert df["age"].to_list() == ["19", "17"]
        assert df["output"].to_list() == [True, False]

    def test_validate_against_column_types(self):
        """Test validation against column types."""
        collection = PredicateExampleCollection()

        example = PredicateExample(
            input={"age": 19, "country": "US"},
            output=True
        )
        collection.create_example(example)

        # Valid types
        column_types = {
            "age": IntegerType,
            "country": StringType
        }
        collection._validate_against_column_types(column_types)

        # Wrong type
        wrong_types = {
            "age": FloatType,  # Wrong
            "country": StringType
        }
        with pytest.raises(InvalidExampleCollectionError, match="type mismatch"):
            collection._validate_against_column_types(wrong_types)


class TestJoinExampleCollection:
    """Test cases for JoinExampleCollection."""

    def test_create_example(self):
        """Test adding examples to the collection."""
        collection = JoinExampleCollection()
        assert len(collection.examples) == 0

        example = JoinExample(left_on="apple", right_on=[1, 2, 3], output=True)
        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].left_on == "apple"
        assert collection.examples[0].right_on == [1, 2, 3]
        assert collection.examples[0].output is True

        # Wrong example type should raise error
        with pytest.raises(InvalidExampleCollectionError):
            collection.create_example(
                MapExample(input={"text": "Hello"}, output="greeting")
            )

    def test_from_polars(self):
        """Test creating collection from a Polars DataFrame."""
        # Valid DataFrame
        df = pl.DataFrame(
            {
                "left_on": ["apple", "apple", "carrot"],
                "right_on": [True, False, True],
                "output": [True, False, True],
                "another_column": [1, 2, 3],
            }
        )

        collection = JoinExampleCollection.from_polars(df)
        assert len(collection.examples) == 3
        assert collection.examples[0].left_on == "apple"
        assert collection.examples[0].right_on
        assert collection.examples[0].output is True

        # Missing left column
        df_missing_left = pl.DataFrame(
            {
                "right_on": [True, False],
                "output": [True, True],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="Join Examples DataFrame missing required 'left_on' column"
        ):
            JoinExampleCollection.from_polars(df_missing_left)

        # Missing right column
        df_missing_right = pl.DataFrame(
            {
                "left_on": ["apple", "carrot"],
                "output": [True, True],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="Join Examples DataFrame missing required 'right_on' column"
        ):
            JoinExampleCollection.from_polars(df_missing_right)

        # Missing output column
        df_missing_output = pl.DataFrame(
            {"left_on": ["apple", "carrot"], "right_on": [True, False]}
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'output' column"
        ):
            JoinExampleCollection.from_polars(df_missing_output)


    def test_to_polars(self):
        """Test converting collection to a Polars DataFrame."""
        collection = JoinExampleCollection()
        collection.create_example(JoinExample(left_on="apple", right_on=True, output=True))
        collection.create_example(
            JoinExample(left_on="apple", right_on=False, output=False)
        )

        df = collection.to_polars()
        assert len(df) == 2
        assert df.columns == ["left_on", "right_on", "output"]
        assert df["left_on"].to_list() == ["apple", "apple"]
        assert df["right_on"].to_list() == [True, False]
        assert df["output"].to_list() == [True, False]

    def test_to_pandas(self):
        """Test converting collection to a Pandas DataFrame."""
        collection = JoinExampleCollection()
        collection.create_example(JoinExample(left_on="apple", right_on=True, output=True))
        collection.create_example(
            JoinExample(left_on="apple", right_on=False, output=False)
        )

        df = collection.to_pandas()
        assert len(df) == 2
        assert list(df.columns) == ["left_on", "right_on", "output"]
        assert df["left_on"].tolist() == ["apple", "apple"]
        assert df["right_on"].tolist() == [True, False]
        assert df["output"].tolist() == [True, False]

    def test_validate_against_join_types(self):
        """Test validation against join column types."""
        collection = JoinExampleCollection()

        example = JoinExample(left_on=123, right_on="CUST-123", output=True)
        collection.create_example(example)

        # Valid types
        collection._validate_against_join_types(IntegerType, StringType)

        # Wrong left type
        with pytest.raises(InvalidExampleCollectionError, match="type mismatch"):
            collection._validate_against_join_types(StringType, StringType)

        # Wrong right type
        with pytest.raises(InvalidExampleCollectionError, match="type mismatch"):
            collection._validate_against_join_types(IntegerType, IntegerType)

    def test_complex_type_validation(self):
        """Test validation with complex types."""
        collection = JoinExampleCollection()

        # Join on complex nested structure
        example1 = JoinExample(
            left_on={"id": 1, "name": "Alice"},
            right_on={"user_id": 1, "username": "alice123"},
            output=True
        )
        collection.create_example(example1)

        # Consistent structure
        example2 = JoinExample(
            left_on={"id": 2, "name": "Bob"},
            right_on={"user_id": 2, "username": "bob456"},
            output=True
        )
        collection.create_example(example2)

        assert len(collection.examples) == 2

        # Inconsistent structure should fail
        example3 = JoinExample(
            left_on={"id": 3, "full_name": "Charlie"},  # Different key
            right_on={"user_id": 3, "username": "charlie789"},
            output=False
        )
        with pytest.raises(InvalidExampleCollectionError, match="Example #3: Field 'left_on' type mismatch."):
            collection.create_example(example3)

class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_collections(self):
        """Test empty collection behavior."""
        collections = [
            MapExampleCollection(),
            ClassifyExampleCollection(),
            PredicateExampleCollection(),
            JoinExampleCollection()
        ]

        for collection in collections:
            df = collection.to_polars()
            assert len(df) == 0

            df_pandas = collection.to_pandas()
            assert len(df_pandas) == 0

    def test_collection_equality(self):
        """Test collection equality comparison."""
        collection1 = MapExampleCollection()
        collection2 = MapExampleCollection()

        example = MapExample(input={"text": "Hello"}, output="greeting")
        collection1.create_example(example)
        collection2.create_example(example)

        assert collection1 == collection2

    def test_unsupported_types(self):
        """Test that unsupported types raise appropriate errors."""
        collection = MapExampleCollection()

        # Custom object type should fail
        class CustomObject:
            pass

        example = MapExample(
            input={"obj": CustomObject()},
            output="result"
        )

        with pytest.raises(TypeInferenceError):  # Will raise from infer_dtype_from_pyobj
            collection.create_example(example)
