import pandas as pd
import polars as pl
import pytest

from fenic.core.types.semantic_examples import (
    BaseExampleCollection,
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


class TestMapExampleCollection:
    """Test cases for MapExampleCollection."""

    def test_create_example(self):
        """Test adding examples to the collection."""
        collection = MapExampleCollection()
        assert len(collection.examples) == 0

        example = MapExample(input={"text": "Hello"}, output="greeting")
        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].input == {"text": "Hello"}
        assert collection.examples[0].output == "greeting"

        # Wrong example type should raise error
        with pytest.raises(InvalidExampleCollectionError):
            collection.create_example(ClassifyExample(input="Hello", output="greeting"))

    def test_from_polars(self):
        """Test creating collection from a Polars DataFrame."""
        # Valid DataFrame
        df = pl.DataFrame(
            {
                "text": ["Hello", "Bonjour", "Hola"],
                "lang": ["en", "fr", "es"],
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
        assert collection.examples[0].input["lang"] == "en"
        assert collection.examples[0].output == "greeting in English"

        # Missing output column
        df_missing_output = pl.DataFrame(
            {"text": ["Hello", "Goodbye"], "lang": ["en", "en"]}
        )

        with pytest.raises(ValueError, match="missing required 'output' column"):
            MapExampleCollection.from_polars(df_missing_output)

        # No input columns
        df_no_inputs = pl.DataFrame({"output": ["result1", "result2"]})

        with pytest.raises(ValueError, match="must have at least one input column"):
            MapExampleCollection.from_polars(df_no_inputs)

        # Null values in output
        df_null_output = pl.DataFrame(
            {"text": ["Hello", "Goodbye"], "output": ["greeting", None]}
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'output' column"
        ):
            MapExampleCollection.from_polars(df_null_output)

    def test_from_pandas(self):
        """Test creating collection from a Pandas DataFrame."""
        # Valid DataFrame
        df = pd.DataFrame(
            {
                "text": ["Hello", "Bonjour", "Hola"],
                "lang": ["en", "fr", "es"],
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
        assert collection.examples[0].input["lang"] == "en"
        assert collection.examples[0].output == "greeting in English"

        # Missing output column
        df_missing_output = pd.DataFrame(
            {"text": ["Hello", "Goodbye"], "lang": ["en", "en"]}
        )

        with pytest.raises(ValueError, match="missing required 'output' column"):
            MapExampleCollection.from_pandas(df_missing_output)

        # No input columns
        df_no_inputs = pd.DataFrame({"output": ["result1", "result2"]})

        with pytest.raises(ValueError, match="must have at least one input column"):
            MapExampleCollection.from_pandas(df_no_inputs)

        # Null values in output
        df_null_output = pd.DataFrame(
            {"text": ["Hello", "Goodbye"], "output": ["greeting", None]}
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'output' column"
        ):
            MapExampleCollection.from_pandas(df_null_output)

    def test_to_polars(self):
        """Test converting collection to a Polars DataFrame."""
        # Empty collection
        empty_collection = MapExampleCollection()
        empty_df = empty_collection.to_polars()
        assert len(empty_df) == 0

        # Collection with examples
        collection = MapExampleCollection()
        collection.create_example(
            MapExample(input={"text": "Hello"}, output="greeting")
        )
        collection.create_example(
            MapExample(input={"text": "Goodbye"}, output="farewell")
        )

        df = collection.to_polars()
        assert len(df) == 2
        assert df.columns == ["text", "output"]
        assert df["text"].to_list() == ["Hello", "Goodbye"]
        assert df["output"].to_list() == ["greeting", "farewell"]

        # Collection with varying schema
        collection = MapExampleCollection()
        collection.create_example(
            MapExample(input={"text": "Hello"}, output="greeting")
        )
        collection.create_example(
            MapExample(input={"text": "Goodbye", "lang": "en"}, output="farewell")
        )

        df = collection.to_polars()
        assert len(df) == 2
        assert set(df.columns) == {"text", "lang", "output"}
        assert df["text"].to_list() == ["Hello", "Goodbye"]
        assert df.filter(pl.col("lang").is_not_null())["lang"].to_list() == ["en"]

    def test_to_pandas(self):
        """Test converting collection to a Pandas DataFrame."""
        collection = MapExampleCollection()
        collection.create_example(
            MapExample(input={"text": "Hello"}, output="greeting")
        )
        collection.create_example(
            MapExample(input={"text": "Goodbye"}, output="farewell")
        )

        df = collection.to_pandas()
        assert len(df) == 2
        assert list(df.columns) == ["text", "output"]
        assert df["text"].tolist() == ["Hello", "Goodbye"]
        assert df["output"].tolist() == ["greeting", "farewell"]

    def test_validate_with_instruction(self):
        """Test validation against instructions."""
        collection = MapExampleCollection()
        collection.create_example(
            MapExample(input={"text": "Hello", "lang": "en"}, output="Hello in English")
        )
        collection.create_example(
            MapExample(
                input={"text": "Bonjour", "lang": "fr"}, output="Hello in French"
            )
        )

        # Valid instruction
        collection._validate_with_instruction(
            "Transform the {text} in {lang} to English"
        )

        # Missing column in instruction
        with pytest.raises(
            InvalidExampleCollectionError, match="collection contains columns not used"
        ):
            collection._validate_with_instruction("Transform the {text} to English")

        # Missing column in collection
        with pytest.raises(
            InvalidExampleCollectionError, match="missing from the collection"
        ):
            collection._validate_with_instruction(
                "Transform the {text} in {lang} with {tone} to English"
            )

        df = pl.DataFrame(
            {
                "text": ["Hello", "Bonjour", "Hola"],
                "lang": ["en", None, "es"],
                "output": [
                    "greeting in English",
                    "greeting in French",
                    "greeting in Spanish",
                ],
            }
        )
        collection_with_nulls = MapExampleCollection.from_polars(df)

        with pytest.raises(
            InvalidExampleCollectionError,
            match="following columns contain null values in one or more examples: lang.",
        ):
            collection_with_nulls._validate_with_instruction(
                "Transform the {text} in {lang} to English"
            )


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

    def test_create_example(self):
        """Test adding examples to the collection."""
        collection = PredicateExampleCollection()
        assert len(collection.examples) == 0

        example = PredicateExample(input={"age": "19"}, output=True)
        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].input["age"] == "19"
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

    def test_validate_with_instruction(self):
        """Test validation against instructions."""
        collection = PredicateExampleCollection()
        collection.create_example(
            PredicateExample(input={"age": "19", "country": "US"}, output=True)
        )
        collection.create_example(
            PredicateExample(input={"age": "17", "country": "UK"}, output=False)
        )

        # Valid instruction
        collection._validate_with_instruction(
            "Is the person at least 18 years old based on {age} and from {country}?"
        )

        # Missing column in instruction
        with pytest.raises(
            InvalidExampleCollectionError, match="collection contains columns not used"
        ):
            collection._validate_with_instruction(
                "Is the person at least 18 years old based on {age}?"
            )

        # Missing column in collection
        with pytest.raises(
            InvalidExampleCollectionError, match="missing from the collection: city."
        ):
            collection._validate_with_instruction(
                "Is the person {age} years old from {country} and {city}?"
            )

        df = pl.DataFrame(
            {
                "age": ["19", "17", "21"],
                "country": ["US", None, "CA"],
                "output": [True, False, True],
            }
        )
        collection_with_nulls = PredicateExampleCollection.from_polars(df)

        with pytest.raises(
            InvalidExampleCollectionError,
            match="following columns contain null values in one or more examples: country.",
        ):
            collection_with_nulls._validate_with_instruction(
                "Is the person at least 18 years old based on {age} and from {country}?"
            )


class TestJoinExampleCollection:
    """Test cases for JoinExampleCollection."""

    def test_create_example(self):
        """Test adding examples to the collection."""
        collection = JoinExampleCollection()
        assert len(collection.examples) == 0

        example = JoinExample(left="apple", right="fruit", output=True)
        collection.create_example(example)
        assert len(collection.examples) == 1
        assert collection.examples[0].left == "apple"
        assert collection.examples[0].right == "fruit"
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
                "left": ["apple", "apple", "carrot"],
                "right": ["fruit", "vegetable", "vegetable"],
                "output": [True, False, True],
            }
        )

        collection = JoinExampleCollection.from_polars(df)
        assert len(collection.examples) == 3
        assert collection.examples[0].left == "apple"
        assert collection.examples[0].right == "fruit"
        assert collection.examples[0].output is True

        # Missing left column
        df_missing_left = pl.DataFrame(
            {
                "item": ["apple", "carrot"],
                "right": ["fruit", "vegetable"],
                "output": [True, True],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'left' column"
        ):
            JoinExampleCollection.from_polars(df_missing_left)

        # Missing right column
        df_missing_right = pl.DataFrame(
            {
                "left": ["apple", "carrot"],
                "category": ["fruit", "vegetable"],
                "output": [True, True],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'right' column"
        ):
            JoinExampleCollection.from_polars(df_missing_right)

        # Missing output column
        df_missing_output = pl.DataFrame(
            {"left": ["apple", "carrot"], "right": ["fruit", "vegetable"]}
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="missing required 'output' column"
        ):
            JoinExampleCollection.from_polars(df_missing_output)

        # Null values in columns
        df_null_left = pl.DataFrame(
            {
                "left": ["apple", None],
                "right": ["fruit", "vegetable"],
                "output": [True, True],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'left' column"
        ):
            JoinExampleCollection.from_polars(df_null_left)

        df_null_right = pl.DataFrame(
            {
                "left": ["apple", "carrot"],
                "right": ["fruit", None],
                "output": [True, True],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'right' column"
        ):
            JoinExampleCollection.from_polars(df_null_right)

        df_null_output = pl.DataFrame(
            {
                "left": ["apple", "carrot"],
                "right": ["fruit", "vegetable"],
                "output": [True, None],
            }
        )

        with pytest.raises(
            InvalidExampleCollectionError, match="null values in 'output' column"
        ):
            JoinExampleCollection.from_polars(df_null_output)

    def test_to_polars(self):
        """Test converting collection to a Polars DataFrame."""
        collection = JoinExampleCollection()
        collection.create_example(JoinExample(left="apple", right="fruit", output=True))
        collection.create_example(
            JoinExample(left="apple", right="vegetable", output=False)
        )

        df = collection.to_polars()
        assert len(df) == 2
        assert df.columns == ["left", "right", "output"]
        assert df["left"].to_list() == ["apple", "apple"]
        assert df["right"].to_list() == ["fruit", "vegetable"]
        assert df["output"].to_list() == [True, False]

    def test_to_pandas(self):
        """Test converting collection to a Pandas DataFrame."""
        collection = JoinExampleCollection()
        collection.create_example(JoinExample(left="apple", right="fruit", output=True))
        collection.create_example(
            JoinExample(left="apple", right="vegetable", output=False)
        )

        df = collection.to_pandas()
        assert len(df) == 2
        assert list(df.columns) == ["left", "right", "output"]
        assert df["left"].tolist() == ["apple", "apple"]
        assert df["right"].tolist() == ["fruit", "vegetable"]
        assert df["output"].tolist() == [True, False]


class TestBaseExampleCollection:
    """Test cases for abstract BaseExampleCollection class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseExampleCollection cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExampleCollection()

    def test_cannot_use_abstract_methods(self):
        """Test that abstract methods must be implemented by subclasses."""

        # Create a minimal concrete subclass implementing only required methods
        class MinimalCollection(BaseExampleCollection):
            example_class = MapExample

            @classmethod
            def from_polars(cls, df):
                return cls()

            def _as_df_input(self):
                return []

        # Should be able to instantiate this class
        _collection = MinimalCollection()

        # Now create a subclass missing an abstract method implementation
        class IncompleteCollection(BaseExampleCollection):
            example_class = MapExample

            # Missing from_polars implementation

            def _as_df_input(self):
                return []

        with pytest.raises(TypeError):
            IncompleteCollection()


class TestEdgeCases:
    """Test edge cases and integration between different collection types."""

    def test_empty_collections(self):
        """Test behavior of empty collections."""
        # Each collection type should handle empty state gracefully
        map_collection = MapExampleCollection()
        classify_collection = ClassifyExampleCollection()
        predicate_collection = PredicateExampleCollection()
        join_collection = JoinExampleCollection()

        # Convert to DataFrames
        map_df = map_collection.to_polars()
        classify_df = classify_collection.to_polars()
        predicate_df = predicate_collection.to_polars()
        join_df = join_collection.to_polars()

        assert len(map_df) == 0
        assert len(classify_df) == 0
        assert len(predicate_df) == 0
        assert len(join_df) == 0

    def test_mixed_types_in_dataframe(self):
        """Test handling of mixed data types in DataFrames."""
        # Create DataFrame with mixed types
        df = pl.DataFrame(
            {
                "numeric": [1, 2, 3],
                "string": ["a", "b", "c"],
                "boolean": [True, False, True],
                "output": ["one", "two", "three"],
            }
        )

        collection = MapExampleCollection.from_polars(df)
        assert len(collection.examples) == 3

        # Ensure types are preserved correctly for map examples (all inputs become strings)
        assert collection.examples[0].input["numeric"] == "1"
        assert collection.examples[0].input["string"] == "a"
        assert collection.examples[0].input["boolean"] == "True"

        # For predicate examples, test proper boolean conversion
        df_predicate = pl.DataFrame(
            {"condition": ["yes", "no", "maybe"], "output": [True, False, True]}
        )

        predicate_collection = PredicateExampleCollection.from_polars(df_predicate)
        assert predicate_collection.examples[0].output is True
        assert predicate_collection.examples[1].output is False

    def test_complex_validation_errors(self):
        """Test complex validation error scenarios."""
        # Create collection with inconsistent schema
        collection = MapExampleCollection()
        collection.create_example(
            MapExample(input={"name": "John"}, output="Person named John")
        )
        collection.create_example(
            MapExample(input={"full_name": "Jane Doe"}, output="Person named Jane Doe")
        )

        # This should fail validation with either instruction
        with pytest.raises(InvalidExampleCollectionError):
            collection._validate_with_instruction("Get the person named {name}")

        with pytest.raises(InvalidExampleCollectionError):
            collection._validate_with_instruction("Get the person with {full_name}")
