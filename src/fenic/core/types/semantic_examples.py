"""Module for handling semantic examples in query processing.

This module provides classes and utilities for building, managing, and validating semantic examples
used in query processing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, List, Mapping, Type, TypeVar, Union

import pandas as pd
import polars as pl
from pydantic import BaseModel

from fenic._constants import (
    EXAMPLE_INPUT_KEY,
    EXAMPLE_OUTPUT_KEY,
    LEFT_ON_KEY,
    RIGHT_ON_KEY,
)
from fenic.core._utils.type_inference import infer_dtype_from_pyobj
from fenic.core.error import InvalidExampleCollectionError
from fenic.core.types.datatypes import DataType

ExampleType = TypeVar("ExampleType")

logger = logging.getLogger(__name__)

class MapExample(BaseModel):
    """A single semantic example for semantic mapping operations.

    Map examples demonstrate the transformation of input variables to a specific output
    string or structured model used in a semantic.map operation.
    """

    input: Mapping[str, Any]
    output: Union[str, BaseModel]

    def __eq__(self, other: MapExample) -> bool:
        """Compare MapExample instances, handling dictionary key order differences."""
        if not isinstance(other, MapExample):
            return False

        # Compare outputs
        if self.output != other.output:
            return False

        # Compare inputs, handling dictionary key order differences
        if len(self.input) != len(other.input):
            return False

        # Convert to sorted items for comparison
        self_items = sorted(self.input.items())
        other_items = sorted(other.input.items())

        return self_items == other_items


class ClassifyExample(BaseModel):
    """A single semantic example for classification operations.

    Classify examples demonstrate the classification of an input string into a specific category string,
    used in a semantic.classify operation.
    """

    input: str
    output: str


class PredicateExample(BaseModel):
    """A single semantic example for semantic predicate operations.

    Predicate examples demonstrate the evaluation of input variables against a specific condition,
    used in a semantic.predicate operation.
    """

    input: Mapping[str, Any]
    output: bool

    def __eq__(self, other: PredicateExample) -> bool:
        """Compare PredicateExample instances, handling dictionary key order differences."""
        if not isinstance(other, PredicateExample):
            return False

        # Compare outputs
        if self.output != other.output:
            return False

        # Compare inputs, handling dictionary key order differences
        if len(self.input) != len(other.input):
            return False

        # Convert to sorted items for comparison
        self_items = sorted(self.input.items())
        other_items = sorted(other.input.items())

        return self_items == other_items


class JoinExample(BaseModel):
    """A single semantic example for semantic join operations.

    Join examples demonstrate the evaluation of two input variables across different
    datasets against a specific condition, used in a semantic.join operation.
    """

    left_on: Any
    right_on: Any
    output: bool

class BaseExampleCollection(ABC, Generic[ExampleType]):
    """Abstract base class for all semantic example collections.

    Semantic examples demonstrate the expected input-output relationship for a given task,
    helping guide language models to produce consistent and accurate responses. Each example
    consists of inputs and the corresponding expected output.

    These examples are particularly valuable for:

    - Demonstrating the expected reasoning pattern
    - Showing correct output formats
    - Handling edge cases through demonstration
    - Improving model performance without changing the underlying model
    """

    example_class: ClassVar[Type[ExampleType]]

    def __init__(self, examples: List[ExampleType] = None):
        """Initialize a collection of semantic examples.

        Args:
            examples: Optional list of examples to add to the collection. Each example
                will be processed through create_example() to ensure proper formatting
                and validation.

        Note:
            The examples list is initialized as empty if no examples are provided.
            Each example in the provided list will be processed through create_example()
            to ensure proper formatting and validation.
        """
        self.examples: List[ExampleType] = []
        if examples:
            for example in examples:
                self.create_example(example)

    @classmethod
    @abstractmethod
    def from_polars(cls, df: pl.DataFrame) -> BaseExampleCollection:
        """Create a collection from a Polars DataFrame.

        Args:
            df: The Polars DataFrame containing example data. The specific
                column structure requirements depend on the concrete collection type.

        Returns:
            A new example collection populated with examples from the DataFrame.

        Raises:
            InvalidExampleCollectionError: If the DataFrame's structure doesn't match
                the expected format for this collection type.
        """
        pass

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> BaseExampleCollection:
        """Create a collection from a Pandas DataFrame.

        Args:
            df: The Pandas DataFrame containing example data. The specific
                column structure requirements depend on the concrete collection type.

        Returns:
            A new example collection populated with examples from the DataFrame.

        Raises:
            InvalidExampleCollectionError: If the DataFrame's structure doesn't match
                the expected format for this collection type.
        """
        polars_df = pl.from_pandas(data=df)
        return cls.from_polars(polars_df)

    @abstractmethod
    def _as_df_input(self) -> List[Dict[str, Any]]:
        """Convert the collection to a list of dictionaries suitable for DataFrame creation."""
        pass

    def to_polars(self) -> pl.DataFrame:
        """Convert the collection to a Polars DataFrame.

        Returns:
            A Polars DataFrame representing the collection's examples.
            Returns an empty DataFrame if the collection contains no examples.
        """
        rows = self._as_df_input()
        return pl.DataFrame(rows)

    def to_pandas(self) -> pd.DataFrame:
        """Convert the collection to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame representing the collection's examples.
            Returns an empty DataFrame if the collection contains no examples.
        """
        rows = self._as_df_input()
        return pd.DataFrame(rows)

    def create_example(self, example: ExampleType) -> BaseExampleCollection:
        """Create an example in the collection.

        Args:
        example: The semantic example to add. Must be an instance of the
                collection's example_class.

        Returns:
            Self for method chaining.
        """
        if not isinstance(example, self.example_class):
            raise InvalidExampleCollectionError(
                f"Expected example of type {self.example_class.__name__}, got {type(example).__name__}"
            )
        self.examples.append(example)
        return self

    def __eq__(self, other: BaseExampleCollection) -> bool:
        """Check if two example collections are equal."""
        if not isinstance(other, BaseExampleCollection):
            return False
        if self.example_class != other.example_class:
            return False
        if len(self.examples) != len(other.examples):
            return False

        # For MapExampleCollection and PredicateExampleCollection, compare examples directly since their __eq__ methods now handle key order differences
        if isinstance(self, MapExampleCollection) and isinstance(other, MapExampleCollection):
            # Compare examples directly since MapExample.__eq__ now handles key order differences
            return all(example1 == example2 for example1, example2 in zip(self.examples, other.examples, strict=True))

        if isinstance(self, PredicateExampleCollection) and isinstance(other, PredicateExampleCollection):
            # Compare examples directly since PredicateExample.__eq__ now handles key order differences
            return all(example1 == example2 for example1, example2 in zip(self.examples, other.examples, strict=True))

        # For other collection types, use the DataFrame comparison
        return self.to_polars().equals(other.to_polars())

class MapExampleCollection(BaseExampleCollection[MapExample]):
    """Collection of input-output examples for semantic map operations.

    Stores examples that demonstrate how input data should be transformed into
    output text or structured data. Each example shows the expected output for
    a given set of input fields.
    """

    example_class = MapExample


    def __init__(self, examples: List[MapExample] = None):
        """Initialize a collection of semantic map examples.

        Args:
            examples: List of examples to add to the collection. Each example
                will be processed through create_example() to ensure proper formatting
                and validation.
        """
        self._type_validator = _ExampleTypeValidator()
        super().__init__(examples)

    def create_example(self, example: MapExample) -> MapExampleCollection:
        """Create an example in the collection with output and input type validation.

        Ensures all examples in the collection have consistent output types
        (either all strings or all BaseModel instances) and validates that input
        fields have consistent types across examples.

        For input validation:
        - The first example establishes the schema and cannot have None values
        - Subsequent examples must have the same fields but can have None values
        - Non-None values must match the established type for each field

        Args:
            example: The MapExample to add.

        Returns:
            Self for method chaining.

        Raises:
            InvalidExampleCollectionError: If the example output type doesn't match
                the existing examples in the collection, if the first example contains
                None values, or if subsequent examples have type mismatches.
        """
        if not isinstance(example, MapExample):
            raise InvalidExampleCollectionError(
                f"Expected example of type {MapExample.__name__}, got {type(example).__name__}"
            )

        # Validate output type consistency
        self._validate_single_example_output_type(example)

        # Validate input types
        example_num = len(self.examples) + 1
        self._type_validator.process_example(example.input, example_num)

        self.examples.append(example)
        return self

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> MapExampleCollection:
        """Create collection from a Polars DataFrame. Must have an 'output' column and at least one input column."""
        collection = cls()

        if EXAMPLE_OUTPUT_KEY not in df.columns:
            raise ValueError(
                f"Map Examples DataFrame missing required '{EXAMPLE_OUTPUT_KEY}' column"
            )

        input_cols = [col for col in df.columns if col != EXAMPLE_OUTPUT_KEY]

        if not input_cols:
            raise ValueError(
                "Map Examples DataFrame must have at least one input column"
            )

        for row in df.iter_rows(named=True):
            input_dict = {col: row[col] for col in input_cols}
            example = MapExample(input=input_dict, output=row[EXAMPLE_OUTPUT_KEY])
            collection.create_example(example)

        return collection

    def _as_df_input(self) -> List[Dict[str, Any]]:
        """Convert examples to a list of dictionaries suitable for DataFrame creation."""
        if not self.examples:
            return []

        rows = []
        for example in self.examples:
            example_dict = example.model_dump()
            # Handle BaseModel instances by converting to JSON string
            output_value = example_dict[EXAMPLE_OUTPUT_KEY]
            if isinstance(example.output, BaseModel):
                output_value = example.output.model_dump_json()

            row = {
                **example_dict[EXAMPLE_INPUT_KEY],
                EXAMPLE_OUTPUT_KEY: output_value,
            }
            rows.append(row)

        return rows

    def _validate_against_column_types(self, column_types: Dict[str, DataType]) -> None:
        """Validate that the collection matches the expected input columns from the instruction."""
        self._type_validator.validate_against_column_types(column_types)

    def _validate_single_example_output_type(self, example: MapExample) -> None:
        if not self.examples:
            return
        first_example = self.examples[0]
        first_is_basemodel = isinstance(first_example.output, BaseModel)
        current_is_basemodel = isinstance(example.output, BaseModel)
        if first_is_basemodel != current_is_basemodel:
            first_type = type(first_example.output).__name__
            current_type = type(example.output).__name__
            raise InvalidExampleCollectionError(
                f"All examples in Example Collection must have consistent output types. "
                f"Existing examples have {first_type} outputs, but new example has {current_type} output."
            )
        # If both are BaseModel, ensure they're the same type
        if first_is_basemodel and current_is_basemodel:
            if not isinstance(first_example.output, type(example.output)):
                raise InvalidExampleCollectionError(
                    f"All BaseModel examples must be of the same type. "
                    f"Existing examples are {type(first_example.output).__name__}, "
                    f"but new example is {type(example.output).__name__}."
                )

class ClassifyExampleCollection(BaseExampleCollection[ClassifyExample]):
    """Collection of text-to-category examples for classification operations.

    Stores examples showing which category each input text should be assigned to.
    Each example contains an input string and its corresponding category label.
    """

    example_class = ClassifyExample

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> ClassifyExampleCollection:
        """Create collection from a Polars DataFrame. Must have an 'output' column and an 'input' column."""
        collection = cls()

        if EXAMPLE_INPUT_KEY not in df.columns:
            raise InvalidExampleCollectionError(
                f"Classify Examples DataFrame missing required '{EXAMPLE_INPUT_KEY}' column"
            )
        if EXAMPLE_OUTPUT_KEY not in df.columns:
            raise InvalidExampleCollectionError(
                f"Classify Examples DataFrame missing required '{EXAMPLE_OUTPUT_KEY}' column"
            )

        for row in df.iter_rows(named=True):
            if row[EXAMPLE_INPUT_KEY] is None:
                raise InvalidExampleCollectionError(
                    f"Classify Examples DataFrame contains null values in '{EXAMPLE_INPUT_KEY}' column"
                )
            if row[EXAMPLE_OUTPUT_KEY] is None:
                raise InvalidExampleCollectionError(
                    f"Classify Examples DataFrame contains null values in '{EXAMPLE_OUTPUT_KEY}' column"
                )

            example = ClassifyExample(
                input=row[EXAMPLE_INPUT_KEY],
                output=row[EXAMPLE_OUTPUT_KEY],
            )
            collection.create_example(example)

        return collection

    def _as_df_input(self) -> List[Dict[str, Any]]:
        """Convert examples to a list of dictionaries suitable for DataFrame creation."""
        if not self.examples:
            return []

        rows = []
        for example in self.examples:
            example_dict = example.model_dump()
            rows.append(
                {
                    EXAMPLE_INPUT_KEY: example_dict[EXAMPLE_INPUT_KEY],
                    EXAMPLE_OUTPUT_KEY: example_dict[EXAMPLE_OUTPUT_KEY],
                }
            )

        return rows

    def _validate_with_labels(self, valid_labels: set[str]) -> None:
        """Validate examples against a set of valid labels."""
        invalid_examples = []
        for i, example in enumerate(self.examples):
            if example.output not in valid_labels:
                invalid_examples.append((i, example.output))

        if invalid_examples:
            valid_labels_str = ", ".join(sorted(valid_labels))
            invalid_str = ", ".join(f"#{i}: '{label}'" for i, label in invalid_examples)
            raise InvalidExampleCollectionError(
                f"Example outputs must match available class labels. "
                f"Valid labels: {valid_labels_str}. "
                f"Invalid examples: {invalid_str}"
            )


class PredicateExampleCollection(BaseExampleCollection[PredicateExample]):
    """Collection of input-to-boolean examples for predicate operations.

    Stores examples showing which inputs should evaluate to True or False
    based on some condition. Each example contains input fields and a
    boolean output indicating whether the condition holds.
    """

    example_class = PredicateExample

    def __init__(self, examples: List[PredicateExample] = None):
        """Initialize a collection of semantic predicate examples.

        Args:
            examples: List of examples to add to the collection. Each example
                will be processed through create_example() to ensure proper formatting
                and validation.
        """
        self._type_validator = _ExampleTypeValidator()
        super().__init__(examples)

    def create_example(self, example: PredicateExample) -> PredicateExampleCollection:
        """Create an example in the collection with input type validation.

        Validates that input fields have consistent types across examples.
        The first example establishes the schema and cannot have None values.
        Subsequent examples must have the same fields but can have None values.

        Args:
            example: The PredicateExample to add.

        Returns:
            Self for method chaining.

        Raises:
            InvalidExampleCollectionError: If the example type is wrong, if the
                first example contains None values, or if subsequent examples
                have type mismatches.
        """
        if not isinstance(example, PredicateExample):
            raise InvalidExampleCollectionError(
                f"Expected example of type {PredicateExample.__name__}, got {type(example).__name__}"
            )

        # Validate input types
        example_num = len(self.examples) + 1
        self._type_validator.process_example(example.input, example_num)

        self.examples.append(example)
        return self

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> PredicateExampleCollection:
        """Create collection from a Polars DataFrame."""
        collection = cls()

        # Validate output column exists
        if EXAMPLE_OUTPUT_KEY not in df.columns:
            raise InvalidExampleCollectionError(
                f"Predicate Examples DataFrame missing required '{EXAMPLE_OUTPUT_KEY}' column"
            )

        input_cols = [col for col in df.columns if col != EXAMPLE_OUTPUT_KEY]

        if not input_cols:
            raise InvalidExampleCollectionError(
                "Predicate Examples DataFrame must have at least one input column"
            )

        for row in df.iter_rows(named=True):
            if row[EXAMPLE_OUTPUT_KEY] is None:
                raise InvalidExampleCollectionError(
                    f"Predicate Examples DataFrame contains null values in '{EXAMPLE_OUTPUT_KEY}' column"
                )

            input_dict = {col: row[col] for col in input_cols if row[col] is not None}

            example = PredicateExample(input=input_dict, output=row[EXAMPLE_OUTPUT_KEY])
            collection.create_example(example)

        return collection

    def _as_df_input(self) -> List[Dict[str, Any]]:
        """Convert examples to a list of dictionaries suitable for DataFrame creation."""
        if not self.examples:
            return []

        rows = []
        for example in self.examples:
            example_dict = example.model_dump()
            row = {
                **example_dict[EXAMPLE_INPUT_KEY],
                EXAMPLE_OUTPUT_KEY: example_dict[EXAMPLE_OUTPUT_KEY],
            }
            rows.append(row)

        return rows

    def _validate_against_column_types(self, column_types: Dict[str, DataType]) -> None:
        """Validate that the collection matches the expected input columns from the instruction."""
        self._type_validator.validate_against_column_types(column_types)


class JoinExampleCollection(BaseExampleCollection[JoinExample]):
    """Collection of comparison examples for semantic join operations.

    Stores examples showing which pairs of values should be considered matches
    for joining data. Each example contains a left value, right value, and
    boolean output indicating whether they match.
    """

    example_class = JoinExample

    def __init__(self, examples: List[JoinExample] = None):
        """Initialize a collection of semantic join examples.

        Args:
            examples: List of examples to add to the collection. Each example
                will be processed through create_example() to ensure proper formatting
                and validation.
        """
        self._type_validator = _ExampleTypeValidator()
        super().__init__(examples)

    def create_example(self, example: JoinExample) -> JoinExampleCollection:
        """Create an example in the collection with type validation.

        Validates that left_on and right_on values have consistent types across
        examples. The first example establishes the types and cannot have None values.
        Subsequent examples must have matching types but can have None values.

        Args:
            example: The JoinExample to add.

        Returns:
            Self for method chaining.

        Raises:
            InvalidExampleCollectionError: If the example type is wrong, if the
                first example contains None values, or if subsequent examples
                have type mismatches.
        """
        if not isinstance(example, JoinExample):
            raise InvalidExampleCollectionError(
                f"Expected example of type {JoinExample.__name__}, got {type(example).__name__}"
            )

        # Convert to dict format for validation
        example_dict = {
            LEFT_ON_KEY: example.left_on,
            RIGHT_ON_KEY: example.right_on
        }

        example_num = len(self.examples) + 1
        self._type_validator.process_example(example_dict, example_num)

        self.examples.append(example)
        return self

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> JoinExampleCollection:
        """Create collection from a Polars DataFrame. Must have 'left_on', 'right_on', and 'output' columns."""
        collection = cls()

        required_columns = [
            LEFT_ON_KEY,
            RIGHT_ON_KEY,
            EXAMPLE_OUTPUT_KEY,
        ]
        for col in required_columns:
            if col not in df.columns:
                raise InvalidExampleCollectionError(
                    f"Join Examples DataFrame missing required '{col}' column"
                )

        for row in df.iter_rows(named=True):
            for col in required_columns:
                if row[col] is None:
                    raise InvalidExampleCollectionError(
                        f"Join Examples DataFrame contains null values in '{col}' column"
                    )

            example = JoinExample(
                left_on=row[LEFT_ON_KEY],
                right_on=row[RIGHT_ON_KEY],
                output=row[EXAMPLE_OUTPUT_KEY],
            )
            collection.create_example(example)

        return collection

    def _validate_against_join_types(self, left_type: DataType, right_type: DataType) -> None:
        """Validate that example types match the actual join column types.

        Args:
            left_type: DataType of the left join column
            right_type: DataType of the right join column

        Raises:
            InvalidExampleCollectionError: If example types don't match column types.
        """
        column_types = {
            LEFT_ON_KEY: left_type,
            RIGHT_ON_KEY: right_type
        }
        self._type_validator.validate_against_column_types(column_types)

    def _as_df_input(self) -> List[Dict[str, Any]]:
        """Convert examples to a list of dictionaries suitable for DataFrame creation."""
        if not self.examples:
            return []

        rows = []
        for example in self.examples:
            example_dict = example.model_dump()
            rows.append(
                {
                    LEFT_ON_KEY: example_dict[LEFT_ON_KEY],
                    RIGHT_ON_KEY: example_dict[RIGHT_ON_KEY],
                    EXAMPLE_OUTPUT_KEY: example_dict[EXAMPLE_OUTPUT_KEY],
                }
            )

        return rows

class _ExampleTypeValidator:
    """Validates types across examples with no None values allowed."""

    def __init__(self):
        self.field_types: Dict[str, DataType] = {}
        self.is_first_example = True

    def process_example(self, example_dict: Dict[str, Any], example_num: int) -> None:
        """Process an example dict, establishing or validating types."""
        # Check for None values in any example
        for key, value in example_dict.items():
            if value is None:
                raise InvalidExampleCollectionError(
                    f"Example #{example_num}: None values are not allowed. "
                    f"Field '{key}' is None."
                )

        if self.is_first_example:
            # First example: establishes the schema
            for key, value in example_dict.items():
                self.field_types[key] = infer_dtype_from_pyobj(value, path=key)
            self.is_first_example = False
        else:
            # Subsequent examples: must have same keys and matching types
            expected_keys = set(self.field_types.keys())
            actual_keys = set(example_dict.keys())

            if expected_keys != actual_keys:
                raise InvalidExampleCollectionError(
                    f"Example #{example_num} has inconsistent fields. "
                    f"Expected: {sorted(expected_keys)}, Got: {sorted(actual_keys)}"
                )

            # Validate types match established schema
            for key, value in example_dict.items():
                inferred_type = infer_dtype_from_pyobj(value, path=key)
                if self.field_types[key] != inferred_type:
                    raise InvalidExampleCollectionError(
                        f"Example #{example_num}: Field '{key}' type mismatch. "
                        f"Expected {self.field_types[key]}, got {inferred_type}"
                    )

    def validate_against_column_types(self, column_types: Dict[str, DataType]) -> None:
        """Validate that column types are a subset of example field types.

        Args:
            column_types: Mapping of column names to their DataType from the query plan.

        Raises:
            InvalidExampleCollectionError: If column types don't match example types
                or if columns are missing from examples.
        """
        example_types = self.field_types

        # Check that all column types exist in examples
        missing_fields = set(column_types.keys()) - set(example_types.keys())
        if missing_fields:
            raise InvalidExampleCollectionError(
                f"The following columns are used in the jinja template but missing from examples: "
                f"{sorted(missing_fields)}. "
                f"Examples have fields: {sorted(example_types.keys())}"
            )

        # Validate types for columns that exist
        for field_name, column_type in column_types.items():
            example_type = example_types[field_name]
            if example_type != column_type:
                raise InvalidExampleCollectionError(
                    f"Field '{field_name}' type mismatch: "
                    f"operator expects {column_type}, but examples have {example_type}"
                )
