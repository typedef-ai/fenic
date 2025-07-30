"""Module for handling semantic examples in query processing.

This module provides classes and utilities for building, managing, and validating semantic examples
used in query processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Generic, List, Type, TypeVar, Union

import pandas as pd
import polars as pl
from pydantic import BaseModel

from fenic._constants import (
    EXAMPLE_INPUT_KEY,
    EXAMPLE_LEFT_KEY,
    EXAMPLE_OUTPUT_KEY,
    EXAMPLE_RIGHT_KEY,
)
from fenic.core._utils.misc import parse_instruction
from fenic.core.error import InvalidExampleCollectionError

ExampleType = TypeVar("ExampleType")


class MapExample(BaseModel):
    """A single semantic example for semantic mapping operations.

    Map examples demonstrate the transformation of input variables to a specific output
    string or structured model used in a semantic.map operation.
    """

    input: Dict[str, str]
    output: Union[str, BaseModel]


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

    input: Dict[str, str]
    output: bool


class JoinExample(BaseModel):
    """A single semantic example for semantic join operations.

    Join examples demonstrate the evaluation of two input strings across different
    datasets against a specific condition, used in a semantic.join operation.
    """

    left: str
    right: str
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

    example_class: ClassVar[Type] = None

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


class MapExampleCollection(BaseExampleCollection[MapExample]):
    """Collection of examples for semantic mapping operations.

    Map operations transform input variables into a text output according to
    specified instructions. This collection manages examples that demonstrate
    the expected transformations for different inputs.

    Examples in this collection can have multiple input variables, each mapped
    to their respective values, with a single output string or structured model
    representing the expected transformation result.
    """

    example_class = MapExample

    def create_example(self, example: MapExample) -> MapExampleCollection:
        """Create an example in the collection with output type validation.

        Ensures all examples in the collection have consistent output types
        (either all strings or all BaseModel instances).

        Args:
            example: The MapExample to add.

        Returns:
            Self for method chaining.

        Raises:
            InvalidExampleCollectionError: If the example output type doesn't match
                the existing examples in the collection.
        """
        if not isinstance(example, MapExample):
            raise InvalidExampleCollectionError(
                f"Expected example of type {MapExample.__name__}, got {type(example).__name__}"
            )
        _validate_single_example_output_type(self.examples, example)
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
            if row[EXAMPLE_OUTPUT_KEY] is None:
                raise InvalidExampleCollectionError(
                    f"Map Examples DataFrame contains null values in '{EXAMPLE_OUTPUT_KEY}' column"
                )

            input_dict = {
                col: str(row[col]) for col in input_cols if row[col] is not None
            }

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

    def _validate_with_instruction(self, instruction: str) -> None:
        """Validate that the collection matches the expected input columns from the instruction."""
        df = self.to_polars()
        expected_cols = set(parse_instruction(instruction))
        actual_cols = set(df.columns) - {EXAMPLE_OUTPUT_KEY}

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            raise InvalidExampleCollectionError(
                f"The following columns are required by the instruction but missing from the collection: "
                f"{', '.join(sorted(missing))}.\nExpected columns: {', '.join(sorted(expected_cols))}.\n"
                f"Actual columns: {', '.join(sorted(actual_cols))}."
            )

        if extra:
            raise InvalidExampleCollectionError(
                f"The examples collection contains columns not used in the instruction: {', '.join(sorted(extra))}.\n"
                f"Only the following columns are expected based on the instruction: "
                f"{', '.join(sorted(expected_cols))}."
            )
        nulls_per_column = df.select(
            [pl.col(col).is_null().any().alias(col) for col in expected_cols]
        )
        null_columns = [col for col in expected_cols if nulls_per_column[0, col]]

        if null_columns:
            raise InvalidExampleCollectionError(
                f"The following columns contain null values in one or more examples: {', '.join(sorted(null_columns))}."
            )


class ClassifyExampleCollection(BaseExampleCollection[ClassifyExample]):
    """Collection of examples for semantic classification operations.

    Classification operations categorize input text into predefined classes.
    This collection manages examples that demonstrate the expected classification
    results for different inputs.

    Examples in this collection have a single input string and an output string
    representing the classification result.
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
    """Collection of examples for semantic predicate operations.

    Predicate operations evaluate conditions on input variables to produce
    boolean (True/False) results. This collection manages examples that
    demonstrate the expected boolean outcomes for different inputs.

    Examples in this collection can have multiple input variables, each mapped
    to their respective values, with a single boolean output representing the
    evaluation result of the predicate.
    """

    example_class = PredicateExample

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

    def _validate_with_instruction(self, instruction: str) -> None:
        """Validate that the collection matches the expected input columns from the instruction."""
        df = self.to_polars()
        expected_cols = set(parse_instruction(instruction))
        actual_cols = set(df.columns) - {EXAMPLE_OUTPUT_KEY}

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            raise InvalidExampleCollectionError(
                f"The following columns are required by the instruction but missing from the collection: "
                f"{', '.join(sorted(missing))}.\nExpected columns: {', '.join(sorted(expected_cols))}.\n"
                f"Actual columns: {', '.join(sorted(actual_cols))}."
            )

        if extra:
            raise InvalidExampleCollectionError(
                f"The examples collection contains columns not used in the instruction: {', '.join(sorted(extra))}.\n"
                f"Only the following columns are expected based on the instruction: "
                f"{', '.join(sorted(expected_cols))}."
            )
        nulls_per_column = df.select(
            [pl.col(col).is_null().any().alias(col) for col in expected_cols]
        )
        null_columns = [col for col in expected_cols if nulls_per_column[0, col]]

        if null_columns:
            raise InvalidExampleCollectionError(
                f"The following columns contain null values in one or more examples: {', '.join(sorted(null_columns))}."
            )


class JoinExampleCollection(BaseExampleCollection[JoinExample]):
    """Collection of examples for semantic join operations."""

    example_class = JoinExample

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> JoinExampleCollection:
        """Create collection from a Polars DataFrame. Must have 'left', 'right', and 'output' columns."""
        collection = cls()

        required_columns = [
            EXAMPLE_LEFT_KEY,
            EXAMPLE_RIGHT_KEY,
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
                left=row[EXAMPLE_LEFT_KEY],
                right=row[EXAMPLE_RIGHT_KEY],
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
                    EXAMPLE_LEFT_KEY: example_dict[EXAMPLE_LEFT_KEY],
                    EXAMPLE_RIGHT_KEY: example_dict[EXAMPLE_RIGHT_KEY],
                    EXAMPLE_OUTPUT_KEY: example_dict[EXAMPLE_OUTPUT_KEY],
                }
            )

        return rows

# when we add semantic.extract examples, this signature can change to
# def validate_single_example_output_type(existing_examples: list[Union[MapExample, ExtractExample]], example: Union[MapExample, ExtractExample]):
def _validate_single_example_output_type(existing_examples: list[MapExample], example: MapExample):
    if not existing_examples:
        return
    first_example = existing_examples[0]
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