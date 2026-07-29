# fenic.core.types.semantic_examples

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/types/semantic_examples/

Module for handling semantic examples in query processing.

This module provides classes and utilities for building, managing, and validating semantic examples
used in query processing.

Classes:

- **`BaseExampleCollection`**
  –

  Abstract base class for all semantic example collections.
- **`ClassifyExample`**
  –

  A single semantic example for classification operations.
- **`ClassifyExampleCollection`**
  –

  Collection of text-to-category examples for classification operations.
- **`JoinExample`**
  –

  A single semantic example for semantic join operations.
- **`JoinExampleCollection`**
  –

  Collection of comparison examples for semantic join operations.
- **`MapExample`**
  –

  A single semantic example for semantic mapping operations.
- **`MapExampleCollection`**
  –

  Collection of input-output examples for semantic map operations.
- **`PredicateExample`**
  –

  A single semantic example for semantic predicate operations.
- **`PredicateExampleCollection`**
  –

  Collection of input-to-boolean examples for predicate operations.

## BaseExampleCollection

```
BaseExampleCollection(examples: List[ExampleType] = None)
```

Bases: `ABC`, `Generic[ExampleType]`

```
              flowchart TD
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

              click fenic.core.types.semantic_examples.BaseExampleCollection href "" "fenic.core.types.semantic_examples.BaseExampleCollection"
```

Abstract base class for all semantic example collections.

Semantic examples demonstrate the expected input-output relationship for a given task,
helping guide language models to produce consistent and accurate responses. Each example
consists of inputs and the corresponding expected output.

These examples are particularly valuable for:

- Demonstrating the expected reasoning pattern
- Showing correct output formats
- Handling edge cases through demonstration
- Improving model performance without changing the underlying model

Initialize a collection of semantic examples.

Parameters:

- **`examples`**
  (`List[ExampleType]`, default:
  `None`
  )
  –

  Optional list of examples to add to the collection. Each example
  will be processed through create_example() to ensure proper formatting
  and validation.

Note

The examples list is initialized as empty if no examples are provided.
Each example in the provided list will be processed through create_example()
to ensure proper formatting and validation.

Methods:

- **`create_example`**
  –

  Create an example in the collection.
- **`from_pandas`**
  –

  Create a collection from a Pandas DataFrame.
- **`from_polars`**
  –

  Create a collection from a Polars DataFrame.
- **`to_pandas`**
  –

  Convert the collection to a Pandas DataFrame.
- **`to_polars`**
  –

  Convert the collection to a Polars DataFrame.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### create_example

```
create_example(example: ExampleType) -> BaseExampleCollection
```

Create an example in the collection.

example: The semantic example to add. Must be an instance of the
collection's example_class.

Returns:

- `BaseExampleCollection`
  –

  Self for method chaining.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### from_pandas

```
from_pandas(df: DataFrame) -> BaseExampleCollection
```

Create a collection from a Pandas DataFrame.

Parameters:

- **`df`**
  (`DataFrame`)
  –

  The Pandas DataFrame containing example data. The specific
  column structure requirements depend on the concrete collection type.

Returns:

- `BaseExampleCollection`
  –

  A new example collection populated with examples from the DataFrame.

Raises:

- `InvalidExampleCollectionError`
  –

  If the DataFrame's structure doesn't match
  the expected format for this collection type.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### from_polars

```
from_polars(df: DataFrame) -> BaseExampleCollection
```

Create a collection from a Polars DataFrame.

Parameters:

- **`df`**
  (`DataFrame`)
  –

  The Polars DataFrame containing example data. The specific
  column structure requirements depend on the concrete collection type.

Returns:

- `BaseExampleCollection`
  –

  A new example collection populated with examples from the DataFrame.

Raises:

- `InvalidExampleCollectionError`
  –

  If the DataFrame's structure doesn't match
  the expected format for this collection type.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### to_pandas

```
to_pandas() -> pd.DataFrame
```

Convert the collection to a Pandas DataFrame.

Returns:

- `DataFrame`
  –

  A Pandas DataFrame representing the collection's examples.
- `DataFrame`
  –

  Returns an empty DataFrame if the collection contains no examples.

Source code in `src/fenic/core/types/semantic_examples.py`

```
def to_pandas(self) -> pd.DataFrame:
    """Convert the collection to a Pandas DataFrame.

    Returns:
        A Pandas DataFrame representing the collection's examples.
        Returns an empty DataFrame if the collection contains no examples.
    """
    rows = self._as_df_input()
    return pd.DataFrame(rows)
```

### to_polars

```
to_polars() -> pl.DataFrame
```

Convert the collection to a Polars DataFrame.

Returns:

- `DataFrame`
  –

  A Polars DataFrame representing the collection's examples.
- `DataFrame`
  –

  Returns an empty DataFrame if the collection contains no examples.

Source code in `src/fenic/core/types/semantic_examples.py`

```
def to_polars(self) -> pl.DataFrame:
    """Convert the collection to a Polars DataFrame.

    Returns:
        A Polars DataFrame representing the collection's examples.
        Returns an empty DataFrame if the collection contains no examples.
    """
    rows = self._as_df_input()
    return pl.DataFrame(rows)
```

## ClassifyExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.semantic_examples.ClassifyExample[ClassifyExample]

              click fenic.core.types.semantic_examples.ClassifyExample href "" "fenic.core.types.semantic_examples.ClassifyExample"
```

A single semantic example for classification operations.

Classify examples demonstrate the classification of an input string into a specific category string,
used in a semantic.classify operation.

## ClassifyExampleCollection

```
ClassifyExampleCollection(examples: List[ExampleType] = None)
```

Bases: `BaseExampleCollection[ClassifyExample]`

```
              flowchart TD
              fenic.core.types.semantic_examples.ClassifyExampleCollection[ClassifyExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.semantic_examples.ClassifyExampleCollection

              click fenic.core.types.semantic_examples.ClassifyExampleCollection href "" "fenic.core.types.semantic_examples.ClassifyExampleCollection"
              click fenic.core.types.semantic_examples.BaseExampleCollection href "" "fenic.core.types.semantic_examples.BaseExampleCollection"
```

Collection of text-to-category examples for classification operations.

Stores examples showing which category each input text should be assigned to.
Each example contains an input string and its corresponding category label.

Methods:

- **`from_polars`**
  –

  Create collection from a Polars DataFrame. Must have an 'output' column and an 'input' column.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### from_polars

```
from_polars(df: DataFrame) -> ClassifyExampleCollection
```

Create collection from a Polars DataFrame. Must have an 'output' column and an 'input' column.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

## JoinExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.semantic_examples.JoinExample[JoinExample]

              click fenic.core.types.semantic_examples.JoinExample href "" "fenic.core.types.semantic_examples.JoinExample"
```

A single semantic example for semantic join operations.

Join examples demonstrate the evaluation of two input variables across different
datasets against a specific condition, used in a semantic.join operation.

## JoinExampleCollection

```
JoinExampleCollection(examples: List[JoinExample] = None)
```

Bases: `BaseExampleCollection[JoinExample]`

```
              flowchart TD
              fenic.core.types.semantic_examples.JoinExampleCollection[JoinExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.semantic_examples.JoinExampleCollection

              click fenic.core.types.semantic_examples.JoinExampleCollection href "" "fenic.core.types.semantic_examples.JoinExampleCollection"
              click fenic.core.types.semantic_examples.BaseExampleCollection href "" "fenic.core.types.semantic_examples.BaseExampleCollection"
```

Collection of comparison examples for semantic join operations.

Stores examples showing which pairs of values should be considered matches
for joining data. Each example contains a left value, right value, and
boolean output indicating whether they match.

Initialize a collection of semantic join examples.

Parameters:

- **`examples`**
  (`List[JoinExample]`, default:
  `None`
  )
  –

  List of examples to add to the collection. Each example
  will be processed through create_example() to ensure proper formatting
  and validation.

Methods:

- **`create_example`**
  –

  Create an example in the collection with type validation.
- **`from_polars`**
  –

  Create collection from a Polars DataFrame. Must have 'left_on', 'right_on', and 'output' columns.

Source code in `src/fenic/core/types/semantic_examples.py`

```
def __init__(self, examples: List[JoinExample] = None):
    """Initialize a collection of semantic join examples.

    Args:
        examples: List of examples to add to the collection. Each example
            will be processed through create_example() to ensure proper formatting
            and validation.
    """
    self._type_validator = _ExampleTypeValidator()
    super().__init__(examples)
```

### create_example

```
create_example(example: JoinExample) -> JoinExampleCollection
```

Create an example in the collection with type validation.

Validates that left_on and right_on values have consistent types across
examples. The first example establishes the types and cannot have None values.
Subsequent examples must have matching types but can have None values.

Parameters:

- **`example`**
  (`JoinExample`)
  –

  The JoinExample to add.

Returns:

- `JoinExampleCollection`
  –

  Self for method chaining.

Raises:

- `InvalidExampleCollectionError`
  –

  If the example type is wrong, if the
  first example contains None values, or if subsequent examples
  have type mismatches.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### from_polars

```
from_polars(df: DataFrame) -> JoinExampleCollection
```

Create collection from a Polars DataFrame. Must have 'left_on', 'right_on', and 'output' columns.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

## MapExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.semantic_examples.MapExample[MapExample]

              click fenic.core.types.semantic_examples.MapExample href "" "fenic.core.types.semantic_examples.MapExample"
```

A single semantic example for semantic mapping operations.

Map examples demonstrate the transformation of input variables to a specific output
string or structured model used in a semantic.map operation.

## MapExampleCollection

```
MapExampleCollection(examples: List[MapExample] = None)
```

Bases: `BaseExampleCollection[MapExample]`

```
              flowchart TD
              fenic.core.types.semantic_examples.MapExampleCollection[MapExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.semantic_examples.MapExampleCollection

              click fenic.core.types.semantic_examples.MapExampleCollection href "" "fenic.core.types.semantic_examples.MapExampleCollection"
              click fenic.core.types.semantic_examples.BaseExampleCollection href "" "fenic.core.types.semantic_examples.BaseExampleCollection"
```

Collection of input-output examples for semantic map operations.

Stores examples that demonstrate how input data should be transformed into
output text or structured data. Each example shows the expected output for
a given set of input fields.

Initialize a collection of semantic map examples.

Parameters:

- **`examples`**
  (`List[MapExample]`, default:
  `None`
  )
  –

  List of examples to add to the collection. Each example
  will be processed through create_example() to ensure proper formatting
  and validation.

Methods:

- **`create_example`**
  –

  Create an example in the collection with output and input type validation.
- **`from_polars`**
  –

  Create collection from a Polars DataFrame. Must have an 'output' column and at least one input column.

Source code in `src/fenic/core/types/semantic_examples.py`

```
def __init__(self, examples: List[MapExample] = None):
    """Initialize a collection of semantic map examples.

    Args:
        examples: List of examples to add to the collection. Each example
            will be processed through create_example() to ensure proper formatting
            and validation.
    """
    self._type_validator = _ExampleTypeValidator()
    super().__init__(examples)
```

### create_example

```
create_example(example: MapExample) -> MapExampleCollection
```

Create an example in the collection with output and input type validation.

Ensures all examples in the collection have consistent output types
(either all strings or all BaseModel instances) and validates that input
fields have consistent types across examples.

For input validation:
- The first example establishes the schema and cannot have None values
- Subsequent examples must have the same fields but can have None values
- Non-None values must match the established type for each field

Parameters:

- **`example`**
  (`MapExample`)
  –

  The MapExample to add.

Returns:

- `MapExampleCollection`
  –

  Self for method chaining.

Raises:

- `InvalidExampleCollectionError`
  –

  If the example output type doesn't match
  the existing examples in the collection, if the first example contains
  None values, or if subsequent examples have type mismatches.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### from_polars

```
from_polars(df: DataFrame) -> MapExampleCollection
```

Create collection from a Polars DataFrame. Must have an 'output' column and at least one input column.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

## PredicateExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.semantic_examples.PredicateExample[PredicateExample]

              click fenic.core.types.semantic_examples.PredicateExample href "" "fenic.core.types.semantic_examples.PredicateExample"
```

A single semantic example for semantic predicate operations.

Predicate examples demonstrate the evaluation of input variables against a specific condition,
used in a semantic.predicate operation.

## PredicateExampleCollection

```
PredicateExampleCollection(examples: List[PredicateExample] = None)
```

Bases: `BaseExampleCollection[PredicateExample]`

```
              flowchart TD
              fenic.core.types.semantic_examples.PredicateExampleCollection[PredicateExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.semantic_examples.PredicateExampleCollection

              click fenic.core.types.semantic_examples.PredicateExampleCollection href "" "fenic.core.types.semantic_examples.PredicateExampleCollection"
              click fenic.core.types.semantic_examples.BaseExampleCollection href "" "fenic.core.types.semantic_examples.BaseExampleCollection"
```

Collection of input-to-boolean examples for predicate operations.

Stores examples showing which inputs should evaluate to True or False
based on some condition. Each example contains input fields and a
boolean output indicating whether the condition holds.

Initialize a collection of semantic predicate examples.

Parameters:

- **`examples`**
  (`List[PredicateExample]`, default:
  `None`
  )
  –

  List of examples to add to the collection. Each example
  will be processed through create_example() to ensure proper formatting
  and validation.

Methods:

- **`create_example`**
  –

  Create an example in the collection with input type validation.
- **`from_polars`**
  –

  Create collection from a Polars DataFrame.

Source code in `src/fenic/core/types/semantic_examples.py`

```
def __init__(self, examples: List[PredicateExample] = None):
    """Initialize a collection of semantic predicate examples.

    Args:
        examples: List of examples to add to the collection. Each example
            will be processed through create_example() to ensure proper formatting
            and validation.
    """
    self._type_validator = _ExampleTypeValidator()
    super().__init__(examples)
```

### create_example

```
create_example(example: PredicateExample) -> PredicateExampleCollection
```

Create an example in the collection with input type validation.

Validates that input fields have consistent types across examples.
The first example establishes the schema and cannot have None values.
Subsequent examples must have the same fields but can have None values.

Parameters:

- **`example`**
  (`PredicateExample`)
  –

  The PredicateExample to add.

Returns:

- `PredicateExampleCollection`
  –

  Self for method chaining.

Raises:

- `InvalidExampleCollectionError`
  –

  If the example type is wrong, if the
  first example contains None values, or if subsequent examples
  have type mismatches.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```

### from_polars

```
from_polars(df: DataFrame) -> PredicateExampleCollection
```

Create collection from a Polars DataFrame.

Source code in `src/fenic/core/types/semantic_examples.py`

```
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
```
