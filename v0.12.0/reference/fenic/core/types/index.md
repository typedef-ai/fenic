# fenic.core.types

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/types/

Schema module for defining and manipulating DataFrame schemas.

Classes:

- **`ArrayType`**
  –

  A type representing a homogeneous variable-length array (list) of elements.
- **`ClassDefinition`**
  –

  Definition of a classification class with optional description.
- **`ClassifyExample`**
  –

  A single semantic example for classification operations.
- **`ClassifyExampleCollection`**
  –

  Collection of text-to-category examples for classification operations.
- **`ColumnField`**
  –

  Represents a typed column in a DataFrame schema.
- **`DataType`**
  –

  Base class for all data types.
- **`DatasetMetadata`**
  –

  Metadata for a dataset (table or view).
- **`DocumentPathType`**
  –

  Represents a string containing a a document's local (file system) or remote (URL) path.
- **`EmbeddingType`**
  –

  A type representing a fixed-length embedding vector.
- **`JoinExample`**
  –

  A single semantic example for semantic join operations.
- **`JoinExampleCollection`**
  –

  Collection of comparison examples for semantic join operations.
- **`KeyPoints`**
  –

  Summary as a concise bulleted list.
- **`MapExample`**
  –

  A single semantic example for semantic mapping operations.
- **`MapExampleCollection`**
  –

  Collection of input-output examples for semantic map operations.
- **`Paragraph`**
  –

  Summary as a cohesive narrative.
- **`PredicateExample`**
  –

  A single semantic example for semantic predicate operations.
- **`PredicateExampleCollection`**
  –

  Collection of input-to-boolean examples for predicate operations.
- **`QueryResult`**
  –

  Container for query execution results and associated metadata.
- **`Schema`**
  –

  Represents the schema of a DataFrame.
- **`StructField`**
  –

  A field in a StructType. Fields are nullable.
- **`StructType`**
  –

  A type representing a struct (record) with named fields.
- **`TranscriptType`**
  –

  Represents a string containing a transcript in a specific format.

Attributes:

- **`BooleanType`**
  –

  Represents a boolean value. (True/False)
- **`BranchSide`**
  –

  Type alias representing the side of a branch in a lineage graph.
- **`DataCollection`**
  –

  Type alias representing provider data collection policies.
- **`DataLike`**
  –

  Union type representing any supported data format for both input and output operations.
- **`DataLikeType`**
  –

  String literal type for specifying data output formats.
- **`DateType`**
  –

  Represents a date value.
- **`DoubleType`**
  –

  Represents a 64-bit floating-point number.
- **`FloatType`**
  –

  Represents a 32-bit floating-point number.
- **`FuzzySimilarityMethod`**
  –

  Type alias representing the supported fuzzy string similarity algorithms.
- **`HtmlType`**
  –

  Represents a string containing raw HTML markup.
- **`IntegerType`**
  –

  Represents a signed integer value.
- **`JsonType`**
  –

  Represents a string containing JSON data.
- **`MarkdownType`**
  –

  Represents a string containing Markdown-formatted text.
- **`ModelQuantization`**
  –

  Type alias representing supported quantization formats for provider models.
- **`ProviderSort`**
  –

  Type alias representing provider sorting strategies used by OpenRouter routing.
- **`SemanticSimilarityMetric`**
  –

  Type alias representing supported semantic similarity metrics.
- **`StringType`**
  –

  Represents a UTF-8 encoded string value.
- **`StructuredOutputStrategy`**
  –

  Type alias representing the strategy to use when a model supports both
- **`TimestampType`**
  –

  Represents a timestamp value.

## BooleanType

```
BooleanType = _BooleanType()
```

Represents a boolean value. (True/False)

## BranchSide

```
BranchSide = Literal['left', 'right']
```

Type alias representing the side of a branch in a lineage graph.

Valid values:

- "left": The left branch of a join.
- "right": The right branch of a join.

## DataCollection

```
DataCollection = Literal['allow', 'deny']
```

Type alias representing provider data collection policies.

Valid values:

- "allow": Permit providers that may retain or train on prompts non-transiently.
- "deny": Restrict to providers that do not collect/store user data.

## DataLike

```
DataLike = Union[pl.DataFrame, pd.DataFrame, Dict[str, List[Any]], List[Dict[str, Any]], pa.Table]
```

Union type representing any supported data format for both input and output operations.

This type encompasses all possible data structures that can be:
1. Used as input when creating DataFrames
2. Returned as output from query results

Supported formats

- pl.DataFrame: Native Polars DataFrame with efficient columnar storage
- pd.DataFrame: Pandas DataFrame, optionally with PyArrow extension arrays
- Dict[str, List[Any]]: Column-oriented dictionary where:
  - Keys are column names (str)
  - Values are lists containing all values for that column
- List[Dict[str, Any]]: Row-oriented list where:
  - Each element is a dictionary representing one row
  - Dictionary keys are column names, values are cell values
- pa.Table: Apache Arrow Table with columnar memory layout

Usage

- Input: Used in create_dataframe() to accept data in various formats
- Output: Used in QueryResult.data to return results in requested format

The specific type returned depends on the DataLikeType format specified
when collecting query results.

## DataLikeType

```
DataLikeType = Literal['polars', 'pandas', 'pydict', 'pylist', 'arrow']
```

String literal type for specifying data output formats.

Valid values

- "polars": Native Polars DataFrame format
- "pandas": Pandas DataFrame with PyArrow extension arrays
- "pydict": Python dictionary with column names as keys, lists as values
- "pylist": Python list of dictionaries, each representing one row
- "arrow": Apache Arrow Table format

Used as input parameter for methods that can return data in multiple formats.

## DateType

```
DateType = _DateType()
```

Represents a date value.

## DoubleType

```
DoubleType = _DoubleType()
```

Represents a 64-bit floating-point number.

## FloatType

```
FloatType = _FloatType()
```

Represents a 32-bit floating-point number.

## FuzzySimilarityMethod

```
FuzzySimilarityMethod = Literal['indel', 'levenshtein', 'damerau_levenshtein', 'jaro_winkler', 'jaro', 'hamming']
```

Type alias representing the supported fuzzy string similarity algorithms.

These algorithms quantify the similarity or difference between two strings using various distance or similarity metrics:

- "indel":
  Computes the Indel (Insertion-Deletion) distance, which counts only insertions and deletions needed to transform one string into another, excluding substitutions. This is equivalent to the Longest Common Subsequence (LCS) problem. Useful when character substitutions should not be considered as valid operations (e.g., DNA sequence alignment where only insertions/deletions occur).
- "levenshtein":
  Computes the Levenshtein distance, which is the minimum number of single-character edits (insertions, deletions, or substitutions) required to transform one string into another. Suitable for general-purpose fuzzy matching where transpositions do not matter.
- "damerau_levenshtein":
  An extension of Levenshtein distance that also accounts for transpositions of adjacent characters (e.g., "ab" → "ba"). This metric is more accurate for real-world typos and keyboard errors.
- "jaro":
  Measures similarity based on the number and order of common characters between two strings. It is particularly effective for short strings such as names. Returns a normalized score between 0 (no similarity) and 1 (exact match).
- "jaro_winkler":
  A variant of the Jaro distance that gives more weight to common prefixes. Designed to improve accuracy on strings with shared beginnings (e.g., first names, surnames).
- "hamming":
  Measures the number of differing characters between two strings of equal length. Only valid when both strings are the same length. It does not support insertions or deletions—only substitutions.

Choose the method based on the type of expected variation (e.g., typos, transpositions, or structural changes).

## HtmlType

```
HtmlType = _HtmlType()
```

Represents a string containing raw HTML markup.

## IntegerType

```
IntegerType = _IntegerType()
```

Represents a signed integer value.

## JsonType

```
JsonType = _JsonType()
```

Represents a string containing JSON data.

## MarkdownType

```
MarkdownType = _MarkdownType()
```

Represents a string containing Markdown-formatted text.

## ModelQuantization

```
ModelQuantization = Literal['int4', 'int8', 'fp4', 'fp6', 'fp8', 'fp16', 'bf16', 'fp32', 'unknown']
```

Type alias representing supported quantization formats for provider models.

Common values:

- "int4", "int8": Integer quantization for smaller, faster models.
- "fp4", "fp6", "fp8": Low-precision floating point formats.
- "fp16", "bf16": Half-precision formats commonly used on GPUs/TPUs.
- "fp32": Full precision floating point.
- "unknown": Provider did not specify a quantization.

## ProviderSort

```
ProviderSort = Literal['price', 'throughput', 'latency']
```

Type alias representing provider sorting strategies used by OpenRouter routing.

Valid values:

- "price": Prefer providers with the lowest recent price.
- "throughput": Prefer providers with the highest recent throughput.
- "latency": Prefer providers with the lowest recent latency.

## SemanticSimilarityMetric

```
SemanticSimilarityMetric = Literal['cosine', 'l2', 'dot']
```

Type alias representing supported semantic similarity metrics.

Valid values:

- "cosine": Cosine similarity, measures the cosine of the angle between two vectors.
- "l2": Euclidean (L2) distance, measures the straight-line distance between two vectors.
- "dot": Dot product similarity, the raw inner product of two vectors.

These metrics are commonly used for comparing embedding vectors in semantic search
and other similarity-based applications.

## StringType

```
StringType = _StringType()
```

Represents a UTF-8 encoded string value.

## StructuredOutputStrategy

```
StructuredOutputStrategy = Literal['prefer_tools', 'prefer_response_format']
```

Type alias representing the strategy to use when a model supports both
tool-calling and response-format-based structured outputs.

Valid values:

- "prefer_tools": Prefer tool/function calling with a JSON schema.
- "prefer_response_format": Prefer response_format structured outputs.

## TimestampType

```
TimestampType = _TimestampType()
```

Represents a timestamp value.

## ArrayType

Bases: `DataType`

```
              flowchart TD
              fenic.core.types.ArrayType[ArrayType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.core.types.ArrayType

              click fenic.core.types.ArrayType href "" "fenic.core.types.ArrayType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

A type representing a homogeneous variable-length array (list) of elements.

Attributes:

- **`element_type`**
  (`DataType`)
  –

  The data type of each element in the array.

Create an array of strings

```
ArrayType(StringType)
ArrayType(element_type=StringType)
```

## ClassDefinition

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.ClassDefinition[ClassDefinition]

              click fenic.core.types.ClassDefinition href "" "fenic.core.types.ClassDefinition"
```

Definition of a classification class with optional description.

Used to define the available classes for semantic classification operations.
The description helps the LLM understand what each class represents.

## ClassifyExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.ClassifyExample[ClassifyExample]

              click fenic.core.types.ClassifyExample href "" "fenic.core.types.ClassifyExample"
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
              fenic.core.types.ClassifyExampleCollection[ClassifyExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.ClassifyExampleCollection

              click fenic.core.types.ClassifyExampleCollection href "" "fenic.core.types.ClassifyExampleCollection"
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

## ColumnField

Represents a typed column in a DataFrame schema.

A ColumnField defines the structure of a single column by specifying its name
and data type. This is used as a building block for DataFrame schemas.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the column.
- **`data_type`**
  (`DataType`)
  –

  The data type of the column, as a DataType instance.

## DataType

Bases: `ABC`

```
              flowchart TD
              fenic.core.types.DataType[DataType]

              click fenic.core.types.DataType href "" "fenic.core.types.DataType"
```

Base class for all data types.

You won't instantiate this class directly. Instead, use one of the
concrete types like `StringType`, `ArrayType`, or `StructType`.

Used for casting, type validation, and schema inference in the DataFrame API.

## DatasetMetadata

Metadata for a dataset (table or view).

Attributes:

- **`schema`**
  (`Schema`)
  –

  The schema of the dataset.
- **`description`**
  (`Optional[str]`)
  –

  The natural language description of the dataset's contents.

## DocumentPathType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.types.DocumentPathType[DocumentPathType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.types.DocumentPathType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.types.DocumentPathType href "" "fenic.core.types.DocumentPathType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a a document's local (file system) or remote (URL) path.

## EmbeddingType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.types.EmbeddingType[EmbeddingType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.types.EmbeddingType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.types.EmbeddingType href "" "fenic.core.types.EmbeddingType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

A type representing a fixed-length embedding vector.

Attributes:

- **`dimensions`**
  (`int`)
  –

  The number of dimensions in the embedding vector.
- **`embedding_model`**
  (`str`)
  –

  Name of the model used to generate the embedding.

Create an embedding type for text-embedding-3-small

```
EmbeddingType(384, embedding_model="text-embedding-3-small")
```

## JoinExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.JoinExample[JoinExample]

              click fenic.core.types.JoinExample href "" "fenic.core.types.JoinExample"
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
              fenic.core.types.JoinExampleCollection[JoinExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.JoinExampleCollection

              click fenic.core.types.JoinExampleCollection href "" "fenic.core.types.JoinExampleCollection"
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

## KeyPoints

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.KeyPoints[KeyPoints]

              click fenic.core.types.KeyPoints href "" "fenic.core.types.KeyPoints"
```

Summary as a concise bulleted list.

Each bullet should capture a distinct and essential idea, with a maximum number of points specified.

Attributes:

- **`max_points`**
  (`int`)
  –

  The maximum number of key points to include in the summary.

Methods:

- **`max_tokens`**
  –

  Calculate the maximum number of tokens for the summary based on the number of key points.

### max_tokens

```
max_tokens() -> int
```

Calculate the maximum number of tokens for the summary based on the number of key points.

Source code in `src/fenic/core/types/summarize.py`

```
def max_tokens(self) -> int:
    """Calculate the maximum number of tokens for the summary based on the number of key points."""
    return self.max_points * 75
```

## MapExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.MapExample[MapExample]

              click fenic.core.types.MapExample href "" "fenic.core.types.MapExample"
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
              fenic.core.types.MapExampleCollection[MapExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.MapExampleCollection

              click fenic.core.types.MapExampleCollection href "" "fenic.core.types.MapExampleCollection"
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

## Paragraph

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.Paragraph[Paragraph]

              click fenic.core.types.Paragraph href "" "fenic.core.types.Paragraph"
```

Summary as a cohesive narrative.

The summary should flow naturally and not exceed a specified maximum word count.

Attributes:

- **`max_words`**
  (`int`)
  –

  The maximum number of words allowed in the summary.

Methods:

- **`max_tokens`**
  –

  Calculate the maximum number of tokens for the summary based on the number of words.

### max_tokens

```
max_tokens() -> int
```

Calculate the maximum number of tokens for the summary based on the number of words.

Source code in `src/fenic/core/types/summarize.py`

```
def max_tokens(self) -> int:
    """Calculate the maximum number of tokens for the summary based on the number of words."""
    return int(self.max_words * 1.5)
```

## PredicateExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.types.PredicateExample[PredicateExample]

              click fenic.core.types.PredicateExample href "" "fenic.core.types.PredicateExample"
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
              fenic.core.types.PredicateExampleCollection[PredicateExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.types.PredicateExampleCollection

              click fenic.core.types.PredicateExampleCollection href "" "fenic.core.types.PredicateExampleCollection"
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

## QueryResult

```
QueryResult(data: DataLike, metrics: QueryMetrics)
```

Container for query execution results and associated metadata.

This dataclass bundles together the materialized data from a query execution
along with metrics about the execution process. It provides a unified interface
for accessing both the computed results and performance information.

Attributes:

- **`data`**
  (`DataLike`)
  –

  The materialized query results in the requested format.
  Can be any of the supported data types (Polars/Pandas DataFrame,
  Arrow Table, or Python dict/list structures).
- **`metrics`**
  (`QueryMetrics`)
  –

  Execution metadata including timing information,
  memory usage, rows processed, and other performance metrics collected
  during query execution.

Access query results and metrics

```
# Execute query and get results with metrics
result = df.filter(col("age") > 25).collect("pandas")
pandas_df = result.data  # Access the Pandas DataFrame
print(result.metrics.execution_time)  # Access execution metrics
print(result.metrics.rows_processed)  # Access row count
```

Work with different data formats

```
# Get results in different formats
polars_result = df.collect("polars")
arrow_result = df.collect("arrow")
dict_result = df.collect("pydict")

# All contain the same data, different formats
print(type(polars_result.data))  # <class 'polars.DataFrame'>
print(type(arrow_result.data))   # <class 'pyarrow.lib.Table'>
print(type(dict_result.data))    # <class 'dict'>
```

Note

The actual type of the `data` attribute depends on the format requested
during collection. Use type checking or isinstance() if you need to
handle the data differently based on its format.

## Schema

Represents the schema of a DataFrame.

A Schema defines the structure of a DataFrame by specifying an ordered collection
of column fields. Each column field defines the name and data type of a column
in the DataFrame.

Attributes:

- **`column_fields`**
  (`List[ColumnField]`)
  –

  An ordered list of ColumnField objects that define the
  structure of the DataFrame.

Methods:

- **`column_names`**
  –

  Get a list of all column names in the schema.

### column_names

```
column_names() -> List[str]
```

Get a list of all column names in the schema.

Returns:

- `List[str]`
  –

  A list of strings containing the names of all columns in the schema.

Source code in `src/fenic/core/types/schema.py`

```
def column_names(self) -> List[str]:
    """Get a list of all column names in the schema.

    Returns:
        A list of strings containing the names of all columns in the schema.
    """
    return [field.name for field in self.column_fields]
```

## StructField

A field in a StructType. Fields are nullable.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the field.
- **`data_type`**
  (`DataType`)
  –

  The data type of the field.

## StructType

Bases: `DataType`

```
              flowchart TD
              fenic.core.types.StructType[StructType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.core.types.StructType

              click fenic.core.types.StructType href "" "fenic.core.types.StructType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

A type representing a struct (record) with named fields.

Attributes:

- **`fields`**
  –

  List of field definitions.

Create a struct with name and age fields

```
StructType([
    StructField("name", StringType),
    StructField("age", IntegerType),
])
```

## TranscriptType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.types.TranscriptType[TranscriptType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.types.TranscriptType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.types.TranscriptType href "" "fenic.core.types.TranscriptType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a transcript in a specific format.
