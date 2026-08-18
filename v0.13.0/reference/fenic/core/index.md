# fenic.core

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/

Core module for Fenic.

Classes:

- **`ArrayType`**
  –

  A type representing a homogeneous variable-length array (list) of elements.
- **`BoundToolParam`**
  –

  A bound tool parameter.
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
- **`LMMetrics`**
  –

  Tracks language model usage metrics including token counts and costs.
- **`MapExample`**
  –

  A single semantic example for semantic mapping operations.
- **`MapExampleCollection`**
  –

  Collection of input-output examples for semantic map operations.
- **`OperatorMetrics`**
  –

  Metrics for a single operator in the query execution plan.
- **`Paragraph`**
  –

  Summary as a cohesive narrative.
- **`PredicateExample`**
  –

  A single semantic example for semantic predicate operations.
- **`PredicateExampleCollection`**
  –

  Collection of input-to-boolean examples for predicate operations.
- **`QueryMetrics`**
  –

  Comprehensive metrics for an executed query.
- **`QueryResult`**
  –

  Container for query execution results and associated metadata.
- **`RMMetrics`**
  –

  Tracks embedding model usage metrics including token counts and costs.
- **`Schema`**
  –

  Represents the schema of a DataFrame.
- **`StructField`**
  –

  A field in a StructType. Fields are nullable.
- **`StructType`**
  –

  A type representing a struct (record) with named fields.
- **`SystemTool`**
  –

  A tool implemented as a regular Python function with explicit parameters.
- **`ToolParam`**
  –

  A parameter for a parameterized view tool.
- **`TranscriptType`**
  –

  Represents a string containing a transcript in a specific format.
- **`UserDefinedTool`**
  –

  A tool that has been bound to a specific Parameterized View.

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
              fenic.core.ArrayType[ArrayType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.core.ArrayType

              click fenic.core.ArrayType href "" "fenic.core.ArrayType"
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

## BoundToolParam

A bound tool parameter.

A bound tool parameter is a parameter that has been bound to a specific, typed,
`tool_param` usage within a Dataframe.

## ClassDefinition

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.ClassDefinition[ClassDefinition]

              click fenic.core.ClassDefinition href "" "fenic.core.ClassDefinition"
```

Definition of a classification class with optional description.

Used to define the available classes for semantic classification operations.
The description helps the LLM understand what each class represents.

## ClassifyExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.ClassifyExample[ClassifyExample]

              click fenic.core.ClassifyExample href "" "fenic.core.ClassifyExample"
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
              fenic.core.ClassifyExampleCollection[ClassifyExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.ClassifyExampleCollection

              click fenic.core.ClassifyExampleCollection href "" "fenic.core.ClassifyExampleCollection"
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
              fenic.core.DataType[DataType]

              click fenic.core.DataType href "" "fenic.core.DataType"
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
              fenic.core.DocumentPathType[DocumentPathType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.DocumentPathType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.DocumentPathType href "" "fenic.core.DocumentPathType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a a document's local (file system) or remote (URL) path.

## EmbeddingType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.EmbeddingType[EmbeddingType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.EmbeddingType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.EmbeddingType href "" "fenic.core.EmbeddingType"
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
              fenic.core.JoinExample[JoinExample]

              click fenic.core.JoinExample href "" "fenic.core.JoinExample"
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
              fenic.core.JoinExampleCollection[JoinExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.JoinExampleCollection

              click fenic.core.JoinExampleCollection href "" "fenic.core.JoinExampleCollection"
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
              fenic.core.KeyPoints[KeyPoints]

              click fenic.core.KeyPoints href "" "fenic.core.KeyPoints"
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

## LMMetrics

```
LMMetrics(num_uncached_input_tokens: int = 0, num_cached_input_tokens: int = 0, num_output_tokens: int = 0, cost: float = 0.0, num_requests: int = 0, num_reserved_output_tokens: int = 0)
```

Tracks language model usage metrics including token counts and costs.

Attributes:

- **`num_uncached_input_tokens`**
  (`int`)
  –

  Number of uncached tokens in the prompt/input.
- **`num_cached_input_tokens`**
  (`int`)
  –

  Number of cached tokens in the prompt/input.
- **`num_output_tokens`**
  (`int`)
  –

  Number of tokens in the completion/output (actual usage).
- **`cost`**
  (`float`)
  –

  Total cost in USD for the LM API call.
- **`num_requests`**
  (`int`)
  –

  Total number of LM API requests made.
- **`num_reserved_output_tokens`**
  (`int`)
  –

  Output tokens debited from the TPM bucket at
  reservation time. Compare against num_output_tokens to measure
  reservation efficiency (actual / reserved → 1 is tight).

## MapExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.MapExample[MapExample]

              click fenic.core.MapExample href "" "fenic.core.MapExample"
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
              fenic.core.MapExampleCollection[MapExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.MapExampleCollection

              click fenic.core.MapExampleCollection href "" "fenic.core.MapExampleCollection"
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

## OperatorMetrics

```
OperatorMetrics(operator_id: str, num_output_rows: int = 0, execution_time_ms: float = 0.0, lm_metrics: LMMetrics = LMMetrics(), rm_metrics: RMMetrics = RMMetrics())
```

Metrics for a single operator in the query execution plan.

Attributes:

- **`operator_id`**
  (`str`)
  –

  Unique identifier for the operator
- **`num_output_rows`**
  (`int`)
  –

  Number of rows output by this operator
- **`execution_time_ms`**
  (`float`)
  –

  Execution time in milliseconds
- **`lm_metrics`**
  (`LMMetrics`)
  –

  Language model usage metrics for this operator

## Paragraph

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.Paragraph[Paragraph]

              click fenic.core.Paragraph href "" "fenic.core.Paragraph"
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
              fenic.core.PredicateExample[PredicateExample]

              click fenic.core.PredicateExample href "" "fenic.core.PredicateExample"
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
              fenic.core.PredicateExampleCollection[PredicateExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.core.PredicateExampleCollection

              click fenic.core.PredicateExampleCollection href "" "fenic.core.PredicateExampleCollection"
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

## QueryMetrics

```
QueryMetrics(execution_id: str, session_id: str, execution_time_ms: float = 0.0, num_output_rows: int = 0, total_lm_metrics: LMMetrics = LMMetrics(), total_rm_metrics: RMMetrics = RMMetrics(), end_ts: datetime = datetime.now(), _operator_metrics: Dict[str, OperatorMetrics] = dict(), _plan_repr: PhysicalPlanRepr = (lambda: PhysicalPlanRepr(operator_id='empty'))())
```

Comprehensive metrics for an executed query.

Includes overall statistics and detailed metrics for each operator
in the execution plan.

Attributes:

- **`execution_id`**
  (`str`)
  –

  Unique identifier for this query execution
- **`session_id`**
  (`str`)
  –

  Identifier for the session this query belongs to
- **`execution_time_ms`**
  (`float`)
  –

  Total query execution time in milliseconds
- **`num_output_rows`**
  (`int`)
  –

  Total number of rows returned by the query
- **`total_lm_metrics`**
  (`LMMetrics`)
  –

  Aggregated language model metrics across all operators
- **`end_ts`**
  (`datetime`)
  –

  Timestamp when query execution completed

Methods:

- **`get_execution_plan_details`**
  –

  Generate a formatted execution plan with detailed metrics.
- **`get_summary`**
  –

  Summarize the query metrics in a single line.
- **`to_dict`**
  –

  Convert QueryMetrics to a dictionary for table storage.

### start_ts

```
start_ts: datetime
```

Calculate start timestamp from end timestamp and execution time.

### get_execution_plan_details

```
get_execution_plan_details() -> str
```

Generate a formatted execution plan with detailed metrics.

Produces a hierarchical representation of the query execution plan,
including performance metrics and language model usage for each operator.

Returns:

- **`str`** ( `str`
  ) –

  A formatted string showing the execution plan with metrics.

Source code in `src/fenic/core/metrics.py`

```
def get_execution_plan_details(self) -> str:
    """Generate a formatted execution plan with detailed metrics.

    Produces a hierarchical representation of the query execution plan,
    including performance metrics and language model usage for each operator.

    Returns:
        str: A formatted string showing the execution plan with metrics.
    """

    def _format_node(node: PhysicalPlanRepr, indent: int = 1) -> str:
        op = self._operator_metrics[node.operator_id]
        indent_str = "  " * indent

        details = [
            f"{indent_str}{op.operator_id}",
            f"{indent_str}  Output Rows: {op.num_output_rows:,}",
            f"{indent_str}  Execution Time: {op.execution_time_ms:.2f}ms",
        ]

        if op.lm_metrics.cost > 0:
            details.extend(
                [
                    f"{indent_str}  Language Model Usage: {op.lm_metrics.num_uncached_input_tokens:,} input tokens, {op.lm_metrics.num_cached_input_tokens:,} cached input tokens, {op.lm_metrics.num_output_tokens:,} output tokens ({op.lm_metrics.num_reserved_output_tokens:,} reserved)",
                    f"{indent_str}  Language Model Cost: ${op.lm_metrics.cost:.6f}",
                ]
            )

        if op.rm_metrics.cost > 0:
            details.extend(
                [
                    f"{indent_str}  Embedding Model Usage: {op.rm_metrics.num_input_tokens:,} input tokens",
                    f"{indent_str}  Embedding Model Cost: ${op.rm_metrics.cost:.6f}",
                ]
            )
        return (
            "\n".join(details)
            + "\n"
            + "".join(_format_node(child, indent + 1) for child in node.children)
        )

    return f"Execution Plan\n{_format_node(self._plan_repr)}"
```

### get_summary

```
get_summary() -> str
```

Summarize the query metrics in a single line.

Returns:

- **`str`** ( `str`
  ) –

  A concise summary of execution time, row count, and LM cost.

Source code in `src/fenic/core/metrics.py`

```
def get_summary(self) -> str:
    """Summarize the query metrics in a single line.

    Returns:
        str: A concise summary of execution time, row count, and LM cost.
    """
    return (
        f"Query executed in {self.execution_time_ms:.2f}ms, "
        f"returned {self.num_output_rows:,} rows, "
        f"language model cost: ${self.total_lm_metrics.cost:.6f}, "
        f"embedding model cost: ${self.total_rm_metrics.cost:.6f}"
    )
```

### to_dict

```
to_dict() -> Dict[str, Any]
```

Convert QueryMetrics to a dictionary for table storage.

Returns:

- `Dict[str, Any]`
  –

  Dict containing all metrics fields suitable for database storage.

Source code in `src/fenic/core/metrics.py`

```
def to_dict(self) -> Dict[str, Any]:
    """Convert QueryMetrics to a dictionary for table storage.

    Returns:
        Dict containing all metrics fields suitable for database storage.
    """
    return {
        "execution_id": self.execution_id,
        "session_id": self.session_id,
        "execution_time_ms": self.execution_time_ms,
        "num_output_rows": self.num_output_rows,
        "start_ts": self.start_ts,
        "end_ts": self.end_ts,
        "total_lm_cost": self.total_lm_metrics.cost,
        "total_lm_uncached_input_tokens": self.total_lm_metrics.num_uncached_input_tokens,
        "total_lm_cached_input_tokens": self.total_lm_metrics.num_cached_input_tokens,
        "total_lm_output_tokens": self.total_lm_metrics.num_output_tokens,
        "total_lm_requests": self.total_lm_metrics.num_requests,
        "total_rm_cost": self.total_rm_metrics.cost,
        "total_rm_input_tokens": self.total_rm_metrics.num_input_tokens,
        "total_rm_requests": self.total_rm_metrics.num_requests,
    }
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

## RMMetrics

```
RMMetrics(num_input_tokens: int = 0, num_requests: int = 0, cost: float = 0.0)
```

Tracks embedding model usage metrics including token counts and costs.

Attributes:

- **`num_input_tokens`**
  (`int`)
  –

  Number of tokens to embed
- **`cost`**
  (`float`)
  –

  Total cost in USD to embed the tokens

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
              fenic.core.StructType[StructType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.core.StructType

              click fenic.core.StructType href "" "fenic.core.StructType"
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

## SystemTool

A tool implemented as a regular Python function with explicit parameters.

The function must be a `Callable[..., LogicalPlan]`
(a function defined with `async def`). Collection/formatting is handled by
the MCP generator wrapper.

## ToolParam

Bases: `BaseModel`

```
              flowchart TD
              fenic.core.ToolParam[ToolParam]

              click fenic.core.ToolParam href "" "fenic.core.ToolParam"
```

A parameter for a parameterized view tool.

A parameter is a named value that can be passed to a tool. These are matched to the
parameter names of the `tool_param` UnresolvedLiteralExpr expressions captured in the Logical Plan.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the parameter.
- **`description`**
  (`str`)
  –

  The description of the parameter.
- **`allowed_values`**
  (`Optional[List[ToolParameterType]]`)
  –

  The allowed values for the parameter.
- **`has_default`**
  (`bool`)
  –

  Whether the parameter has a default value.
- **`default_value`**
  (`Optional[ToolParameterType]`)
  –

  The default value for the parameter.

### required

```
required: bool
```

Whether the parameter is required.

Returns:

- `bool`
  –

  True if the parameter is required, False otherwise.

## TranscriptType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.core.TranscriptType[TranscriptType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.core.TranscriptType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.core.TranscriptType href "" "fenic.core.TranscriptType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a transcript in a specific format.

## UserDefinedTool

A tool that has been bound to a specific Parameterized View.
