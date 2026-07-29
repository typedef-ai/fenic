# fenic

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/

Fenic is an opinionated, PySpark-inspired DataFrame framework for building production AI and agentic applications.

Classes:

- **`AdaptiveTokenEstimationConfig`**
  –

  Tunes adaptive output-token reservation for rate limiting.
- **`AnthropicLanguageModel`**
  –

  Configuration for Anthropic language models.
- **`ArrayType`**
  –

  A type representing a homogeneous variable-length array (list) of elements.
- **`BoundToolParam`**
  –

  A bound tool parameter.
- **`Catalog`**
  –

  Entry point for catalog operations.
- **`ClassDefinition`**
  –

  Definition of a classification class with optional description.
- **`ClassifyExample`**
  –

  A single semantic example for classification operations.
- **`ClassifyExampleCollection`**
  –

  Collection of text-to-category examples for classification operations.
- **`CloudConfig`**
  –

  Configuration for cloud-based execution.
- **`CohereEmbeddingModel`**
  –

  Configuration for Cohere embedding models.
- **`Column`**
  –

  A column expression in a DataFrame.
- **`ColumnField`**
  –

  Represents a typed column in a DataFrame schema.
- **`DataFrame`**
  –

  A data collection organized into named columns.
- **`DataFrameReader`**
  –

  Interface used to load a DataFrame from external storage systems.
- **`DataFrameWriter`**
  –

  Interface used to write a DataFrame to external storage systems.
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
- **`GoogleDeveloperEmbeddingModel`**
  –

  Configuration for Google Developer embedding models.
- **`GoogleDeveloperLanguageModel`**
  –

  Configuration for Gemini models accessible through Google Developer AI Studio.
- **`GoogleVertexEmbeddingModel`**
  –

  Configuration for Google Vertex AI embedding models.
- **`GoogleVertexLanguageModel`**
  –

  Configuration for Google Vertex AI models.
- **`GroupedData`**
  –

  Methods for aggregations on a grouped DataFrame.
- **`InvalidExampleCollectionError`**
  –

  Exception raised when a semantic example collection is invalid.
- **`JoinExample`**
  –

  A single semantic example for semantic join operations.
- **`JoinExampleCollection`**
  –

  Collection of comparison examples for semantic join operations.
- **`KeyPoints`**
  –

  Summary as a concise bulleted list.
- **`LLMResponseCacheConfig`**
  –

  Configuration for LLM response caching.
- **`LMMetrics`**
  –

  Tracks language model usage metrics including token counts and costs.
- **`Lineage`**
  –

  Query interface for tracing data lineage through a query plan.
- **`MapExample`**
  –

  A single semantic example for semantic mapping operations.
- **`MapExampleCollection`**
  –

  Collection of input-output examples for semantic map operations.
- **`ModelAlias`**
  –

  A combination of a model name and a required profile for that model.
- **`OpenAIEmbeddingModel`**
  –

  Configuration for OpenAI embedding models.
- **`OpenAILanguageModel`**
  –

  Configuration for OpenAI language models.
- **`OpenRouterLanguageModel`**
  –

  Configuration for OpenRouter language models.
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
- **`SemanticConfig`**
  –

  Configuration for semantic language and embedding models.
- **`SemanticExtensions`**
  –

  A namespace for semantic dataframe operators.
- **`Session`**
  –

  The entry point to programming with the DataFrame API. Similar to PySpark's SparkSession.
- **`SessionConfig`**
  –

  Configuration for a user session.
- **`StructField`**
  –

  A field in a StructType. Fields are nullable.
- **`StructType`**
  –

  A type representing a struct (record) with named fields.
- **`SystemTool`**
  –

  A tool implemented as a regular Python function with explicit parameters.
- **`SystemToolConfig`**
  –

  Configuration for canonical system tools.
- **`ToolParam`**
  –

  A parameter for a parameterized view tool.
- **`TranscriptType`**
  –

  Represents a string containing a transcript in a specific format.
- **`UserDefinedTool`**
  –

  A tool that has been bound to a specific Parameterized View.

Functions:

- **`approx_count_distinct`**
  –

  Aggregate function: returns an approximate count (HyperLogLog++) of distinct non-null values.
- **`array_agg`**
  –

  Alias for collect_list().
- **`asc`**
  –

  Mark this column for ascending sort order with nulls first.
- **`asc_nulls_first`**
  –

  Alias for asc().
- **`asc_nulls_last`**
  –

  Mark this column for ascending sort order with nulls last.
- **`async_udf`**
  –

  A decorator for creating async user-defined functions (UDFs) with configurable concurrency and retries.
- **`avg`**
  –

  Aggregate function: returns the average (mean) of all values in the specified column. Applies to numeric and embedding types.
- **`coalesce`**
  –

  Returns the first non-null value from the given columns for each row.
- **`col`**
  –

  Creates a Column expression referencing a column in the DataFrame.
- **`collect_list`**
  –

  Aggregate function: collects all values from the specified column into a list.
- **`configure_logging`**
  –

  Configure logging for the library and root logger in interactive environments.
- **`count`**
  –

  Aggregate function: returns the count of non-null values in the specified column.
- **`count_distinct`**
  –

  Aggregate function: returns the number of distinct non-null rows across one or more columns.
- **`create_mcp_server`**
  –

  Create an MCP server from datasets and tools.
- **`desc`**
  –

  Mark this column for descending sort order with nulls first.
- **`desc_nulls_first`**
  –

  Alias for desc().
- **`desc_nulls_last`**
  –

  Mark this column for descending sort order with nulls last.
- **`empty`**
  –

  Creates a Column expression representing an empty value of the given type.
- **`first`**
  –

  Aggregate function: returns the first non-null value in the specified column.
- **`flatten`**
  –

  Flattens an array of arrays into a single array (one level deep).
- **`greatest`**
  –

  Returns the greatest value from the given columns for each row.
- **`least`**
  –

  Returns the least value from the given columns for each row.
- **`lit`**
  –

  Creates a Column expression representing a literal value.
- **`max`**
  –

  Aggregate function: returns the maximum value in the specified column.
- **`mean`**
  –

  Aggregate function: returns the mean (average) of all values in the specified column.
- **`min`**
  –

  Aggregate function: returns the minimum value in the specified column.
- **`null`**
  –

  Creates a Column expression representing a null value of the specified data type.
- **`run_mcp_server_asgi`**
  –

  Run an MCP server as a Starlette ASGI app.
- **`run_mcp_server_async`**
  –

  Run an MCP server asynchronously.
- **`run_mcp_server_sync`**
  –

  Run an MCP server synchronously.
- **`stddev`**
  –

  Aggregate function: returns the sample standard deviation of the specified column.
- **`struct`**
  –

  Creates a new struct column from multiple input columns.
- **`sum`**
  –

  Aggregate function: returns the sum of all values in the specified column.
- **`sum_distinct`**
  –

  Aggregate function: returns the sum of distinct numeric values in the specified column.
- **`tool_param`**
  –

  Creates an unresolved literal placeholder column with a declared data type.
- **`udf`**
  –

  A decorator or function for creating user-defined functions (UDFs) that can be applied to DataFrame rows.
- **`when`**
  –

  Evaluates a conditional expression (like if-then).

Attributes:

- **`BooleanType`**
  –

  Represents a boolean value. (True/False)
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

## AdaptiveTokenEstimationConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.AdaptiveTokenEstimationConfig[AdaptiveTokenEstimationConfig]

              click fenic.AdaptiveTokenEstimationConfig href "" "fenic.AdaptiveTokenEstimationConfig"
```

Tunes adaptive output-token reservation for rate limiting.

Output-token reservations are learned from observed usage and clamped to the
request's max_completion_tokens ceiling, then corrected after each response
(settlement). Enabled by default.

Setting `enabled=False` disables adaptive *estimation* — reservations fall
back to the static worst-case ceiling instead of the learned distribution.
Settlement (reconciling the token bucket to actual usage after each response)
is **always on** regardless of this flag. It corrects the bucket in both
directions — refunding the over-reservation (the common case) and debiting
further when a request exceeds its reservation — and neither direction increases
429 risk: a refund only returns capacity the provider never charged, and a debit
only makes the limiter more conservative.

## AnthropicLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.AnthropicLanguageModel[AnthropicLanguageModel]

              click fenic.AnthropicLanguageModel href "" "fenic.AnthropicLanguageModel"
```

Configuration for Anthropic language models.

This class defines the configuration settings for Anthropic language models,
including model selection and separate rate limiting parameters for input and output tokens.

Attributes:

- **`model_name`**
  (`AnthropicLanguageModelName`)
  –

  The name of the Anthropic model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`input_tpm`**
  (`int`)
  –

  Input tokens per minute limit; must be greater than 0.
- **`output_tpm`**
  (`int`)
  –

  Output tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring an Anthropic model with separate input/output rate limits:

```
config = AnthropicLanguageModel(
    model_name="claude-haiku-4-5", rpm=100, input_tpm=100, output_tpm=100
)
```

Configuring an Anthropic model with profiles:

```
config = SessionConfig(
    semantic=SemanticConfig(
        language_models={
            "claude": AnthropicLanguageModel(
                model_name="claude-sonnet-4-6",
                rpm=100,
                input_tpm=100,
                output_tpm=100,
                profiles={
                    "thinking_disabled": AnthropicLanguageModel.Profile(),
                    "fast": AnthropicLanguageModel.Profile(thinking_token_budget=1024),
                    "thorough": AnthropicLanguageModel.Profile(thinking_token_budget=4096)
                },
                default_profile="fast"
            )
        },
        default_language_model="claude"
)

# Using the default "fast" profile for the "claude" model
semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias="claude")

# Using the "thorough" profile for the "claude" model
semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias=ModelAlias(name="claude", profile="thorough"))
```

Classes:

- **`Profile`**
  –

  Anthropic-specific profile configurations.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.AnthropicLanguageModel.Profile[Profile]

              click fenic.AnthropicLanguageModel.Profile href "" "fenic.AnthropicLanguageModel.Profile"
```

Anthropic-specific profile configurations.

This class defines profile configurations for Anthropic models, allowing
different thinking and effort settings to be applied to the same model.

Attributes:

- **`thinking_token_budget`**
  (`Optional[int]`)
  –

  Provide a default thinking budget in tokens. If not provided,
  thinking will be disabled for the profile. The minimum token budget supported by Anthropic is 1024 tokens.
  For Claude models that use adaptive thinking, use `effort` instead.
- **`effort`**
  (`Optional[AnthropicReasoningEffortType]`)
  –

  Provider-native Anthropic effort level. Supported values vary by model:
  low, medium, high, xhigh, and max.
  On adaptive-thinking models the thinking budget shares the request's
  output token window rather than being reserved on top of it, so very
  high effort levels can consume part of the visible completion budget.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Note

If `thinking_token_budget` or adaptive `effort` enables thinking,
`temperature` cannot be customized -- any changes to `temperature`
will be ignored. Effort-only profiles on non-adaptive models
configure Anthropic `output_config` without enabling thinking, so
custom `temperature` remains available when the model supports it.

Example

Configuring a profile with a thinking budget:

```
profile = AnthropicLanguageModel.Profile(thinking_token_budget=2048)
```

Configuring a profile with a large thinking budget:

```
profile = AnthropicLanguageModel.Profile(thinking_token_budget=8192)
```

Configuring a profile with effort:

```
profile = AnthropicLanguageModel.Profile(effort="xhigh")
```

## ArrayType

Bases: `DataType`

```
              flowchart TD
              fenic.ArrayType[ArrayType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.ArrayType

              click fenic.ArrayType href "" "fenic.ArrayType"
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

## Catalog

```
Catalog(catalog: BaseCatalog)
```

Entry point for catalog operations.

Provides methods to manage catalogs, databases, and tables, as well as
read-only access to system tables such as `fenic_system.query_metrics`.

##### Catalog and Database Management

Example:
```python
# Create a catalog
session.catalog.create_catalog("my_catalog") # → True

```
# Set active catalog
session.catalog.set_current_catalog("my_catalog")

# Create a database
session.catalog.create_database("my_database")  # → True

# Set active database
session.catalog.set_current_database("my_database")

# Create a table
session.catalog.create_table(
    "my_table",
    Schema([ColumnField("id", IntegerType)])
)  # → True
```
```

##### Metrics Table (Local Sessions Only)

Query metrics are recorded for each session and stored locally
in `fenic_system.query_metrics`. Metrics can be loaded into a DataFrame
for analysis.

Example

```
# Load all metrics for the current application
metrics_df = session.table("fenic_system.query_metrics")

# Show the 10 most recent queries in the application
recent_queries = session.sql("""
    SELECT *
    FROM {df}
    ORDER BY CAST(end_ts AS TIMESTAMP) DESC
    LIMIT 10
""", df=metrics_df)
recent_queries.show()

# Find query metrics for a specific session with non-zero LM costs
specific_session_queries = session.sql("""
    SELECT *
    FROM {df}
    WHERE session_id = '9e7e256f-fad9-4cd9-844e-399d795aaea0'
        AND total_lm_cost > 0
    ORDER BY CAST(end_ts AS TIMESTAMP) ASC
""", df=metrics_df)
specific_session_queries.show()

# Aggregate total LM costs and requests between a specific time window
metrics_window = session.sql("""
    SELECT
        CAST(SUM(total_lm_cost) AS DOUBLE) AS total_lm_cost_in_window,
        CAST(SUM(total_lm_requests) AS DOUBLE) AS total_lm_requests_in_window
    FROM {df}
    WHERE CAST(end_ts AS TIMESTAMP) BETWEEN
        CAST('2025-08-29 10:00:00' AS TIMESTAMP)
        AND CAST('2025-08-29 12:00:00' AS TIMESTAMP)
""", df=metrics_df)

metrics_window.show()
```

Initialize a Catalog instance.

Parameters:

- **`catalog`**
  (`BaseCatalog`)
  –

  The underlying catalog implementation.

Methods:

- **`create_catalog`**
  –

  Creates a new catalog.
- **`create_database`**
  –

  Creates a new database.
- **`create_table`**
  –

  Creates a new table.
- **`create_tool`**
  –

  Creates a new tool in the current catalog.
- **`describe_table`**
  –

  Returns the schema of the specified table.
- **`describe_tool`**
  –

  Returns the tool with the specified name from the current catalog.
- **`describe_view`**
  –

  Returns the schema and description of the specified view.
- **`does_catalog_exist`**
  –

  Checks if a catalog with the specified name exists.
- **`does_database_exist`**
  –

  Checks if a database with the specified name exists.
- **`does_table_exist`**
  –

  Checks if a table with the specified name exists.
- **`does_view_exist`**
  –

  Checks if a view with the specified name exists.
- **`drop_catalog`**
  –

  Drops a catalog.
- **`drop_database`**
  –

  Drops a database.
- **`drop_table`**
  –

  Drops the specified table.
- **`drop_tool`**
  –

  Drops the specified tool from the current catalog.
- **`drop_view`**
  –

  Drops the specified view.
- **`get_current_catalog`**
  –

  Returns the name of the current catalog.
- **`get_current_database`**
  –

  Returns the name of the current database in the current catalog.
- **`list_catalogs`**
  –

  Returns a list of available catalogs.
- **`list_databases`**
  –

  Returns a list of databases in the current catalog.
- **`list_tables`**
  –

  Returns a list of tables stored in the current database.
- **`list_tools`**
  –

  Lists the tools available in the current catalog.
- **`list_views`**
  –

  Returns a list of views stored in the current database.
- **`set_current_catalog`**
  –

  Sets the current catalog.
- **`set_current_database`**
  –

  Sets the current database.
- **`set_table_description`**
  –

  Set or unset the description for a table.
- **`set_view_description`**
  –

  Set the description for a view.

Source code in `src/fenic/api/catalog.py`

```
def __init__(self, catalog: BaseCatalog):
    """Initialize a Catalog instance.

    Args:
        catalog: The underlying catalog implementation.
    """
    self.catalog = catalog
```

### create_catalog

```
create_catalog(catalog_name: str, ignore_if_exists: bool = True) -> bool
```

Creates a new catalog.

Parameters:

- **`catalog_name`**
  (`str`)
  –

  Name of the catalog to create.
- **`ignore_if_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, return False when the catalog already exists.
  If False, raise an error when the catalog already exists.
  Defaults to True.

Raises:

- `CatalogAlreadyExistsError`
  –

  If the catalog already exists and ignore_if_exists is False.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the catalog was created successfully, False if the catalog
- `bool`
  –

  already exists and ignore_if_exists is True.

Create a new catalog

```
# Create a new catalog named 'my_catalog'
session.catalog.create_catalog('my_catalog')
# Returns: True
```

Create an existing catalog with ignore_if_exists

```
# Try to create an existing catalog with ignore_if_exists=True
session.catalog.create_catalog('my_catalog', ignore_if_exists=True)
# Returns: False
```

Create an existing catalog without ignore_if_exists

```
# Try to create an existing catalog with ignore_if_exists=False
session.catalog.create_catalog('my_catalog', ignore_if_exists=False)
# Raises: CatalogAlreadyExistsError
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def create_catalog(self, catalog_name: str, ignore_if_exists: bool = True) -> bool:
    """Creates a new catalog.

    Args:
        catalog_name (str): Name of the catalog to create.
        ignore_if_exists (bool): If True, return False when the catalog already exists.
            If False, raise an error when the catalog already exists.
            Defaults to True.

    Raises:
        CatalogAlreadyExistsError: If the catalog already exists and ignore_if_exists is False.

    Returns:
        bool: True if the catalog was created successfully, False if the catalog
        already exists and ignore_if_exists is True.

    Example: Create a new catalog
        ```python
        # Create a new catalog named 'my_catalog'
        session.catalog.create_catalog('my_catalog')
        # Returns: True
        ```

    Example: Create an existing catalog with ignore_if_exists
        ```python
        # Try to create an existing catalog with ignore_if_exists=True
        session.catalog.create_catalog('my_catalog', ignore_if_exists=True)
        # Returns: False
        ```

    Example: Create an existing catalog without ignore_if_exists
        ```python
        # Try to create an existing catalog with ignore_if_exists=False
        session.catalog.create_catalog('my_catalog', ignore_if_exists=False)
        # Raises: CatalogAlreadyExistsError
        ```
    """
    return self.catalog.create_catalog(catalog_name, ignore_if_exists)
```

### create_database

```
create_database(database_name: str, ignore_if_exists: bool = True) -> bool
```

Creates a new database.

Parameters:

- **`database_name`**
  (`str`)
  –

  Fully qualified or relative database name to create.
- **`ignore_if_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, return False when the database already exists.
  If False, raise an error when the database already exists.
  Defaults to True.

Raises:

- `DatabaseAlreadyExistsError`
  –

  If the database already exists and ignore_if_exists is False.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the database was created successfully, False if the database
- `bool`
  –

  already exists and ignore_if_exists is True.

Create a new database

```
# Create a new database named 'my_database'
session.catalog.create_database('my_database')
# Returns: True
```

Create an existing database with ignore_if_exists

```
# Try to create an existing database with ignore_if_exists=True
session.catalog.create_database('my_database', ignore_if_exists=True)
# Returns: False
```

Create an existing database without ignore_if_exists

```
# Try to create an existing database with ignore_if_exists=False
session.catalog.create_database('my_database', ignore_if_exists=False)
# Raises: DatabaseAlreadyExistsError
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def create_database(
    self, database_name: str, ignore_if_exists: bool = True
) -> bool:
    """Creates a new database.

    Args:
        database_name (str): Fully qualified or relative database name to create.
        ignore_if_exists (bool): If True, return False when the database already exists.
            If False, raise an error when the database already exists.
            Defaults to True.

    Raises:
        DatabaseAlreadyExistsError: If the database already exists and ignore_if_exists is False.

    Returns:
        bool: True if the database was created successfully, False if the database
        already exists and ignore_if_exists is True.

    Example: Create a new database
        ```python
        # Create a new database named 'my_database'
        session.catalog.create_database('my_database')
        # Returns: True
        ```

    Example: Create an existing database with ignore_if_exists
        ```python
        # Try to create an existing database with ignore_if_exists=True
        session.catalog.create_database('my_database', ignore_if_exists=True)
        # Returns: False
        ```

    Example: Create an existing database without ignore_if_exists
        ```python
        # Try to create an existing database with ignore_if_exists=False
        session.catalog.create_database('my_database', ignore_if_exists=False)
        # Raises: DatabaseAlreadyExistsError
        ```
    """
    return self.catalog.create_database(database_name, ignore_if_exists)
```

### create_table

```
create_table(table_name: str, schema: Schema, ignore_if_exists: bool = True, description: Optional[str] = None) -> bool
```

Creates a new table.

Parameters:

- **`table_name`**
  (`str`)
  –

  Fully qualified or relative table name to create.
- **`schema`**
  (`Schema`)
  –

  Schema of the table to create.
- **`ignore_if_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, return False when the table already exists.
  If False, raise an error when the table already exists.
  Defaults to True.
- **`description`**
  (`Optional[str]`, default:
  `None`
  )
  –

  Description of the table to create.
  Defaults to None.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the table was created successfully, False if the table
- `bool`
  –

  already exists and ignore_if_exists is True.

Raises:

- `TableAlreadyExistsError`
  –

  If the table already exists and ignore_if_exists is False

Create a new table

```
# Create a new table with an integer column
session.catalog.create_table('my_table', Schema([
    ColumnField('id', IntegerType),
]), description='My table description')
# Returns: True
```

Create an existing table with ignore_if_exists

```
# Try to create an existing table with ignore_if_exists=True
session.catalog.create_table('my_table', Schema([
    ColumnField('id', IntegerType),
]), ignore_if_exists=True, description='My table description')
# Returns: False
```

Create an existing table without ignore_if_exists

```
# Try to create an existing table with ignore_if_exists=False
session.catalog.create_table('my_table', Schema([
    ColumnField('id', IntegerType),
]), ignore_if_exists=False, description='My table description')
# Raises: TableAlreadyExistsError
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def create_table(
    self, table_name: str, schema: Schema, ignore_if_exists: bool = True, description: Optional[str] = None
) -> bool:
    """Creates a new table.

    Args:
        table_name (str): Fully qualified or relative table name to create.
        schema (Schema): Schema of the table to create.
        ignore_if_exists (bool): If True, return False when the table already exists.
            If False, raise an error when the table already exists.
            Defaults to True.
        description (Optional[str]): Description of the table to create.
            Defaults to None.

    Returns:
        bool: True if the table was created successfully, False if the table
        already exists and ignore_if_exists is True.

    Raises:
        TableAlreadyExistsError: If the table already exists and ignore_if_exists is False

    Example: Create a new table
        ```python
        # Create a new table with an integer column
        session.catalog.create_table('my_table', Schema([
            ColumnField('id', IntegerType),
        ]), description='My table description')
        # Returns: True
        ```

    Example: Create an existing table with ignore_if_exists
        ```python
        # Try to create an existing table with ignore_if_exists=True
        session.catalog.create_table('my_table', Schema([
            ColumnField('id', IntegerType),
        ]), ignore_if_exists=True, description='My table description')
        # Returns: False
        ```

    Example: Create an existing table without ignore_if_exists
        ```python
        # Try to create an existing table with ignore_if_exists=False
        session.catalog.create_table('my_table', Schema([
            ColumnField('id', IntegerType),
        ]), ignore_if_exists=False, description='My table description')
        # Raises: TableAlreadyExistsError
        ```
    """
    return self.catalog.create_table(table_name, schema, ignore_if_exists, description)
```

### create_tool

```
create_tool(tool_name: str, tool_description: str, tool_query: DataFrame, tool_params: List[ToolParam], result_limit: int = 50, ignore_if_exists: bool = True) -> bool
```

Creates a new tool in the current catalog.

Parameters:

- **`tool_name`**
  (`str`)
  –

  The name of the tool.
- **`tool_description`**
  (`str`)
  –

  The description of the tool.
- **`tool_query`**
  (`DataFrame`)
  –

  The query to execute when the tool is called.
- **`tool_params`**
  (`Sequence[ToolParam]`)
  –

  The parameters of the tool.
- **`result_limit`**
  (`int`, default:
  `50`
  )
  –

  The maximum number of rows to return from the tool.
- **`ignore_if_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, return False when the tool already exists.
  If False, raise an error when the tool already exists.
  Defaults to True.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the tool was created successfully, False otherwise.

Raises:

- `ToolAlreadyExistsError`
  –

  If the tool already exists.

Examples:

```
# Create a new tool with a single parameter
df = session.create_dataframe(...)

session.catalog.create_tool(
    tool_name="my_tool",
    tool_description="A tool that does something",
    tool_query=df,
    result_limit=100,
    tool_params=[ToolParam(name="param1", description="A parameter", allowed_values=["value1", "value2"])],
)
# Returns: True
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def create_tool(
    self,
    tool_name: str,
    tool_description: str,
    tool_query: DataFrame,
    tool_params: List[ToolParam],
    result_limit: int = 50,
    ignore_if_exists: bool = True
) -> bool:
    """Creates a new tool in the current catalog.

    Args:
        tool_name (str): The name of the tool.
        tool_description (str): The description of the tool.
        tool_query (DataFrame): The query to execute when the tool is called.
        tool_params (Sequence[ToolParam]): The parameters of the tool.
        result_limit (int): The maximum number of rows to return from the tool.
        ignore_if_exists (bool): If True, return False when the tool already exists.
            If False, raise an error when the tool already exists.
            Defaults to True.

    Returns:
        bool: True if the tool was created successfully, False otherwise.

    Raises:
        ToolAlreadyExistsError: If the tool already exists.

    Examples:
        ```python
        # Create a new tool with a single parameter
        df = session.create_dataframe(...)

        session.catalog.create_tool(
            tool_name="my_tool",
            tool_description="A tool that does something",
            tool_query=df,
            result_limit=100,
            tool_params=[ToolParam(name="param1", description="A parameter", allowed_values=["value1", "value2"])],
        )
        # Returns: True
        ```
    """
    return self.catalog.create_tool(
        tool_name,
        tool_description,
        tool_params,
        tool_query._logical_plan,
        result_limit,
        ignore_if_exists,
    )
```

### describe_table

```
describe_table(table_name: str) -> DatasetMetadata
```

Returns the schema of the specified table.

Parameters:

- **`table_name`**
  (`str`)
  –

  Fully qualified or relative table name to describe.

Returns:

- **`DatasetMetadata`** ( `DatasetMetadata`
  ) –

  An object containing:
  schema: A schema object describing the table's structure with field names and types.
  description: A natural language description of the table's contents and uses.

Raises:

- `TableNotFoundError`
  –

  If the table doesn't exist.

Describe a table's schema

```
# For a table created with: create_table('t1', Schema([ColumnField('id', IntegerType)]), description='My table description')
session.catalog.describe_table('t1')
# Returns: DatasetMetadata(schema=Schema([
#     ColumnField('id', IntegerType),
# ]), description="My table description")
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def describe_table(self, table_name: str) -> DatasetMetadata:
    """Returns the schema of the specified table.

    Args:
        table_name (str): Fully qualified or relative table name to describe.

    Returns:
        DatasetMetadata: An object containing:
            schema: A schema object describing the table's structure with field names and types.
            description: A natural language description of the table's contents and uses.

    Raises:
        TableNotFoundError: If the table doesn't exist.

    Example: Describe a table's schema
        ```python
        # For a table created with: create_table('t1', Schema([ColumnField('id', IntegerType)]), description='My table description')
        session.catalog.describe_table('t1')
        # Returns: DatasetMetadata(schema=Schema([
        #     ColumnField('id', IntegerType),
        # ]), description="My table description")
        ```
    """
    return self.catalog.describe_table(table_name)
```

### describe_tool

```
describe_tool(tool_name: str) -> UserDefinedTool
```

Returns the tool with the specified name from the current catalog.

Parameters:

- **`tool_name`**
  (`str`)
  –

  The name of the tool to get.

Raises:

- `ToolNotFoundError`
  –

  If the tool doesn't exist.

Returns:

- **`Tool`** ( `UserDefinedTool`
  ) –

  The tool with the specified name.

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def describe_tool(self, tool_name: str) -> UserDefinedTool:
    """Returns the tool with the specified name from the current catalog.

    Args:
        tool_name (str): The name of the tool to get.

    Raises:
        ToolNotFoundError: If the tool doesn't exist.

    Returns:
        Tool: The tool with the specified name.
    """
    return self.catalog.describe_tool(tool_name)
```

### describe_view

```
describe_view(view_name: str) -> DatasetMetadata
```

Returns the schema and description of the specified view.

Parameters:

- **`view_name`**
  (`str`)
  –

  Fully qualified or relative view name to describe.

Returns:

- **`DatasetMetadata`** ( `DatasetMetadata`
  ) –

  An object containing:
  schema: A schema object describing the view's structure with field names and types.
  description: A natural language description of the view's contents and uses.

Raises:

- `TableNotFoundError`
  –

  If the view doesn't exist.

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def describe_view(self, view_name: str) -> DatasetMetadata:
    """Returns the schema and description of the specified view.

    Args:
        view_name (str): Fully qualified or relative view name to describe.

    Returns:
        DatasetMetadata: An object containing:
            schema: A schema object describing the view's structure with field names and types.
            description: A natural language description of the view's contents and uses.

    Raises:
        TableNotFoundError: If the view doesn't exist.

    """
    return self.catalog.describe_view(view_name)
```

### does_catalog_exist

```
does_catalog_exist(catalog_name: str) -> bool
```

Checks if a catalog with the specified name exists.

Parameters:

- **`catalog_name`**
  (`str`)
  –

  Name of the catalog to check.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the catalog exists, False otherwise.

Check if a catalog exists

```
# Check if 'my_catalog' exists
session.catalog.does_catalog_exist('my_catalog')
# Returns: True
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def does_catalog_exist(self, catalog_name: str) -> bool:
    """Checks if a catalog with the specified name exists.

    Args:
        catalog_name (str): Name of the catalog to check.

    Returns:
        bool: True if the catalog exists, False otherwise.

    Example: Check if a catalog exists
        ```python
        # Check if 'my_catalog' exists
        session.catalog.does_catalog_exist('my_catalog')
        # Returns: True
        ```
    """
    return self.catalog.does_catalog_exist(catalog_name)
```

### does_database_exist

```
does_database_exist(database_name: str) -> bool
```

Checks if a database with the specified name exists.

Parameters:

- **`database_name`**
  (`str`)
  –

  Fully qualified or relative database name to check.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the database exists, False otherwise.

Check if a database exists

```
# Check if 'my_database' exists
session.catalog.does_database_exist('my_database')
# Returns: True
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def does_database_exist(self, database_name: str) -> bool:
    """Checks if a database with the specified name exists.

    Args:
        database_name (str): Fully qualified or relative database name to check.

    Returns:
        bool: True if the database exists, False otherwise.

    Example: Check if a database exists
        ```python
        # Check if 'my_database' exists
        session.catalog.does_database_exist('my_database')
        # Returns: True
        ```
    """
    return self.catalog.does_database_exist(database_name)
```

### does_table_exist

```
does_table_exist(table_name: str) -> bool
```

Checks if a table with the specified name exists.

Parameters:

- **`table_name`**
  (`str`)
  –

  Fully qualified or relative table name to check.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the table exists, False otherwise.

Check if a table exists

```
# Check if 'my_table' exists
session.catalog.does_table_exist('my_table')
# Returns: True
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def does_table_exist(self, table_name: str) -> bool:
    """Checks if a table with the specified name exists.

    Args:
        table_name (str): Fully qualified or relative table name to check.

    Returns:
        bool: True if the table exists, False otherwise.

    Example: Check if a table exists
        ```python
        # Check if 'my_table' exists
        session.catalog.does_table_exist('my_table')
        # Returns: True
        ```
    """
    return self.catalog.does_table_exist(table_name)
```

### does_view_exist

```
does_view_exist(view_name: str) -> bool
```

Checks if a view with the specified name exists.

Parameters:

- **`view_name`**
  (`str`)
  –

  Fully qualified or relative view name to check.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the view exists, False otherwise.

Example
> > > session.catalog.does_view_exist('my_view')
> > > True.

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def does_view_exist(self, view_name: str) -> bool:
    """Checks if a view with the specified name exists.

    Args:
        view_name (str): Fully qualified or relative view name to check.

    Returns:
        bool: True if the view exists, False otherwise.

    Example:
        >>> session.catalog.does_view_exist('my_view')
        True.
    """
    return self.catalog.does_view_exist(view_name)
```

### drop_catalog

```
drop_catalog(catalog_name: str, ignore_if_not_exists: bool = True) -> bool
```

Drops a catalog.

Parameters:

- **`catalog_name`**
  (`str`)
  –

  Name of the catalog to drop.
- **`ignore_if_not_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, silently return if the catalog doesn't exist.
  If False, raise an error if the catalog doesn't exist.
  Defaults to True.

Raises:

- `CatalogNotFoundError`
  –

  If the catalog does not exist and ignore_if_not_exists is False

Returns:

- **`bool`** ( `bool`
  ) –

  True if the catalog was dropped successfully, False if the catalog
- `bool`
  –

  didn't exist and ignore_if_not_exists is True.

Drop a non-existent catalog

```
# Try to drop a non-existent catalog
session.catalog.drop_catalog('my_catalog')
# Returns: False
```

Drop a non-existent catalog without ignore_if_not_exists

```
# Try to drop a non-existent catalog with ignore_if_not_exists=False
session.catalog.drop_catalog('my_catalog', ignore_if_not_exists=False)
# Raises: CatalogNotFoundError
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def drop_catalog(
    self, catalog_name: str, ignore_if_not_exists: bool = True
) -> bool:
    """Drops a catalog.

    Args:
        catalog_name (str): Name of the catalog to drop.
        ignore_if_not_exists (bool): If True, silently return if the catalog doesn't exist.
            If False, raise an error if the catalog doesn't exist.
            Defaults to True.

    Raises:
        CatalogNotFoundError: If the catalog does not exist and ignore_if_not_exists is False

    Returns:
        bool: True if the catalog was dropped successfully, False if the catalog
        didn't exist and ignore_if_not_exists is True.

    Example: Drop a non-existent catalog
        ```python
        # Try to drop a non-existent catalog
        session.catalog.drop_catalog('my_catalog')
        # Returns: False
        ```

    Example: Drop a non-existent catalog without ignore_if_not_exists
        ```python
        # Try to drop a non-existent catalog with ignore_if_not_exists=False
        session.catalog.drop_catalog('my_catalog', ignore_if_not_exists=False)
        # Raises: CatalogNotFoundError
        ```
    """
    return self.catalog.drop_catalog(catalog_name, ignore_if_not_exists)
```

### drop_database

```
drop_database(database_name: str, cascade: bool = False, ignore_if_not_exists: bool = True) -> bool
```

Drops a database.

Parameters:

- **`database_name`**
  (`str`)
  –

  Fully qualified or relative database name to drop.
- **`cascade`**
  (`bool`, default:
  `False`
  )
  –

  If True, drop all tables in the database.
  Defaults to False.
- **`ignore_if_not_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, silently return if the database doesn't exist.
  If False, raise an error if the database doesn't exist.
  Defaults to True.

Raises:

- `DatabaseNotFoundError`
  –

  If the database does not exist and ignore_if_not_exists is False
- `CatalogError`
  –

  If the current database is being dropped, if the database is not empty and cascade is False

Returns:

- **`bool`** ( `bool`
  ) –

  True if the database was dropped successfully, False if the database
- `bool`
  –

  didn't exist and ignore_if_not_exists is True.

Drop a non-existent database

```
# Try to drop a non-existent database
session.catalog.drop_database('my_database')
# Returns: False
```

Drop a non-existent database without ignore_if_not_exists

```
# Try to drop a non-existent database with ignore_if_not_exists=False
session.catalog.drop_database('my_database', ignore_if_not_exists=False)
# Raises: DatabaseNotFoundError
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def drop_database(
    self,
    database_name: str,
    cascade: bool = False,
    ignore_if_not_exists: bool = True,
) -> bool:
    """Drops a database.

    Args:
        database_name (str): Fully qualified or relative database name to drop.
        cascade (bool): If True, drop all tables in the database.
            Defaults to False.
        ignore_if_not_exists (bool): If True, silently return if the database doesn't exist.
            If False, raise an error if the database doesn't exist.
            Defaults to True.

    Raises:
        DatabaseNotFoundError: If the database does not exist and ignore_if_not_exists is False
        CatalogError: If the current database is being dropped, if the database is not empty and cascade is False

    Returns:
        bool: True if the database was dropped successfully, False if the database
        didn't exist and ignore_if_not_exists is True.

    Example: Drop a non-existent database
        ```python
        # Try to drop a non-existent database
        session.catalog.drop_database('my_database')
        # Returns: False
        ```

    Example: Drop a non-existent database without ignore_if_not_exists
        ```python
        # Try to drop a non-existent database with ignore_if_not_exists=False
        session.catalog.drop_database('my_database', ignore_if_not_exists=False)
        # Raises: DatabaseNotFoundError
        ```
    """
    return self.catalog.drop_database(database_name, cascade, ignore_if_not_exists)
```

### drop_table

```
drop_table(table_name: str, ignore_if_not_exists: bool = True) -> bool
```

Drops the specified table.

By default this method will return False if the table doesn't exist.

Parameters:

- **`table_name`**
  (`str`)
  –

  Fully qualified or relative table name to drop.
- **`ignore_if_not_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, return False when the table doesn't exist.
  If False, raise an error when the table doesn't exist.
  Defaults to True.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the table was dropped successfully, False if the table
- `bool`
  –

  didn't exist and ignore_if_not_exist is True.

Raises:

- `TableNotFoundError`
  –

  If the table doesn't exist and ignore_if_not_exists is False

Drop an existing table

```
# Drop an existing table 't1'
session.catalog.drop_table('t1')
# Returns: True
```

Drop a non-existent table with ignore_if_not_exists

```
# Try to drop a non-existent table with ignore_if_not_exists=True
session.catalog.drop_table('t2', ignore_if_not_exists=True)
# Returns: False
```

Drop a non-existent table without ignore_if_not_exists

```
# Try to drop a non-existent table with ignore_if_not_exists=False
session.catalog.drop_table('t2', ignore_if_not_exists=False)
# Raises: TableNotFoundError
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def drop_table(self, table_name: str, ignore_if_not_exists: bool = True) -> bool:
    """Drops the specified table.

    By default this method will return False if the table doesn't exist.

    Args:
        table_name (str): Fully qualified or relative table name to drop.
        ignore_if_not_exists (bool): If True, return False when the table doesn't exist.
            If False, raise an error when the table doesn't exist.
            Defaults to True.

    Returns:
        bool: True if the table was dropped successfully, False if the table
        didn't exist and ignore_if_not_exist is True.

    Raises:
        TableNotFoundError: If the table doesn't exist and ignore_if_not_exists is False

    Example: Drop an existing table
        ```python
        # Drop an existing table 't1'
        session.catalog.drop_table('t1')
        # Returns: True
        ```

    Example: Drop a non-existent table with ignore_if_not_exists
        ```python
        # Try to drop a non-existent table with ignore_if_not_exists=True
        session.catalog.drop_table('t2', ignore_if_not_exists=True)
        # Returns: False
        ```

    Example: Drop a non-existent table without ignore_if_not_exists
        ```python
        # Try to drop a non-existent table with ignore_if_not_exists=False
        session.catalog.drop_table('t2', ignore_if_not_exists=False)
        # Raises: TableNotFoundError
        ```
    """
    return self.catalog.drop_table(table_name, ignore_if_not_exists)
```

### drop_tool

```
drop_tool(tool_name: str, ignore_if_not_exists: bool = True) -> bool
```

Drops the specified tool from the current catalog.

Parameters:

- **`tool_name`**
  (`str`)
  –

  The name of the tool to drop.
- **`ignore_if_not_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, return False when the tool doesn't exist.
  If False, raise an error when the tool doesn't exist.
  Defaults to True.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the tool was dropped successfully, False if the tool
  didn't exist and ignore_if_not_exists is True.

Raises:

- `ToolNotFoundError`
  –

  If the tool doesn't exist and ignore_if_not_exists is False

Example
> > > session.catalog.drop_tool('my_tool')
> > > True
> > > session.catalog.drop_tool('my_tool', ignore_if_not_exists=True)
> > > False
> > > session.catalog.drop_tool('my_tool', ignore_if_not_exists=False)

#### Raises ToolNotFoundError.

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def drop_tool(self, tool_name: str, ignore_if_not_exists: bool = True) -> bool:
    """Drops the specified tool from the current catalog.

    Args:
        tool_name (str): The name of the tool to drop.
        ignore_if_not_exists (bool): If True, return False when the tool doesn't exist.
            If False, raise an error when the tool doesn't exist.
            Defaults to True.

    Returns:
        bool: True if the tool was dropped successfully, False if the tool
            didn't exist and ignore_if_not_exists is True.

    Raises:
        ToolNotFoundError: If the tool doesn't exist and ignore_if_not_exists is False

    Example:
        >>> session.catalog.drop_tool('my_tool')
        True
        >>> session.catalog.drop_tool('my_tool', ignore_if_not_exists=True)
        False
        >>> session.catalog.drop_tool('my_tool', ignore_if_not_exists=False)
        # Raises ToolNotFoundError.
    """
    return self.catalog.drop_tool(tool_name, ignore_if_not_exists)
```

### drop_view

```
drop_view(view_name: str, ignore_if_not_exists: bool = True) -> bool
```

Drops the specified view.

By default this method will return False if the view doesn't exist.

Parameters:

- **`view_name`**
  (`str`)
  –

  Fully qualified or relative view name to drop.
- **`ignore_if_not_exists`**
  (`bool`, default:
  `True`
  )
  –

  If True, return False when the view
  doesn't exist. If False, raise an error when the view doesn't exist.
  Defaults to True.

Returns:

- **`bool`** ( `bool`
  ) –

  True if the view was dropped successfully, False if the view
  didn't exist and ignore_if_not_exist is True.

Raises:

- `TableNotFoundError`
  –

  If the view doesn't exist and ignore_if_not_exists is False

Example:
>>> # For an existing view 'v1'
>>> session.catalog.drop_table('v1')
True
>>> # For a non-existent table 'v2'
>>> session.catalog.drop_table('v2', ignore_if_not_exists=True)
False
>>> session.catalog.drop_table('v2', ignore_if_not_exists=False)
# Raises TableNotFoundError.

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def drop_view(self, view_name: str, ignore_if_not_exists: bool = True) -> bool:
    """Drops the specified view.

    By default this method will return False if the view doesn't exist.

    Args:
        view_name (str): Fully qualified or relative view name to drop.
        ignore_if_not_exists (bool, optional): If True, return False when the view
            doesn't exist. If False, raise an error when the view doesn't exist.
            Defaults to True.

    Returns:
        bool: True if the view was dropped successfully, False if the view
            didn't exist and ignore_if_not_exist is True.

    Raises:
        TableNotFoundError: If the view doesn't exist and ignore_if_not_exists is False
    Example:
        >>> # For an existing view 'v1'
        >>> session.catalog.drop_table('v1')
        True
        >>> # For a non-existent table 'v2'
        >>> session.catalog.drop_table('v2', ignore_if_not_exists=True)
        False
        >>> session.catalog.drop_table('v2', ignore_if_not_exists=False)
        # Raises TableNotFoundError.
    """
    return self.catalog.drop_view(view_name, ignore_if_not_exists)
```

### get_current_catalog

```
get_current_catalog() -> str
```

Returns the name of the current catalog.

Returns:

- **`str`** ( `str`
  ) –

  The name of the current catalog.

Get current catalog name

```
# Get the name of the current catalog
session.catalog.get_current_catalog()
# Returns: 'default'
```

Source code in `src/fenic/api/catalog.py`

```
def get_current_catalog(self) -> str:
    """Returns the name of the current catalog.

    Returns:
        str: The name of the current catalog.

    Example: Get current catalog name
        ```python
        # Get the name of the current catalog
        session.catalog.get_current_catalog()
        # Returns: 'default'
        ```
    """
    return self.catalog.get_current_catalog()
```

### get_current_database

```
get_current_database() -> str
```

Returns the name of the current database in the current catalog.

Returns:

- **`str`** ( `str`
  ) –

  The name of the current database.

Get current database name

```
# Get the name of the current database
session.catalog.get_current_database()
# Returns: 'default'
```

Source code in `src/fenic/api/catalog.py`

```
def get_current_database(self) -> str:
    """Returns the name of the current database in the current catalog.

    Returns:
        str: The name of the current database.

    Example: Get current database name
        ```python
        # Get the name of the current database
        session.catalog.get_current_database()
        # Returns: 'default'
        ```
    """
    return self.catalog.get_current_database()
```

### list_catalogs

```
list_catalogs() -> List[str]
```

Returns a list of available catalogs.

Returns:

- `List[str]`
  –

  List[str]: A list of catalog names available in the system.
- `List[str]`
  –

  Returns an empty list if no catalogs are found.

List all catalogs

```
# Get all available catalogs
session.catalog.list_catalogs()
# Returns: ['default', 'my_catalog', 'other_catalog']
```

Source code in `src/fenic/api/catalog.py`

```
def list_catalogs(self) -> List[str]:
    """Returns a list of available catalogs.

    Returns:
        List[str]: A list of catalog names available in the system.
        Returns an empty list if no catalogs are found.

    Example: List all catalogs
        ```python
        # Get all available catalogs
        session.catalog.list_catalogs()
        # Returns: ['default', 'my_catalog', 'other_catalog']
        ```
    """
    return self.catalog.list_catalogs()
```

### list_databases

```
list_databases() -> List[str]
```

Returns a list of databases in the current catalog.

Returns:

- `List[str]`
  –

  List[str]: A list of database names in the current catalog.
- `List[str]`
  –

  Returns an empty list if no databases are found.

List all databases

```
# Get all databases in the current catalog
session.catalog.list_databases()
# Returns: ['default', 'my_database', 'other_database']
```

Source code in `src/fenic/api/catalog.py`

```
def list_databases(self) -> List[str]:
    """Returns a list of databases in the current catalog.

    Returns:
        List[str]: A list of database names in the current catalog.
        Returns an empty list if no databases are found.

    Example: List all databases
        ```python
        # Get all databases in the current catalog
        session.catalog.list_databases()
        # Returns: ['default', 'my_database', 'other_database']
        ```
    """
    return self.catalog.list_databases()
```

### list_tables

```
list_tables() -> List[str]
```

Returns a list of tables stored in the current database.

This method queries the current database to retrieve all available table names.

Returns:

- `List[str]`
  –

  List[str]: A list of table names stored in the database.
- `List[str]`
  –

  Returns an empty list if no tables are found.

List all tables

```
# Get all tables in the current database
session.catalog.list_tables()
# Returns: ['table1', 'table2', 'table3']
```

Source code in `src/fenic/api/catalog.py`

```
def list_tables(self) -> List[str]:
    """Returns a list of tables stored in the current database.

    This method queries the current database to retrieve all available table names.

    Returns:
        List[str]: A list of table names stored in the database.
        Returns an empty list if no tables are found.

    Example: List all tables
        ```python
        # Get all tables in the current database
        session.catalog.list_tables()
        # Returns: ['table1', 'table2', 'table3']
        ```
    """
    return self.catalog.list_tables()
```

### list_tools

```
list_tools() -> List[UserDefinedTool]
```

Lists the tools available in the current catalog.

Source code in `src/fenic/api/catalog.py`

```
def list_tools(self) -> List[UserDefinedTool]:
    """Lists the tools available in the current catalog."""
    return self.catalog.list_tools()
```

### list_views

```
list_views() -> List[str]
```

Returns a list of views stored in the current database.

This method queries the current database to retrieve all available view names.

Returns:

- `List[str]`
  –

  List[str]: A list of view names stored in the database.
- `List[str]`
  –

  Returns an empty list if no views are found.

Example
> > > session.catalog.list_views()
> > > ['view1', 'view2', 'view3'].

Source code in `src/fenic/api/catalog.py`

```
def list_views(self) -> List[str]:
    """Returns a list of views stored in the current database.

    This method queries the current database to retrieve all available view names.

    Returns:
        List[str]: A list of view names stored in the database.
        Returns an empty list if no views are found.

    Example:
        >>> session.catalog.list_views()
        ['view1', 'view2', 'view3'].
    """
    return self.catalog.list_views()
```

### set_current_catalog

```
set_current_catalog(catalog_name: str) -> None
```

Sets the current catalog.

Parameters:

- **`catalog_name`**
  (`str`)
  –

  Name of the catalog to set as current.

Raises:

- `ValueError`
  –

  If the specified catalog doesn't exist.

Set current catalog

```
# Set 'my_catalog' as the current catalog
session.catalog.set_current_catalog('my_catalog')
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def set_current_catalog(self, catalog_name: str) -> None:
    """Sets the current catalog.

    Args:
        catalog_name (str): Name of the catalog to set as current.

    Raises:
        ValueError: If the specified catalog doesn't exist.

    Example: Set current catalog
        ```python
        # Set 'my_catalog' as the current catalog
        session.catalog.set_current_catalog('my_catalog')
        ```
    """
    self.catalog.set_current_catalog(catalog_name)
```

### set_current_database

```
set_current_database(database_name: str) -> None
```

Sets the current database.

Parameters:

- **`database_name`**
  (`str`)
  –

  Fully qualified or relative database name to set as current.

Raises:

- `DatabaseNotFoundError`
  –

  If the specified database doesn't exist.

Set current database

```
# Set 'my_database' as the current database
session.catalog.set_current_database('my_database')
```

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def set_current_database(self, database_name: str) -> None:
    """Sets the current database.

    Args:
        database_name (str): Fully qualified or relative database name to set as current.

    Raises:
        DatabaseNotFoundError: If the specified database doesn't exist.

    Example: Set current database
        ```python
        # Set 'my_database' as the current database
        session.catalog.set_current_database('my_database')
        ```
    """
    self.catalog.set_current_database(database_name)
```

### set_table_description

```
set_table_description(table_name: str, description: Optional[str] = None) -> None
```

Set or unset the description for a table.

Parameters:

- **`table_name`**
  (`str`)
  –

  Fully qualified or relative table name to set the description for.
- **`description`**
  (`Optional[str]`, default:
  `None`
  )
  –

  The description to set for the table.

Raises:

- `TableNotFoundError`
  –

  If the table doesn't exist.

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def set_table_description(self, table_name: str, description: Optional[str] = None) -> None:
    """Set or unset the description for a table.

    Args:
        table_name: Fully qualified or relative table name to set the description for.
        description: The description to set for the table.

    Raises:
        TableNotFoundError: If the table doesn't exist.
    """
    self.catalog.set_table_description(table_name, description)
```

### set_view_description

```
set_view_description(view_name: str, description: Optional[str] = None) -> None
```

Set the description for a view.

Parameters:

- **`view_name`**
  (`str`)
  –

  Fully qualified or relative view name to set the description for.
- **`description`**
  (`str`, default:
  `None`
  )
  –

  The description to set for the view.

Raises:

- `TableNotFoundError`
  –

  If the view doesn't exist.
- `ValidationError`
  –

  If the description is empty.

Set a description for a view

```python

#### Set a description for a view 'v1'

session.catalog.set_view_description('v1', 'My view description')

Source code in `src/fenic/api/catalog.py`

```
@validate_call(config=ConfigDict(strict=True))
def set_view_description(self, view_name: str, description: Optional[str] = None) -> None:
    """Set the description for a view.

    Args:
        view_name (str): Fully qualified or relative view name to set the description for.
        description (str): The description to set for the view.

    Raises:
        TableNotFoundError: If the view doesn't exist.
        ValidationError: If the description is empty.

    Example: Set a description for a view
        ```python
        # Set a description for a view 'v1'
        session.catalog.set_view_description('v1', 'My view description')
    """
    self.catalog.set_view_description(view_name, description)
```

## ClassDefinition

Bases: `BaseModel`

```
              flowchart TD
              fenic.ClassDefinition[ClassDefinition]

              click fenic.ClassDefinition href "" "fenic.ClassDefinition"
```

Definition of a classification class with optional description.

Used to define the available classes for semantic classification operations.
The description helps the LLM understand what each class represents.

## ClassifyExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.ClassifyExample[ClassifyExample]

              click fenic.ClassifyExample href "" "fenic.ClassifyExample"
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
              fenic.ClassifyExampleCollection[ClassifyExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.ClassifyExampleCollection

              click fenic.ClassifyExampleCollection href "" "fenic.ClassifyExampleCollection"
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

## CloudConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.CloudConfig[CloudConfig]

              click fenic.CloudConfig href "" "fenic.CloudConfig"
```

Configuration for cloud-based execution.

This class defines settings for running operations in a cloud environment,
allowing for scalable and distributed processing of language model operations.

Attributes:

- **`size`**
  (`Optional[CloudExecutorSize]`)
  –

  Size of the cloud executor instance.
  If None, the default size will be used.

Example

Configuring cloud execution with a specific size:

```
config = CloudConfig(size=CloudExecutorSize.MEDIUM)
```

Using default cloud configuration:

```
config = CloudConfig()
```

## CohereEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.CohereEmbeddingModel[CohereEmbeddingModel]

              click fenic.CohereEmbeddingModel href "" "fenic.CohereEmbeddingModel"
```

Configuration for Cohere embedding models.

This class defines the configuration settings for Cohere embedding models,
including model selection and rate limiting parameters.

Attributes:

- **`model_name`**
  (`CohereEmbeddingModelName`)
  –

  The name of the Cohere model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit for the model.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit for the model.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional dictionary of profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  Default profile name to use if none specified.

Example

Configuring a Cohere embedding model with profiles:

```
cohere_config = CohereEmbeddingModel(
    model_name="embed-v4.0",
    rpm=100,
    tpm=50_000,
    profiles={
        "high_dim": CohereEmbeddingModel.Profile(
            embedding_dimensionality=1536, embedding_task_type="search_document"
        ),
        "classification": CohereEmbeddingModel.Profile(
            embedding_dimensionality=1024, embedding_task_type="classification"
        ),
    },
    default_profile="high_dim",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Cohere embedding models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.CohereEmbeddingModel.Profile[Profile]

              click fenic.CohereEmbeddingModel.Profile href "" "fenic.CohereEmbeddingModel.Profile"
```

Profile configurations for Cohere embedding models.

This class defines profile configurations for Cohere embedding models, allowing
different output dimensionality and task type settings to be applied to the same model.

Attributes:

- **`output_dimensionality`**
  (`Optional[int]`)
  –

  The dimensionality of the embedding created by this model.
  If not provided, the model will use its default dimensionality.
- **`input_type`**
  (`CohereEmbeddingTaskType`)
  –

  The type of input text (search_query, search_document, classification, clustering)

Example

Configuring a profile with custom dimensionality:

```
profile = CohereEmbeddingModel.Profile(output_dimensionality=1536)
```

Configuring a profile with default settings:

```
profile = CohereEmbeddingModel.Profile()
```

## Column

A column expression in a DataFrame.

This class represents a column expression that can be used in DataFrame operations.
It provides methods for accessing, transforming, and combining column data.

Create a column reference

```
# Reference a column by name using col() function
col("column_name")
```

Use column in operations

```
# Perform arithmetic operations
df.select(col("price") * col("quantity"))
```

Chain column operations

```
# Chain multiple operations
df.select(col("name").upper().contains("John"))
```

Methods:

- **`alias`**
  –

  Create an alias for this column.
- **`asc`**
  –

  Mark this column for ascending sort order.
- **`asc_nulls_first`**
  –

  Alias for asc().
- **`asc_nulls_last`**
  –

  Mark this column for ascending sort order with nulls last.
- **`cast`**
  –

  Cast the column to a new data type.
- **`contains`**
  –

  Check if the column contains a substring.
- **`contains_any`**
  –

  Check if the column contains any of the specified substrings.
- **`desc`**
  –

  Mark this column for descending sort order.
- **`desc_nulls_first`**
  –

  Alias for desc().
- **`desc_nulls_last`**
  –

  Mark this column for descending sort order with nulls last.
- **`ends_with`**
  –

  Check if the column ends with a substring.
- **`get_item`**
  –

  Access an item in a struct or array column.
- **`ilike`**
  –

  Check if the column matches a SQL LIKE pattern (case-insensitive).
- **`is_in`**
  –

  Check if the column is in a list of values or a column expression.
- **`is_not_null`**
  –

  Check if the column contains non-NULL values.
- **`is_null`**
  –

  Check if the column contains NULL values.
- **`like`**
  –

  Check if the column matches a SQL LIKE pattern.
- **`otherwise`**
  –

  Returns a value when no prior conditions are True.
- **`rlike`**
  –

  Check if the column matches a regular expression pattern.
- **`starts_with`**
  –

  Check if the column starts with a substring.
- **`when`**
  –

  Evaluates a condition for each row and returns a value when true.

### alias

```
alias(name: str) -> Column
```

Create an alias for this column.

This method assigns a new name to the column expression, which is useful
for renaming columns or providing names for complex expressions.

Parameters:

- **`name`**
  (`str`)
  –

  The alias name to assign

Returns:

- **`Column`** ( `Column`
  ) –

  Column with the specified alias

Rename a column

```
# Rename a column to a new name
df.select(col("original_name").alias("new_name"))
```

Name a complex expression

```
# Give a name to a calculated column
df.select((col("price") * col("quantity")).alias("total_value"))
```

Source code in `src/fenic/api/column.py`

```
def alias(self, name: str) -> Column:
    """Create an alias for this column.

    This method assigns a new name to the column expression, which is useful
    for renaming columns or providing names for complex expressions.

    Args:
        name (str): The alias name to assign

    Returns:
        Column: Column with the specified alias

    Example: Rename a column
        ```python
        # Rename a column to a new name
        df.select(col("original_name").alias("new_name"))
        ```

    Example: Name a complex expression
        ```python
        # Give a name to a calculated column
        df.select((col("price") * col("quantity")).alias("total_value"))
        ```
    """
    return Column._from_logical_expr(AliasExpr(self._logical_expr, name))
```

### asc

```
asc() -> Column
```

Mark this column for ascending sort order.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with ascending order and nulls first.

Sort by age in ascending order

```
# Sort a dataframe by age in ascending order
df.sort(col("age").asc()).show()
```

Source code in `src/fenic/api/column.py`

```
def asc(self) -> Column:
    """Mark this column for ascending sort order.

    Returns:
        Column: A sort expression with ascending order and nulls first.

    Example: Sort by age in ascending order
        ```python
        # Sort a dataframe by age in ascending order
        df.sort(col("age").asc()).show()
        ```
    """
    return Column._from_logical_expr(SortExpr(self._logical_expr, ascending=True))
```

### asc_nulls_first

```
asc_nulls_first() -> Column
```

Alias for asc().

Returns:

- **`Column`** ( `Column`
  ) –

  A Column expression that provides a column and sort order to the sort function

Source code in `src/fenic/api/column.py`

```
def asc_nulls_first(self) -> Column:
    """Alias for asc().

    Returns:
        Column: A Column expression that provides a column and sort order to the sort function
    """
    return self.asc()
```

### asc_nulls_last

```
asc_nulls_last() -> Column
```

Mark this column for ascending sort order with nulls last.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with ascending order and nulls last.

Sort by age in ascending order with nulls last

```
# Sort a dataframe by age in ascending order, with nulls appearing last
df.sort(col("age").asc_nulls_last()).show()
```

Source code in `src/fenic/api/column.py`

```
def asc_nulls_last(self) -> Column:
    """Mark this column for ascending sort order with nulls last.

    Returns:
        Column: A sort expression with ascending order and nulls last.

    Example: Sort by age in ascending order with nulls last
        ```python
        # Sort a dataframe by age in ascending order, with nulls appearing last
        df.sort(col("age").asc_nulls_last()).show()
        ```
    """
    return Column._from_logical_expr(
        SortExpr(self._logical_expr, ascending=True, nulls_last=True)
    )
```

### cast

```
cast(data_type: DataType) -> Column
```

Cast the column to a new data type.

This method creates an expression that casts the column to a specified data type.
The casting behavior depends on the source and target types:

Primitive type casting:

- Numeric types (IntegerType, FloatType, DoubleType) can be cast between each other
- Numeric types can be cast to/from StringType
- DateType and TimestampType can be cast between each other
- DateType and TimestampType can be cast to/from numeric types and StringType
- BooleanType can be cast to/from numeric types and StringType
- StringType cannot be directly cast to BooleanType (will raise TypeError)
- BooleanType cannot be cast to DateType or TimestampType (will raise TypeError)
- DateType and TimestampType cannot be cast to BooleanType (will raise TypeError)

Complex type casting:

- ArrayType can only be cast to another ArrayType (with castable element types)
- StructType can only be cast to another StructType (with matching/castable fields)
- Primitive types cannot be cast to/from complex types

Parameters:

- **`data_type`**
  (`DataType`)
  –

  The target DataType to cast the column to

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing the casted expression

Cast integer to string

```
# Convert an integer column to string type
df.select(col("int_col").cast(StringType))
```

Cast array of integers to array of strings

```
# Convert an array of integers to an array of strings
df.select(col("int_array").cast(ArrayType(element_type=StringType)))
```

Cast struct fields to different types

```
# Convert struct fields to different types
new_type = StructType([
    StructField("id", StringType),
    StructField("value", FloatType)
])
df.select(col("data_struct").cast(new_type))
```

Raises:

- `TypeError`
  –

  If the requested cast operation is not supported

Source code in `src/fenic/api/column.py`

```
def cast(self, data_type: DataType) -> Column:
    """Cast the column to a new data type.

    This method creates an expression that casts the column to a specified data type.
    The casting behavior depends on the source and target types:

    Primitive type casting:

    - Numeric types (IntegerType, FloatType, DoubleType) can be cast between each other
    - Numeric types can be cast to/from StringType
    - DateType and TimestampType can be cast between each other
    - DateType and TimestampType can be cast to/from numeric types and StringType
    - BooleanType can be cast to/from numeric types and StringType
    - StringType cannot be directly cast to BooleanType (will raise TypeError)
    - BooleanType cannot be cast to DateType or TimestampType (will raise TypeError)
    - DateType and TimestampType cannot be cast to BooleanType (will raise TypeError)

    Complex type casting:

    - ArrayType can only be cast to another ArrayType (with castable element types)
    - StructType can only be cast to another StructType (with matching/castable fields)
    - Primitive types cannot be cast to/from complex types

    Args:
        data_type (DataType): The target DataType to cast the column to

    Returns:
        Column: A Column representing the casted expression

    Example: Cast integer to string
        ```python
        # Convert an integer column to string type
        df.select(col("int_col").cast(StringType))
        ```

    Example: Cast array of integers to array of strings
        ```python
        # Convert an array of integers to an array of strings
        df.select(col("int_array").cast(ArrayType(element_type=StringType)))
        ```

    Example: Cast struct fields to different types
        ```python
        # Convert struct fields to different types
        new_type = StructType([
            StructField("id", StringType),
            StructField("value", FloatType)
        ])
        df.select(col("data_struct").cast(new_type))
        ```

    Raises:
        TypeError: If the requested cast operation is not supported
    """
    return Column._from_logical_expr(CastExpr(self._logical_expr, data_type))
```

### contains

```
contains(other: Union[str, Column]) -> Column
```

Check if the column contains a substring.

This method creates a boolean expression that checks if each value in the column
contains the specified substring. The substring can be either a literal string
or a column expression.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The substring to search for (can be a string or column expression)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value contains the substring

Find rows where name contains "john"

```
# Filter rows where the name column contains "john"
df.filter(col("name").contains("john"))
```

Find rows where text contains a dynamic pattern

```
# Filter rows where text contains a value from another column
df.filter(col("text").contains(col("pattern")))
```

Source code in `src/fenic/api/column.py`

```
def contains(self, other: Union[str, Column]) -> Column:
    """Check if the column contains a substring.

    This method creates a boolean expression that checks if each value in the column
    contains the specified substring. The substring can be either a literal string
    or a column expression.

    Args:
        other (Union[str, Column]): The substring to search for (can be a string or column expression)

    Returns:
        Column: A boolean column indicating whether each value contains the substring

    Example: Find rows where name contains "john"
        ```python
        # Filter rows where the name column contains "john"
        df.filter(col("name").contains("john"))
        ```

    Example: Find rows where text contains a dynamic pattern
        ```python
        # Filter rows where text contains a value from another column
        df.filter(col("text").contains(col("pattern")))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(ContainsExpr(self._logical_expr, other_expr))
```

### contains_any

```
contains_any(others: List[str], case_insensitive: bool = True) -> Column
```

Check if the column contains any of the specified substrings.

This method creates a boolean expression that checks if each value in the column
contains any of the specified substrings. The matching can be case-sensitive or
case-insensitive.

Parameters:

- **`others`**
  (`List[str]`)
  –

  List of substrings to search for
- **`case_insensitive`**
  (`bool`, default:
  `True`
  )
  –

  Whether to perform case-insensitive matching (default: True)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value contains any substring

Find rows where name contains "john" or "jane" (case-insensitive)

```
# Filter rows where name contains either "john" or "jane"
df.filter(col("name").contains_any(["john", "jane"]))
```

Case-sensitive matching

```
# Filter rows with case-sensitive matching
df.filter(col("name").contains_any(["John", "Jane"], case_insensitive=False))
```

Source code in `src/fenic/api/column.py`

```
def contains_any(self, others: List[str], case_insensitive: bool = True) -> Column:
    """Check if the column contains any of the specified substrings.

    This method creates a boolean expression that checks if each value in the column
    contains any of the specified substrings. The matching can be case-sensitive or
    case-insensitive.

    Args:
        others (List[str]): List of substrings to search for
        case_insensitive (bool): Whether to perform case-insensitive matching (default: True)

    Returns:
        Column: A boolean column indicating whether each value contains any substring

    Example: Find rows where name contains "john" or "jane" (case-insensitive)
        ```python
        # Filter rows where name contains either "john" or "jane"
        df.filter(col("name").contains_any(["john", "jane"]))
        ```

    Example: Case-sensitive matching
        ```python
        # Filter rows with case-sensitive matching
        df.filter(col("name").contains_any(["John", "Jane"], case_insensitive=False))
        ```
    """
    return Column._from_logical_expr(
        ContainsAnyExpr(self._logical_expr, others, case_insensitive)
    )
```

### desc

```
desc() -> Column
```

Mark this column for descending sort order.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with descending order.

Sort by age in descending order

```
# Sort a dataframe by age in descending order
df.sort(col("age").desc()).show()
```

Source code in `src/fenic/api/column.py`

```
def desc(self) -> Column:
    """Mark this column for descending sort order.

    Returns:
        Column: A sort expression with descending order.

    Example: Sort by age in descending order
        ```python
        # Sort a dataframe by age in descending order
        df.sort(col("age").desc()).show()
        ```
    """
    return Column._from_logical_expr(
        SortExpr(self._logical_expr, ascending=False)
    )
```

### desc_nulls_first

```
desc_nulls_first() -> Column
```

Alias for desc().

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with descending order and nulls first.

Sort by age in descending order with nulls first

```
df.sort(col("age").desc_nulls_first()).show()
```

Source code in `src/fenic/api/column.py`

```
def desc_nulls_first(self) -> Column:
    """Alias for desc().

    Returns:
        Column: A sort expression with descending order and nulls first.

    Example: Sort by age in descending order with nulls first
        ```python
        df.sort(col("age").desc_nulls_first()).show()
        ```
    """
    return self.desc()
```

### desc_nulls_last

```
desc_nulls_last() -> Column
```

Mark this column for descending sort order with nulls last.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with descending order and nulls last.

Sort by age in descending order with nulls last

```
# Sort a dataframe by age in descending order, with nulls appearing last
df.sort(col("age").desc_nulls_last()).show()
```

Source code in `src/fenic/api/column.py`

```
def desc_nulls_last(self) -> Column:
    """Mark this column for descending sort order with nulls last.

    Returns:
        Column: A sort expression with descending order and nulls last.

    Example: Sort by age in descending order with nulls last
        ```python
        # Sort a dataframe by age in descending order, with nulls appearing last
        df.sort(col("age").desc_nulls_last()).show()
        ```
    """
    return Column._from_logical_expr(
        SortExpr(self._logical_expr, ascending=False, nulls_last=True)
    )
```

### ends_with

```
ends_with(other: Union[str, Column]) -> Column
```

Check if the column ends with a substring.

This method creates a boolean expression that checks if each value in the column
ends with the specified substring. The substring can be either a literal string
or a column expression.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The substring to check for at the end (can be a string or column expression)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value ends with the substring

Find rows where email ends with "@gmail.com"

```
df.filter(col("email").ends_with("@gmail.com"))
```

Find rows where text ends with a dynamic pattern

```
df.filter(col("text").ends_with(col("suffix")))
```

Raises:

- `ValueError`
  –

  If the substring ends with a regular expression anchor ($)

Source code in `src/fenic/api/column.py`

```
def ends_with(self, other: Union[str, Column]) -> Column:
    """Check if the column ends with a substring.

    This method creates a boolean expression that checks if each value in the column
    ends with the specified substring. The substring can be either a literal string
    or a column expression.

    Args:
        other (Union[str, Column]): The substring to check for at the end (can be a string or column expression)

    Returns:
        Column: A boolean column indicating whether each value ends with the substring

    Example: Find rows where email ends with "@gmail.com"
        ```python
        df.filter(col("email").ends_with("@gmail.com"))
        ```

    Example: Find rows where text ends with a dynamic pattern
        ```python
        df.filter(col("text").ends_with(col("suffix")))
        ```

    Raises:
        ValueError: If the substring ends with a regular expression anchor ($)
    """
    if isinstance(other, str):
        if other.endswith("$"):
            raise ValidationError("substr should not end with a regular expression anchor")
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(EndsWithExpr(self._logical_expr, other_expr))
```

### get_item

```
get_item(key: Union[str, int, Column]) -> Column
```

Access an item in a struct or array column.

This method allows accessing elements in complex data types:

- For array columns, the key should be an integer index or a column expression that evaluates to an integer
- For struct columns, the key should be a literal field name

Parameters:

- **`key`**
  (`Union[str, int]`)
  –

  The index (for arrays) or field name (for structs) to access

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing the accessed item

Access an array element

```
# Get the first element from an array column
df.select(col("array_column").get_item(0))
```

Access a struct field

```
# Get a field from a struct column
df.select(col("struct_column").get_item("field_name"))
```

Source code in `src/fenic/api/column.py`

```
def get_item(self, key: Union[str, int, Column]) -> Column:
    """Access an item in a struct or array column.

    This method allows accessing elements in complex data types:

    - For array columns, the key should be an integer index or a column expression that evaluates to an integer
    - For struct columns, the key should be a literal field name

    Args:
        key (Union[str, int]): The index (for arrays) or field name (for structs) to access

    Returns:
        Column: A Column representing the accessed item

    Example: Access an array element
        ```python
        # Get the first element from an array column
        df.select(col("array_column").get_item(0))
        ```

    Example: Access a struct field
        ```python
        # Get a field from a struct column
        df.select(col("struct_column").get_item("field_name"))
        ```
    """
    if isinstance(key, Column):
        return Column._from_logical_expr(IndexExpr(self._logical_expr, key._logical_expr))
    elif isinstance(key, str):
        return Column._from_logical_expr(IndexExpr(self._logical_expr, LiteralExpr(key, StringType)))
    else:
        return Column._from_logical_expr(IndexExpr(self._logical_expr, LiteralExpr(key, IntegerType)))
```

### ilike

```
ilike(other: Union[str, Column]) -> Column
```

Check if the column matches a SQL LIKE pattern (case-insensitive).

This method creates a boolean expression that checks if each value in the column
matches the specified SQL LIKE pattern, ignoring case.
The pattern can be a string or a a column expression that resolves to a string.

SQL LIKE pattern syntax:

- % matches any sequence of characters
- _ matches any single character

Parameters:

- **`other`**
  (`str`)
  –

  The SQL LIKE pattern to match against

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value matches the pattern

Find rows where name starts with "j" and ends with "n" (case-insensitive)

```
# Filter rows where name matches the pattern "j%n" (case-insensitive)
df.filter(col("name").ilike("j%n"))
```

Find rows where code matches pattern (case-insensitive)

```
# Filter rows where code matches the pattern "a_b%" (case-insensitive)
df.filter(col("code").ilike("a_b%"))
```

Source code in `src/fenic/api/column.py`

```
def ilike(self, other: Union[str, Column]) -> Column:
    r"""Check if the column matches a SQL LIKE pattern (case-insensitive).

    This method creates a boolean expression that checks if each value in the column
    matches the specified SQL LIKE pattern, ignoring case.
    The pattern can be a string or a a column expression that resolves to a string.

    SQL LIKE pattern syntax:

    - % matches any sequence of characters
    - _ matches any single character

    Args:
        other (str): The SQL LIKE pattern to match against

    Returns:
        Column: A boolean column indicating whether each value matches the pattern

    Example: Find rows where name starts with "j" and ends with "n" (case-insensitive)
        ```python
        # Filter rows where name matches the pattern "j%n" (case-insensitive)
        df.filter(col("name").ilike("j%n"))
        ```

    Example: Find rows where code matches pattern (case-insensitive)
        ```python
        # Filter rows where code matches the pattern "a_b%" (case-insensitive)
        df.filter(col("code").ilike("a_b%"))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(ILikeExpr(self._logical_expr, other_expr))
```

### is_in

```
is_in(other: Union[List[Any], ColumnOrName]) -> Column
```

Check if the column is in a list of values or a column expression.

Parameters:

- **`other`**
  (`Union[List[Any], ColumnOrName]`)
  –

  A list of values or a Column expression

Returns:

- **`Column`** ( `Column`
  ) –

  A Column expression representing whether each element of Column is in the list

Check if name is in a list of values

```
# Filter rows where name is in a list of values
df.filter(col("name").is_in(["Alice", "Bob"]))
```

Check if value is in another column

```
# Filter rows where name is in another column
df.filter(col("name").is_in(col("other_column")))
```

Source code in `src/fenic/api/column.py`

```
def is_in(self, other: Union[List[Any], ColumnOrName]) -> Column:
    """Check if the column is in a list of values or a column expression.

    Args:
        other (Union[List[Any], ColumnOrName]): A list of values or a Column expression

    Returns:
        Column: A Column expression representing whether each element of Column is in the list

    Example: Check if name is in a list of values
        ```python
        # Filter rows where name is in a list of values
        df.filter(col("name").is_in(["Alice", "Bob"]))
        ```

    Example: Check if value is in another column
        ```python
        # Filter rows where name is in another column
        df.filter(col("name").is_in(col("other_column")))
        ```
    """
    if isinstance(other, list):
        try:
            type_ = infer_dtype_from_pyobj(other)
            return Column._from_logical_expr(InExpr(self._logical_expr, LiteralExpr(other, type_)))
        except TypeInferenceError as e:
            raise ValidationError(f"Cannot apply IN on {other}. List argument to IN must be be a valid Python List literal.") from e
    else:
        return Column._from_logical_expr(InExpr(self._logical_expr, other._logical_expr))
```

### is_not_null

```
is_not_null() -> Column
```

Check if the column contains non-NULL values.

This method creates an expression that evaluates to TRUE when the column value is not NULL.

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing a boolean expression that is TRUE when this column is not NULL

Filter rows where a column is not NULL

```
df.filter(col("some_column").is_not_null())
```

Use in a complex condition

```
df.filter(col("col1").is_not_null() & (col("col2") <= 50))
```

Source code in `src/fenic/api/column.py`

```
def is_not_null(self) -> Column:
    """Check if the column contains non-NULL values.

    This method creates an expression that evaluates to TRUE when the column value is not NULL.

    Returns:
        Column: A Column representing a boolean expression that is TRUE when this column is not NULL

    Example: Filter rows where a column is not NULL
        ```python
        df.filter(col("some_column").is_not_null())
        ```

    Example: Use in a complex condition
        ```python
        df.filter(col("col1").is_not_null() & (col("col2") <= 50))
        ```
    """
    return Column._from_logical_expr(IsNullExpr(self._logical_expr, False))
```

### is_null

```
is_null() -> Column
```

Check if the column contains NULL values.

This method creates an expression that evaluates to TRUE when the column value is NULL.

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing a boolean expression that is TRUE when this column is NULL

Filter rows where a column is NULL

```
# Filter rows where some_column is NULL
df.filter(col("some_column").is_null())
```

Use in a complex condition

```
# Filter rows where col1 is NULL or col2 is greater than 100
df.filter(col("col1").is_null() | (col("col2") > 100))
```

Source code in `src/fenic/api/column.py`

```
def is_null(self) -> Column:
    """Check if the column contains NULL values.

    This method creates an expression that evaluates to TRUE when the column value is NULL.

    Returns:
        Column: A Column representing a boolean expression that is TRUE when this column is NULL

    Example: Filter rows where a column is NULL
        ```python
        # Filter rows where some_column is NULL
        df.filter(col("some_column").is_null())
        ```

    Example: Use in a complex condition
        ```python
        # Filter rows where col1 is NULL or col2 is greater than 100
        df.filter(col("col1").is_null() | (col("col2") > 100))
        ```
    """
    return Column._from_logical_expr(IsNullExpr(self._logical_expr, True))
```

### like

```
like(other: Union[str, Column]) -> Column
```

Check if the column matches a SQL LIKE pattern.

This method creates a boolean expression that checks if each value in the column
matches the specified SQL LIKE pattern.
The pattern can be a string or a a column expression that resolves to a string.

SQL LIKE pattern syntax:

- % matches any sequence of characters
- _ matches any single character

Parameters:

- **`other`**
  (`str`)
  –

  The SQL LIKE pattern to match against

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value matches the pattern

Find rows where name starts with "J" and ends with "n"

```
# Filter rows where name matches the pattern "J%n"
df.filter(col("name").like("J%n"))
```

Find rows where code matches specific pattern

```
# Filter rows where code matches the pattern "A_B%"
df.filter(col("code").like("A_B%"))
```

Source code in `src/fenic/api/column.py`

```
def like(self, other: Union[str, Column]) -> Column:
    r"""Check if the column matches a SQL LIKE pattern.

    This method creates a boolean expression that checks if each value in the column
    matches the specified SQL LIKE pattern.
    The pattern can be a string or a a column expression that resolves to a string.

    SQL LIKE pattern syntax:

    - % matches any sequence of characters
    - _ matches any single character

    Args:
        other (str): The SQL LIKE pattern to match against

    Returns:
        Column: A boolean column indicating whether each value matches the pattern

    Example: Find rows where name starts with "J" and ends with "n"
        ```python
        # Filter rows where name matches the pattern "J%n"
        df.filter(col("name").like("J%n"))
        ```

    Example: Find rows where code matches specific pattern
        ```python
        # Filter rows where code matches the pattern "A_B%"
        df.filter(col("code").like("A_B%"))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(LikeExpr(self._logical_expr, other_expr))
```

### otherwise

```
otherwise(value: Column) -> Column
```

Returns a value when no prior conditions are True.

This is the final part of a when-chain, like the 'else' in an if-elif-else.
All branches must return the same type.

Parameters:

- **`value`**
  (`Column`)
  –

  Value to return when no prior conditions are True

Returns:

- **`Column`** ( `Column`
  ) –

  The complete conditional expression chain

Raises:

- `ValidationError`
  –

  If called on a non-when expression

Example

```
# Create age-based categories
df = session.createDataFrame({"age": [8, 25, 67]})

result = df.select(
    fc.when(col("age") < 18, fc.lit("minor"))
    .when(col("age") < 65, fc.lit("adult"))
    .otherwise(fc.lit("senior"))
    .alias("category")
)

result.show()
# +--------+
# |category|
# +--------+
# |   minor|
# |   adult|
# |  senior|
# +--------+
```

Note

- If otherwise() is not called, unmatched rows return null
- Can only be called once per when-chain
- Typically the last method in the chain

Source code in `src/fenic/api/column.py`

```
def otherwise(self, value: Column) -> Column:
    """Returns a value when no prior conditions are True.

    This is the final part of a when-chain, like the 'else' in an if-elif-else.
    All branches must return the same type.

    Args:
        value: Value to return when no prior conditions are True

    Returns:
        Column: The complete conditional expression chain

    Raises:
        ValidationError: If called on a non-when expression

    Example:
        ```python
        # Create age-based categories
        df = session.createDataFrame({"age": [8, 25, 67]})

        result = df.select(
            fc.when(col("age") < 18, fc.lit("minor"))
            .when(col("age") < 65, fc.lit("adult"))
            .otherwise(fc.lit("senior"))
            .alias("category")
        )

        result.show()
        # +--------+
        # |category|
        # +--------+
        # |   minor|
        # |   adult|
        # |  senior|
        # +--------+
        ```

    Note:
        - If otherwise() is not called, unmatched rows return null
        - Can only be called once per when-chain
        - Typically the last method in the chain
    """
    return Column._from_logical_expr(OtherwiseExpr(self._logical_expr, value._logical_expr))
```

### rlike

```
rlike(other: Union[str, Column]) -> Column
```

Check if the column matches a regular expression pattern.

This method creates a boolean expression that checks if each value in the column
matches the specified regular expression pattern.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The regular expression pattern to match against.
  Can be a string or a a column expression that resolves to a string.

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value matches the pattern

Find rows where phone number matches pattern

```
# Filter rows where phone number matches a specific pattern
df.filter(col("phone").rlike(r"^\d{3}-\d{3}-\d{4}$"))
```

Find rows where text contains word boundaries

```
# Filter rows where text contains a word with boundaries
df.filter(col("text").rlike(r"\bhello\b"))
```

Source code in `src/fenic/api/column.py`

```
def rlike(self, other: Union[str, Column]) -> Column:
    r"""Check if the column matches a regular expression pattern.

    This method creates a boolean expression that checks if each value in the column
    matches the specified regular expression pattern.

    Args:
        other (Union[str, Column]): The regular expression pattern to match against.
              Can be a string or a a column expression that resolves to a string.

    Returns:
        Column: A boolean column indicating whether each value matches the pattern

    Example: Find rows where phone number matches pattern
        ```python
        # Filter rows where phone number matches a specific pattern
        df.filter(col("phone").rlike(r"^\d{3}-\d{3}-\d{4}$"))
        ```

    Example: Find rows where text contains word boundaries
        ```python
        # Filter rows where text contains a word with boundaries
        df.filter(col("text").rlike(r"\bhello\b"))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(RLikeExpr(self._logical_expr, other_expr))
```

### starts_with

```
starts_with(other: Union[str, Column]) -> Column
```

Check if the column starts with a substring.

This method creates a boolean expression that checks if each value in the column
starts with the specified substring. The substring can be either a literal string
or a column expression.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The substring to check for at the start (can be a string or column expression)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value starts with the substring

Find rows where name starts with "Mr"

```
# Filter rows where name starts with "Mr"
df.filter(col("name").starts_with("Mr"))
```

Find rows where text starts with a dynamic pattern

```
# Filter rows where text starts with a value from another column
df.filter(col("text").starts_with(col("prefix")))
```

Raises:

- `ValueError`
  –

  If the substring starts with a regular expression anchor (^)

Source code in `src/fenic/api/column.py`

```
def starts_with(self, other: Union[str, Column]) -> Column:
    """Check if the column starts with a substring.

    This method creates a boolean expression that checks if each value in the column
    starts with the specified substring. The substring can be either a literal string
    or a column expression.

    Args:
        other (Union[str, Column]): The substring to check for at the start (can be a string or column expression)

    Returns:
        Column: A boolean column indicating whether each value starts with the substring

    Example: Find rows where name starts with "Mr"
        ```python
        # Filter rows where name starts with "Mr"
        df.filter(col("name").starts_with("Mr"))
        ```

    Example: Find rows where text starts with a dynamic pattern
        ```python
        # Filter rows where text starts with a value from another column
        df.filter(col("text").starts_with(col("prefix")))
        ```

    Raises:
        ValueError: If the substring starts with a regular expression anchor (^)
    """
    if isinstance(other, str):
        if other.startswith("^"):
            raise ValidationError("substr should not start with a regular expression anchor")
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(StartsWithExpr(self._logical_expr, other_expr))
```

### when

```
when(condition: Column, value: Column) -> Column
```

Evaluates a condition for each row and returns a value when true.

Can be chained with more .when() calls or finished with .otherwise().
All branches must return the same type.

Parameters:

- **`condition`**
  (`Column`)
  –

  Boolean expression to test
- **`value`**
  (`Column`)
  –

  Value to return when condition is True

Returns:

- **`Column`** ( `Column`
  ) –

  A new when expression with this condition added to the chain

Raises:

- `ValidationError`
  –

  If called on a non-when expression (e.g., regular columns)

Example

```
# Build a multi-condition expression:
result = (
    fc.when(col("age") < 18, fc.lit("minor"))
    .when(col("age") < 65, fc.lit("adult"))  # Add another condition
    .when(col("age") >= 65, fc.lit("senior"))
    .otherwise(fc.lit("unknown"))
)

# This evaluates like:
# if age < 18: return "minor"
# elif age < 65: return "adult"
# elif age >= 65: return "senior"
# else: return "unknown"

# ERROR - cannot call .when() on non-when expressions:
# col("age").when(...)  # ValidationError!
```

Note: Conditions are evaluated in order. The first True condition wins.

Source code in `src/fenic/api/column.py`

```
def when(self, condition: Column, value: Column) -> Column:
    """Evaluates a condition for each row and returns a value when true.

    Can be chained with more .when() calls or finished with .otherwise().
    All branches must return the same type.

    Args:
        condition: Boolean expression to test
        value: Value to return when condition is True

    Returns:
        Column: A new when expression with this condition added to the chain

    Raises:
        ValidationError: If called on a non-when expression (e.g., regular columns)

    Example:
        ```python
        # Build a multi-condition expression:
        result = (
            fc.when(col("age") < 18, fc.lit("minor"))
            .when(col("age") < 65, fc.lit("adult"))  # Add another condition
            .when(col("age") >= 65, fc.lit("senior"))
            .otherwise(fc.lit("unknown"))
        )

        # This evaluates like:
        # if age < 18: return "minor"
        # elif age < 65: return "adult"
        # elif age >= 65: return "senior"
        # else: return "unknown"

        # ERROR - cannot call .when() on non-when expressions:
        # col("age").when(...)  # ValidationError!
        ```

    Note: Conditions are evaluated in order. The first True condition wins.
    """
    return Column._from_logical_expr(WhenExpr(self._logical_expr, condition._logical_expr, value._logical_expr))
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

## DataFrame

A data collection organized into named columns.

The DataFrame class represents a lazily evaluated computation on data. Operations on
DataFrame build up a logical query plan that is only executed when an action like
show(), to_polars(), to_pandas(), to_arrow(), to_pydict(), to_pylist(), or count() is called.

The DataFrame supports method chaining for building complex transformations.

Create and transform a DataFrame

```
# Create a DataFrame from a dictionary
df = session.create_dataframe({"id": [1, 2, 3], "value": ["a", "b", "c"]})

# Chain transformations
result = df.filter(col("id") > 1).select("id", "value")

# Show results
result.show()
# Output:
# +---+-----+
# | id|value|
# +---+-----+
# |  2|    b|
# |  3|    c|
# +---+-----+
```

Methods:

- **`agg`**
  –

  Aggregate on the entire DataFrame without groups.
- **`cache`**
  –

  Alias for persist(). Mark DataFrame for caching after first computation.
- **`collect`**
  –

  Execute the DataFrame computation and return the result as a QueryResult.
- **`count`**
  –

  Count the number of rows in the DataFrame.
- **`distinct`**
  –

  Return a DataFrame with duplicate rows removed. Alias for drop_duplicates(subset=None).
- **`drop`**
  –

  Remove one or more columns from this DataFrame.
- **`drop_duplicates`**
  –

  Return a DataFrame with duplicate rows removed.
- **`explain`**
  –

  Display the logical plan of the DataFrame.
- **`explode`**
  –

  Create a new row for each element in an array column.
- **`explode_outer`**
  –

  Create a new row for each element in an array column, containing the element's position in the array and its value, and preserving null/empty arrays.
- **`explode_with_index`**
  –

  Create a new row for each element in an array column, with the element's position in the array and its value.
- **`filter`**
  –

  Filters rows using the given condition.
- **`group_by`**
  –

  Groups the DataFrame using the specified columns.
- **`join`**
  –

  Joins this DataFrame with another DataFrame.
- **`limit`**
  –

  Limits the number of rows to the specified number.
- **`lineage`**
  –

  Create a Lineage object to trace data through transformations.
- **`order_by`**
  –

  Sort the DataFrame by the specified columns. Alias for sort().
- **`persist`**
  –

  Mark this DataFrame to be persisted after first computation.
- **`posexplode`**
  –

  Create a new row for each element in an array column, with the element's position in the array and its value.
- **`posexplode_outer`**
  –

  Create a new row for each element in an array column with position and value, preserving null/empty arrays.
- **`select`**
  –

  Projects a set of Column expressions or column names.
- **`show`**
  –

  Display the DataFrame content in a tabular form.
- **`sort`**
  –

  Sort the DataFrame by the specified columns.
- **`to_arrow`**
  –

  Execute the DataFrame computation and return an Apache Arrow Table.
- **`to_pandas`**
  –

  Execute the DataFrame computation and return a Pandas DataFrame.
- **`to_polars`**
  –

  Execute the DataFrame computation and return the result as a Polars DataFrame.
- **`to_pydict`**
  –

  Execute the DataFrame computation and return a dictionary of column arrays.
- **`to_pylist`**
  –

  Execute the DataFrame computation and return a list of row dictionaries.
- **`union`**
  –

  Return a new DataFrame containing the union of rows in this and another DataFrame.
- **`unnest`**
  –

  Unnest the specified struct columns into separate columns.
- **`where`**
  –

  Filters rows using the given condition (alias for filter()).
- **`with_column`**
  –

  Add a new column or replace an existing column.
- **`with_column_renamed`**
  –

  Rename a column. No-op if the column does not exist.
- **`with_columns`**
  –

  Add multiple new columns or replace existing columns.

Attributes:

- **`columns`**
  (`List[str]`)
  –

  Get list of column names.
- **`schema`**
  (`Schema`)
  –

  Get the schema of this DataFrame.
- **`semantic`**
  (`SemanticExtensions`)
  –

  Interface for semantic operations on the DataFrame.
- **`write`**
  (`DataFrameWriter`)
  –

  Interface for saving the content of the DataFrame.

### columns

```
columns: List[str]
```

Get list of column names.

Returns:

- `List[str]`
  –

  List[str]: List of all column names in the DataFrame

Examples:

```
>>> df.columns
['name', 'age', 'city']
```

### schema

```
schema: Schema
```

Get the schema of this DataFrame.

Returns:

- **`Schema`** ( `Schema`
  ) –

  Schema containing field names and data types

Examples:

```
>>> df.schema
Schema([
    ColumnField('name', StringType),
    ColumnField('age', IntegerType)
])
```

### semantic

```
semantic: SemanticExtensions
```

Interface for semantic operations on the DataFrame.

### write

```
write: DataFrameWriter
```

Interface for saving the content of the DataFrame.

Returns:

- **`DataFrameWriter`** ( `DataFrameWriter`
  ) –

  Writer interface to write DataFrame.

### agg

```
agg(*exprs: Union[Column, Dict[str, str]]) -> DataFrame
```

Aggregate on the entire DataFrame without groups.

This is equivalent to group_by() without any grouping columns.

Parameters:

- **`*exprs`**
  (`Union[Column, Dict[str, str]]`, default:
  `()`
  )
  –

  Aggregation expressions or dictionary of aggregations.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Aggregation results.

Multiple aggregations

```
# Create sample DataFrame
df = session.create_dataframe({
    "salary": [80000, 70000, 90000, 75000, 85000],
    "age": [25, 30, 35, 28, 32]
})

# Multiple aggregations
df.agg(
    count().alias("total_rows"),
    avg(col("salary")).alias("avg_salary")
).show()
# Output:
# +----------+-----------+
# |total_rows|avg_salary|
# +----------+-----------+
# |         5|   80000.0|
# +----------+-----------+
```

Dictionary style

```
# Dictionary style
df.agg({col("salary"): "avg", col("age"): "max"}).show()
# Output:
# +-----------+--------+
# |avg(salary)|max(age)|
# +-----------+--------+
# |    80000.0|      35|
# +-----------+--------+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def agg(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
    """Aggregate on the entire DataFrame without groups.

    This is equivalent to group_by() without any grouping columns.

    Args:
        *exprs: Aggregation expressions or dictionary of aggregations.

    Returns:
        DataFrame: Aggregation results.

    Example: Multiple aggregations
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "salary": [80000, 70000, 90000, 75000, 85000],
            "age": [25, 30, 35, 28, 32]
        })

        # Multiple aggregations
        df.agg(
            count().alias("total_rows"),
            avg(col("salary")).alias("avg_salary")
        ).show()
        # Output:
        # +----------+-----------+
        # |total_rows|avg_salary|
        # +----------+-----------+
        # |         5|   80000.0|
        # +----------+-----------+
        ```

    Example: Dictionary style
        ```python
        # Dictionary style
        df.agg({col("salary"): "avg", col("age"): "max"}).show()
        # Output:
        # +-----------+--------+
        # |avg(salary)|max(age)|
        # +-----------+--------+
        # |    80000.0|      35|
        # +-----------+--------+
        ```
    """
    return self.group_by().agg(*exprs)
```

### cache

```
cache() -> DataFrame
```

Alias for persist(). Mark DataFrame for caching after first computation.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Same DataFrame, but marked for caching

See Also

persist(): Full documentation of caching behavior

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def cache(self) -> DataFrame:
    """Alias for persist(). Mark DataFrame for caching after first computation.

    Returns:
        DataFrame: Same DataFrame, but marked for caching

    See Also:
        persist(): Full documentation of caching behavior
    """
    return self.persist()
```

### collect

```
collect(data_type: DataLikeType = 'polars') -> QueryResult
```

Execute the DataFrame computation and return the result as a QueryResult.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a QueryResult, which contains both the result data and the query metrics.

Parameters:

- **`data_type`**
  (`DataLikeType`, default:
  `'polars'`
  )
  –

  The type of data to return

Returns:

- **`QueryResult`** ( `QueryResult`
  ) –

  A QueryResult with materialized data and query metrics

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def collect(self, data_type: DataLikeType = "polars") -> QueryResult:
    """Execute the DataFrame computation and return the result as a QueryResult.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a QueryResult, which contains both the result data and the query metrics.

    Args:
        data_type: The type of data to return

    Returns:
        QueryResult: A QueryResult with materialized data and query metrics
    """
    result: Tuple[pl.DataFrame, QueryMetrics] = self._session_state.execution.collect(self._logical_plan)
    df, metrics = result
    logger.info(metrics.get_summary())

    if data_type == "polars":
        return QueryResult(df, metrics)
    elif data_type == "pandas":
        return QueryResult(df.to_pandas(use_pyarrow_extension_array=True), metrics)
    elif data_type == "arrow":
        return QueryResult(df.to_arrow(), metrics)
    elif data_type == "pydict":
        return QueryResult(df.to_dict(as_series=False), metrics)
    elif data_type == "pylist":
        return QueryResult(df.to_dicts(), metrics)
    else:
        raise ValidationError(f"Invalid data type: {data_type} in collect(). Valid data types are: polars, pandas, arrow, pydict, pylist")
```

### count

```
count() -> int
```

Count the number of rows in the DataFrame.

This is an action that triggers computation of the DataFrame.
The output is an integer representing the number of rows.

Returns:

- **`int`** ( `int`
  ) –

  The number of rows in the DataFrame

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def count(self) -> int:
    """Count the number of rows in the DataFrame.

    This is an action that triggers computation of the DataFrame.
    The output is an integer representing the number of rows.

    Returns:
        int: The number of rows in the DataFrame
    """
    return self._session_state.execution.count(self._logical_plan)[0]
```

### distinct

```
distinct() -> DataFrame
```

Return a DataFrame with duplicate rows removed. Alias for drop_duplicates(subset=None).

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with duplicate rows removed.

Example

```
# Create sample DataFrame
df = session.create_dataframe({
    "c1": [1, 2, 3, 1],
    "c2": ["a", "a", "a", "a"],
    "c3": ["b", "b", "b", "b"]
})

# Remove duplicates considering all columns
df.distinct().show()
# Output:
# +---+---+---+
# | c1| c2| c3|
# +---+---+---+
# |  1|  a|  b|
# |  2|  a|  b|
# |  3|  a|  b|
# +---+---+---+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def distinct(self) -> DataFrame:
    """Return a DataFrame with duplicate rows removed. Alias for drop_duplicates(subset=None).

    Returns:
        DataFrame: A new DataFrame with duplicate rows removed.

    Example:
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "c1": [1, 2, 3, 1],
            "c2": ["a", "a", "a", "a"],
            "c3": ["b", "b", "b", "b"]
        })

        # Remove duplicates considering all columns
        df.distinct().show()
        # Output:
        # +---+---+---+
        # | c1| c2| c3|
        # +---+---+---+
        # |  1|  a|  b|
        # |  2|  a|  b|
        # |  3|  a|  b|
        # +---+---+---+
        ```
    """
    return self.drop_duplicates()
```

### drop

```
drop(*col_names: str) -> DataFrame
```

Remove one or more columns from this DataFrame.

Parameters:

- **`*col_names`**
  (`str`, default:
  `()`
  )
  –

  Names of columns to drop.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame without specified columns.

Raises:

- `ValueError`
  –

  If any specified column doesn't exist in the DataFrame.
- `ValueError`
  –

  If dropping the columns would result in an empty DataFrame.

Drop single column

```
# Create sample DataFrame
df = session.create_dataframe({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35]
})

# Drop single column
df.drop("age").show()
# Output:
# +---+-------+
# | id|   name|
# +---+-------+
# |  1|  Alice|
# |  2|    Bob|
# |  3|Charlie|
# +---+-------+
```

Drop multiple columns

```
# Drop multiple columns
df.drop(col("id"), "age").show()
# Output:
# +-------+
# |   name|
# +-------+
# |  Alice|
# |    Bob|
# |Charlie|
# +-------+
```

Error when dropping non-existent column

```
# This will raise a ValueError
df.drop("non_existent_column")
# ValueError: Column 'non_existent_column' not found in DataFrame
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def drop(self, *col_names: str) -> DataFrame:
    """Remove one or more columns from this DataFrame.

    Args:
        *col_names: Names of columns to drop.

    Returns:
        DataFrame: New DataFrame without specified columns.

    Raises:
        ValueError: If any specified column doesn't exist in the DataFrame.
        ValueError: If dropping the columns would result in an empty DataFrame.

    Example: Drop single column
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })

        # Drop single column
        df.drop("age").show()
        # Output:
        # +---+-------+
        # | id|   name|
        # +---+-------+
        # |  1|  Alice|
        # |  2|    Bob|
        # |  3|Charlie|
        # +---+-------+
        ```

    Example: Drop multiple columns
        ```python
        # Drop multiple columns
        df.drop(col("id"), "age").show()
        # Output:
        # +-------+
        # |   name|
        # +-------+
        # |  Alice|
        # |    Bob|
        # |Charlie|
        # +-------+
        ```

    Example: Error when dropping non-existent column
        ```python
        # This will raise a ValueError
        df.drop("non_existent_column")
        # ValueError: Column 'non_existent_column' not found in DataFrame
        ```
    """
    if not col_names:
        return self

    current_cols = set(self.columns)
    to_drop = set(col_names)
    missing = to_drop - current_cols

    if missing:
        missing_str = (
            f"Column '{next(iter(missing))}'"
            if len(missing) == 1
            else f"Columns {sorted(missing)}"
        )
        raise ValueError(f"{missing_str} not found in DataFrame")

    remaining_cols = [
        col(c)._logical_expr for c in self.columns if c not in to_drop
    ]

    if not remaining_cols:
        raise ValueError("Cannot drop all columns from DataFrame")

    return self._from_logical_plan(
        Projection.from_session_state(self._logical_plan, remaining_cols, self._session_state),
        self._session_state,
    )
```

### drop_duplicates

```
drop_duplicates(subset: Optional[List[str]] = None) -> DataFrame
```

Return a DataFrame with duplicate rows removed.

Parameters:

- **`subset`**
  (`Optional[List[str]]`, default:
  `None`
  )
  –

  Column names to consider when identifying duplicates. If not provided, all columns are considered.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with duplicate rows removed.

Raises:

- `ValueError`
  –

  If a specified column is not present in the current DataFrame schema.

Remove duplicates considering specific columns

```
# Create sample DataFrame
df = session.create_dataframe({
    "c1": [1, 2, 3, 1],
    "c2": ["a", "a", "a", "a"],
    "c3": ["b", "b", "b", "b"]
})

# Remove duplicates considering all columns
df.drop_duplicates([col("c1"), col("c2"), col("c3")]).show()
# Output:
# +---+---+---+
# | c1| c2| c3|
# +---+---+---+
# |  1|  a|  b|
# |  2|  a|  b|
# |  3|  a|  b|
# +---+---+---+

# Remove duplicates considering only c1
df.drop_duplicates([col("c1")]).show()
# Output:
# +---+---+---+
# | c1| c2| c3|
# +---+---+---+
# |  1|  a|  b|
# |  2|  a|  b|
# |  3|  a|  b|
# +---+---+---+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def drop_duplicates(
    self,
    subset: Optional[List[str]] = None,
) -> DataFrame:
    """Return a DataFrame with duplicate rows removed.

    Args:
        subset: Column names to consider when identifying duplicates. If not provided, all columns are considered.

    Returns:
        DataFrame: A new DataFrame with duplicate rows removed.

    Raises:
        ValueError: If a specified column is not present in the current DataFrame schema.

    Example: Remove duplicates considering specific columns
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "c1": [1, 2, 3, 1],
            "c2": ["a", "a", "a", "a"],
            "c3": ["b", "b", "b", "b"]
        })

        # Remove duplicates considering all columns
        df.drop_duplicates([col("c1"), col("c2"), col("c3")]).show()
        # Output:
        # +---+---+---+
        # | c1| c2| c3|
        # +---+---+---+
        # |  1|  a|  b|
        # |  2|  a|  b|
        # |  3|  a|  b|
        # +---+---+---+

        # Remove duplicates considering only c1
        df.drop_duplicates([col("c1")]).show()
        # Output:
        # +---+---+---+
        # | c1| c2| c3|
        # +---+---+---+
        # |  1|  a|  b|
        # |  2|  a|  b|
        # |  3|  a|  b|
        # +---+---+---+
        ```
    """
    exprs = []
    if subset:
        for c in subset:
            if c not in self.columns:
                raise TypeError(f"Column {c} not found in DataFrame.")
            exprs.append(col(c)._logical_expr)

    return self._from_logical_plan(
        DropDuplicates.from_session_state(self._logical_plan, exprs, self._session_state),
        self._session_state,
    )
```

### explain

```
explain() -> None
```

Display the logical plan of the DataFrame.

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def explain(self) -> None:
    """Display the logical plan of the DataFrame."""
    print(str(self._logical_plan))
```

### explode

```
explode(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column.

This operation is useful for flattening nested data structures. For each row in the
input DataFrame that contains an array/list in the specified column, this method will:
1. Create N new rows, where N is the length of the array
2. Each new row will be identical to the original row, except the array column will
contain just a single element from the original array
3. Rows with NULL values or empty arrays in the specified column are filtered out

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with the array column exploded into multiple rows.

Raises:

- `TypeError`
  –

  If column argument is not a string or Column.

Explode array column

```
# Create sample DataFrame
df = session.create_dataframe({
    "id": [1, 2, 3, 4],
    "tags": [["red", "blue"], ["green"], [], None],
    "name": ["Alice", "Bob", "Carol", "Dave"]
})

# Explode the tags column
df.explode("tags").show()
# Output:
# +---+-----+-----+
# | id| tags| name|
# +---+-----+-----+
# |  1|  red|Alice|
# |  1| blue|Alice|
# |  2|green|  Bob|
# +---+-----+-----+
```

Using column expression

```
# Explode using column expression
df.explode(col("tags")).show()
# Output:
# +---+-----+-----+
# | id| tags| name|
# +---+-----+-----+
# |  1|  red|Alice|
# |  1| blue|Alice|
# |  2|green|  Bob|
# +---+-----+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def explode(self, column: ColumnOrName) -> DataFrame:
    """Create a new row for each element in an array column.

    This operation is useful for flattening nested data structures. For each row in the
    input DataFrame that contains an array/list in the specified column, this method will:
    1. Create N new rows, where N is the length of the array
    2. Each new row will be identical to the original row, except the array column will
       contain just a single element from the original array
    3. Rows with NULL values or empty arrays in the specified column are filtered out

    Args:
        column: Name of array column to explode (as string) or Column expression.

    Returns:
        DataFrame: New DataFrame with the array column exploded into multiple rows.

    Raises:
        TypeError: If column argument is not a string or Column.

    Example: Explode array column
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "id": [1, 2, 3, 4],
            "tags": [["red", "blue"], ["green"], [], None],
            "name": ["Alice", "Bob", "Carol", "Dave"]
        })

        # Explode the tags column
        df.explode("tags").show()
        # Output:
        # +---+-----+-----+
        # | id| tags| name|
        # +---+-----+-----+
        # |  1|  red|Alice|
        # |  1| blue|Alice|
        # |  2|green|  Bob|
        # +---+-----+-----+
        ```

    Example: Using column expression
        ```python
        # Explode using column expression
        df.explode(col("tags")).show()
        # Output:
        # +---+-----+-----+
        # | id| tags| name|
        # +---+-----+-----+
        # |  1|  red|Alice|
        # |  1| blue|Alice|
        # |  2|green|  Bob|
        # +---+-----+-----+
        ```
    """
    return self._from_logical_plan(
        Explode.from_session_state(self._logical_plan, Column._from_col_or_name(column)._logical_expr, self._session_state),
        self._session_state,
    )
```

### explode_outer

```
explode_outer(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column, containing the element's position in the array and its value, and preserving null/empty arrays.

This operation is similar to explode(), but keeps rows where the array column
is null or empty, producing a row with null in the exploded column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with the array column exploded into multiple rows.
- `DataFrame`
  –

  Rows with null or empty arrays are preserved with null in the exploded column.

Explode with outer join behavior

```
df = session.create_dataframe({
    "id": [1, 2, 3],
    "tags": [["red", "blue"], [], None],
})

df.explode_outer("tags").show()
# Output:
# +---+-----+
# | id| tags|
# +---+-----+
# |  1|  red|
# |  1| blue|
# |  2| NULL|  # empty array preserved as null
# |  3| NULL|  # null array preserved as null
# +---+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def explode_outer(self, column: ColumnOrName) -> DataFrame:
    """Create a new row for each element in an array column, containing the element's position in the array and its value, and preserving null/empty arrays.

    This operation is similar to explode(), but keeps rows where the array column
    is null or empty, producing a row with null in the exploded column.

    Args:
        column: Name of array column to explode (as string) or Column expression.

    Returns:
        DataFrame: New DataFrame with the array column exploded into multiple rows.
        Rows with null or empty arrays are preserved with null in the exploded column.

    Example: Explode with outer join behavior
        ```python
        df = session.create_dataframe({
            "id": [1, 2, 3],
            "tags": [["red", "blue"], [], None],
        })

        df.explode_outer("tags").show()
        # Output:
        # +---+-----+
        # | id| tags|
        # +---+-----+
        # |  1|  red|
        # |  1| blue|
        # |  2| NULL|  # empty array preserved as null
        # |  3| NULL|  # null array preserved as null
        # +---+-----+
        ```
    """
    return self._from_logical_plan(
        Explode.from_session_state(
            self._logical_plan,
            Column._from_col_or_name(column)._logical_expr,
            self._session_state,
            keep_null_and_empty=True
        ),
        self._session_state,
    )
```

### explode_with_index

```
explode_with_index(column: ColumnOrName, index_col_name: str = 'pos', value_col_name: str = 'col', keep_null_and_empty: bool = False) -> DataFrame
```

Create a new row for each element in an array column, with the element's position in the array and its value.

This operation is similar to explode(), but also adds a column containing the 0-based
position of each element within its original array. By default, the position column is named "pos".
and the value column is named "col". These columns replace the original column in the output DataFrame.
If keep_null_and_empty is True, the position column will be null for rows where the array is null or empty.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.
- **`index_col_name`**
  (`str`, default:
  `'pos'`
  )
  –

  Name for the column containing 0-based array positions (default: "pos").
- **`value_col_name`**
  (`str`, default:
  `'col'`
  )
  –

  Name for the exploded value column (default: "col").
- **`keep_null_and_empty`**
  (`bool`, default:
  `False`
  )
  –

  If True, preserves rows where the array is null or empty (default: False).
  Mimicks the behavior of posexplode (false) vs posexplode_outer (true).

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with:
  - An integer column (named `index_col_name`) containing 0-based positions
  - The exploded array column (named `value_col_name`)
  - All other columns from the original DataFrame

Explode with index

```
df = session.create_dataframe({
    "id": [1, 2, 3],
    "tags": [["red", "blue"], ["green"], []],
})

df.explode_with_index("tags").show()
# Output:
# +-----+---+-----+
# | pos| id| tags|
# +-----+---+-----+
# |    0|  1|  red|
# |    1|  1| blue|
# |    0|  2|green|
# +-----+---+-----+
```

Custom column names

```
df.explode_with_index("tags", index_col_name="index", value_name="tag").show()
# Output:
# +-----+---+-----+
# |index| id|  tag|
# +-----+---+-----+
# |    0|  1|  red|
# |    1|  1| blue|
# |    0|  2|green|
# +-----+---+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def explode_with_index(
    self,
    column: ColumnOrName,
    index_col_name: str = "pos",
    value_col_name: str = "col",
    keep_null_and_empty: bool = False,
) -> DataFrame:
    """Create a new row for each element in an array column, with the element's position in the array and its value.

    This operation is similar to explode(), but also adds a column containing the 0-based
    position of each element within its original array. By default, the position column is named "pos".
    and the value column is named "col". These columns replace the original column in the output DataFrame.
    If keep_null_and_empty is True, the position column will be null for rows where the array is null or empty.

    Args:
        column: Name of array column to explode (as string) or Column expression.
        index_col_name: Name for the column containing 0-based array positions (default: "pos").
        value_col_name: Name for the exploded value column (default: "col").
        keep_null_and_empty: If True, preserves rows where the array is null or empty (default: False).
            Mimicks the behavior of posexplode (false) vs posexplode_outer (true).

    Returns:
        DataFrame: New DataFrame with:
            - An integer column (named `index_col_name`) containing 0-based positions
            - The exploded array column (named `value_col_name`)
            - All other columns from the original DataFrame

    Example: Explode with index
        ```python
        df = session.create_dataframe({
            "id": [1, 2, 3],
            "tags": [["red", "blue"], ["green"], []],
        })

        df.explode_with_index("tags").show()
        # Output:
        # +-----+---+-----+
        # | pos| id| tags|
        # +-----+---+-----+
        # |    0|  1|  red|
        # |    1|  1| blue|
        # |    0|  2|green|
        # +-----+---+-----+
        ```

    Example: Custom column names
        ```python
        df.explode_with_index("tags", index_col_name="index", value_name="tag").show()
        # Output:
        # +-----+---+-----+
        # |index| id|  tag|
        # +-----+---+-----+
        # |    0|  1|  red|
        # |    1|  1| blue|
        # |    0|  2|green|
        # +-----+---+-----+
        ```
    """
    return self._from_logical_plan(
        ExplodeWithIndex.from_session_state(
            self._logical_plan,
            Column._from_col_or_name(column)._logical_expr,
            index_col_name,
            value_col_name,
            self._session_state,
            keep_null_and_empty,
        ),
        self._session_state,
    )
```

### filter

```
filter(condition: Column) -> DataFrame
```

Filters rows using the given condition.

Parameters:

- **`condition`**
  (`Column`)
  –

  A Column expression that evaluates to a boolean

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Filtered DataFrame

Filter with numeric comparison

```
# Create a DataFrame
df = session.create_dataframe({"age": [25, 30, 35], "name": ["Alice", "Bob", "Charlie"]})

# Filter with numeric comparison
df.filter(col("age") > 25).show()
# Output:
# +---+-------+
# |age|   name|
# +---+-------+
# | 30|    Bob|
# | 35|Charlie|
# +---+-------+
```

Filter with semantic predicate

```
# Filter with semantic predicate
df.filter((col("age") > 25) & semantic.predicate("This {feedback} mentions problems with the user interface or navigation")).show()
# Output:
# +---+-------+
# |age|   name|
# +---+-------+
# | 30|    Bob|
# | 35|Charlie|
# +---+-------+
```

Filter with multiple conditions

```
# Filter with multiple conditions
df.filter((col("age") > 25) & (col("age") <= 35)).show()
# Output:
# +---+-------+
# |age|   name|
# +---+-------+
# | 30|    Bob|
# | 35|Charlie|
# +---+-------+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def filter(self, condition: Column) -> DataFrame:
    """Filters rows using the given condition.

    Args:
        condition: A Column expression that evaluates to a boolean

    Returns:
        DataFrame: Filtered DataFrame

    Example: Filter with numeric comparison
        ```python
        # Create a DataFrame
        df = session.create_dataframe({"age": [25, 30, 35], "name": ["Alice", "Bob", "Charlie"]})

        # Filter with numeric comparison
        df.filter(col("age") > 25).show()
        # Output:
        # +---+-------+
        # |age|   name|
        # +---+-------+
        # | 30|    Bob|
        # | 35|Charlie|
        # +---+-------+
        ```

    Example: Filter with semantic predicate
        ```python
        # Filter with semantic predicate
        df.filter((col("age") > 25) & semantic.predicate("This {feedback} mentions problems with the user interface or navigation")).show()
        # Output:
        # +---+-------+
        # |age|   name|
        # +---+-------+
        # | 30|    Bob|
        # | 35|Charlie|
        # +---+-------+
        ```

    Example: Filter with multiple conditions
        ```python
        # Filter with multiple conditions
        df.filter((col("age") > 25) & (col("age") <= 35)).show()
        # Output:
        # +---+-------+
        # |age|   name|
        # +---+-------+
        # | 30|    Bob|
        # | 35|Charlie|
        # +---+-------+
        ```
    """
    return self._from_logical_plan(
        Filter.from_session_state(self._logical_plan, condition._logical_expr, self._session_state),
        self._session_state,
    )
```

### group_by

```
group_by(*cols: ColumnOrName) -> GroupedData
```

Groups the DataFrame using the specified columns.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Columns to group by. Can be column names as strings or Column expressions.

Returns:

- **`GroupedData`** ( `GroupedData`
  ) –

  Object for performing aggregations on the grouped data.

Group by single column

```
# Create sample DataFrame
df = session.create_dataframe({
    "department": ["IT", "HR", "IT", "HR", "IT"],
    "salary": [80000, 70000, 90000, 75000, 85000]
})

# Group by single column
df.group_by(col("department")).agg(count("*")).show()
# Output:
# +----------+-----+
# |department|count|
# +----------+-----+
# |        IT|    3|
# |        HR|    2|
# +----------+-----+
```

Group by multiple columns

```
# Group by multiple columns
df.group_by(col("department"), col("location")).agg({"salary": "avg"}).show()
# Output:
# +----------+--------+-----------+
# |department|location|avg(salary)|
# +----------+--------+-----------+
# |        IT|    NYC|    85000.0|
# |        HR|    NYC|    72500.0|
# +----------+--------+-----------+
```

Group by expression

```
# Group by expression
df.group_by(lower(col("department")).alias("department")).agg(count("*")).show()
# Output:
# +---------+-----+
# |department|count|
# +----------+-----+
# |        it|    3|
# |        hr|    2|
# +---------+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def group_by(self, *cols: ColumnOrName) -> GroupedData:
    """Groups the DataFrame using the specified columns.

    Args:
        *cols: Columns to group by. Can be column names as strings or Column expressions.

    Returns:
        GroupedData: Object for performing aggregations on the grouped data.

    Example: Group by single column
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "department": ["IT", "HR", "IT", "HR", "IT"],
            "salary": [80000, 70000, 90000, 75000, 85000]
        })

        # Group by single column
        df.group_by(col("department")).agg(count("*")).show()
        # Output:
        # +----------+-----+
        # |department|count|
        # +----------+-----+
        # |        IT|    3|
        # |        HR|    2|
        # +----------+-----+
        ```

    Example: Group by multiple columns
        ```python
        # Group by multiple columns
        df.group_by(col("department"), col("location")).agg({"salary": "avg"}).show()
        # Output:
        # +----------+--------+-----------+
        # |department|location|avg(salary)|
        # +----------+--------+-----------+
        # |        IT|    NYC|    85000.0|
        # |        HR|    NYC|    72500.0|
        # +----------+--------+-----------+
        ```

    Example: Group by expression
        ```python
        # Group by expression
        df.group_by(lower(col("department")).alias("department")).agg(count("*")).show()
        # Output:
        # +---------+-----+
        # |department|count|
        # +----------+-----+
        # |        it|    3|
        # |        hr|    2|
        # +---------+-----+
        ```
    """
    return GroupedData(self, list(cols) if cols else None)
```

### join

```
join(other: DataFrame, on: Union[str, List[str]], *, how: JoinType = 'inner') -> DataFrame
```

```
join(other: DataFrame, *, left_on: Union[ColumnOrName, List[ColumnOrName]], right_on: Union[ColumnOrName, List[ColumnOrName]], how: JoinType = 'inner') -> DataFrame
```

```
join(other: DataFrame, on: Optional[Union[str, List[str]]] = None, *, left_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None, right_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None, how: JoinType = 'inner') -> DataFrame
```

Joins this DataFrame with another DataFrame.

The Dataframes must have no duplicate column names between them. This API only supports equi-joins.
For non-equi-joins, use session.sql().

Parameters:

- **`other`**
  (`DataFrame`)
  –

  DataFrame to join with.
- **`on`**
  (`Optional[Union[str, List[str]]]`, default:
  `None`
  )
  –

  Join condition(s). Can be:
  - A column name (str)
  - A list of column names (List[str])
  - A Column expression (e.g., col('a'))
  - A list of Column expressions
  - `None` for cross joins
- **`left_on`**
  (`Optional[Union[ColumnOrName, List[ColumnOrName]]]`, default:
  `None`
  )
  –

  Column(s) from the left DataFrame to join on. Can be:
  - A column name (str)
  - A Column expression (e.g., col('a'), col('a') + 1)
  - A list of column names or expressions
- **`right_on`**
  (`Optional[Union[ColumnOrName, List[ColumnOrName]]]`, default:
  `None`
  )
  –

  Column(s) from the right DataFrame to join on. Can be:
  - A column name (str)
  - A Column expression (e.g., col('b'), upper(col('b')))
  - A list of column names or expressions
- **`how`**
  (`JoinType`, default:
  `'inner'`
  )
  –

  Type of join to perform.

Returns:

- `DataFrame`
  –

  Joined DataFrame.

Raises:

- `ValidationError`
  –

  If cross join is used with an ON clause.
- `ValidationError`
  –

  If join condition is invalid.
- `ValidationError`
  –

  If both 'on' and 'left_on'/'right_on' parameters are provided.
- `ValidationError`
  –

  If only one of 'left_on' or 'right_on' is provided.
- `ValidationError`
  –

  If 'left_on' and 'right_on' have different lengths

Inner join on column name

```
# Create sample DataFrames
df1 = session.create_dataframe({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})
df2 = session.create_dataframe({
    "id": [1, 2, 4],
    "age": [25, 30, 35]
})

# Join on single column
df1.join(df2, on=col("id")).show()
# Output:
# +---+-----+---+
# | id| name|age|
# +---+-----+---+
# |  1|Alice| 25|
# |  2|  Bob| 30|
# +---+-----+---+
```

Join with expression

```
# Join with Column expressions
df1.join(
    df2,
    left_on=col("id"),
    right_on=col("id"),
).show()
# Output:
# +---+-----+---+
# | id| name|age|
# +---+-----+---+
# |  1|Alice| 25|
# |  2|  Bob| 30|
# +---+-----+---+
```

Cross join

```
# Cross join (cartesian product)
df1.join(df2, how="cross").show()
# Output:
# +---+-----+---+---+
# | id| name| id|age|
# +---+-----+---+---+
# |  1|Alice|  1| 25|
# |  1|Alice|  2| 30|
# |  1|Alice|  4| 35|
# |  2|  Bob|  1| 25|
# |  2|  Bob|  2| 30|
# |  2|  Bob|  4| 35|
# |  3|Charlie| 1| 25|
# |  3|Charlie| 2| 30|
# |  3|Charlie| 4| 35|
# +---+-----+---+---+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def join(
    self,
    other: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    *,
    left_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None,
    right_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None,
    how: JoinType = "inner",
) -> DataFrame:
    """Joins this DataFrame with another DataFrame.

    The Dataframes must have no duplicate column names between them. This API only supports equi-joins.
    For non-equi-joins, use session.sql().

    Args:
        other: DataFrame to join with.
        on: Join condition(s). Can be:
            - A column name (str)
            - A list of column names (List[str])
            - A Column expression (e.g., col('a'))
            - A list of Column expressions
            - `None` for cross joins
        left_on: Column(s) from the left DataFrame to join on. Can be:
            - A column name (str)
            - A Column expression (e.g., col('a'), col('a') + 1)
            - A list of column names or expressions
        right_on: Column(s) from the right DataFrame to join on. Can be:
            - A column name (str)
            - A Column expression (e.g., col('b'), upper(col('b')))
            - A list of column names or expressions
        how: Type of join to perform.

    Returns:
        Joined DataFrame.

    Raises:
        ValidationError: If cross join is used with an ON clause.
        ValidationError: If join condition is invalid.
        ValidationError: If both 'on' and 'left_on'/'right_on' parameters are provided.
        ValidationError: If only one of 'left_on' or 'right_on' is provided.
        ValidationError: If 'left_on' and 'right_on' have different lengths

    Example: Inner join on column name
        ```python
        # Create sample DataFrames
        df1 = session.create_dataframe({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        df2 = session.create_dataframe({
            "id": [1, 2, 4],
            "age": [25, 30, 35]
        })

        # Join on single column
        df1.join(df2, on=col("id")).show()
        # Output:
        # +---+-----+---+
        # | id| name|age|
        # +---+-----+---+
        # |  1|Alice| 25|
        # |  2|  Bob| 30|
        # +---+-----+---+
        ```

    Example: Join with expression
        ```python
        # Join with Column expressions
        df1.join(
            df2,
            left_on=col("id"),
            right_on=col("id"),
        ).show()
        # Output:
        # +---+-----+---+
        # | id| name|age|
        # +---+-----+---+
        # |  1|Alice| 25|
        # |  2|  Bob| 30|
        # +---+-----+---+
        ```

    Example: Cross join
        ```python
        # Cross join (cartesian product)
        df1.join(df2, how="cross").show()
        # Output:
        # +---+-----+---+---+
        # | id| name| id|age|
        # +---+-----+---+---+
        # |  1|Alice|  1| 25|
        # |  1|Alice|  2| 30|
        # |  1|Alice|  4| 35|
        # |  2|  Bob|  1| 25|
        # |  2|  Bob|  2| 30|
        # |  2|  Bob|  4| 35|
        # |  3|Charlie| 1| 25|
        # |  3|Charlie| 2| 30|
        # |  3|Charlie| 4| 35|
        # +---+-----+---+---+
        ```
    """
    validate_join_parameters(self, on, left_on, right_on, how)

    # Build join conditions
    left_conditions, right_conditions = build_join_conditions(on, left_on, right_on)

    self._ensure_same_session(self._session_state, [other._session_state])
    return self._from_logical_plan(
        Join.from_session_state(
            self._logical_plan,
            other._logical_plan,
            left_conditions,
            right_conditions,
            how,
            self._session_state),
        self._session_state,
    )
```

### limit

```
limit(n: int) -> DataFrame
```

Limits the number of rows to the specified number.

Parameters:

- **`n`**
  (`int`)
  –

  Maximum number of rows to return.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  DataFrame with at most n rows.

Raises:

- `TypeError`
  –

  If n is not an integer.

Limit rows

```
# Create sample DataFrame
df = session.create_dataframe({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"]
})

# Get first 3 rows
df.limit(3).show()
# Output:
# +---+-------+
# | id|   name|
# +---+-------+
# |  1|  Alice|
# |  2|    Bob|
# |  3|Charlie|
# +---+-------+
```

Limit with other operations

```
# Limit after filtering
df.filter(col("id") > 2).limit(2).show()
# Output:
# +---+-------+
# | id|   name|
# +---+-------+
# |  3|Charlie|
# |  4|   Dave|
# +---+-------+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def limit(self, n: int) -> DataFrame:
    """Limits the number of rows to the specified number.

    Args:
        n: Maximum number of rows to return.

    Returns:
        DataFrame: DataFrame with at most n rows.

    Raises:
        TypeError: If n is not an integer.

    Example: Limit rows
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"]
        })

        # Get first 3 rows
        df.limit(3).show()
        # Output:
        # +---+-------+
        # | id|   name|
        # +---+-------+
        # |  1|  Alice|
        # |  2|    Bob|
        # |  3|Charlie|
        # +---+-------+
        ```

    Example: Limit with other operations
        ```python
        # Limit after filtering
        df.filter(col("id") > 2).limit(2).show()
        # Output:
        # +---+-------+
        # | id|   name|
        # +---+-------+
        # |  3|Charlie|
        # |  4|   Dave|
        # +---+-------+
        ```
    """
    return self._from_logical_plan(
        Limit.from_session_state(self._logical_plan, n, self._session_state),
        self._session_state)
```

### lineage

```
lineage() -> Lineage
```

Create a Lineage object to trace data through transformations.

The Lineage interface allows you to trace how specific rows are transformed
through your DataFrame operations, both forwards and backwards through the
computation graph.

Returns:

- **`Lineage`** ( `Lineage`
  ) –

  Interface for querying data lineage

Example

```
# Create lineage query
lineage = df.lineage()

# Trace specific rows backwards through transformations
source_rows = lineage.backward(["result_uuid1", "result_uuid2"])

# Or trace forwards to see outputs
result_rows = lineage.forward(["source_uuid1"])
```

See Also

LineageQuery: Full documentation of lineage querying capabilities

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def lineage(self) -> Lineage:
    """Create a Lineage object to trace data through transformations.

    The Lineage interface allows you to trace how specific rows are transformed
    through your DataFrame operations, both forwards and backwards through the
    computation graph.

    Returns:
        Lineage: Interface for querying data lineage

    Example:
        ```python
        # Create lineage query
        lineage = df.lineage()

        # Trace specific rows backwards through transformations
        source_rows = lineage.backward(["result_uuid1", "result_uuid2"])

        # Or trace forwards to see outputs
        result_rows = lineage.forward(["source_uuid1"])
        ```

    See Also:
        LineageQuery: Full documentation of lineage querying capabilities
    """
    return Lineage(self._session_state.execution.build_lineage(self._logical_plan))
```

### order_by

```
order_by(cols: Union[ColumnOrName, List[ColumnOrName], None] = None, ascending: Optional[Union[bool, List[bool]]] = None) -> DataFrame
```

Sort the DataFrame by the specified columns. Alias for sort().

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  sorted Dataframe.

See Also

sort(): Full documentation of sorting behavior and parameters.

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def order_by(
    self,
    cols: Union[ColumnOrName, List[ColumnOrName], None] = None,
    ascending: Optional[Union[bool, List[bool]]] = None,
) -> DataFrame:
    """Sort the DataFrame by the specified columns. Alias for sort().

    Returns:
        DataFrame: sorted Dataframe.

    See Also:
        sort(): Full documentation of sorting behavior and parameters.
    """
    return self.sort(cols, ascending)
```

### persist

```
persist() -> DataFrame
```

Mark this DataFrame to be persisted after first computation.

The persisted DataFrame will be cached after its first computation,
avoiding recomputation in subsequent operations. This is useful for:
- DataFrames that are created once and reused multiple times in your workflow
- DataFrames that are computationally expensive (large joins, aggregations, etc.)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Same DataFrame, but marked for persistence

Example

```
# Cache intermediate results for reuse
filtered_df = (df
    .filter(col("age") > 25)
    .persist()  # Cache these results
)

# Both operations will use cached results
result1 = filtered_df.group_by("department").count()
result2 = filtered_df.select("name", "salary")
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def persist(self) -> DataFrame:
    """Mark this DataFrame to be persisted after first computation.

    The persisted DataFrame will be cached after its first computation,
    avoiding recomputation in subsequent operations. This is useful for:
        - DataFrames that are created once and reused multiple times in your workflow
        - DataFrames that are computationally expensive (large joins, aggregations, etc.)

    Returns:
        DataFrame: Same DataFrame, but marked for persistence

    Example:
        ```python
        # Cache intermediate results for reuse
        filtered_df = (df
            .filter(col("age") > 25)
            .persist()  # Cache these results
        )

        # Both operations will use cached results
        result1 = filtered_df.group_by("department").count()
        result2 = filtered_df.select("name", "salary")
        ```
    """
    cache_info = CacheInfo(cache_key=f"cache_{uuid.uuid4().hex}")
    self._logical_plan.set_cache_info(cache_info)
    return self._from_logical_plan(
        self._logical_plan,
        self._session_state)
```

### posexplode

```
posexplode(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column, with the element's position in the array and its value.

This is a PySpark-compatible alias for explode_with_index.
Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
These columns replace the original column in the output DataFrame.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with 'pos' and 'col' columns, plus all other original columns.

PySpark-style posexplode

```
df = session.create_dataframe({
    "id": [1, 2],
    "tags": [["red", "blue"], ["green"]],
})

df.posexplode("tags").show()
# Output:
# +---+---+-----+
# |pos| id|  col|
# +---+---+-----+
# |  0|  1|  red|
# |  1|  1| blue|
# |  0|  2|green|
# +---+---+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def posexplode(self, column: ColumnOrName) -> DataFrame:
    """Create a new row for each element in an array column, with the element's position in the array and its value.

    This is a PySpark-compatible alias for explode_with_index.
    Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
    These columns replace the original column in the output DataFrame.

    Args:
        column: Name of array column to explode (as string) or Column expression.

    Returns:
        DataFrame: New DataFrame with 'pos' and 'col' columns, plus all other original columns.

    Example: PySpark-style posexplode
        ```python
        df = session.create_dataframe({
            "id": [1, 2],
            "tags": [["red", "blue"], ["green"]],
        })

        df.posexplode("tags").show()
        # Output:
        # +---+---+-----+
        # |pos| id|  col|
        # +---+---+-----+
        # |  0|  1|  red|
        # |  1|  1| blue|
        # |  0|  2|green|
        # +---+---+-----+
        ```
    """
    return self.explode_with_index(column)
```

### posexplode_outer

```
posexplode_outer(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column with position and value, preserving null/empty arrays.

This is a PySpark-compatible alias for explode_with_index with keep_null_and_empty=True.
Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
Rows with null or empty arrays produce (null, null).

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with 'pos' and 'col' columns, plus all other original columns.
- `DataFrame`
  –

  Rows with null or empty arrays are preserved with (null, null).

PySpark-style posexplode_outer

```
df = session.create_dataframe({
    "id": [1, 2, 3],
    "tags": [["red", "blue"], [], None],
})

df.posexplode_outer("tags").show()
# Output:
# +---+---+-----+
# |pos| id|  col|
# +---+---+-----+
# |  0|  1|  red|
# |  1|  1| blue|
# |NULL|  2| NULL|  # empty array preserved
# |NULL|  3| NULL|  # null array preserved
# +---+---+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def posexplode_outer(self, column: ColumnOrName) -> DataFrame:
    """Create a new row for each element in an array column with position and value, preserving null/empty arrays.

    This is a PySpark-compatible alias for explode_with_index with keep_null_and_empty=True.
    Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
    Rows with null or empty arrays produce (null, null).

    Args:
        column: Name of array column to explode (as string) or Column expression.

    Returns:
        DataFrame: New DataFrame with 'pos' and 'col' columns, plus all other original columns.
        Rows with null or empty arrays are preserved with (null, null).

    Example: PySpark-style posexplode_outer
        ```python
        df = session.create_dataframe({
            "id": [1, 2, 3],
            "tags": [["red", "blue"], [], None],
        })

        df.posexplode_outer("tags").show()
        # Output:
        # +---+---+-----+
        # |pos| id|  col|
        # +---+---+-----+
        # |  0|  1|  red|
        # |  1|  1| blue|
        # |NULL|  2| NULL|  # empty array preserved
        # |NULL|  3| NULL|  # null array preserved
        # +---+---+-----+
        ```
    """
    return self.explode_with_index(
        column, keep_null_and_empty=True
    )
```

### select

```
select(*cols: ColumnOrName) -> DataFrame
```

Projects a set of Column expressions or column names.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions to select. Can be:
  - String column names (e.g., "id", "name")
  - Column objects (e.g., col("id"), col("age") + 1)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with selected columns

Select by column names

```
# Create a DataFrame
df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

# Select by column names
df.select(col("name"), col("age")).show()
# Output:
# +-----+---+
# | name|age|
# +-----+---+
# |Alice| 25|
# |  Bob| 30|
# +-----+---+
```

Select with expressions

```
# Select with expressions
df.select(col("name"), col("age") + 1).show()
# Output:
# +-----+-------+
# | name|age + 1|
# +-----+-------+
# |Alice|     26|
# |  Bob|     31|
# +-----+-------+
```

Mix strings and expressions

```
# Mix strings and expressions
df.select(col("name"), col("age") * 2).show()
# Output:
# +-----+-------+
# | name|age * 2|
# +-----+-------+
# |Alice|     50|
# |  Bob|     60|
# +-----+-------+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def select(self, *cols: ColumnOrName) -> DataFrame:
    """Projects a set of Column expressions or column names.

    Args:
        *cols: Column expressions to select. Can be:
            - String column names (e.g., "id", "name")
            - Column objects (e.g., col("id"), col("age") + 1)

    Returns:
        DataFrame: A new DataFrame with selected columns

    Example: Select by column names
        ```python
        # Create a DataFrame
        df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Select by column names
        df.select(col("name"), col("age")).show()
        # Output:
        # +-----+---+
        # | name|age|
        # +-----+---+
        # |Alice| 25|
        # |  Bob| 30|
        # +-----+---+
        ```

    Example: Select with expressions
        ```python
        # Select with expressions
        df.select(col("name"), col("age") + 1).show()
        # Output:
        # +-----+-------+
        # | name|age + 1|
        # +-----+-------+
        # |Alice|     26|
        # |  Bob|     31|
        # +-----+-------+
        ```

    Example: Mix strings and expressions
        ```python
        # Mix strings and expressions
        df.select(col("name"), col("age") * 2).show()
        # Output:
        # +-----+-------+
        # | name|age * 2|
        # +-----+-------+
        # |Alice|     50|
        # |  Bob|     60|
        # +-----+-------+
        ```
    """
    exprs = []
    if not cols:
        return self
    for c in cols:
        if isinstance(c, str):
            if c == "*":
                exprs.extend(col(field)._logical_expr for field in self.columns)
            else:
                exprs.append(col(c)._logical_expr)
        else:
            exprs.append(c._logical_expr)

    return self._from_logical_plan(
        Projection.from_session_state(self._logical_plan, exprs, self._session_state),
        self._session_state,
    )
```

### show

```
show(n: int = 10, explain_analyze: bool = False) -> None
```

Display the DataFrame content in a tabular form.

This is an action that triggers computation of the DataFrame.
The output is printed to stdout in a formatted table.

Parameters:

- **`n`**
  (`int`, default:
  `10`
  )
  –

  Number of rows to display
- **`explain_analyze`**
  (`bool`, default:
  `False`
  )
  –

  Whether to print the explain analyze plan

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def show(self, n: int = 10, explain_analyze: bool = False) -> None:
    """Display the DataFrame content in a tabular form.

    This is an action that triggers computation of the DataFrame.
    The output is printed to stdout in a formatted table.

    Args:
        n: Number of rows to display
        explain_analyze: Whether to print the explain analyze plan
    """
    output, metrics = self._session_state.execution.show(self._logical_plan, n)
    logger.info(metrics.get_summary())
    print(output)
    if explain_analyze:
        print(metrics.get_execution_plan_details())
```

### sort

```
sort(cols: Union[ColumnOrName, List[ColumnOrName], None] = None, ascending: Optional[Union[bool, List[bool]]] = None) -> DataFrame
```

Sort the DataFrame by the specified columns.

Parameters:

- **`cols`**
  (`Union[ColumnOrName, List[ColumnOrName], None]`, default:
  `None`
  )
  –

  Columns to sort by. This can be:
  - A single column name (str)
  - A Column expression (e.g., `col("name")`)
  - A list of column names or Column expressions
  - Column expressions may include sorting directives such as `asc("col")`, `desc("col")`,
  `asc_nulls_last("col")`, etc.
  - If no columns are provided, the operation is a no-op.
- **`ascending`**
  (`Optional[Union[bool, List[bool]]]`, default:
  `None`
  )
  –

  A boolean or list of booleans indicating sort order.
  - If `True`, sorts in ascending order; if `False`, descending.
  - If a list is provided, its length must match the number of columns.
  - Cannot be used if any of the columns use `asc()`/`desc()` expressions.
  - If not specified and no sort expressions are used, columns will be sorted in ascending order by default.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame sorted by the specified columns.

Raises:

- `ValueError`
  –

  - If `ascending` is provided and its length does not match `cols`
  - If both `ascending` and column expressions like `asc()`/`desc()` are used
- `TypeError`
  –

  - If `cols` is not a column name, Column, or list of column names/Columns
  - If `ascending` is not a boolean or list of booleans

Sort in ascending order

```
# Create sample DataFrame
df = session.create_dataframe([(2, "Alice"), (5, "Bob")], schema=["age", "name"])

# Sort by age in ascending order
df.sort(asc(col("age"))).show()
# Output:
# +---+-----+
# |age| name|
# +---+-----+
# |  2|Alice|
# |  5|  Bob|
# +---+-----+
```

Sort in descending order

```
# Sort by age in descending order
df.sort(col("age").desc()).show()
# Output:
# +---+-----+
# |age| name|
# +---+-----+
# |  5|  Bob|
# |  2|Alice|
# +---+-----+
```

Sort with boolean ascending parameter

```
# Sort by age in descending order using boolean
df.sort(col("age"), ascending=False).show()
# Output:
# +---+-----+
# |age| name|
# +---+-----+
# |  5|  Bob|
# |  2|Alice|
# +---+-----+
```

Multiple columns with different sort orders

```
# Create sample DataFrame
df = session.create_dataframe([(2, "Alice"), (2, "Bob"), (5, "Bob")], schema=["age", "name"])

# Sort by age descending, then name ascending
df.sort(desc(col("age")), col("name")).show()
# Output:
# +---+-----+
# |age| name|
# +---+-----+
# |  5|  Bob|
# |  2|Alice|
# |  2|  Bob|
# +---+-----+
```

Multiple columns with list of ascending strategies

```
# Sort both columns in descending order
df.sort([col("age"), col("name")], ascending=[False, False]).show()
# Output:
# +---+-----+
# |age| name|
# +---+-----+
# |  5|  Bob|
# |  2|  Bob|
# |  2|Alice|
# +---+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def sort(
    self,
    cols: Union[ColumnOrName, List[ColumnOrName], None] = None,
    ascending: Optional[Union[bool, List[bool]]] = None,
) -> DataFrame:
    """Sort the DataFrame by the specified columns.

    Args:
        cols: Columns to sort by. This can be:
            - A single column name (str)
            - A Column expression (e.g., `col("name")`)
            - A list of column names or Column expressions
            - Column expressions may include sorting directives such as `asc("col")`, `desc("col")`,
            `asc_nulls_last("col")`, etc.
            - If no columns are provided, the operation is a no-op.

        ascending: A boolean or list of booleans indicating sort order.
            - If `True`, sorts in ascending order; if `False`, descending.
            - If a list is provided, its length must match the number of columns.
            - Cannot be used if any of the columns use `asc()`/`desc()` expressions.
            - If not specified and no sort expressions are used, columns will be sorted in ascending order by default.

    Returns:
        DataFrame: A new DataFrame sorted by the specified columns.

    Raises:
        ValueError:
            - If `ascending` is provided and its length does not match `cols`
            - If both `ascending` and column expressions like `asc()`/`desc()` are used
        TypeError:
            - If `cols` is not a column name, Column, or list of column names/Columns
            - If `ascending` is not a boolean or list of booleans

    Example: Sort in ascending order
        ```python
        # Create sample DataFrame
        df = session.create_dataframe([(2, "Alice"), (5, "Bob")], schema=["age", "name"])

        # Sort by age in ascending order
        df.sort(asc(col("age"))).show()
        # Output:
        # +---+-----+
        # |age| name|
        # +---+-----+
        # |  2|Alice|
        # |  5|  Bob|
        # +---+-----+
        ```

    Example: Sort in descending order
        ```python
        # Sort by age in descending order
        df.sort(col("age").desc()).show()
        # Output:
        # +---+-----+
        # |age| name|
        # +---+-----+
        # |  5|  Bob|
        # |  2|Alice|
        # +---+-----+
        ```

    Example: Sort with boolean ascending parameter
        ```python
        # Sort by age in descending order using boolean
        df.sort(col("age"), ascending=False).show()
        # Output:
        # +---+-----+
        # |age| name|
        # +---+-----+
        # |  5|  Bob|
        # |  2|Alice|
        # +---+-----+
        ```

    Example: Multiple columns with different sort orders
        ```python
        # Create sample DataFrame
        df = session.create_dataframe([(2, "Alice"), (2, "Bob"), (5, "Bob")], schema=["age", "name"])

        # Sort by age descending, then name ascending
        df.sort(desc(col("age")), col("name")).show()
        # Output:
        # +---+-----+
        # |age| name|
        # +---+-----+
        # |  5|  Bob|
        # |  2|Alice|
        # |  2|  Bob|
        # +---+-----+
        ```

    Example: Multiple columns with list of ascending strategies
        ```python
        # Sort both columns in descending order
        df.sort([col("age"), col("name")], ascending=[False, False]).show()
        # Output:
        # +---+-----+
        # |age| name|
        # +---+-----+
        # |  5|  Bob|
        # |  2|  Bob|
        # |  2|Alice|
        # +---+-----+
        ```
    """
    col_args = cols
    if cols is None:
        return self._from_logical_plan(
            Sort.from_session_state(self._logical_plan, [], self._session_state),
            self._session_state,
        )
    elif not isinstance(cols, List):
        col_args = [cols]

    # parse the ascending arguments
    bool_ascending = []
    using_default_ascending = False
    if ascending is None:
        using_default_ascending = True
        bool_ascending = [True] * len(col_args)
    elif isinstance(ascending, bool):
        bool_ascending = [ascending] * len(col_args)
    elif isinstance(ascending, List):
        bool_ascending = ascending
        if len(bool_ascending) != len(cols):
            raise ValueError(
                f"the list length of ascending sort strategies must match the specified sort columns"
                f"Got {len(cols)} column expressions and {len(bool_ascending)} ascending strategies. "
            )
    else:
        raise TypeError(
            f"Invalid ascending strategy type: {type(ascending)}.  Must be a boolean or list of booleans."
        )

    # create our list of sort expressions, for each column expression
    # that isn't already provided as a asc()/desc() SortExpr
    sort_exprs = []
    for c, asc_bool in zip(col_args, bool_ascending, strict=True):
        if isinstance(c, ColumnOrName):
            c_expr = Column._from_col_or_name(c)._logical_expr
        else:
            raise TypeError(
                f"Invalid column type: {type(c).__name__}.  Must be a string or Column Expression."
            )
        if not isinstance(asc_bool, bool):
            raise TypeError(
                f"Invalid ascending strategy type: {type(asc_bool).__name__}.  Must be a boolean."
            )
        if isinstance(c_expr, SortExpr):
            if not using_default_ascending:
                raise TypeError(
                    "Cannot specify both asc()/desc() expressions and boolean ascending strategies."
                    f"Got expression: {c_expr} and ascending argument: {bool_ascending}"
                )
            sort_exprs.append(c_expr)
        else:
            sort_exprs.append(SortExpr(c_expr, ascending=asc_bool))

    return self._from_logical_plan(
        Sort.from_session_state(self._logical_plan, sort_exprs, self._session_state),
        self._session_state,
    )
```

### to_arrow

```
to_arrow() -> pa.Table
```

Execute the DataFrame computation and return an Apache Arrow Table.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into an Apache Arrow Table with columnar memory layout
optimized for analytics and zero-copy data exchange.

Returns:

- `Table`
  –

  pa.Table: An Apache Arrow Table containing the computed results

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def to_arrow(self) -> pa.Table:
    """Execute the DataFrame computation and return an Apache Arrow Table.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into an Apache Arrow Table with columnar memory layout
    optimized for analytics and zero-copy data exchange.

    Returns:
        pa.Table: An Apache Arrow Table containing the computed results
    """
    return self.collect("arrow").data
```

### to_pandas

```
to_pandas() -> pd.DataFrame
```

Execute the DataFrame computation and return a Pandas DataFrame.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Pandas DataFrame.

Returns:

- `DataFrame`
  –

  pd.DataFrame: A Pandas DataFrame containing the computed results with

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def to_pandas(self) -> pd.DataFrame:
    """Execute the DataFrame computation and return a Pandas DataFrame.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a Pandas DataFrame.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the computed results with
    """
    return self.collect("pandas").data
```

### to_polars

```
to_polars() -> pl.DataFrame
```

Execute the DataFrame computation and return the result as a Polars DataFrame.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Polars DataFrame.

Returns:

- `DataFrame`
  –

  pl.DataFrame: A Polars DataFrame with materialized results

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def to_polars(self) -> pl.DataFrame:
    """Execute the DataFrame computation and return the result as a Polars DataFrame.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a Polars DataFrame.

    Returns:
        pl.DataFrame: A Polars DataFrame with materialized results
    """
    return self.collect("polars").data
```

### to_pydict

```
to_pydict() -> Dict[str, List[Any]]
```

Execute the DataFrame computation and return a dictionary of column arrays.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Python dictionary where each column becomes a list of values.

Returns:

- `Dict[str, List[Any]]`
  –

  Dict[str, List[Any]]: A dictionary containing the computed results with:
  - Keys: Column names as strings
  - Values: Lists containing all values for each column

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def to_pydict(self) -> Dict[str, List[Any]]:
    """Execute the DataFrame computation and return a dictionary of column arrays.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a Python dictionary where each column becomes a list of values.

    Returns:
        Dict[str, List[Any]]: A dictionary containing the computed results with:
            - Keys: Column names as strings
            - Values: Lists containing all values for each column
    """
    return self.collect("pydict").data
```

### to_pylist

```
to_pylist() -> List[Dict[str, Any]]
```

Execute the DataFrame computation and return a list of row dictionaries.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Python list where each element is a dictionary
representing one row with column names as keys.

Returns:

- `List[Dict[str, Any]]`
  –

  List[Dict[str, Any]]: A list containing the computed results with:
  - Each element: A dictionary representing one row
  - Dictionary keys: Column names as strings
  - Dictionary values: Cell values in Python native types
  - List length equals number of rows in the result

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def to_pylist(self) -> List[Dict[str, Any]]:
    """Execute the DataFrame computation and return a list of row dictionaries.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a Python list where each element is a dictionary
    representing one row with column names as keys.

    Returns:
        List[Dict[str, Any]]: A list containing the computed results with:
            - Each element: A dictionary representing one row
            - Dictionary keys: Column names as strings
            - Dictionary values: Cell values in Python native types
            - List length equals number of rows in the result
    """
    return self.collect("pylist").data
```

### union

```
union(other: DataFrame) -> DataFrame
```

Return a new DataFrame containing the union of rows in this and another DataFrame.

This is equivalent to UNION ALL in SQL. To remove duplicates, use drop_duplicates() after union().

Parameters:

- **`other`**
  (`DataFrame`)
  –

  Another DataFrame with the same schema.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame containing rows from both DataFrames.

Raises:

- `ValueError`
  –

  If the DataFrames have different schemas.
- `TypeError`
  –

  If other is not a DataFrame.

Union two DataFrames

```
# Create two DataFrames
df1 = session.create_dataframe({
    "id": [1, 2],
    "value": ["a", "b"]
})
df2 = session.create_dataframe({
    "id": [3, 4],
    "value": ["c", "d"]
})

# Union the DataFrames
df1.union(df2).show()
# Output:
# +---+-----+
# | id|value|
# +---+-----+
# |  1|    a|
# |  2|    b|
# |  3|    c|
# |  4|    d|
# +---+-----+
```

Union with duplicates

```
# Create DataFrames with overlapping data
df1 = session.create_dataframe({
    "id": [1, 2],
    "value": ["a", "b"]
})
df2 = session.create_dataframe({
    "id": [2, 3],
    "value": ["b", "c"]
})

# Union with duplicates
df1.union(df2).show()
# Output:
# +---+-----+
# | id|value|
# +---+-----+
# |  1|    a|
# |  2|    b|
# |  2|    b|
# |  3|    c|
# +---+-----+

# Remove duplicates after union
df1.union(df2).drop_duplicates().show()
# Output:
# +---+-----+
# | id|value|
# +---+-----+
# |  1|    a|
# |  2|    b|
# |  3|    c|
# +---+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def union(self, other: DataFrame) -> DataFrame:
    """Return a new DataFrame containing the union of rows in this and another DataFrame.

    This is equivalent to UNION ALL in SQL. To remove duplicates, use drop_duplicates() after union().

    Args:
        other: Another DataFrame with the same schema.

    Returns:
        DataFrame: A new DataFrame containing rows from both DataFrames.

    Raises:
        ValueError: If the DataFrames have different schemas.
        TypeError: If other is not a DataFrame.

    Example: Union two DataFrames
        ```python
        # Create two DataFrames
        df1 = session.create_dataframe({
            "id": [1, 2],
            "value": ["a", "b"]
        })
        df2 = session.create_dataframe({
            "id": [3, 4],
            "value": ["c", "d"]
        })

        # Union the DataFrames
        df1.union(df2).show()
        # Output:
        # +---+-----+
        # | id|value|
        # +---+-----+
        # |  1|    a|
        # |  2|    b|
        # |  3|    c|
        # |  4|    d|
        # +---+-----+
        ```

    Example: Union with duplicates
        ```python
        # Create DataFrames with overlapping data
        df1 = session.create_dataframe({
            "id": [1, 2],
            "value": ["a", "b"]
        })
        df2 = session.create_dataframe({
            "id": [2, 3],
            "value": ["b", "c"]
        })

        # Union with duplicates
        df1.union(df2).show()
        # Output:
        # +---+-----+
        # | id|value|
        # +---+-----+
        # |  1|    a|
        # |  2|    b|
        # |  2|    b|
        # |  3|    c|
        # +---+-----+

        # Remove duplicates after union
        df1.union(df2).drop_duplicates().show()
        # Output:
        # +---+-----+
        # | id|value|
        # +---+-----+
        # |  1|    a|
        # |  2|    b|
        # |  3|    c|
        # +---+-----+
        ```
    """
    self._ensure_same_session(self._session_state, [other._session_state])
    return self._from_logical_plan(
        UnionLogicalPlan.from_session_state([self._logical_plan, other._logical_plan], self._session_state),
        self._session_state,
    )
```

### unnest

```
unnest(*col_names: str) -> DataFrame
```

Unnest the specified struct columns into separate columns.

This operation flattens nested struct data by expanding each field of a struct
into its own top-level column.

For each specified column containing a struct:
1. Each field in the struct becomes a separate column.
2. New columns are named after the corresponding struct fields.
3. The new columns are inserted into the DataFrame in place of the original struct column.
4. The overall column order is preserved.

Parameters:

- **`*col_names`**
  (`str`, default:
  `()`
  )
  –

  One or more struct columns to unnest. Each can be a string (column name)
  or a Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with the specified struct columns expanded.

Raises:

- `TypeError`
  –

  If any argument is not a string or Column.
- `ValueError`
  –

  If a specified column does not contain struct data.

Unnest struct column

```
# Create sample DataFrame
df = session.create_dataframe({
    "id": [1, 2],
    "tags": [{"red": 1, "blue": 2}, {"red": 3}],
    "name": ["Alice", "Bob"]
})

# Unnest the tags column
df.unnest(col("tags")).show()
# Output:
# +---+---+----+-----+
# | id| red|blue| name|
# +---+---+----+-----+
# |  1|  1|   2|Alice|
# |  2|  3|null|  Bob|
# +---+---+----+-----+
```

Unnest multiple struct columns

```
# Create sample DataFrame with multiple struct columns
df = session.create_dataframe({
    "id": [1, 2],
    "tags": [{"red": 1, "blue": 2}, {"red": 3}],
    "info": [{"age": 25, "city": "NY"}, {"age": 30, "city": "LA"}],
    "name": ["Alice", "Bob"]
})

# Unnest multiple struct columns
df.unnest(col("tags"), col("info")).show()
# Output:
# +---+---+----+---+----+-----+
# | id| red|blue|age|city| name|
# +---+---+----+---+----+-----+
# |  1|  1|   2| 25|  NY|Alice|
# |  2|  3|null| 30|  LA|  Bob|
# +---+---+----+---+----+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def unnest(self, *col_names: str) -> DataFrame:
    """Unnest the specified struct columns into separate columns.

    This operation flattens nested struct data by expanding each field of a struct
    into its own top-level column.

    For each specified column containing a struct:
    1. Each field in the struct becomes a separate column.
    2. New columns are named after the corresponding struct fields.
    3. The new columns are inserted into the DataFrame in place of the original struct column.
    4. The overall column order is preserved.

    Args:
        *col_names: One or more struct columns to unnest. Each can be a string (column name)
            or a Column expression.

    Returns:
        DataFrame: A new DataFrame with the specified struct columns expanded.

    Raises:
        TypeError: If any argument is not a string or Column.
        ValueError: If a specified column does not contain struct data.

    Example: Unnest struct column
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "id": [1, 2],
            "tags": [{"red": 1, "blue": 2}, {"red": 3}],
            "name": ["Alice", "Bob"]
        })

        # Unnest the tags column
        df.unnest(col("tags")).show()
        # Output:
        # +---+---+----+-----+
        # | id| red|blue| name|
        # +---+---+----+-----+
        # |  1|  1|   2|Alice|
        # |  2|  3|null|  Bob|
        # +---+---+----+-----+
        ```

    Example: Unnest multiple struct columns
        ```python
        # Create sample DataFrame with multiple struct columns
        df = session.create_dataframe({
            "id": [1, 2],
            "tags": [{"red": 1, "blue": 2}, {"red": 3}],
            "info": [{"age": 25, "city": "NY"}, {"age": 30, "city": "LA"}],
            "name": ["Alice", "Bob"]
        })

        # Unnest multiple struct columns
        df.unnest(col("tags"), col("info")).show()
        # Output:
        # +---+---+----+---+----+-----+
        # | id| red|blue|age|city| name|
        # +---+---+----+---+----+-----+
        # |  1|  1|   2| 25|  NY|Alice|
        # |  2|  3|null| 30|  LA|  Bob|
        # +---+---+----+---+----+-----+
        ```
    """
    if not col_names:
        return self
    exprs = []
    for c in col_names:
        if c not in self.columns:
            raise TypeError(f"Column {c} not found in DataFrame.")
        exprs.append(col(c)._logical_expr)
    return self._from_logical_plan(
        Unnest.from_session_state(self._logical_plan, exprs, self._session_state),
        self._session_state,
    )
```

### where

```
where(condition: Column) -> DataFrame
```

Filters rows using the given condition (alias for filter()).

Parameters:

- **`condition`**
  (`Column`)
  –

  A Column expression that evaluates to a boolean

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Filtered DataFrame

See Also

filter(): Full documentation of filtering behavior

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def where(self, condition: Column) -> DataFrame:
    """Filters rows using the given condition (alias for filter()).

    Args:
        condition: A Column expression that evaluates to a boolean

    Returns:
        DataFrame: Filtered DataFrame

    See Also:
        filter(): Full documentation of filtering behavior
    """
    return self.filter(condition)
```

### with_column

```
with_column(col_name: str, col: Union[Any, Column, Series, Series]) -> DataFrame
```

Add a new column or replace an existing column.

Parameters:

- **`col_name`**
  (`str`)
  –

  Name of the new column
- **`col`**
  (`Union[Any, Column, Series, Series]`)
  –

  Column expression, Series, or value to assign to the column:

  - Column: A Column expression (e.g., `col("age") + 1`)
  - `pl.Series` or `pd.Series`: A Polars or pandas Series with data
    - **Note: Series length MUST match the DataFrame height**
  - Any other value: Treated as a literal value (broadcast to all rows)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with added/replaced column

Raises:

- `ExecutionError`
  –

  - If a Series length does not match the DataFrame height
- `ValidationError`
  –

  - If the Series contains all null values and no dtype is specified
  - If the Series has length 0

Notes:
- The name of the created column will be the name defined in col_name, even if input is a Series with a different name.

Add literal column

```
# Create a DataFrame
df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

# Add literal column
df.with_column("constant", lit(1)).show()
# Output:
# +-----+---+--------+
# | name|age|constant|
# +-----+---+--------+
# |Alice| 25|       1|
# |  Bob| 30|       1|
# +-----+---+--------+
```

Add computed column

```
# Add computed column
df.with_column("double_age", col("age") * 2).show()
# Output:
# +-----+---+----------+
# | name|age|double_age|
# +-----+---+----------+
# |Alice| 25|        50|
# |  Bob| 30|        60|
# +-----+---+----------+
```

Replace existing column

```
# Replace existing column
df.with_column("age", col("age") + 1).show()
# Output:
# +-----+---+
# | name|age|
# +-----+---+
# |Alice| 26|
# |  Bob| 31|
# +-----+---+
```

Add column with complex expression

```
# Add column with complex expression
df.with_column(
    "age_category",
    when(col("age") < 30, "young")
    .when(col("age") < 50, "middle")
    .otherwise("senior")
).show()
# Output:
# +-----+---+------------+
# | name|age|age_category|
# +-----+---+------------+
# |Alice| 25|       young|
# |  Bob| 30|     middle|
# +-----+---+------------+
```

Add column from Polars Series

```
import polars as pl

# Create a DataFrame
df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

# Add column from Polars Series
bonus = pl.Series([100, 200])
df.with_column("bonus", bonus).show()
# Output:
# +-----+---+-----+
# | name|age|bonus|
# +-----+---+-----+
# |Alice| 25|  100|
# |  Bob| 30|  200|
# +-----+---+-----+
```

Add column from pandas Series

```
import pandas as pd

# Create a DataFrame
df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

# Add column from pandas Series
score = pd.Series([85.5, 92.0])
df.with_column("score", score).show()
# Output:
# +-----+---+-----+
# | name|age|score|
# +-----+---+-----+
# |Alice| 25| 85.5|
# |  Bob| 30| 92.0|
# +-----+---+-----+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def with_column(self, col_name: str, col: Union[Any, Column, pl.Series, pd.Series]) -> DataFrame:
    """Add a new column or replace an existing column.

    Args:
        col_name: Name of the new column
        col: Column expression, Series, or value to assign to the column:

            - Column: A Column expression (e.g., `col("age") + 1`)
            - `pl.Series` or `pd.Series`: A Polars or pandas Series with data
                - **Note: Series length MUST match the DataFrame height**
            - Any other value: Treated as a literal value (broadcast to all rows)

    Returns:
        DataFrame: New DataFrame with added/replaced column

    Raises:
        ExecutionError:
            - If a Series length does not match the DataFrame height
        ValidationError:
            - If the Series contains all null values and no dtype is specified
            - If the Series has length 0
    Notes:
        - The name of the created column will be the name defined in col_name, even if input is a Series with a different name.

    Example: Add literal column
        ```python
        # Create a DataFrame
        df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Add literal column
        df.with_column("constant", lit(1)).show()
        # Output:
        # +-----+---+--------+
        # | name|age|constant|
        # +-----+---+--------+
        # |Alice| 25|       1|
        # |  Bob| 30|       1|
        # +-----+---+--------+
        ```

    Example: Add computed column
        ```python
        # Add computed column
        df.with_column("double_age", col("age") * 2).show()
        # Output:
        # +-----+---+----------+
        # | name|age|double_age|
        # +-----+---+----------+
        # |Alice| 25|        50|
        # |  Bob| 30|        60|
        # +-----+---+----------+
        ```

    Example: Replace existing column
        ```python
        # Replace existing column
        df.with_column("age", col("age") + 1).show()
        # Output:
        # +-----+---+
        # | name|age|
        # +-----+---+
        # |Alice| 26|
        # |  Bob| 31|
        # +-----+---+
        ```

    Example: Add column with complex expression
        ```python
        # Add column with complex expression
        df.with_column(
            "age_category",
            when(col("age") < 30, "young")
            .when(col("age") < 50, "middle")
            .otherwise("senior")
        ).show()
        # Output:
        # +-----+---+------------+
        # | name|age|age_category|
        # +-----+---+------------+
        # |Alice| 25|       young|
        # |  Bob| 30|     middle|
        # +-----+---+------------+
        ```

    Example: Add column from Polars Series
        ```python
        import polars as pl

        # Create a DataFrame
        df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Add column from Polars Series
        bonus = pl.Series([100, 200])
        df.with_column("bonus", bonus).show()
        # Output:
        # +-----+---+-----+
        # | name|age|bonus|
        # +-----+---+-----+
        # |Alice| 25|  100|
        # |  Bob| 30|  200|
        # +-----+---+-----+
        ```

    Example: Add column from pandas Series
        ```python
        import pandas as pd

        # Create a DataFrame
        df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Add column from pandas Series
        score = pd.Series([85.5, 92.0])
        df.with_column("score", score).show()
        # Output:
        # +-----+---+-----+
        # | name|age|score|
        # +-----+---+-----+
        # |Alice| 25| 85.5|
        # |  Bob| 30| 92.0|
        # +-----+---+-----+
        ```
    """
    exprs = []

    # Handle different input types: Column, Series, or literal value
    if isinstance(col, (pl.Series, pd.Series)):
        # Wrap Series in SeriesLiteralExpr and then in Column
        col = Column._from_logical_expr(SeriesLiteralExpr(col))
    elif not isinstance(col, Column):
        # Wrap other values as literals
        col = lit(col)

    for field in self.columns:
        if field != col_name:
            exprs.append(Column._from_column_name(field)._logical_expr)

    # Add the new column with alias
    exprs.append(col.alias(col_name)._logical_expr)

    return self._from_logical_plan(
        Projection.from_session_state(self._logical_plan, exprs, self._session_state),
        self._session_state,
    )
```

### with_column_renamed

```
with_column_renamed(col_name: str, new_col_name: str) -> DataFrame
```

Rename a column. No-op if the column does not exist.

Parameters:

- **`col_name`**
  (`str`)
  –

  Name of the column to rename.
- **`new_col_name`**
  (`str`)
  –

  New name for the column.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with the column renamed.

Rename a column

```
# Create sample DataFrame
df = session.create_dataframe({
    "age": [25, 30, 35],
    "name": ["Alice", "Bob", "Charlie"]
})

# Rename a column
df.with_column_renamed("age", "age_in_years").show()
# Output:
# +------------+-------+
# |age_in_years|   name|
# +------------+-------+
# |         25|  Alice|
# |         30|    Bob|
# |         35|Charlie|
# +------------+-------+
```

Rename multiple columns

```
# Rename multiple columns
df = (df
    .with_column_renamed("age", "age_in_years")
    .with_column_renamed("name", "full_name")
).show()
# Output:
# +------------+----------+
# |age_in_years|full_name |
# +------------+----------+
# |         25|     Alice|
# |         30|       Bob|
# |         35|   Charlie|
# +------------+----------+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def with_column_renamed(self, col_name: str, new_col_name: str) -> DataFrame:
    """Rename a column. No-op if the column does not exist.

    Args:
        col_name: Name of the column to rename.
        new_col_name: New name for the column.

    Returns:
        DataFrame: New DataFrame with the column renamed.

    Example: Rename a column
        ```python
        # Create sample DataFrame
        df = session.create_dataframe({
            "age": [25, 30, 35],
            "name": ["Alice", "Bob", "Charlie"]
        })

        # Rename a column
        df.with_column_renamed("age", "age_in_years").show()
        # Output:
        # +------------+-------+
        # |age_in_years|   name|
        # +------------+-------+
        # |         25|  Alice|
        # |         30|    Bob|
        # |         35|Charlie|
        # +------------+-------+
        ```

    Example: Rename multiple columns
        ```python
        # Rename multiple columns
        df = (df
            .with_column_renamed("age", "age_in_years")
            .with_column_renamed("name", "full_name")
        ).show()
        # Output:
        # +------------+----------+
        # |age_in_years|full_name |
        # +------------+----------+
        # |         25|     Alice|
        # |         30|       Bob|
        # |         35|   Charlie|
        # +------------+----------+
        ```
    """
    exprs = []
    renamed = False

    for field in self.schema.column_fields:
        name = field.name
        if name == col_name:
            exprs.append(col(name).alias(new_col_name)._logical_expr)
            renamed = True
        else:
            exprs.append(col(name)._logical_expr)

    if not renamed:
        return self

    return self._from_logical_plan(
        Projection.from_session_state(self._logical_plan, exprs, self._session_state),
        self._session_state,
    )
```

### with_columns

```
with_columns(cols_map: Dict[str, Union[Any, Column, Series, Series]]) -> DataFrame
```

Add multiple new columns or replace existing columns.

Parameters:

- **`cols_map`**
  (`Dict[str, Union[Any, Column, Series, Series]]`)
  –

  A dictionary where keys are column names and values are:

  - Column: Column expressions (e.g., col("age") + 1)
  - pl.Series or pd.Series: Series with data
    - **Note: Series length MUST match the DataFrame height**
  - Any other value: Treated as literal values (broadcast to all rows)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with added/replaced columns

Raises:

- `ValueError`
  –

  - If two columns being created in the same `with_columns` call depend on each other
- `ExecutionError`
  –

  - If any Series length does not match the DataFrame height
- `ValidationError`
  –

  - If any Series contains all null values and no dtype is specified
  - If any Series has length 0

Notes:
- All columns are created at once, so new columns cannot depend on each other.
- The name of the created column will be the name defined in cols_map, even if input is a Series with a different name.

Add multiple columns

```
# Create a DataFrame
df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

# Add multiple columns at once
df.with_columns({
    "double_age": col("age") * 2,
    "constant": lit(1),
    "age_plus_10": col("age") + 10
}).show()
# Output:
# +-----+---+----------+--------+-----------+
# | name|age|double_age|constant|age_plus_10|
# +-----+---+----------+--------+-----------+
# |Alice| 25|        50|       1|         35|
# |  Bob| 30|        60|       1|         40|
# +-----+---+----------+--------+-----------+
```

Replace and add columns

```
# Replace existing column and add new ones
df.with_columns({
    "age": col("age") + 1,
    "is_adult": col("age") >= 18
}).show()
# Output:
# +-----+---+--------+
# | name|age|is_adult|
# +-----+---+--------+
# |Alice| 26|    true|
# |  Bob| 31|    true|
# +-----+---+--------+
```

Complex expressions

```
# Add multiple columns with complex expressions
df.with_columns({
    "age_category": when(col("age") < 30, "young")
        .when(col("age") < 50, "middle")
        .otherwise("senior"),
    "name_length": length(col("name")),
    "name_upper": upper(col("name"))
}).show()
# Output:
# +-----+---+------------+-----------+----------+
# | name|age|age_category|name_length|name_upper|
# +-----+---+------------+-----------+----------+
# |Alice| 25|       young|          5|     ALICE|
# |  Bob| 30|      middle|          3|       BOB|
# +-----+---+------------+-----------+----------+
```

Add columns from Series

```
import polars as pl

# Create a DataFrame
df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

# Add multiple columns from Series
df.with_columns({
    "bonus": pl.Series([100, 200]),
    "score": pl.Series([85.5, 92.0])
}).show()
# Output:
# +-----+---+-----+-----+
# | name|age|bonus|score|
# +-----+---+-----+-----+
# |Alice| 25|  100| 85.5|
# |  Bob| 30|  200| 92.0|
# +-----+---+-----+-----+
```

Mix Series with Column expressions

```
import polars as pl

# Mix Series with Column expressions
df.with_columns({
    "bonus": pl.Series([100, 200]),
    "double_age": col("age") * 2,
    "constant": 1
}).show()
# Output:
# +-----+---+-----+----------+--------+
# | name|age|bonus|double_age|constant|
# +-----+---+-----+----------+--------+
# |Alice| 25|  100|        50|       1|
# |  Bob| 30|  200|        60|       1|
# +-----+---+-----+----------+--------+
```

Error when adding columns that depend on each other

```
df.with_columns({
    "age_plus_1": col("age") + 1,
    "age_plus_2": col("age_plus_1") + 1
})
# ValueError: Column 'age_plus_1' not found in schema

# Instead, use a single with_column call
df = df.with_column(
    "age_plus_1", col("age") + 1
).with_column(
    "age_plus_2", col("age_plus_1") + 1
)
df.show()
# Output:
# +-----+---+----------+----------+
# | name|age|age_plus_1|age_plus_2|
# +-----+---+----------+----------+
# |Alice| 25|        26|        27|
# |  Bob| 30|        31|        32|
# +-----+---+----------+----------+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def with_columns(self, cols_map: Dict[str, Union[Any, Column, pl.Series, pd.Series]]) -> DataFrame:
    """Add multiple new columns or replace existing columns.

    Args:
        cols_map: A dictionary where keys are column names and values are:

            - Column: Column expressions (e.g., col("age") + 1)
            - pl.Series or pd.Series: Series with data
                - **Note: Series length MUST match the DataFrame height**
            - Any other value: Treated as literal values (broadcast to all rows)

    Returns:
        DataFrame: New DataFrame with added/replaced columns

    Raises:
        ValueError:
            - If two columns being created in the same `with_columns` call depend on each other
        ExecutionError:
            - If any Series length does not match the DataFrame height
        ValidationError:
            - If any Series contains all null values and no dtype is specified
            - If any Series has length 0
    Notes:
        - All columns are created at once, so new columns cannot depend on each other.
        - The name of the created column will be the name defined in cols_map, even if input is a Series with a different name.

    Example: Add multiple columns
        ```python
        # Create a DataFrame
        df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Add multiple columns at once
        df.with_columns({
            "double_age": col("age") * 2,
            "constant": lit(1),
            "age_plus_10": col("age") + 10
        }).show()
        # Output:
        # +-----+---+----------+--------+-----------+
        # | name|age|double_age|constant|age_plus_10|
        # +-----+---+----------+--------+-----------+
        # |Alice| 25|        50|       1|         35|
        # |  Bob| 30|        60|       1|         40|
        # +-----+---+----------+--------+-----------+
        ```

    Example: Replace and add columns
        ```python
        # Replace existing column and add new ones
        df.with_columns({
            "age": col("age") + 1,
            "is_adult": col("age") >= 18
        }).show()
        # Output:
        # +-----+---+--------+
        # | name|age|is_adult|
        # +-----+---+--------+
        # |Alice| 26|    true|
        # |  Bob| 31|    true|
        # +-----+---+--------+
        ```

    Example: Complex expressions
        ```python
        # Add multiple columns with complex expressions
        df.with_columns({
            "age_category": when(col("age") < 30, "young")
                .when(col("age") < 50, "middle")
                .otherwise("senior"),
            "name_length": length(col("name")),
            "name_upper": upper(col("name"))
        }).show()
        # Output:
        # +-----+---+------------+-----------+----------+
        # | name|age|age_category|name_length|name_upper|
        # +-----+---+------------+-----------+----------+
        # |Alice| 25|       young|          5|     ALICE|
        # |  Bob| 30|      middle|          3|       BOB|
        # +-----+---+------------+-----------+----------+
        ```

    Example: Add columns from Series
        ```python
        import polars as pl

        # Create a DataFrame
        df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Add multiple columns from Series
        df.with_columns({
            "bonus": pl.Series([100, 200]),
            "score": pl.Series([85.5, 92.0])
        }).show()
        # Output:
        # +-----+---+-----+-----+
        # | name|age|bonus|score|
        # +-----+---+-----+-----+
        # |Alice| 25|  100| 85.5|
        # |  Bob| 30|  200| 92.0|
        # +-----+---+-----+-----+
        ```

    Example: Mix Series with Column expressions
        ```python
        import polars as pl

        # Mix Series with Column expressions
        df.with_columns({
            "bonus": pl.Series([100, 200]),
            "double_age": col("age") * 2,
            "constant": 1
        }).show()
        # Output:
        # +-----+---+-----+----------+--------+
        # | name|age|bonus|double_age|constant|
        # +-----+---+-----+----------+--------+
        # |Alice| 25|  100|        50|       1|
        # |  Bob| 30|  200|        60|       1|
        # +-----+---+-----+----------+--------+
        ```

    Example: Error when adding columns that depend on each other
        ```python
        df.with_columns({
            "age_plus_1": col("age") + 1,
            "age_plus_2": col("age_plus_1") + 1
        })
        # ValueError: Column 'age_plus_1' not found in schema

        # Instead, use a single with_column call
        df = df.with_column(
            "age_plus_1", col("age") + 1
        ).with_column(
            "age_plus_2", col("age_plus_1") + 1
        )
        df.show()
        # Output:
        # +-----+---+----------+----------+
        # | name|age|age_plus_1|age_plus_2|
        # +-----+---+----------+----------+
        # |Alice| 25|        26|        27|
        # |  Bob| 30|        31|        32|
        # +-----+---+----------+----------+
        ```
    """
    if not cols_map:
        return self

    exprs = []
    new_col_names = set(cols_map.keys())

    # Add existing columns that are not being replaced
    for field in self.columns:
        if field not in new_col_names:
            exprs.append(Column._from_column_name(field)._logical_expr)

    # Add all new columns with aliases
    for col_name, col_expr in cols_map.items():
        # Handle different input types: Column, Series, or literal value
        if isinstance(col_expr, (pl.Series, pd.Series)):
            # Wrap Series in SeriesLiteralExpr and then in Column
            col_expr = Column._from_logical_expr(SeriesLiteralExpr(col_expr))
        elif not isinstance(col_expr, Column):
            # Automatically wrap non-Column values (literals) with lit() for convenience
            # This allows users to pass raw Python values like: {"constant": 100, "status": "active"}
            # instead of requiring: {"constant": lit(100), "status": lit("active")}
            col_expr = lit(col_expr)
        exprs.append(col_expr.alias(col_name)._logical_expr)

    return self._from_logical_plan(
        Projection.from_session_state(self._logical_plan, exprs, self._session_state),
        self._session_state,
    )
```

## DataFrameReader

```
DataFrameReader(session_state: BaseSessionState)
```

Interface used to load a DataFrame from external storage systems.

Similar to PySpark's DataFrameReader.

Supported External Storage Schemes:
- Amazon S3 (s3://)
- Format: s3://{bucket_name}/{path_to_file}

```
- Notes:
    - Uses boto3 to aquire AWS credentials.

- Examples:
    - s3://my-bucket/data.csv
    - s3://my-bucket/data/*.parquet
```

- Hugging Face Datasets (hf://)

  - Format: hf://{repo_type}/{repo_id}/{path_to_file}
  - Notes:

    - Supports glob patterns (*,* \*)
    - Supports dataset revisions and branch aliases (e.g., @refs/convert/parquet, @~parquet)
    - HF_TOKEN environment variable is required to read private datasets.
  - Examples:

    - hf://datasets/datasets-examples/doc-formats-csv-1/data.csv
    - hf://datasets/cais/mmlu/astronomy/\*.parquet
    - hf://datasets/datasets-examples/doc-formats-csv-1@~parquet/\**/*.parquet
- Local Files (file:// or implicit)

  - Format: file://{absolute_or_relative_path}
  - Notes:

    - Paths without a scheme (e.g., ./data.csv or /tmp/data.parquet) are treated as local files
  - Examples:
    - file:///home/user/data.csv
    - ./data/\*.parquet

Creates a DataFrameReader.

Parameters:

- **`session_state`**
  (`BaseSessionState`)
  –

  The session state to use for reading

Methods:

- **`csv`**
  –

  Load a DataFrame from one or more CSV files.
- **`docs`**
  –

  Load a DataFrame with the document contents of a list of paths (markdown or json).
- **`parquet`**
  –

  Load a DataFrame from one or more Parquet files.
- **`pdf_metadata`**
  –

  Load a DataFrame with metadata of PDF files in a list of paths.

Source code in `src/fenic/api/io/reader.py`

```
def __init__(self, session_state: BaseSessionState):
    """Creates a DataFrameReader.

    Args:
        session_state: The session state to use for reading
    """
    self._options: Dict[str, Any] = {}
    self._session_state = session_state
```

### csv

```
csv(paths: Union[str, Path, list[Union[str, Path]]], schema: Optional[Schema] = None, merge_schemas: bool = False) -> DataFrame
```

Load a DataFrame from one or more CSV files.

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  A single file path, a glob pattern (e.g., "data/\*.csv"), or a list of paths.
- **`schema`**
  (`Optional[Schema]`, default:
  `None`
  )
  –

  (optional) A complete schema definition of column names and their types. Only primitive types are supported.
  - For e.g.:
  - Schema([ColumnField(name="id", data_type=IntegerType), ColumnField(name="name", data_type=StringType)])
  - If provided, all files must match this schema exactly—all column names must be present, and values must be
  convertible to the specified types. Partial schemas are not allowed.
- **`merge_schemas`**
  (`bool`, default:
  `False`
  )
  –

  Whether to merge schemas across all files.
  - If True: Column names are unified across files. Missing columns are filled with nulls. Column types are
  inferred and widened as needed.
  - If False (default): Only accepts columns from the first file. Column types from the first file are
  inferred and applied across all files. If subsequent files do not have the same column name and order as the first file, an error is raised.
  - The "first file" is defined as:
  - The first file in lexicographic order (for glob patterns), or
  - The first file in the provided list (for lists of paths).

Notes

- The first row in each file is assumed to be a header row.
- Delimiters (e.g., comma, tab) are automatically inferred.
- You may specify either `schema` or `merge_schemas=True`, but not both.
- Any date/datetime columns are cast to strings during ingestion.

Raises:

- `ValidationError`
  –

  If both `schema` and `merge_schemas=True` are provided.
- `ValidationError`
  –

  If any path does not end with `.csv`.
- `PlanError`
  –

  If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

Read a single CSV file

```
df = session.read.csv("file.csv")
```

Read multiple CSV files with schema merging

```
df = session.read.csv("data/*.csv", merge_schemas=True)
```

Read CSV files with explicit schema

`python
df = session.read.csv(
["a.csv", "b.csv"],
schema=Schema([
ColumnField(name="id", data_type=IntegerType),
ColumnField(name="value", data_type=FloatType)
])
)`

Source code in `src/fenic/api/io/reader.py`

```
def csv(
    self,
    paths: Union[str, Path, list[Union[str, Path]]],
    schema: Optional[Schema] = None,
    merge_schemas: bool = False,
) -> DataFrame:
    """Load a DataFrame from one or more CSV files.

    Args:
        paths: A single file path, a glob pattern (e.g., "data/*.csv"), or a list of paths.
        schema: (optional) A complete schema definition of column names and their types. Only primitive types are supported.
            - For e.g.:
                - Schema([ColumnField(name="id", data_type=IntegerType), ColumnField(name="name", data_type=StringType)])
            - If provided, all files must match this schema exactly—all column names must be present, and values must be
            convertible to the specified types. Partial schemas are not allowed.
        merge_schemas: Whether to merge schemas across all files.
            - If True: Column names are unified across files. Missing columns are filled with nulls. Column types are
            inferred and widened as needed.
            - If False (default): Only accepts columns from the first file. Column types from the first file are
            inferred and applied across all files. If subsequent files do not have the same column name and order as the first file, an error is raised.
            - The "first file" is defined as:
                - The first file in lexicographic order (for glob patterns), or
                - The first file in the provided list (for lists of paths).

    Notes:
        - The first row in each file is assumed to be a header row.
        - Delimiters (e.g., comma, tab) are automatically inferred.
        - You may specify either `schema` or `merge_schemas=True`, but not both.
        - Any date/datetime columns are cast to strings during ingestion.

    Raises:
        ValidationError: If both `schema` and `merge_schemas=True` are provided.
        ValidationError: If any path does not end with `.csv`.
        PlanError: If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

    Example: Read a single CSV file
        ```python
        df = session.read.csv("file.csv")
        ```

    Example: Read multiple CSV files with schema merging
        ```python
        df = session.read.csv("data/*.csv", merge_schemas=True)
        ```

    Example: Read CSV files with explicit schema
        ```python
        df = session.read.csv(
            ["a.csv", "b.csv"],
            schema=Schema([
                ColumnField(name="id", data_type=IntegerType),
                ColumnField(name="value", data_type=FloatType)
            ])
        )            ```
    """
    if schema is not None and merge_schemas:
        raise ValidationError(
            "Cannot specify both 'schema' and 'merge_schemas=True' - these options conflict. "
            "Choose one approach: "
            "1) Use 'schema' to enforce a specific schema: csv(paths, schema=your_schema), "
            "2) Use 'merge_schemas=True' to automatically merge schemas: csv(paths, merge_schemas=True), "
            "3) Use neither to inherit schema from the first file: csv(paths)"
        )
    if schema is not None:
        for col_field in schema.column_fields:
            if not isinstance(
                col_field.data_type,
                _PrimitiveType,
            ):
                raise ValidationError(
                    f"CSV files only support primitive data types in schema definitions. "
                    f"Column '{col_field.name}' has type {type(col_field.data_type).__name__}, but CSV schemas must use: "
                    f"IntegerType, FloatType, DoubleType, BooleanType, or StringType. "
                    f"Example: Schema([ColumnField(name='id', data_type=IntegerType), ColumnField(name='name', data_type=StringType)])"
                )
    options = {
        "merge_schemas": merge_schemas,
    }
    if schema:
        options["schema"] = schema
    return self._read_file(
        paths, file_format="csv", file_extension=".csv", **options
    )
```

### docs

```
docs(paths: Union[str, Path, list[Union[str, Path]]], content_type: Literal['markdown', 'json'], exclude: Optional[str] = None, recursive: bool = False) -> DataFrame
```

Load a DataFrame with the document contents of a list of paths (markdown or json).

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  Glob pattern (or list of glob patterns) to the folder(s) to load.
- **`content_type`**
  (`Literal['markdown', 'json']`)
  –

  Content type of the files. One of "markdown" or "json".
- **`exclude`**
  (`Optional[str]`, default:
  `None`
  )
  –

  A regex pattern to exclude files.
  If it is not provided no files will be excluded.
- **`recursive`**
  (`bool`, default:
  `False`
  )
  –

  Whether to recursively load files from the folder.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A dataframe with all the documents found in the paths.
  The content of each document is a row in the dataframe.

Raises:

- `ValidationError`
  –

  If any file does not have a `.md` or `.json` depending on the content_type.
- `UnsupportedFileTypeError`
  –

  If the specified content_type is not "markdown" or "json" .

Notes

- Each row in the dataframe corresponds to a file in the list of paths.
- The dataframe has the following columns:
  - file_path: The path to the file.
  - error: The error message if the file failed to be loaded.
  - content: The content of the file casted to the content_type.
- Recursive loading is supported in conjunction with the '\**' glob pattern,
  e.g. `data/**/*.md` will load all markdown files in the `data` folder and all subfolders
  when recursive is set to True.
  Without recursive = True, then \*\* behaves like a single '*' pattern.

Read all the markdown files in a folder and all its subfolders.

```
df = session.read.docs("data/docs/**/*.md", content_type="markdown", recursive=True)
```

Read a folder of markdown files excluding some files.

```
df = session.read.docs("data/docs/*.md", content_type="markdown", exclude=r"\.bak.md$")
```

Source code in `src/fenic/api/io/reader.py`

```
def docs(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        content_type: Literal["markdown", "json"],
        exclude: Optional[str] = None,
        recursive: bool = False,
) -> DataFrame:
    r"""Load a DataFrame with the document contents of a list of paths (markdown or json).

    Args:
        paths: Glob pattern (or list of glob patterns) to the folder(s) to load.
        content_type: Content type of the files. One of "markdown" or "json".
        exclude: A regex pattern to exclude files.
                 If it is not provided no files will be excluded.
        recursive: Whether to recursively load files from the folder.

    Returns:
        DataFrame: A dataframe with all the documents found in the paths.
                   The content of each document is a row in the dataframe.

    Raises:
        ValidationError: If any file does not have a `.md` or `.json` depending on the content_type.
        UnsupportedFileTypeError: If the specified content_type is not "markdown" or "json" .

    Notes:
        - Each row in the dataframe corresponds to a file in the list of paths.
        - The dataframe has the following columns:
            - file_path: The path to the file.
            - error: The error message if the file failed to be loaded.
            - content: The content of the file casted to the content_type.
        - Recursive loading is supported in conjunction with the '**' glob pattern,
          e.g. `data/**/*.md` will load all markdown files in the `data` folder and all subfolders
               when recursive is set to True.
          Without recursive = True, then ** behaves like a single '*' pattern.

    Example: Read all the markdown files in a folder and all its subfolders.
        ```python
        df = session.read.docs("data/docs/**/*.md", content_type="markdown", recursive=True)
        ```

    Example: Read a folder of markdown files excluding some files.
        ```python
        df = session.read.docs("data/docs/*.md", content_type="markdown", exclude=r"\.bak.md$")
        ```

    """
    path_str_list = validate_paths_and_return_list_of_strings(paths)

    if content_type not in ["markdown", "json"]:
        raise UnsupportedFileTypeError(f"{content_type}, must be 'markdown' or 'json'")

    logical_node = DocSource.from_session_state(
        paths=path_str_list,
        content_type=content_type,
        exclude=exclude,
        recursive=recursive,
        session_state=self._session_state,
    )
    from fenic.api.dataframe import DataFrame

    return DataFrame._from_logical_plan(logical_node, self._session_state)
```

### parquet

```
parquet(paths: Union[str, Path, list[Union[str, Path]]], merge_schemas: bool = False) -> DataFrame
```

Load a DataFrame from one or more Parquet files.

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  A single file path, a glob pattern (e.g., "data/\*.parquet"), or a list of paths.
- **`merge_schemas`**
  (`bool`, default:
  `False`
  )
  –

  If True, infers and merges schemas across all files.
  Missing columns are filled with nulls, and differing types are widened to a common supertype.

Behavior

- If `merge_schemas=False` (default), all files must match the schema of the first file exactly.
  Subsequent files must contain all columns from the first file with compatible data types.
  If any column is missing or has incompatible types, an error is raised.
- If `merge_schemas=True`, column names are unified across all files, and data types are automatically
  widened to accommodate all values.
- The "first file" is defined as:
  - The first file in lexicographic order (for glob patterns), or
  - The first file in the provided list (for lists of paths).

Notes

- Date and datetime columns are cast to strings during ingestion.

Raises:

- `ValidationError`
  –

  If any file does not have a `.parquet` extension.
- `PlanError`
  –

  If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

Read a single Parquet file

```
df = session.read.parquet("file.parquet")
```

Read multiple Parquet files

```
df = session.read.parquet("data/*.parquet")
```

Read Parquet files with schema merging

```
df = session.read.parquet(["a.parquet", "b.parquet"], merge_schemas=True)
```

Source code in `src/fenic/api/io/reader.py`

```
def parquet(
    self,
    paths: Union[str, Path, list[Union[str, Path]]],
    merge_schemas: bool = False,
) -> DataFrame:
    """Load a DataFrame from one or more Parquet files.

    Args:
        paths: A single file path, a glob pattern (e.g., "data/*.parquet"), or a list of paths.
        merge_schemas: If True, infers and merges schemas across all files.
            Missing columns are filled with nulls, and differing types are widened to a common supertype.

    Behavior:
        - If `merge_schemas=False` (default), all files must match the schema of the first file exactly.
        Subsequent files must contain all columns from the first file with compatible data types.
        If any column is missing or has incompatible types, an error is raised.
        - If `merge_schemas=True`, column names are unified across all files, and data types are automatically
        widened to accommodate all values.
        - The "first file" is defined as:
            - The first file in lexicographic order (for glob patterns), or
            - The first file in the provided list (for lists of paths).

    Notes:
        - Date and datetime columns are cast to strings during ingestion.

    Raises:
        ValidationError: If any file does not have a `.parquet` extension.
        PlanError: If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

    Example: Read a single Parquet file
        ```python
        df = session.read.parquet("file.parquet")
        ```

    Example: Read multiple Parquet files
        ```python
        df = session.read.parquet("data/*.parquet")
        ```

    Example: Read Parquet files with schema merging
        ```python
        df = session.read.parquet(["a.parquet", "b.parquet"], merge_schemas=True)
        ```
    """
    options = {
        "merge_schemas": merge_schemas,
    }
    return self._read_file(
        paths, file_format="parquet", file_extension=".parquet", **options
    )
```

### pdf_metadata

```
pdf_metadata(paths: Union[str, Path, list[Union[str, Path]]], exclude: Optional[str] = None, recursive: bool = False) -> DataFrame
```

Load a DataFrame with metadata of PDF files in a list of paths.

Note

Local execution requires the `pdf` extra: `pip install "fenic[pdf]"`.

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  Glob pattern (or list of glob patterns) to the folder(s) to load.
- **`exclude`**
  (`Optional[str]`, default:
  `None`
  )
  –

  A regex pattern to exclude files.
  If it is not provided no files will be excluded.
- **`recursive`**
  (`bool`, default:
  `False`
  )
  –

  Whether to recursively load files from the folder.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A dataframe with the metadata of all the PDF files found in the paths.
  the metadata from a single PDF document is a row in the dataframe.

Raises:

- `ValidationError`
  –

  If any file does not have a `.pdf` extension.

Notes

- Each row in the dataframe corresponds to a file in the list of paths.
- The metadata columns are:
  - file_path: The path to the document.
  - error: The error message if the file failed to be loaded.
  - size: Size of the PDF file in bytes.
  - title: Title of the PDF document.
  - author: Author of the PDF document.
  - creation_date: Creation date of the PDF.
  - mod_date: Modification date of the PDF.
  - page_count: Number of pages in the PDF.
  - has_forms: Whether the PDF contains form fields, or fields that accept user input.
  - has_signature_fields: Whether the PDF contains signature fields.
  - image_count: Number of images in the PDF.
  - is_encrypted: Whether the PDF is encrypted.
- Recursive loading is supported in conjunction with the '\**' glob pattern,
  e.g. `data/**/*.pdf` will load all PDF files in the `data` folder and all subfolders
  when recursive is set to True.
  Without recursive = True, then \*\* behaves like a single '*' pattern.

Read the metadata of all the PDF files in a folder and all its subfolders.

```
df = session.read.pdf_metadata("data/docs/**/*.pdf", recursive=True)
```

Read a metadata of PDFS in a folder, excluding some files.

```
df = session.read.pdf_metadata("data/docs/*.pdf", exclude=r"\.backup.pdf$")
```

Source code in `src/fenic/api/io/reader.py`

```
def pdf_metadata(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        exclude: Optional[str] = None,
        recursive: bool = False,
) -> DataFrame:
    r"""Load a DataFrame with metadata of PDF files in a list of paths.

    Note:
        Local execution requires the `pdf` extra: `pip install "fenic[pdf]"`.

    Args:
        paths: Glob pattern (or list of glob patterns) to the folder(s) to load.
        exclude: A regex pattern to exclude files.
                 If it is not provided no files will be excluded.
        recursive: Whether to recursively load files from the folder.

    Returns:
        DataFrame: A dataframe with the metadata of all the PDF files found in the paths.
                   the metadata from a single PDF document is a row in the dataframe.

    Raises:
        ValidationError: If any file does not have a `.pdf` extension.

    Notes:
        - Each row in the dataframe corresponds to a file in the list of paths.
        - The metadata columns are:
            - file_path: The path to the document.
            - error: The error message if the file failed to be loaded.
            - size: Size of the PDF file in bytes.
            - title: Title of the PDF document.
            - author: Author of the PDF document.
            - creation_date: Creation date of the PDF.
            - mod_date: Modification date of the PDF.
            - page_count: Number of pages in the PDF.
            - has_forms: Whether the PDF contains form fields, or fields that accept user input.
            - has_signature_fields: Whether the PDF contains signature fields.
            - image_count: Number of images in the PDF.
            - is_encrypted: Whether the PDF is encrypted.
        - Recursive loading is supported in conjunction with the '**' glob pattern,
          e.g. `data/**/*.pdf` will load all PDF files in the `data` folder and all subfolders
               when recursive is set to True.
          Without recursive = True, then ** behaves like a single '*' pattern.

    Example: Read the metadata of all the PDF files in a folder and all its subfolders.
        ```python
        df = session.read.pdf_metadata("data/docs/**/*.pdf", recursive=True)
        ```

    Example: Read a metadata of PDFS in a folder, excluding some files.
        ```python
        df = session.read.pdf_metadata("data/docs/*.pdf", exclude=r"\.backup.pdf$")
        ```

    """
    path_str_list = validate_paths_and_return_list_of_strings(paths)

    logical_node = DocSource.from_session_state(
        paths=path_str_list,
        content_type="pdf",
        exclude=exclude,
        recursive=recursive,
        session_state=self._session_state,
    )
    from fenic.api.dataframe import DataFrame

    return DataFrame._from_logical_plan(logical_node, self._session_state)
```

## DataFrameWriter

```
DataFrameWriter(dataframe: DataFrame)
```

Interface used to write a DataFrame to external storage systems.

Similar to PySpark's DataFrameWriter.

Supported External Storage Schemes:
- Amazon S3 (s3://)
- Format: s3://{bucket_name}/{path_to_file}

```
- Notes:
    - Uses boto3 to aquire AWS credentials.

- Examples:
    - s3://my-bucket/data.csv
    - s3://my-bucket/data/*.parquet
```

- Local Files (file:// or implicit)

  - Format: file://{absolute_or_relative_path}
  - Notes:

    - Paths without a scheme (e.g., ./data.csv or /tmp/data.parquet) are treated as local files
  - Examples:
    - file:///home/user/data.csv
    - ./data/\*.parquet

Initialize a DataFrameWriter.

Parameters:

- **`dataframe`**
  (`DataFrame`)
  –

  The DataFrame to write.

Methods:

- **`csv`**
  –

  Saves the content of the DataFrame as a single CSV file with comma as the delimiter and headers in the first row.
- **`parquet`**
  –

  Saves the content of the DataFrame as a single Parquet file.
- **`save_as_table`**
  –

  Saves the content of the DataFrame as the specified table.
- **`save_as_view`**
  –

  Saves the content of the DataFrame as a view.

Source code in `src/fenic/api/io/writer.py`

```
def __init__(self, dataframe: DataFrame):
    """Initialize a DataFrameWriter.

    Args:
        dataframe: The DataFrame to write.
    """
    self._dataframe = dataframe
```

### csv

```
csv(file_path: Union[str, Path], mode: Literal['error', 'overwrite', 'ignore'] = 'overwrite') -> QueryMetrics
```

Saves the content of the DataFrame as a single CSV file with comma as the delimiter and headers in the first row.

Parameters:

- **`file_path`**
  (`Union[str, Path]`)
  –

  Path to save the CSV file to
- **`mode`**
  (`Literal['error', 'overwrite', 'ignore']`, default:
  `'overwrite'`
  )
  –

  Write mode. Default is "overwrite".
  - error: Raises an error if file exists
  - overwrite: Overwrites the file if it exists
  - ignore: Silently ignores operation if file exists

Returns:

- **`QueryMetrics`** ( `QueryMetrics`
  ) –

  The query metrics

Save with overwrite mode (default)

```
df.write.csv("output.csv")  # Overwrites if exists
```

Save with error mode

```
df.write.csv("output.csv", mode="error")  # Raises error if exists
```

Save with ignore mode

```
df.write.csv("output.csv", mode="ignore")  # Skips if exists
```

Source code in `src/fenic/api/io/writer.py`

```
def csv(
    self,
    file_path: Union[str, Path],
    mode: Literal["error", "overwrite", "ignore"] = "overwrite",
) -> QueryMetrics:
    """Saves the content of the DataFrame as a single CSV file with comma as the delimiter and headers in the first row.

    Args:
        file_path: Path to save the CSV file to
        mode: Write mode. Default is "overwrite".
             - error: Raises an error if file exists
             - overwrite: Overwrites the file if it exists
             - ignore: Silently ignores operation if file exists

    Returns:
        QueryMetrics: The query metrics

    Example: Save with overwrite mode (default)
        ```python
        df.write.csv("output.csv")  # Overwrites if exists
        ```

    Example: Save with error mode
        ```python
        df.write.csv("output.csv", mode="error")  # Raises error if exists
        ```

    Example: Save with ignore mode
        ```python
        df.write.csv("output.csv", mode="ignore")  # Skips if exists
        ```
    """
    file_path = str(file_path)
    if not file_path.endswith(".csv"):
        raise ValidationError(
            f"CSV writer requires a '.csv' file extension. "
            f"Your path '{file_path}' is missing the extension."
        )

    sink_plan = FileSink.from_session_state(
        child=self._dataframe._logical_plan,
        sink_type="csv",
        path=file_path,
        mode=mode,
        session_state=self._dataframe._session_state,
    )

    metrics = self._dataframe._session_state.execution.save_to_file(
        sink_plan, file_path=file_path, mode=mode
    )
    logger.info(metrics.get_summary())
    return metrics
```

### parquet

```
parquet(file_path: Union[str, Path], mode: Literal['error', 'overwrite', 'ignore'] = 'overwrite') -> QueryMetrics
```

Saves the content of the DataFrame as a single Parquet file.

Parameters:

- **`file_path`**
  (`Union[str, Path]`)
  –

  Path to save the Parquet file to
- **`mode`**
  (`Literal['error', 'overwrite', 'ignore']`, default:
  `'overwrite'`
  )
  –

  Write mode. Default is "overwrite".
  - error: Raises an error if file exists
  - overwrite: Overwrites the file if it exists
  - ignore: Silently ignores operation if file exists

Returns:

- **`QueryMetrics`** ( `QueryMetrics`
  ) –

  The query metrics

Save with overwrite mode (default)

```
df.write.parquet("output.parquet")  # Overwrites if exists
```

Save with error mode

```
df.write.parquet("output.parquet", mode="error")  # Raises error if exists
```

Save with ignore mode

```
df.write.parquet("output.parquet", mode="ignore")  # Skips if exists
```

Source code in `src/fenic/api/io/writer.py`

```
def parquet(
    self,
    file_path: Union[str, Path],
    mode: Literal["error", "overwrite", "ignore"] = "overwrite",
) -> QueryMetrics:
    """Saves the content of the DataFrame as a single Parquet file.

    Args:
        file_path: Path to save the Parquet file to
        mode: Write mode. Default is "overwrite".
             - error: Raises an error if file exists
             - overwrite: Overwrites the file if it exists
             - ignore: Silently ignores operation if file exists

    Returns:
        QueryMetrics: The query metrics

    Example: Save with overwrite mode (default)
        ```python
        df.write.parquet("output.parquet")  # Overwrites if exists
        ```

    Example: Save with error mode
        ```python
        df.write.parquet("output.parquet", mode="error")  # Raises error if exists
        ```

    Example: Save with ignore mode
        ```python
        df.write.parquet("output.parquet", mode="ignore")  # Skips if exists
        ```
    """
    file_path = str(file_path)
    if not file_path.endswith(".parquet"):
        raise ValidationError(
            f"Parquet writer requires a '.parquet' file extension. "
            f"Your path '{file_path}' is missing the extension."
        )

    sink_plan = FileSink.from_session_state(
        child=self._dataframe._logical_plan,
        sink_type="parquet",
        path=file_path,
        mode=mode,
        session_state=self._dataframe._session_state,
    )

    metrics = self._dataframe._session_state.execution.save_to_file(
        sink_plan, file_path=file_path, mode=mode
    )
    logger.info(metrics.get_summary())
    return metrics
```

### save_as_table

```
save_as_table(table_name: str, mode: Literal['error', 'append', 'overwrite', 'ignore'] = 'error') -> QueryMetrics
```

Saves the content of the DataFrame as the specified table.

Parameters:

- **`table_name`**
  (`str`)
  –

  Name of the table to save to
- **`mode`**
  (`Literal['error', 'append', 'overwrite', 'ignore']`, default:
  `'error'`
  )
  –

  Write mode. Default is "error".
  - error: Raises an error if table exists
  - append: Appends data to table if it exists
  - overwrite: Overwrites existing table
  - ignore: Silently ignores operation if table exists

Returns:

- **`QueryMetrics`** ( `QueryMetrics`
  ) –

  The query metrics

Save with error mode (default)

```
df.write.save_as_table("my_table")  # Raises error if table exists
```

Save with append mode

```
df.write.save_as_table("my_table", mode="append")  # Adds to existing table
```

Save with overwrite mode

```
df.write.save_as_table("my_table", mode="overwrite")  # Replaces existing table
```

Source code in `src/fenic/api/io/writer.py`

```
def save_as_table(
    self,
    table_name: str,
    mode: Literal["error", "append", "overwrite", "ignore"] = "error",
) -> QueryMetrics:
    """Saves the content of the DataFrame as the specified table.

    Args:
        table_name: Name of the table to save to
        mode: Write mode. Default is "error".
             - error: Raises an error if table exists
             - append: Appends data to table if it exists
             - overwrite: Overwrites existing table
             - ignore: Silently ignores operation if table exists

    Returns:
        QueryMetrics: The query metrics

    Example: Save with error mode (default)
        ```python
        df.write.save_as_table("my_table")  # Raises error if table exists
        ```

    Example: Save with append mode
        ```python
        df.write.save_as_table("my_table", mode="append")  # Adds to existing table
        ```

    Example: Save with overwrite mode
        ```python
        df.write.save_as_table("my_table", mode="overwrite")  # Replaces existing table
        ```
    """
    sink_plan = TableSink.from_session_state(
        child=self._dataframe._logical_plan,
        table_name=table_name,
        mode=mode,
        session_state=self._dataframe._session_state,
    )

    metrics = self._dataframe._session_state.execution.save_as_table(
        sink_plan, table_name=table_name, mode=mode
    )
    logger.info(metrics.get_summary())
    return metrics
```

### save_as_view

```
save_as_view(view_name: str, description: str | None = None) -> None
```

Saves the content of the DataFrame as a view.

Parameters:

- **`view_name`**
  (`str`)
  –

  Name of the view to save to
- **`description`**
  (`str | None`, default:
  `None`
  )
  –

  Optional human-readable view description to store in the catalog.

Returns:

- `None`
  –

  None.

Source code in `src/fenic/api/io/writer.py`

```
def save_as_view(
    self,
    view_name: str,
    description: str | None = None,
) -> None:
    """Saves the content of the DataFrame as a view.

    Args:
        view_name: Name of the view to save to
        description: Optional human-readable view description to store in the catalog.

    Returns:
        None.
    """
    self._dataframe._session_state.execution.save_as_view(
        logical_plan=self._dataframe._logical_plan, view_name=view_name, view_description=description
    )
```

## DataType

Bases: `ABC`

```
              flowchart TD
              fenic.DataType[DataType]

              click fenic.DataType href "" "fenic.DataType"
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
              fenic.DocumentPathType[DocumentPathType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.DocumentPathType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.DocumentPathType href "" "fenic.DocumentPathType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a a document's local (file system) or remote (URL) path.

## EmbeddingType

Bases: `_LogicalType`

```
              flowchart TD
              fenic.EmbeddingType[EmbeddingType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.EmbeddingType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.EmbeddingType href "" "fenic.EmbeddingType"
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

## GoogleDeveloperEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleDeveloperEmbeddingModel[GoogleDeveloperEmbeddingModel]

              click fenic.GoogleDeveloperEmbeddingModel href "" "fenic.GoogleDeveloperEmbeddingModel"
```

Configuration for Google Developer embedding models.

This class defines the configuration settings for Google embedding models available in Google Developer AI Studio,
including model selection and rate limiting parameters. These models are accessible using a GOOGLE_API_KEY environment variable.

Attributes:

- **`model_name`**
  (`GoogleDeveloperEmbeddingModelName`)
  –

  The name of the Google Developer embedding model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Developer embedding model with rate limits:

```
config = GoogleDeveloperEmbeddingModelConfig(
    model_name="gemini-embedding-001", rpm=100, tpm=1000
)
```

Configuring a Google Developer embedding model with profiles:

```
config = GoogleDeveloperEmbeddingModelConfig(
    model_name="gemini-embedding-001",
    rpm=100,
    tpm=1000,
    profiles={
        "default": GoogleDeveloperEmbeddingModelConfig.Profile(),
        "high_dim": GoogleDeveloperEmbeddingModelConfig.Profile(
            output_dimensionality=3072
        ),
    },
    default_profile="default",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Developer embedding models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleDeveloperEmbeddingModel.Profile[Profile]

              click fenic.GoogleDeveloperEmbeddingModel.Profile href "" "fenic.GoogleDeveloperEmbeddingModel.Profile"
```

Profile configurations for Google Developer embedding models.

This class defines profile configurations for Google embedding models, allowing
different output dimensionality and task type settings to be applied to the same model.

Attributes:

- **`output_dimensionality`**
  (`Optional[int]`)
  –

  The dimensionality of the embedding created by this model.
  If not provided, the model will use its default dimensionality.
- **`task_type`**
  (`GoogleEmbeddingTaskType`)
  –

  The type of task for the embedding model.

Example

Configuring a profile with custom dimensionality:

```
profile = GoogleDeveloperEmbeddingModelConfig.Profile(
    output_dimensionality=3072
)
```

Configuring a profile with default settings:

```
profile = GoogleDeveloperEmbeddingModelConfig.Profile()
```

## GoogleDeveloperLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleDeveloperLanguageModel[GoogleDeveloperLanguageModel]

              click fenic.GoogleDeveloperLanguageModel href "" "fenic.GoogleDeveloperLanguageModel"
```

Configuration for Gemini models accessible through Google Developer AI Studio.

This class defines the configuration settings for Google Gemini models available in Google Developer AI Studio,
including model selection and rate limiting parameters. These models are accessible using a GOOGLE_API_KEY environment variable.

Attributes:

- **`model_name`**
  (`GoogleDeveloperLanguageModelName`)
  –

  The name of the Google Developer model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Developer model with rate limits:

```
config = GoogleDeveloperLanguageModel(
    model_name="gemini-2.5-flash",
    rpm=100,
    tpm=1000
)
```

Configuring a reasoning Google Developer model with profiles:

```
config = GoogleDeveloperLanguageModel(
    model_name="gemini-2.5-flash",
    rpm=100,
    tpm=1000,
    profiles={
        "thinking_disabled": GoogleDeveloperLanguageModel.Profile(),
        "fast": GoogleDeveloperLanguageModel.Profile(
            thinking_token_budget=1024
        ),
        "thorough": GoogleDeveloperLanguageModel.Profile(
            thinking_token_budget=8192
        ),
    },
    default_profile="fast",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Developer models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleDeveloperLanguageModel.Profile[Profile]

              click fenic.GoogleDeveloperLanguageModel.Profile href "" "fenic.GoogleDeveloperLanguageModel.Profile"
```

Profile configurations for Google Developer models.

This class defines profile configurations for Google Gemini models, allowing
different thinking/reasoning settings to be applied to the same model.

Attributes:

- **`thinking_token_budget`**
  (`Optional[int]`)
  –

  If configuring a reasoning model, provide a thinking budget in tokens.
  If not provided, or if set to 0, thinking will be disabled for the profile (not supported on gemini-2.5-pro).
  To have the model automatically determine a thinking budget based on the complexity of
  the prompt, set this to -1. Note that Gemini models take this as a suggestion -- and not a hard limit.
  It is very possible for the model to generate far more thinking tokens than the suggested budget, and for the
  model to generate reasoning tokens even if thinking is disabled.
  Note: For gemini-3 models, use thinking_level instead.
- **`thinking_level`**
  (`Optional[ThinkingLevelType]`)
  –

  For gemini-3+ models, set the thinking level to high, medium, low, or minimal.
  This parameter is mutually exclusive with thinking_token_budget.
- **`media_resolution`**
  (`Optional[MediaResolutionType]`)
  –

  For gemini-3+ models, set the media resolution for PDF processing.
  Can be "low", "medium", or "high". Affects token cost per page.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Example

Configuring a profile with a fixed thinking budget (gemini-2.5 and earlier):

```
profile = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=4096)
```

Configuring a profile with thinking level (gemini-3+):

```
profile = GoogleDeveloperLanguageModel.Profile(thinking_level="high")
```

## GoogleVertexEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleVertexEmbeddingModel[GoogleVertexEmbeddingModel]

              click fenic.GoogleVertexEmbeddingModel href "" "fenic.GoogleVertexEmbeddingModel"
```

Configuration for Google Vertex AI embedding models.

This class defines the configuration settings for Google embedding models available in Google Vertex AI,
including model selection and rate limiting parameters. These models are accessible using Google Cloud credentials.

Attributes:

- **`model_name`**
  (`GoogleVertexEmbeddingModelName`)
  –

  The name of the Google Vertex embedding model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Vertex embedding model with rate limits:

```
embedding_model = GoogleVertexEmbeddingModel(
    model_name="gemini-embedding-001", rpm=100, tpm=1000
)
```

Configuring a Google Vertex embedding model with profiles:

```
embedding_model = GoogleVertexEmbeddingModel(
    model_name="gemini-embedding-001",
    rpm=100,
    tpm=1000,
    profiles={
        "default": GoogleVertexEmbeddingModel.Profile(),
        "high_dim": GoogleVertexEmbeddingModel.Profile(
            output_dimensionality=3072
        ),
    },
    default_profile="default",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Vertex embedding models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleVertexEmbeddingModel.Profile[Profile]

              click fenic.GoogleVertexEmbeddingModel.Profile href "" "fenic.GoogleVertexEmbeddingModel.Profile"
```

Profile configurations for Google Vertex embedding models.

This class defines profile configurations for Google embedding models, allowing
different output dimensionality and task type settings to be applied to the same model.

Attributes:

- **`output_dimensionality`**
  (`Optional[int]`)
  –

  The dimensionality of the embedding created by this model.
  If not provided, the model will use its default dimensionality.
- **`task_type`**
  (`GoogleEmbeddingTaskType`)
  –

  The type of task for the embedding model.

Example

Configuring a profile with custom dimensionality:

```
profile = GoogleVertexEmbeddingModelConfig.Profile(
    output_dimensionality=3072
)
```

Configuring a profile with default settings:

```
profile = GoogleVertexEmbeddingModelConfig.Profile()
```

## GoogleVertexLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleVertexLanguageModel[GoogleVertexLanguageModel]

              click fenic.GoogleVertexLanguageModel href "" "fenic.GoogleVertexLanguageModel"
```

Configuration for Google Vertex AI models.

This class defines the configuration settings for Google Gemini models available in Google Vertex AI,
including model selection and rate limiting parameters. These models are accessible using Google Cloud credentials.

Attributes:

- **`model_name`**
  (`GoogleVertexLanguageModelName`)
  –

  The name of the Google Vertex model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Vertex model with rate limits:

```
config = GoogleVertexLanguageModel(
    model_name="gemini-2.5-flash", rpm=100, tpm=1000
)
```

Configuring a reasoning Google Vertex model with profiles:

```
config = GoogleVertexLanguageModel(
    model_name="gemini-2.5-flash",
    rpm=100,
    tpm=1000,
    profiles={
        "thinking_disabled": GoogleVertexLanguageModel.Profile(),
        "fast": GoogleVertexLanguageModel.Profile(thinking_token_budget=1024),
        "thorough": GoogleVertexLanguageModel.Profile(
            thinking_token_budget=8192
        ),
    },
    default_profile="fast",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Vertex models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.GoogleVertexLanguageModel.Profile[Profile]

              click fenic.GoogleVertexLanguageModel.Profile href "" "fenic.GoogleVertexLanguageModel.Profile"
```

Profile configurations for Google Vertex models.

This class defines profile configurations for Google Gemini models, allowing
different thinking/reasoning settings to be applied to the same underlying model.

Attributes:

- **`thinking_token_budget`**
  (`Optional[int]`)
  –

  If configuring a reasoning model, provide a thinking budget in tokens.
  If not provided, or if set to 0, thinking will be disabled for the profile (not supported on gemini-2.5-pro).
  To have the model automatically determine a thinking budget based on the complexity of
  the prompt, set this to -1. Note that Gemini models take this as a suggestion -- and not a hard limit.
  It is very possible for the model to generate far more thinking tokens than the suggested budget, and for the
  model to generate reasoning tokens even if thinking is disabled.
  Note: For gemini-3 models, use thinking_level instead.
- **`thinking_level`**
  (`Optional[ThinkingLevelType]`)
  –

  For gemini-3+ models, set the thinking level to high, medium, low, or minimal.
  This parameter is mutually exclusive with thinking_token_budget.
- **`media_resolution`**
  (`Optional[MediaResolutionType]`)
  –

  For gemini-3+ models, set the media resolution for PDF processing.
  Can be "low", "medium", or "high". Affects token cost per page.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Example

Configuring a profile with a fixed thinking budget (gemini-2.5 and earlier):

```
profile = GoogleVertexLanguageModel.Profile(thinking_token_budget=4096)
```

Configuring a profile with thinking level (gemini-3+):

```
profile = GoogleVertexLanguageModel.Profile(thinking_level="high")
```

## GroupedData

```
GroupedData(df: DataFrame, by: Optional[List[ColumnOrName]] = None)
```

Bases: `BaseGroupedData`

```
              flowchart TD
              fenic.GroupedData[GroupedData]
              fenic.api.dataframe._base_grouped_data.BaseGroupedData[BaseGroupedData]

                              fenic.api.dataframe._base_grouped_data.BaseGroupedData --> fenic.GroupedData

              click fenic.GroupedData href "" "fenic.GroupedData"
              click fenic.api.dataframe._base_grouped_data.BaseGroupedData href "" "fenic.api.dataframe._base_grouped_data.BaseGroupedData"
```

Methods for aggregations on a grouped DataFrame.

Initialize grouped data.

Parameters:

- **`df`**
  (`DataFrame`)
  –

  The DataFrame to group.
- **`by`**
  (`Optional[List[ColumnOrName]]`, default:
  `None`
  )
  –

  Optional list of columns to group by.

Methods:

- **`agg`**
  –

  Compute aggregations on grouped data and return the result as a DataFrame.

Source code in `src/fenic/api/dataframe/grouped_data.py`

```
def __init__(self, df: DataFrame, by: Optional[List[ColumnOrName]] = None):
    """Initialize grouped data.

    Args:
        df: The DataFrame to group.
        by: Optional list of columns to group by.
    """
    super().__init__(df)
    self._by: List[Column] = []
    for c in by or []:
        if isinstance(c, str):
            self._by.append(col(c))
        elif isinstance(c, Column):
            # Allow any expression except literals
            if isinstance(c._logical_expr, LiteralExpr):
                raise ValueError(f"Cannot group by literal value: {c}")
            self._by.append(c)
        else:
            raise TypeError(
                f"Group by expressions must be string or Column, got {type(c)}"
            )
    self._by_exprs = [c._logical_expr for c in self._by]
```

### agg

```
agg(*exprs: Union[Column, Dict[str, str]]) -> DataFrame
```

Compute aggregations on grouped data and return the result as a DataFrame.

This method applies aggregate functions to the grouped data.

Parameters:

- **`*exprs`**
  (`Union[Column, Dict[str, str]]`, default:
  `()`
  )
  –

  Aggregation expressions. Can be:

  - Column expressions with aggregate functions (e.g., `count("*")`, `sum("amount")`)
  - A dictionary mapping column names to aggregate function names (e.g., `{"amount": "sum", "age": "avg"}`)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with one row per group and columns for group keys and aggregated values

Raises:

- `ValueError`
  –

  If arguments are not Column expressions or a dictionary
- `ValueError`
  –

  If dictionary values are not valid aggregate function names

Count employees by department

```
# Group by department and count employees
df.group_by("department").agg(count("*").alias("employee_count"))
```

Multiple aggregations

```
# Multiple aggregations
df.group_by("department").agg(
    count("*").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    max("age").alias("max_age")
)
```

Dictionary style aggregations

```
# Dictionary style for simple aggregations
df.group_by("department", "location").agg({"salary": "avg", "age": "max"})
```

Source code in `src/fenic/api/dataframe/grouped_data.py`

```
def agg(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
    """Compute aggregations on grouped data and return the result as a DataFrame.

    This method applies aggregate functions to the grouped data.

    Args:
        *exprs: Aggregation expressions. Can be:

            - Column expressions with aggregate functions (e.g., `count("*")`, `sum("amount")`)
            - A dictionary mapping column names to aggregate function names (e.g., `{"amount": "sum", "age": "avg"}`)

    Returns:
        DataFrame: A new DataFrame with one row per group and columns for group keys and aggregated values

    Raises:
        ValueError: If arguments are not Column expressions or a dictionary
        ValueError: If dictionary values are not valid aggregate function names

    Example: Count employees by department
        ```python
        # Group by department and count employees
        df.group_by("department").agg(count("*").alias("employee_count"))
        ```

    Example: Multiple aggregations
        ```python
        # Multiple aggregations
        df.group_by("department").agg(
            count("*").alias("employee_count"),
            avg("salary").alias("avg_salary"),
            max("age").alias("max_age")
        )
        ```

    Example: Dictionary style aggregations
        ```python
        # Dictionary style for simple aggregations
        df.group_by("department", "location").agg({"salary": "avg", "age": "max"})
        ```
    """
    self._validate_agg_exprs(*exprs)
    if len(exprs) == 1 and isinstance(exprs[0], dict):
        agg_dict = exprs[0]
        return self.agg(*self._process_agg_dict(agg_dict))

    agg_exprs = self._process_agg_exprs(exprs)
    return self._df._from_logical_plan(
        Aggregate.from_session_state(self._df._logical_plan, self._by_exprs, agg_exprs, self._df._session_state),
        self._df._session_state,
    )
```

## InvalidExampleCollectionError

Bases: `ValidationError`

```
              flowchart TD
              fenic.InvalidExampleCollectionError[InvalidExampleCollectionError]
              fenic.core.error.ValidationError[ValidationError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.ValidationError --> fenic.InvalidExampleCollectionError
                                fenic.core.error.FenicError --> fenic.core.error.ValidationError

              click fenic.InvalidExampleCollectionError href "" "fenic.InvalidExampleCollectionError"
              click fenic.core.error.ValidationError href "" "fenic.core.error.ValidationError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Exception raised when a semantic example collection is invalid.

## JoinExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.JoinExample[JoinExample]

              click fenic.JoinExample href "" "fenic.JoinExample"
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
              fenic.JoinExampleCollection[JoinExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.JoinExampleCollection

              click fenic.JoinExampleCollection href "" "fenic.JoinExampleCollection"
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
              fenic.KeyPoints[KeyPoints]

              click fenic.KeyPoints href "" "fenic.KeyPoints"
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

## LLMResponseCacheConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.LLMResponseCacheConfig[LLMResponseCacheConfig]

              click fenic.LLMResponseCacheConfig href "" "fenic.LLMResponseCacheConfig"
```

Configuration for LLM response caching.

LLM response caching stores the results of language model API calls to reduce
costs and improve performance for repeated queries. This is distinct from
DataFrame caching (the `.cache()` operator).

Attributes:

- **`enabled`**
  –

  Whether caching is enabled (default: True).
- **`backend`**
  (`CacheBackend`)
  –

  Cache backend to use (default: LOCAL).
- **`ttl`**
  (`str`)
  –

  Time-to-live duration string (default: "1h").
  Format:  where unit is s/m/h/d.
  Examples: "30s", "15m", "2h", "7d".
  Maximum: 30 days, Minimum: 1 second.
- **`max_size_mb`**
  (`int`)
  –

  Maximum cache size in MB before LRU eviction (default: 128MB).
- **`namespace`**
  (`str`)
  –

  Cache namespace for isolation (default: "default").

Example

Basic configuration within SemanticConfig:

```
config = SessionConfig(
    app_name="my_app",
    semantic=SemanticConfig(
        language_models={
            "gpt": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=1000)
        },
        llm_response_cache=LLMResponseCacheConfig(
            enabled=True,
            ttl="1h",
            max_size_mb=1000,
        )
    )
)
```

Custom TTL and larger cache:

```
llm_response_cache=LLMResponseCacheConfig(
    enabled=True,
    ttl="7d",  # 7 days
    max_size_mb=5000,
)
```

Disabled caching:

```
llm_response_cache=LLMResponseCacheConfig(enabled=False)
```

Methods:

- **`ttl_seconds`**
  –

  Convert TTL string to seconds.
- **`validate_ttl`**
  –

  Validate TTL duration string format.

### ttl_seconds

```
ttl_seconds() -> int
```

Convert TTL string to seconds.

Returns:

- `int`
  –

  TTL duration in seconds.

Raises:

- `ValueError`
  –

  If TTL format is invalid.

Source code in `src/fenic/api/session/config.py`

```
def ttl_seconds(self) -> int:
    """Convert TTL string to seconds.

    Returns:
        TTL duration in seconds.

    Raises:
        ValueError: If TTL format is invalid.
    """
    pattern = r"^(\d+)([smhd])$"
    match = re.match(pattern, self.ttl.lower())

    if not match:
        raise ValueError(f"Invalid TTL format: '{self.ttl}'")

    value, unit = match.groups()
    value = int(value)

    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]
```

### validate_ttl

```
validate_ttl(v: str) -> str
```

Validate TTL duration string format.

Format:  where unit is s/m/h/d
Examples: "30s", "15m", "2h", "7d"

Parameters:

- **`v`**
  (`str`)
  –

  TTL duration string to validate.

Returns:

- `str`
  –

  The validated TTL string.

Raises:

- `ValueError`
  –

  If format is invalid or value is out of range.

Source code in `src/fenic/api/session/config.py`

```
@field_validator("ttl")
@classmethod
def validate_ttl(cls, v: str) -> str:
    """Validate TTL duration string format.

    Format: <number><unit> where unit is s/m/h/d
    Examples: "30s", "15m", "2h", "7d"

    Args:
        v: TTL duration string to validate.

    Returns:
        The validated TTL string.

    Raises:
        ValueError: If format is invalid or value is out of range.
    """
    pattern = r"^(\d+)([smhd])$"
    match = re.match(pattern, v.lower())

    if not match:
        raise ValueError(
            f"Invalid TTL format: '{v}'. "
            "Expected: <number><unit> where unit is s/m/h/d. "
            "Examples: '30m', '2h', '1d'"
        )

    value, unit = match.groups()
    value = int(value)

    # Validate ranges
    if unit == "s" and value < 1:
        raise ValueError("TTL must be at least 1 second")
    if unit == "h" and value > 720:  # 30 days
        raise ValueError("TTL cannot exceed 720 hours (30 days)")
    if unit == "d" and value > 30:
        raise ValueError("TTL cannot exceed 30 days")

    return v
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

## Lineage

```
Lineage(lineage: BaseLineage)
```

Query interface for tracing data lineage through a query plan.

This class allows you to navigate through the query plan both forwards and backwards,
tracing how specific rows are transformed through each operation.

Example

```
# Create a lineage query starting from the root
query = LineageQuery(lineage, session.execution)

# Or start from a specific source
query.start_from_source("my_table")

# Trace rows backwards through a transformation
result = query.backward(["uuid1", "uuid2"])

# Trace rows forward to see their outputs
result = query.forward(["uuid3", "uuid4"])
```

Initialize a Lineage instance.

Parameters:

- **`lineage`**
  (`BaseLineage`)
  –

  The underlying lineage implementation.

Methods:

- **`backwards`**
  –

  Trace rows backwards to see which input rows produced them.
- **`forwards`**
  –

  Trace rows forward to see how they are transformed by the next operation.
- **`get_result_df`**
  –

  Get the result of the query as a Polars DataFrame.
- **`get_source_df`**
  –

  Get a query source by name as a Polars DataFrame.
- **`get_source_names`**
  –

  Get the names of all sources in the query plan. Used to determine where to start the lineage traversal.
- **`show`**
  –

  Print the operator tree of the query.
- **`skip_backwards`**
  –

  [Not Implemented] Trace rows backwards through multiple operations at once.
- **`skip_forwards`**
  –

  [Not Implemented] Trace rows forward through multiple operations at once.
- **`start_from_source`**
  –

  Set the current position to a specific source in the query plan.

Source code in `src/fenic/api/lineage.py`

```
def __init__(self, lineage: BaseLineage):
    """Initialize a Lineage instance.

    Args:
        lineage: The underlying lineage implementation.
    """
    self.lineage = lineage
```

### backwards

```
backwards(ids: List[str], branch_side: Optional[BranchSide] = None) -> pl.DataFrame
```

Trace rows backwards to see which input rows produced them.

Parameters:

- **`ids`**
  (`List[str]`)
  –

  List of UUIDs identifying the rows to trace back
- **`branch_side`**
  (`Optional[BranchSide]`, default:
  `None`
  )
  –

  For operators with multiple inputs (like joins), specify which
  input to trace ("left" or "right"). Not needed for single-input operations.

Returns:

- `DataFrame`
  –

  DataFrame containing the source rows that produced the specified outputs

Raises:

- `ValueError`
  –

  If invalid ids format or incorrect branch_side specification

Example

```
# Simple backward trace
source_rows = query.backward(["result_uuid1"])

# Trace back through a join
left_rows = query.backward(["join_uuid1"], branch_side="left")
right_rows = query.backward(["join_uuid1"], branch_side="right")
```

Source code in `src/fenic/api/lineage.py`

```
@validate_call(config=ConfigDict(strict=True))
def backwards(
    self, ids: List[str], branch_side: Optional[BranchSide] = None
) -> pl.DataFrame:
    """Trace rows backwards to see which input rows produced them.

    Args:
        ids: List of UUIDs identifying the rows to trace back
        branch_side: For operators with multiple inputs (like joins), specify which
            input to trace ("left" or "right"). Not needed for single-input operations.

    Returns:
        DataFrame containing the source rows that produced the specified outputs

    Raises:
        ValueError: If invalid ids format or incorrect branch_side specification

    Example:
        ```python
        # Simple backward trace
        source_rows = query.backward(["result_uuid1"])

        # Trace back through a join
        left_rows = query.backward(["join_uuid1"], branch_side="left")
        right_rows = query.backward(["join_uuid1"], branch_side="right")
        ```
    """
    return self.lineage.backwards(ids, branch_side)
```

### forwards

```
forwards(row_ids: List[str]) -> pl.DataFrame
```

Trace rows forward to see how they are transformed by the next operation.

Parameters:

- **`row_ids`**
  (`List[str]`)
  –

  List of UUIDs identifying the rows to trace

Returns:

- `DataFrame`
  –

  DataFrame containing the transformed rows in the next operation

Raises:

- `ValueError`
  –

  If at root node or if row_ids format is invalid

Example

```
# Trace how specific customer rows are transformed
transformed = query.forward(["customer_uuid1", "customer_uuid2"])
```

Source code in `src/fenic/api/lineage.py`

```
@validate_call(config=ConfigDict(strict=True))
def forwards(self, row_ids: List[str]) -> pl.DataFrame:
    """Trace rows forward to see how they are transformed by the next operation.

    Args:
        row_ids: List of UUIDs identifying the rows to trace

    Returns:
        DataFrame containing the transformed rows in the next operation

    Raises:
        ValueError: If at root node or if row_ids format is invalid

    Example:
        ```python
        # Trace how specific customer rows are transformed
        transformed = query.forward(["customer_uuid1", "customer_uuid2"])
        ```
    """
    return self.lineage.forwards(row_ids)
```

### get_result_df

```
get_result_df() -> pl.DataFrame
```

Get the result of the query as a Polars DataFrame.

Source code in `src/fenic/api/lineage.py`

```
def get_result_df(self) -> pl.DataFrame:
    """Get the result of the query as a Polars DataFrame."""
    return self.lineage.get_result_df()
```

### get_source_df

```
get_source_df(source_name: str) -> pl.DataFrame
```

Get a query source by name as a Polars DataFrame.

Source code in `src/fenic/api/lineage.py`

```
@validate_call(config=ConfigDict(strict=True))
def get_source_df(self, source_name: str) -> pl.DataFrame:
    """Get a query source by name as a Polars DataFrame."""
    return self.lineage.get_source_df(source_name)
```

### get_source_names

```
get_source_names() -> List[str]
```

Get the names of all sources in the query plan. Used to determine where to start the lineage traversal.

Source code in `src/fenic/api/lineage.py`

```
@validate_call(config=ConfigDict(strict=True))
def get_source_names(self) -> List[str]:
    """Get the names of all sources in the query plan. Used to determine where to start the lineage traversal."""
    return self.lineage.get_source_names()
```

### show

```
show() -> None
```

Print the operator tree of the query.

Source code in `src/fenic/api/lineage.py`

```
def show(self) -> None:
    """Print the operator tree of the query."""
    print(self.lineage.stringify_graph())
```

### skip_backwards

```
skip_backwards(ids: List[str]) -> Dict[str, pl.DataFrame]
```

[Not Implemented] Trace rows backwards through multiple operations at once.

This method will allow efficient tracing through multiple operations without
intermediate results.

Parameters:

- **`ids`**
  (`List[str]`)
  –

  List of UUIDs identifying the rows to trace back

Returns:

- `Dict[str, DataFrame]`
  –

  Dictionary mapping operation names to their source DataFrames

Raises:

- `NotImplementedError`
  –

  This method is not yet implemented

Source code in `src/fenic/api/lineage.py`

```
def skip_backwards(self, ids: List[str]) -> Dict[str, pl.DataFrame]:
    """[Not Implemented] Trace rows backwards through multiple operations at once.

    This method will allow efficient tracing through multiple operations without
    intermediate results.

    Args:
        ids: List of UUIDs identifying the rows to trace back

    Returns:
        Dictionary mapping operation names to their source DataFrames

    Raises:
        NotImplementedError: This method is not yet implemented
    """
    raise NotImplementedError("Skip backwards not yet implemented")
```

### skip_forwards

```
skip_forwards(row_ids: List[str]) -> pl.DataFrame
```

[Not Implemented] Trace rows forward through multiple operations at once.

This method will allow efficient tracing through multiple operations without
intermediate results.

Parameters:

- **`row_ids`**
  (`List[str]`)
  –

  List of UUIDs identifying the rows to trace

Returns:

- `DataFrame`
  –

  DataFrame containing the final transformed rows

Raises:

- `NotImplementedError`
  –

  This method is not yet implemented

Source code in `src/fenic/api/lineage.py`

```
def skip_forwards(self, row_ids: List[str]) -> pl.DataFrame:
    """[Not Implemented] Trace rows forward through multiple operations at once.

    This method will allow efficient tracing through multiple operations without
    intermediate results.

    Args:
        row_ids: List of UUIDs identifying the rows to trace

    Returns:
        DataFrame containing the final transformed rows

    Raises:
        NotImplementedError: This method is not yet implemented
    """
    raise NotImplementedError("Skip forwards not yet implemented")
```

### start_from_source

```
start_from_source(source_name: str) -> None
```

Set the current position to a specific source in the query plan.

Parameters:

- **`source_name`**
  (`str`)
  –

  Name of the source table to start from

Example

```
query.start_from_source("customers")
# Now you can trace forward from the customers table
```

Source code in `src/fenic/api/lineage.py`

```
@validate_call(config=ConfigDict(strict=True))
def start_from_source(self, source_name: str) -> None:
    """Set the current position to a specific source in the query plan.

    Args:
        source_name: Name of the source table to start from

    Example:
        ```python
        query.start_from_source("customers")
        # Now you can trace forward from the customers table
        ```
    """
    self.lineage.start_from_source(source_name)
```

## MapExample

Bases: `BaseModel`

```
              flowchart TD
              fenic.MapExample[MapExample]

              click fenic.MapExample href "" "fenic.MapExample"
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
              fenic.MapExampleCollection[MapExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.MapExampleCollection

              click fenic.MapExampleCollection href "" "fenic.MapExampleCollection"
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

## ModelAlias

Bases: `BaseModel`

```
              flowchart TD
              fenic.ModelAlias[ModelAlias]

              click fenic.ModelAlias href "" "fenic.ModelAlias"
```

A combination of a model name and a required profile for that model.

Model aliases are used to select a specific model to use in a semantic operation.
Both the model name and profile must be specified when creating a ModelAlias.

Attributes:

- **`name`**
  (`str`)
  –

  The name of the model.
- **`profile`**
  (`str`)
  –

  The name of a profile configuration to use for the model.

Example

```
model_alias = ModelAlias(name="o4-mini", profile="low")
```

## OpenAIEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.OpenAIEmbeddingModel[OpenAIEmbeddingModel]

              click fenic.OpenAIEmbeddingModel href "" "fenic.OpenAIEmbeddingModel"
```

Configuration for OpenAI embedding models.

This class defines the configuration settings for OpenAI embedding models,
including model selection and rate limiting parameters.

Attributes:

- **`model_name`**
  (`OpenAIEmbeddingModelName`)
  –

  The name of the OpenAI embedding model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.

Example

Configuring an OpenAI embedding model with rate limits:

```
config = OpenAIEmbeddingModel(
    model_name="text-embedding-3-small", rpm=100, tpm=100
)
```

## OpenAILanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.OpenAILanguageModel[OpenAILanguageModel]

              click fenic.OpenAILanguageModel href "" "fenic.OpenAILanguageModel"
```

Configuration for OpenAI language models.

This class defines the configuration settings for OpenAI language models,
including model selection and rate limiting parameters.

Attributes:

- **`model_name`**
  (`OpenAILanguageModelName`)
  –

  The name of the OpenAI model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Note

When using an o-series or gpt5 reasoning model without specifying a reasoning effort in
a Profile, the `reasoning_effort` will default to `low` (for o-series models) or `minimal`
(for gpt5 models).

Example

Configuring an OpenAI language model with rate limits:

```
config = OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100)
```

Configuring an OpenAI model with profiles:

```
config = OpenAILanguageModel(
    model_name="o4-mini",
    rpm=100,
    tpm=100,
    profiles={
        "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
        "thorough": OpenAILanguageModel.Profile(reasoning_effort="high"),
    },
    default_profile="fast",
)
```

Using a profile in a semantic operation:

```
config = SemanticConfig(
    language_models={
        "o4": OpenAILanguageModel(
            model_name="o4-mini",
            rpm=1_000,
            tpm=1_000_000,
            profiles={
                "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                "thorough": OpenAILanguageModel.Profile(
                    reasoning_effort="high"
                ),
            },
            default_profile="fast",
        )
    },
    default_language_model="o4",
)

# Will use the default "fast" profile for the "o4" model
semantic.map(
    instruction="Construct a formal proof of the {hypothesis}.",
    model_alias="o4",
)

# Will use the "thorough" profile for the "o4" model
semantic.map(
    instruction="Construct a formal proof of the {hypothesis}.",
    model_alias=ModelAlias(name="o4", profile="thorough"),
)
```

Classes:

- **`Profile`**
  –

  OpenAI-specific profile configurations.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.OpenAILanguageModel.Profile[Profile]

              click fenic.OpenAILanguageModel.Profile href "" "fenic.OpenAILanguageModel.Profile"
```

OpenAI-specific profile configurations.

This class defines profile configurations for OpenAI models, allowing a user to reference
the same underlying model in semantic operations with different settings.

Attributes:

- **`reasoning_effort`**
  (`Optional[ReasoningEffort]`)
  –

  Provide a reasoning effort. Only for gpt5 and o-series models.
  Valid values: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'.
  - For gpt-5.6 models: supports 'xhigh' and 'max'
  - For gpt-5.5 models: defaults to 'medium', supports 'xhigh'
  - For gpt-5.4 models: defaults to 'none' (disabled reasoning), supports 'xhigh'
  - For gpt-5.1 and gpt-5.2 models: defaults to 'none' (disabled reasoning), does NOT support 'minimal' or 'xhigh'
  - For gpt-5 models: defaults to 'minimal', does NOT support 'none'
  - For o-series models: defaults to 'low', does NOT support 'none' or 'minimal'
- **`verbosity`**
  (`Optional[Verbosity]`)
  –

  Provide a verbosity level. Only for gpt5/gpt5.1 models.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Note

When using an o-series or gpt5 reasoning model with reasoning enabled, the `temperature` cannot be customized.
For gpt-5.1 models with reasoning_effort='none', temperature CAN be customized.

Example

Configuring a profile with medium reasoning effort:

```
profile = OpenAILanguageModel.Profile(reasoning_effort="medium")
```

## OpenRouterLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.OpenRouterLanguageModel[OpenRouterLanguageModel]

              click fenic.OpenRouterLanguageModel href "" "fenic.OpenRouterLanguageModel"
```

Configuration for OpenRouter language models.

This class defines the configuration settings for OpenRouter language models,
including model selection and rate limiting parameters. When fetching available models from OpenRouter, results
will be filtered to only include models from providers that are not in the user’s ignored providers list and are either
in the user’s allowed providers list (if configured) or from any provider (if no allowed providers are specified).

Attributes:

- **`model_name`**
  (`str`)
  –

  `{family}/{model}` identifier (e.g., `anthropic/claude-3-5-sonnet`).
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The key in `profiles` to select by default.
- **`structured_output_strategy`**
  (`Optional[StructuredOutputStrategy]`)
  –

  The strategy to use for structured output if a model supports both tool calling and structured outputs.

  - `prefer_tools`: prefer using tools over response format.
  - `prefer_response_format`: prefer using response format over tools.

Requirements

- Set `OPENROUTER_API_KEY` in your environment.

Example:

```
OpenRouterLanguageModel(
    model_name="openai/gpt-oss-20b",
    profiles={
        "default": OpenRouterLanguageModel.Profile(
            provider=OpenRouterLanguageModel.Provider(
                sort="price"  # Routes to the cheapest available provider
            )
        )
    },
)
```

Example:

```
OpenRouterLanguageModel(
    model_name="anthropic/claude-sonnet-4",
    profiles={
        "default": OpenRouterLanguageModel.Profile(
            provider=OpenRouterLanguageModel.Provider(
                only=[
                    "Anthropic"
                ]  # ensures the request will only be routed to Anthropic and not AWS Bedrock or Google Vertex
            )
        )
    },
)
```

Example:

```
OpenRouterLanguageModel(
    model_name="qwen/qwen3-next-80b-a3b-instruct",
    profiles={
        "default": OpenRouterLanguageModel.Profile(
            provider=OpenRouterLanguageModel.Provider(
                sort="throughput", # routes to the provider with the highest overall throughput
                data_collection="deny" # eliminates providers that retain prompt data (would only route to DeepInfra/AtlasCloud, in this example)
                # Eliminate providers that offer an fp8 quantized version of the model, only allowing bf16.
                # Note that many providers have an `unknown` quantization, so you may be excluding more providers than you expect.
                quantizations=["bf16"]
            )
        )
    }
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for OpenRouter language models.
- **`Provider`**
  –

  Provider routing configuration for OpenRouter language models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.OpenRouterLanguageModel.Profile[Profile]

              click fenic.OpenRouterLanguageModel.Profile href "" "fenic.OpenRouterLanguageModel.Profile"
```

Profile configurations for OpenRouter language models.

Attributes:

- **`models`**
  (`Optional[list[str]]`)
  –

  A list of fallback models to use if the primary model is unavailable.
  ([OpenRouter Documentation](https://openrouter.ai/docs/features/model-routing#the-models-parameter)).
- **`provider`**
  (`Optional[Provider]`)
  –

  Provider routing preferences (include/exclude specific providers, set provider ranking method preference)
  ([OpenRouter Documentation](https://openrouter.ai/docs/features/provider-routing)).
- **`reasoning_effort`**
  (`Optional[OpenRouterReasoningEffort]`)
  –

  OpenRouter reasoning effort configuration (none, minimal, low, medium, high, xhigh, max).
  If the model does support reasoning, but not `reasoning_effort`, a `reasoning_max_tokens` will be calculated
  that is roughly equivalent as a percentage of the model's maximum output size
  ([OpenRouter Documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level))
- **`reasoning_max_tokens`**
  (`Optional[int]`)
  –

  Supported by Anthropic, Gemini, etc., sets a token budget for reasoning
  If the model does support reasoning, but not `reasoning_max_tokens`, a `reasoning_effort_ will be automatically
  calculated based on`reasoning_max_tokens` as a percentage of the model's maximum output size
  ([OpenRouter Documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning))
- **`parsing_engine`**
  (`Optional[ParsingEngine]`)
  –

  The parsing engine to use for processing PDF files. By default, the model's native parsing engine will be used. If the model doesn't support PDF processing and the parsing engine is not provided, an error will be raised. Note: 'mistral-ocr' incurs additional costs.
  ([OpenRouter Documentation](https://openrouter.ai/docs/features/multimodal/pdfs))

### Provider

Bases: `BaseModel`

```
              flowchart TD
              fenic.OpenRouterLanguageModel.Provider[Provider]

              click fenic.OpenRouterLanguageModel.Provider href "" "fenic.OpenRouterLanguageModel.Provider"
```

Provider routing configuration for OpenRouter language models.

[Provider Routing Documentation](https://openrouter.ai/docs/features/provider-routing)

Attributes:

- **`order`**
  (`Optional[list[str]]`)
  –

  List of providers to try in order (e.g. ['Anthropic', 'Amazon Bedrock']).
- **`sort`**
  (`Optional[ProviderSort]`)
  –

  Provider routing preference (e.g. 'price', 'throughput', 'latency').
  "price" will route to the cheapest available provider first, progressing through the list of providers in order of price.
  "throughput" will route to the provider with the highest overall recent throughput, progressing through the list of providers in order of throughput.
  "latency" will route to the provider with the lowest overall recent latency, progressing through the list of providers in order of latency.
- **`quantizations`**
  (`Optional[list[ModelQuantization]]`)
  –

  Allowed quantizations. Note: many providers report `unknown`.
- **`data_collection`**
  (`Optional[DataCollection]`)
  –

  Data collection preference. `allow`: allows the use of providers which store prompt data
  non-transiently and may train on it. `deny`: use only providers which do not collect/store prompt data.
- **`only`**
  (`Optional[list[str]]`)
  –

  Only include these providers when performing provider routing.
- **`exclude`**
  (`Optional[list[str]]`)
  –

  Exclude these providers when performing provider routing.
- **`max_prompt_price`**
  (`Optional[float]`)
  –

  Maximum prompt price ($USD per 1M tokens).
- **`max_completion_price`**
  (`Optional[float]`)
  –

  Maximum completion price ($USD per 1M tokens).

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
              fenic.Paragraph[Paragraph]

              click fenic.Paragraph href "" "fenic.Paragraph"
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
              fenic.PredicateExample[PredicateExample]

              click fenic.PredicateExample href "" "fenic.PredicateExample"
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
              fenic.PredicateExampleCollection[PredicateExampleCollection]
              fenic.core.types.semantic_examples.BaseExampleCollection[BaseExampleCollection]

                              fenic.core.types.semantic_examples.BaseExampleCollection --> fenic.PredicateExampleCollection

              click fenic.PredicateExampleCollection href "" "fenic.PredicateExampleCollection"
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

## SemanticConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.SemanticConfig[SemanticConfig]

              click fenic.SemanticConfig href "" "fenic.SemanticConfig"
```

Configuration for semantic language and embedding models.

This class defines the configuration for both language models and optional
embedding models used in semantic operations. It ensures that all configured
models are valid and supported by their respective providers.

Attributes:

- **`language_models`**
  (`Optional[dict[str, LanguageModel]]`)
  –

  Mapping of model aliases to language model configurations.
- **`default_language_model`**
  (`Optional[str]`)
  –

  The alias of the default language model to use for semantic operations. Not required
  if only one language model is configured.
- **`embedding_models`**
  (`Optional[dict[str, EmbeddingModel]]`)
  –

  Optional mapping of model aliases to embedding model configurations.
- **`default_embedding_model`**
  (`Optional[str]`)
  –

  The alias of the default embedding model to use for semantic operations.

Note

The embedding model is optional and only required for operations that
need semantic search or embedding capabilities.

Example

Configuring semantic models with a single language model:

```
config = SemanticConfig(
    language_models={
        "gpt4": OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100)
    }
)
```

Configuring semantic models with multiple language models and an embedding model:

```
config = SemanticConfig(
    language_models={
        "gpt4": OpenAILanguageModel(
            model_name="gpt-4.1-nano", rpm=100, tpm=100
        ),
        "claude": AnthropicLanguageModel(
            model_name="claude-haiku-4-5",
            rpm=100,
            input_tpm=100,
            output_tpm=100,
        ),
        "gemini": GoogleDeveloperLanguageModel(
            model_name="gemini-2.5-flash", rpm=100, tpm=1000
        ),
    },
    default_language_model="gpt4",
    embedding_models={
        "openai_embeddings": OpenAIEmbeddingModel(
            model_name="text-embedding-3-small", rpm=100, tpm=100
        )
    },
    default_embedding_model="openai_embeddings",
)
```

Configuring models with profiles:

```
config = SemanticConfig(
    language_models={
        "gpt4": OpenAILanguageModel(
            model_name="gpt-4o-mini",
            rpm=100,
            tpm=100,
            profiles={
                "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                "thorough": OpenAILanguageModel.Profile(
                    reasoning_effort="high"
                ),
            },
            default_profile="fast",
        ),
        "claude": AnthropicLanguageModel(
            model_name="claude-haiku-4-5",
            rpm=100,
            input_tpm=100,
            output_tpm=100,
            profiles={
                "fast": AnthropicLanguageModel.Profile(effort="low"),
                "thorough": AnthropicLanguageModel.Profile(effort="high"),
            },
            default_profile="fast",
        ),
    },
    default_language_model="gpt4",
)
```

Methods:

- **`model_post_init`**
  –

  Post initialization hook to set defaults.
- **`validate_models`**
  –

  Validates that the selected models are supported by the system.

### model_post_init

```
model_post_init(__context) -> None
```

Post initialization hook to set defaults.

This hook runs after the model is initialized and validated.
It sets the default language and embedding models if they are not set
and there is only one model available. For Google models that support
thinking_level, it auto-creates "low" and "high" profiles if no profiles
are configured.

Source code in `src/fenic/api/session/config.py`

```
def model_post_init(self, __context) -> None:
    """Post initialization hook to set defaults.

    This hook runs after the model is initialized and validated.
    It sets the default language and embedding models if they are not set
    and there is only one model available. For Google models that support
    thinking_level, it auto-creates "low" and "high" profiles if no profiles
    are configured.
    """
    if self.language_models:
        # Set default language model if not set and only one model exists
        if self.default_language_model is None and len(self.language_models) == 1:
            self.default_language_model = list(self.language_models.keys())[0]

        # Auto-create profiles for Google models that support thinking_level
        for model_config in self.language_models.values():
            if isinstance(model_config, (GoogleDeveloperLanguageModel, GoogleVertexLanguageModel)):
                model_provider = _get_model_provider_for_model_config(model_config)
                model_params = model_catalog.get_completion_model_parameters(
                    model_provider, model_config.model_name
                )
                if model_params and model_params.supported_thinking_levels and model_config.profiles is None:
                    # Auto-create profiles for each supported thinking level
                    if isinstance(model_config, GoogleDeveloperLanguageModel):
                        model_config.profiles = {
                            level: GoogleDeveloperLanguageModel.Profile(thinking_level=level)
                            for level in model_params.supported_thinking_levels
                        }
                    else:
                        model_config.profiles = {
                            level: GoogleVertexLanguageModel.Profile(thinking_level=level)
                            for level in model_params.supported_thinking_levels
                        }
                    model_config.default_profile = "low"

        # Set default profile for each model if not set and only one profile exists
        for model_config in self.language_models.values():
            if model_config.profiles is not None:
                profile_names = list(model_config.profiles.keys())
                if model_config.default_profile is None and len(profile_names) == 1:
                    model_config.default_profile = profile_names[0]

    # Set default embedding model if not set and only one model exists
    if self.embedding_models:
        if self.default_embedding_model is None and len(self.embedding_models) == 1:
            self.default_embedding_model = list(self.embedding_models.keys())[0]
        # Set default profile for each model if not set and only one preset exists
        for model_config in self.embedding_models.values():
            if (
                hasattr(model_config, "profiles")
                and model_config.profiles is not None
            ):
                preset_names = list(model_config.profiles.keys())
                if model_config.default_profile is None and len(preset_names) == 1:
                    model_config.default_profile = preset_names[0]
```

### validate_models

```
validate_models() -> SemanticConfig
```

Validates that the selected models are supported by the system.

This validator checks that both the language model and embedding model (if provided)
are valid and supported by their respective providers.

Returns:

- `SemanticConfig`
  –

  The validated SemanticConfig instance.

Raises:

- `ConfigurationError`
  –

  If any of the models are not supported.

Source code in `src/fenic/api/session/config.py`

```
@model_validator(mode="after")
def validate_models(self) -> SemanticConfig:
    """Validates that the selected models are supported by the system.

    This validator checks that both the language model and embedding model (if provided)
    are valid and supported by their respective providers.

    Returns:
        The validated SemanticConfig instance.

    Raises:
        ConfigurationError: If any of the models are not supported.
    """
    # Skip validation if no models configured (embedding-only or empty config)
    if not self.language_models and not self.embedding_models:
        return self

    # Validate language models if provided
    if self.language_models:
        available_language_model_aliases = list(self.language_models.keys())
        if self.default_language_model is None and len(self.language_models) > 1:
            raise ConfigurationError(
                f"default_language_model is not set, and multiple language models are configured. Please specify one of: {available_language_model_aliases} as a default_language_model."
            )

        if (
            self.default_language_model is not None
            and self.default_language_model not in self.language_models
        ):
            raise ConfigurationError(
                f"default_language_model {self.default_language_model} is not in configured map of language models. Available models: {available_language_model_aliases} ."
            )

        for model_alias, language_model in self.language_models.items():
            language_model_name = language_model.model_name
            language_model_provider = _get_model_provider_for_model_config(
                language_model
            )

            completion_model_params = model_catalog.get_completion_model_parameters(
                language_model_provider, language_model_name
            )
            if completion_model_params is None:
                raise ConfigurationError(
                    model_catalog.generate_unsupported_completion_model_error_message(
                        language_model_provider, language_model_name
                    )
                )
            if language_model.profiles is not None:
                if not completion_model_params.supports_profiles:
                    raise ConfigurationError(
                        f"Model '{model_alias}' does not support parameter profiles. Please remove the Profile configuration."
                    )
                profile_names = list(language_model.profiles.keys())
                if (
                    language_model.default_profile is None
                    and len(profile_names) > 0
                ):
                    raise ConfigurationError(
                        f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile."
                    )
                if (
                    language_model.default_profile is not None
                    and language_model.default_profile not in profile_names
                ):
                    raise ConfigurationError(
                        f"default_profile {language_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}"
                    )
                for profile_alias, profile in language_model.profiles.items():
                    _validate_language_profile(
                        language_model,
                        model_alias,
                        completion_model_params,
                        profile,
                        profile_alias,
                    )

    if self.embedding_models is not None:
        available_embedding_model_aliases = list(self.embedding_models.keys())
        if self.default_embedding_model is None and len(self.embedding_models) > 1:
            raise ConfigurationError(
                f"default_embedding_model is not set, and multiple embedding models are configured. Please specify one of: {available_embedding_model_aliases} as a default_embedding_model."
            )

        if (
            self.default_embedding_model is not None
            and self.default_embedding_model not in self.embedding_models
        ):
            raise ConfigurationError(
                f"default_embedding_model {self.default_embedding_model} is not in configured map of embedding models. Available models: {available_embedding_model_aliases} ."
            )
        for model_alias, embedding_model in self.embedding_models.items():
            embedding_model_provider = _get_model_provider_for_model_config(
                embedding_model
            )
            embedding_model_name = embedding_model.model_name
            embedding_model_parameters = (
                model_catalog.get_embedding_model_parameters(
                    embedding_model_provider, embedding_model_name
                )
            )
            if embedding_model_parameters is None:
                raise ConfigurationError(
                    model_catalog.generate_unsupported_embedding_model_error_message(
                        embedding_model_provider, embedding_model_name
                    )
                )
            if hasattr(embedding_model, "profiles") and embedding_model.profiles:
                profile_names = list(embedding_model.profiles.keys())
                if (
                    embedding_model.default_profile is None
                    and len(profile_names) > 0
                ):
                    raise ConfigurationError(
                        f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile."
                    )
                if (
                    embedding_model.default_profile is not None
                    and embedding_model.default_profile not in profile_names
                ):
                    raise ConfigurationError(
                        f"default_profile {embedding_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}"
                    )

                for profile_alias, profile in embedding_model.profiles.items():
                    _validate_embedding_profile(
                        embedding_model_parameters,
                        model_alias,
                        profile_alias,
                        profile,
                    )

    return self
```

## SemanticExtensions

```
SemanticExtensions(df: DataFrame)
```

A namespace for semantic dataframe operators.

Initialize semantic extensions.

Parameters:

- **`df`**
  (`DataFrame`)
  –

  The DataFrame to extend with semantic operations.

Methods:

- **`join`**
  –

  Performs a semantic join between two DataFrames using a natural language predicate.
- **`sim_join`**
  –

  Performs a semantic similarity join between two DataFrames using embedding expressions.
- **`with_cluster_labels`**
  –

  Cluster rows using K-means and add cluster metadata columns.

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def __init__(self, df: DataFrame):
    """Initialize semantic extensions.

    Args:
        df: The DataFrame to extend with semantic operations.
    """
    self._df = df
```

### join

```
join(other: DataFrame, predicate: str, left_on: Column, right_on: Column, strict: bool = True, examples: Optional[JoinExampleCollection] = None, model_alias: Optional[Union[str, ModelAlias]] = None, request_timeout: Optional[float] = None) -> DataFrame
```

Performs a semantic join between two DataFrames using a natural language predicate.

This method evaluates a boolean predicate for each potential row pair between the two DataFrames,
including only those pairs where the predicate evaluates to True.

The join process:
1. For each row in the left DataFrame, evaluates the predicate in the jinja template against each row in the right DataFrame
2. Includes row pairs where the predicate returns True
3. Excludes row pairs where the predicate returns False
4. Returns a new DataFrame containing all columns from both DataFrames for the matched pairs

The jinja template must use exactly two column placeholders:
- One from the left DataFrame: `{{ left_on }}`
- One from the right DataFrame: `{{ right_on }}`

Parameters:

- **`other`**
  (`DataFrame`)
  –

  The DataFrame to join with.
- **`predicate`**
  (`str`)
  –

  A Jinja2 template containing the natural language predicate.
  Must include placeholders for exactly one column from each DataFrame.
  The template is evaluated as a boolean - True includes the pair, False excludes it.
- **`left_on`**
  (`Column`)
  –

  The column from the left DataFrame (self) to use in the join predicate.
- **`right_on`**
  (`Column`)
  –

  The column from the right DataFrame (other) to use in the join predicate.
- **`strict`**
  (`bool`, default:
  `True`
  )
  –

  If True, when either the left_on or right_on column has a None value for a row pair,
  that pair is automatically excluded from the join (predicate is not evaluated).
  If False, None values are rendered according to Jinja2's null rendering behavior.
  Default is True.
- **`examples`**
  (`Optional[JoinExampleCollection]`, default:
  `None`
  )
  –

  Optional JoinExampleCollection containing labeled examples to guide the join.
  Each example should have:
  - left: Sample value from the left column
  - right: Sample value from the right column
  - output: Boolean indicating whether this pair should be joined (True) or not (False)
- **`model_alias`**
  (`Optional[Union[str, ModelAlias]]`, default:
  `None`
  )
  –

  Optional alias for the language model to use. If None, uses the default model.
- **`request_timeout`**
  (`Optional[float]`, default:
  `None`
  )
  –

  Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame containing matched row pairs with all columns from both DataFrames.

Basic semantic join

```
# Match job listings with candidate resumes based on title/skills
# Only includes pairs where the predicate evaluates to True
df_jobs.semantic.join(df_resumes,
    predicate=dedent('''                    Job Description: {{left_on}}
        Candidate Background: {{right_on}}
        The candidate is qualified for the job.'''),
    left_on=col("job_description"),
    right_on=col("work_experience"),
    examples=examples
)
```

Semantic join with examples

```
# Improve join quality with examples
examples = JoinExampleCollection()
examples.create_example(JoinExample(
    left="5 years experience building backend services in Python using asyncio, FastAPI, and PostgreSQL",
    right="Senior Software Engineer - Backend",
    output=True))  # This pair WILL be included in similar cases
examples.create_example(JoinExample(
    left="5 years experience with growth strategy, private equity due diligence, and M&A",
    right="Product Manager - Hardware",
    output=False))  # This pair will NOT be included in similar cases
df_jobs.semantic.join(
    other=df_resumes,
    predicate=dedent('''                    Job Description: {{left_on}}
        Candidate Background: {{right_on}}
        The candidate is qualified for the job.'''),
    left_on=col("job_description"),
    right_on=col("work_experience"),
    examples=examples
)
```

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def join(
    self,
    other: DataFrame,
    predicate: str,
    left_on: Column,
    right_on: Column,
    strict: bool = True,
    examples: Optional[JoinExampleCollection] = None,
    model_alias: Optional[Union[str, ModelAlias]] = None,
    request_timeout: Optional[float] = None,
) -> DataFrame:
    """Performs a semantic join between two DataFrames using a natural language predicate.

    This method evaluates a boolean predicate for each potential row pair between the two DataFrames,
    including only those pairs where the predicate evaluates to True.

    The join process:
    1. For each row in the left DataFrame, evaluates the predicate in the jinja template against each row in the right DataFrame
    2. Includes row pairs where the predicate returns True
    3. Excludes row pairs where the predicate returns False
    4. Returns a new DataFrame containing all columns from both DataFrames for the matched pairs

    The jinja template must use exactly two column placeholders:
    - One from the left DataFrame: `{{ left_on }}`
    - One from the right DataFrame: `{{ right_on }}`

    Args:
        other: The DataFrame to join with.
        predicate: A Jinja2 template containing the natural language predicate.
            Must include placeholders for exactly one column from each DataFrame.
            The template is evaluated as a boolean - True includes the pair, False excludes it.
        left_on: The column from the left DataFrame (self) to use in the join predicate.
        right_on: The column from the right DataFrame (other) to use in the join predicate.
        strict: If True, when either the left_on or right_on column has a None value for a row pair,
                that pair is automatically excluded from the join (predicate is not evaluated).
                If False, None values are rendered according to Jinja2's null rendering behavior.
                Default is True.
        examples: Optional JoinExampleCollection containing labeled examples to guide the join.
            Each example should have:
            - left: Sample value from the left column
            - right: Sample value from the right column
            - output: Boolean indicating whether this pair should be joined (True) or not (False)
        model_alias: Optional alias for the language model to use. If None, uses the default model.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        DataFrame: A new DataFrame containing matched row pairs with all columns from both DataFrames.

    Example: Basic semantic join
        ```python
        # Match job listings with candidate resumes based on title/skills
        # Only includes pairs where the predicate evaluates to True
        df_jobs.semantic.join(df_resumes,
            predicate=dedent('''\
                Job Description: {{left_on}}
                Candidate Background: {{right_on}}
                The candidate is qualified for the job.'''),
            left_on=col("job_description"),
            right_on=col("work_experience"),
            examples=examples
        )
        ```

    Example: Semantic join with examples
        ```python
        # Improve join quality with examples
        examples = JoinExampleCollection()
        examples.create_example(JoinExample(
            left="5 years experience building backend services in Python using asyncio, FastAPI, and PostgreSQL",
            right="Senior Software Engineer - Backend",
            output=True))  # This pair WILL be included in similar cases
        examples.create_example(JoinExample(
            left="5 years experience with growth strategy, private equity due diligence, and M&A",
            right="Product Manager - Hardware",
            output=False))  # This pair will NOT be included in similar cases
        df_jobs.semantic.join(
            other=df_resumes,
            predicate=dedent('''\
                Job Description: {{left_on}}
                Candidate Background: {{right_on}}
                The candidate is qualified for the job.'''),
            left_on=col("job_description"),
            right_on=col("work_experience"),
            examples=examples
        )
        ```
    """
    from fenic.api.dataframe.dataframe import DataFrame

    if not isinstance(other, DataFrame):
        raise ValidationError(f"other argument must be a DataFrame, got {type(other)}")

    if not isinstance(predicate, str):
        raise ValidationError(
            f"The `predicate` argument to `semantic.join` must be a string, got {type(predicate)}"
        )
    if not isinstance(left_on, Column):
        raise ValidationError(f"`left_on` argument must be a Column, got {type(left_on)} instead.")
    if not isinstance(right_on, Column):
        raise ValidationError(f"`right_on` argument must be a Column, got {type(right_on)} instead.")
    if examples is not None and not isinstance(examples, JoinExampleCollection):
        raise ValidationError(f"`examples` argument must be a JoinExampleCollection, got {type(examples)} instead.")
    if model_alias is not None and not isinstance(model_alias, (str, ModelAlias)):
        raise ValidationError(f"`model_alias` argument must be a string or ModelAlias, got {type(model_alias)} instead.")

    # Validate request_timeout
    request_timeout = validate_timeout(request_timeout)

    resolved_model_alias = _resolve_model_alias(model_alias)
    DataFrame._ensure_same_session(self._df._session_state, [other._session_state])

    return self._df._from_logical_plan(
        SemanticJoin.from_session_state(
            left=self._df._logical_plan,
            right=other._logical_plan,
            left_on=left_on._logical_expr,
            right_on=right_on._logical_expr,
            jinja_template=predicate,
            strict=strict,
            model_alias=resolved_model_alias,
            examples=examples,
            request_timeout=request_timeout,
            session_state=self._df._session_state,
        ),
        self._df._session_state,
    )
```

### sim_join

```
sim_join(other: DataFrame, left_on: ColumnOrName, right_on: ColumnOrName, k: int = 1, similarity_metric: SemanticSimilarityMetric = 'cosine', similarity_score_column: Optional[str] = None, request_timeout: Optional[float] = None) -> DataFrame
```

Performs a semantic similarity join between two DataFrames using embedding expressions.

For each row in the left DataFrame, returns the top `k` most semantically similar rows
from the right DataFrame based on the specified similarity metric.

Note

Local execution requires the `sim-join` extra: `pip install "fenic[sim-join]"`.

Parameters:

- **`other`**
  (`DataFrame`)
  –

  The right-hand DataFrame to join with.
- **`left_on`**
  (`ColumnOrName`)
  –

  Expression or column representing embeddings in the left DataFrame.
  If this is a named column, that column is treated as a user column and is
  included in the output. If this is an expression, it is treated as a
  temporary join key and is not included as an output column.
- **`right_on`**
  (`ColumnOrName`)
  –

  Expression or column representing embeddings in the right DataFrame.
  Named columns are included in the output; expression-derived join keys
  are temporary and are not included in the output.
- **`k`**
  (`int`, default:
  `1`
  )
  –

  Number of most similar matches to return per row.
- **`similarity_metric`**
  (`SemanticSimilarityMetric`, default:
  `'cosine'`
  )
  –

  Similarity metric to use: "l2", "cosine", or "dot".
- **`similarity_score_column`**
  (`Optional[str]`, default:
  `None`
  )
  –

  If set, adds a column with this name containing similarity scores.
  If None, the scores are omitted.
- **`request_timeout`**
  (`Optional[float]`, default:
  `None`
  )
  –

  Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

Returns:

- `DataFrame`
  –

  A DataFrame containing one row for each of the top-k matches per row in the left DataFrame.
- `DataFrame`
  –

  The result includes all columns from both input DataFrames and, when `similarity_score_column`
- `DataFrame`
  –

  is provided, a similarity score column. Join key columns that already exist in the input
- `DataFrame`
  –

  DataFrames are preserved as normal user columns. Join keys derived from expressions are
- `DataFrame`
  –

  temporary execution columns and are not included in the result schema.

Raises:

- `ValidationError`
  –

  If `k` is not positive or if the columns are invalid.
- `ValidationError`
  –

  If `similarity_metric` is not one of "l2", "cosine", "dot"

Match queries to FAQ entries

```
# Match customer queries to FAQ entries
df_queries.semantic.sim_join(
    df_faqs,
    left_on=embeddings(col("query_text")),
    right_on=embeddings(col("faq_question")),
    k=1
)
```

Link headlines to articles

```
# Link news headlines to full articles
df_headlines.semantic.sim_join(
    df_articles,
    left_on=embeddings(col("headline")),
    right_on=embeddings(col("content")),
    k=3,
    return_similarity_scores=True
)
```

Find similar job postings

```
# Find similar job postings across two sources
df_linkedin.semantic.sim_join(
    df_indeed,
    left_on=embeddings(col("job_title")),
    right_on=embeddings(col("job_description")),
    k=2
)
```

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def sim_join(
    self,
    other: DataFrame,
    left_on: ColumnOrName,
    right_on: ColumnOrName,
    k: int = 1,
    similarity_metric: SemanticSimilarityMetric = "cosine",
    similarity_score_column: Optional[str] = None,
    request_timeout: Optional[float] = None,
) -> DataFrame:
    """Performs a semantic similarity join between two DataFrames using embedding expressions.

    For each row in the left DataFrame, returns the top `k` most semantically similar rows
    from the right DataFrame based on the specified similarity metric.

    Note:
        Local execution requires the `sim-join` extra: `pip install "fenic[sim-join]"`.

    Args:
        other: The right-hand DataFrame to join with.
        left_on: Expression or column representing embeddings in the left DataFrame.
            If this is a named column, that column is treated as a user column and is
            included in the output. If this is an expression, it is treated as a
            temporary join key and is not included as an output column.
        right_on: Expression or column representing embeddings in the right DataFrame.
            Named columns are included in the output; expression-derived join keys
            are temporary and are not included in the output.
        k: Number of most similar matches to return per row.
        similarity_metric: Similarity metric to use: "l2", "cosine", or "dot".
        similarity_score_column: If set, adds a column with this name containing similarity scores.
            If None, the scores are omitted.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        A DataFrame containing one row for each of the top-k matches per row in the left DataFrame.
        The result includes all columns from both input DataFrames and, when `similarity_score_column`
        is provided, a similarity score column. Join key columns that already exist in the input
        DataFrames are preserved as normal user columns. Join keys derived from expressions are
        temporary execution columns and are not included in the result schema.

    Raises:
        ValidationError: If `k` is not positive or if the columns are invalid.
        ValidationError: If `similarity_metric` is not one of "l2", "cosine", "dot"

    Example: Match queries to FAQ entries
        ```python
        # Match customer queries to FAQ entries
        df_queries.semantic.sim_join(
            df_faqs,
            left_on=embeddings(col("query_text")),
            right_on=embeddings(col("faq_question")),
            k=1
        )
        ```

    Example: Link headlines to articles
        ```python
        # Link news headlines to full articles
        df_headlines.semantic.sim_join(
            df_articles,
            left_on=embeddings(col("headline")),
            right_on=embeddings(col("content")),
            k=3,
            return_similarity_scores=True
        )
        ```

    Example: Find similar job postings
        ```python
        # Find similar job postings across two sources
        df_linkedin.semantic.sim_join(
            df_indeed,
            left_on=embeddings(col("job_title")),
            right_on=embeddings(col("job_description")),
            k=2
        )
        ```
    """
    from fenic.api.dataframe.dataframe import DataFrame

    if not isinstance(right_on, ColumnOrName):
        raise ValidationError(
            f"The `right_on` argument must be a `Column` or a string representing a column name, "
            f"but got `{type(right_on).__name__}` instead."
        )
    if not isinstance(other, DataFrame):
        raise ValidationError(
                        f"The `other` argument to `sim_join()` must be a DataFrame`, but got `{type(other).__name__}`."
                    )
    if not (isinstance(k, int) and k > 0):
        raise ValidationError(
            f"The parameter `k` must be a positive integer, but received `{k}`."
        )
    args = get_args(SemanticSimilarityMetric)
    if similarity_metric not in args:
        raise ValidationError(
            f"The `similarity_metric` argument must be one of {args}, but got `{similarity_metric}`."
        )

    def _validate_column(column: ColumnOrName, name: str):
        if column is None:
            raise ValidationError(f"The `{name}` argument must not be None.")
        if not isinstance(column, ColumnOrName):
            raise ValidationError(
                f"The `{name}` argument must be a `Column` or a string representing a column name, "
                f"but got `{type(column).__name__}` instead."
            )

    _validate_column(left_on, "left_on")
    _validate_column(right_on, "right_on")

    # Validate request_timeout
    request_timeout = validate_timeout(request_timeout)
    DataFrame._ensure_same_session(self._df._session_state, [other._session_state])
    return self._df._from_logical_plan(
        SemanticSimilarityJoin.from_session_state(
            self._df._logical_plan,
            other._logical_plan,
            Column._from_col_or_name(left_on)._logical_expr,
            Column._from_col_or_name(right_on)._logical_expr,
            k,
            similarity_metric=similarity_metric,
            similarity_score_column=similarity_score_column,
            session_state=self._df._session_state,
            request_timeout=request_timeout,
        ),
        self._df._session_state,
    )
```

### with_cluster_labels

```
with_cluster_labels(by: ColumnOrName, num_clusters: int, max_iter: int = 300, num_init: int = 1, label_column: str = 'cluster_label', centroid_column: Optional[str] = None, request_timeout: Optional[float] = None) -> DataFrame
```

Cluster rows using K-means and add cluster metadata columns.

This method clusters rows based on the given embedding column or expression using K-means.
It adds a new column with cluster assignments, and optionally includes the centroid embedding
for each assigned cluster.

Note

Local execution requires the `cluster` extra: `pip install "fenic[cluster]"`.

Parameters:

- **`by`**
  (`ColumnOrName`)
  –

  Column or expression producing embeddings to cluster (e.g., `embed(col("text"))`).
- **`num_clusters`**
  (`int`)
  –

  Number of clusters to compute (must be > 0).
- **`max_iter`**
  (`int`, default:
  `300`
  )
  –

  Maximum iterations for a single run of the k-means algorithm. The algorithm stops when it either converges or reaches this limit.
- **`num_init`**
  (`int`, default:
  `1`
  )
  –

  Number of independent runs of k-means with different centroid seeds. The best result is selected.
- **`label_column`**
  (`str`, default:
  `'cluster_label'`
  )
  –

  Name of the output column for cluster IDs. Default is "cluster_label".
- **`centroid_column`**
  (`Optional[str]`, default:
  `None`
  )
  –

  If provided, adds a column with this name containing the centroid embedding
  for each row's assigned cluster.
- **`request_timeout`**
  (`Optional[float]`, default:
  `None`
  )
  –

  Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

Returns:

- `DataFrame`
  –

  A DataFrame with all original columns plus:
- `DataFrame`
  –

  - `<label_column>`: integer cluster assignment (0 to num_clusters - 1)
- `DataFrame`
  –

  - `<centroid_column>`: cluster centroid embedding, if specified

Basic clustering

```
# Cluster customer feedback and add cluster metadata
clustered_df = df.semantic.with_cluster_labels("feedback_embeddings", num_clusters=5)

# Then use regular operations to analyze clusters
clustered_df.group_by("cluster_label").agg(count("*"), avg("rating"))
```

Filter outliers using centroids

```
# Cluster and filter out rows far from their centroid
clustered_df = df.semantic.with_cluster_labels("embeddings", num_clusters=3, num_init=10, centroid_column="cluster_centroid")
clean_df = clustered_df.filter(
    embedding.compute_similarity("embeddings", "cluster_centroid", metric="cosine") > 0.7
)
```

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def with_cluster_labels(
    self,
    by: ColumnOrName,
    num_clusters: int,
    max_iter: int = 300,
    num_init: int = 1,
    label_column: str = "cluster_label",
    centroid_column: Optional[str] = None,
    request_timeout: Optional[float] = None,
) -> DataFrame:
    """Cluster rows using K-means and add cluster metadata columns.

    This method clusters rows based on the given embedding column or expression using K-means.
    It adds a new column with cluster assignments, and optionally includes the centroid embedding
    for each assigned cluster.

    Note:
        Local execution requires the `cluster` extra: `pip install "fenic[cluster]"`.

    Args:
        by: Column or expression producing embeddings to cluster (e.g., `embed(col("text"))`).
        num_clusters: Number of clusters to compute (must be > 0).
        max_iter: Maximum iterations for a single run of the k-means algorithm. The algorithm stops when it either converges or reaches this limit.
        num_init: Number of independent runs of k-means with different centroid seeds. The best result is selected.
        label_column: Name of the output column for cluster IDs. Default is "cluster_label".
        centroid_column: If provided, adds a column with this name containing the centroid embedding
                        for each row's assigned cluster.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        A DataFrame with all original columns plus:
        - `<label_column>`: integer cluster assignment (0 to num_clusters - 1)
        - `<centroid_column>`: cluster centroid embedding, if specified

    Example: Basic clustering
        ```python
        # Cluster customer feedback and add cluster metadata
        clustered_df = df.semantic.with_cluster_labels("feedback_embeddings", num_clusters=5)

        # Then use regular operations to analyze clusters
        clustered_df.group_by("cluster_label").agg(count("*"), avg("rating"))
        ```

    Example: Filter outliers using centroids
        ```python
        # Cluster and filter out rows far from their centroid
        clustered_df = df.semantic.with_cluster_labels("embeddings", num_clusters=3, num_init=10, centroid_column="cluster_centroid")
        clean_df = clustered_df.filter(
            embedding.compute_similarity("embeddings", "cluster_centroid", metric="cosine") > 0.7
        )
        ```
    """
    # Validate request_timeout
    request_timeout = validate_timeout(request_timeout)

    # Validate num_clusters
    if not isinstance(num_clusters, int) or num_clusters <= 0:
        raise ValidationError("`num_clusters` must be a positive integer.")

    # Validate max_iter
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValidationError("`max_iter` must be a positive integer.")

    # Validate num_init
    if not isinstance(num_init, int) or num_init <= 0:
        raise ValidationError("`num_init` must be a positive integer.")

    # Validate clustering target
    if not isinstance(by, ColumnOrName):
        raise ValidationError(
            f"Invalid cluster by: expected a column name (str) or Column object, got {type(by).__name__}."
        )

    # Validate label_column
    if not isinstance(label_column, str) or not label_column:
        raise ValidationError("`label_column` must be a non-empty string.")

    # Validate centroid_column if provided
    if centroid_column is not None:
        if not isinstance(centroid_column, str) or not centroid_column:
            raise ValidationError("`centroid_column` must be a non-empty string if provided.")

    # Check that the expression isn't a literal
    by_expr = Column._from_col_or_name(by)._logical_expr
    if isinstance(by_expr, LiteralExpr):
        raise ValidationError(
            f"Invalid cluster by: Cannot cluster by a literal value: {by_expr}."
        )

    return self._df._from_logical_plan(
        SemanticCluster.from_session_state(
            self._df._logical_plan,
            by_expr,
            num_clusters=num_clusters,
            max_iter=max_iter,
            num_init=num_init,
            label_column=label_column,
            centroid_column=centroid_column,
            session_state=self._df._session_state,
            request_timeout=request_timeout,
        ),
        self._df._session_state,
    )
```

## Session

The entry point to programming with the DataFrame API. Similar to PySpark's SparkSession.

Create a session with default configuration

```
session = Session.get_or_create(SessionConfig(app_name="my_app"))
```

Create a session with cloud configuration

```
config = SessionConfig(
    app_name="my_app",
    cloud=True,
    api_key="your_api_key"
)
session = Session.get_or_create(config)
```

Methods:

- **`create_dataframe`**
  –

  Create a DataFrame from a variety of Python-native data formats.
- **`get_or_create`**
  –

  Gets an existing Session or creates a new one with the configured settings.
- **`sql`**
  –

  Execute a read-only SQL query against one or more DataFrames using named placeholders.
- **`stop`**
  –

  Stops the session and closes all connections.
- **`table`**
  –

  Returns the specified table as a DataFrame.
- **`view`**
  –

  Returns the specified view as a DataFrame.

Attributes:

- **`catalog`**
  (`Catalog`)
  –

  Interface for catalog operations on the Session.
- **`read`**
  (`DataFrameReader`)
  –

  Returns a DataFrameReader that can be used to read data in as a DataFrame.

### catalog

```
catalog: Catalog
```

Interface for catalog operations on the Session.

### read

```
read: DataFrameReader
```

Returns a DataFrameReader that can be used to read data in as a DataFrame.

Returns:

- **`DataFrameReader`** ( `DataFrameReader`
  ) –

  A reader interface to read data into DataFrame

Raises:

- `RuntimeError`
  –

  If the session has been stopped

### create_dataframe

```
create_dataframe(data: DataLike, schema: Schema | None = None) -> DataFrame
```

Create a DataFrame from a variety of Python-native data formats.

Parameters:

- **`data`**
  (`DataLike`)
  –

  Input data. Must be one of:
  - Polars DataFrame
  - Pandas DataFrame
  - dict of column_name -> list of values
  - list of dicts (each dict representing a row)
  - pyarrow Table
- **`schema`**
  (`Schema | None`, default:
  `None`
  )
  –

  Optional complete top-level fenic schema. When provided,
  field names are authoritative, result columns are ordered to
  match the schema, values are physically coerced to the schema's
  Polars representation, and the logical DataFrame schema is
  preserved exactly. Use this for logical string-backed types
  such as JSON and Markdown, and for preserving fixed-size
  embedding arrays through local and cloud execution.

Returns:

- `DataFrame`
  –

  A new DataFrame instance

Raises:

- `ValidationError`
  –

  If the input format is unsupported or the provided
  columns do not match the schema.
- `PlanError`
  –

  If the input data cannot be coerced to the provided
  schema, or the schema is invalid for plan construction.

Create from Polars DataFrame

```
import polars as pl
df = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
session.create_dataframe(df)
```

Create from Pandas DataFrame

```
import pandas as pd
df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
session.create_dataframe(df)
```

Create from dictionary

```
session.create_dataframe({"col1": [1, 2], "col2": ["a", "b"]})
```

Create from list of dictionaries

```
session.create_dataframe([
    {"col1": 1, "col2": "a"},
    {"col1": 2, "col2": "b"}
])
```

Create from pyarrow Table

```
import pyarrow as pa
table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"]})
session.create_dataframe(table)
```

Create with an explicit schema

```
import fenic as fc

schema = fc.Schema([
    fc.ColumnField("age", fc.IntegerType),
    fc.ColumnField("name", fc.StringType),
])
session.create_dataframe({"name": ["Alice"], "age": ["42"]}, schema=schema)
```

Source code in `src/fenic/api/session/session.py`

```
def create_dataframe(
    self,
    data: DataLike,
    schema: Schema | None = None,
) -> DataFrame:
    """Create a DataFrame from a variety of Python-native data formats.

    Args:
        data: Input data. Must be one of:
            - Polars DataFrame
            - Pandas DataFrame
            - dict of column_name -> list of values
            - list of dicts (each dict representing a row)
            - pyarrow Table
        schema: Optional complete top-level fenic schema. When provided,
            field names are authoritative, result columns are ordered to
            match the schema, values are physically coerced to the schema's
            Polars representation, and the logical DataFrame schema is
            preserved exactly. Use this for logical string-backed types
            such as JSON and Markdown, and for preserving fixed-size
            embedding arrays through local and cloud execution.

    Returns:
        A new DataFrame instance

    Raises:
        ValidationError: If the input format is unsupported or the provided
            columns do not match the schema.
        PlanError: If the input data cannot be coerced to the provided
            schema, or the schema is invalid for plan construction.

    Example: Create from Polars DataFrame
        ```python
        import polars as pl
        df = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        session.create_dataframe(df)
        ```

    Example: Create from Pandas DataFrame
        ```python
        import pandas as pd
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        session.create_dataframe(df)
        ```

    Example: Create from dictionary
        ```python
        session.create_dataframe({"col1": [1, 2], "col2": ["a", "b"]})
        ```

    Example: Create from list of dictionaries
        ```python
        session.create_dataframe([
            {"col1": 1, "col2": "a"},
            {"col1": 2, "col2": "b"}
        ])
        ```

    Example: Create from pyarrow Table
        ```python
        import pyarrow as pa
        table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"]})
        session.create_dataframe(table)
        ```

    Example: Create with an explicit schema
        ```python
        import fenic as fc

        schema = fc.Schema([
            fc.ColumnField("age", fc.IntegerType),
            fc.ColumnField("name", fc.StringType),
        ])
        session.create_dataframe({"name": ["Alice"], "age": ["42"]}, schema=schema)
        ```
    """
    pl_df, row_field_names = _normalize_data_like_to_polars(
        data,
        allow_empty_list=schema is not None,
        validate_all_rows=schema is not None,
    )
    if schema is None:
        return DataFrame._from_logical_plan(
            InMemorySource.from_session_state(pl_df, self._session_state),
            self._session_state,
        )

    coerced_pl_df = _coerce_to_schema(pl_df, schema, row_field_names=row_field_names)

    return DataFrame._from_logical_plan(
        InMemorySource.from_schema(coerced_pl_df, schema),
        self._session_state,
    )
```

### get_or_create

```
get_or_create(config: SessionConfig) -> Session
```

Gets an existing Session or creates a new one with the configured settings.

Returns:

- `Session`
  –

  A Session instance configured with the provided settings

Source code in `src/fenic/api/session/session.py`

```
@classmethod
def get_or_create(
    cls,
    config: SessionConfig,
) -> Session:
    """Gets an existing Session or creates a new one with the configured settings.

    Returns:
        A Session instance configured with the provided settings
    """
    if config.cloud:
        from fenic._backends.cloud.manager import CloudSessionManager

        cloud_session_manager = CloudSessionManager()
        if not cloud_session_manager.initialized:
            session_manager_dependencies = (
                CloudSessionManager.create_global_session_dependencies()
            )
            cloud_session_manager.configure(session_manager_dependencies)
        future = asyncio.run_coroutine_threadsafe(
            cloud_session_manager.get_or_create_session_state(config),
            cloud_session_manager._asyncio_loop,
        )
        cloud_session_state = future.result()
        return Session._create_cloud_session(cloud_session_state)

    local_session_state: LocalSessionState = LocalSessionManager().get_or_create_session_state(config._to_resolved_config())
    return Session._create_local_session(local_session_state)
```

### sql

```
sql(query: str, /, **tables: DataFrame) -> DataFrame
```

Execute a read-only SQL query against one or more DataFrames using named placeholders.

This allows you to execute ad hoc SQL queries using familiar syntax when it's more convenient than the DataFrame API.
Placeholders in the SQL string (e.g. `{df}`) should correspond to keyword arguments (e.g. `df=my_dataframe`).

For supported SQL syntax and functions, refer to the DuckDB SQL documentation:
https://duckdb.org/docs/sql/introduction.

Parameters:

- **`query`**
  (`str`)
  –

  A SQL query string with placeholders like `{df}`
- **`**tables`**
  (`DataFrame`, default:
  `{}`
  )
  –

  Keyword arguments mapping placeholder names to DataFrames

Returns:

- `DataFrame`
  –

  A lazy DataFrame representing the result of the SQL query

Raises:

- `ValidationError`
  –

  If a placeholder is used in the query but not passed
  as a keyword argument

Simple join between two DataFrames

```
df1 = session.create_dataframe({"id": [1, 2]})
df2 = session.create_dataframe({"id": [2, 3]})
result = session.sql(
    "SELECT * FROM {df1} JOIN {df2} USING (id)",
    df1=df1,
    df2=df2
)
```

Complex query with multiple DataFrames

```
users = session.create_dataframe({"user_id": [1, 2], "name": ["Alice", "Bob"]})
orders = session.create_dataframe({"order_id": [1, 2], "user_id": [1, 2]})
products = session.create_dataframe({"product_id": [1, 2], "name": ["Widget", "Gadget"]})

result = session.sql("""
    SELECT u.name, p.name as product
    FROM {users} u
    JOIN {orders} o ON u.user_id = o.user_id
    JOIN {products} p ON o.product_id = p.product_id
""", users=users, orders=orders, products=products)
```

Source code in `src/fenic/api/session/session.py`

```
def sql(self, query: str, /, **tables: DataFrame) -> DataFrame:
    """Execute a read-only SQL query against one or more DataFrames using named placeholders.

    This allows you to execute ad hoc SQL queries using familiar syntax when it's more convenient than the DataFrame API.
    Placeholders in the SQL string (e.g. `{df}`) should correspond to keyword arguments (e.g. `df=my_dataframe`).

    For supported SQL syntax and functions, refer to the DuckDB SQL documentation:
    https://duckdb.org/docs/sql/introduction.

    Args:
        query: A SQL query string with placeholders like `{df}`
        **tables: Keyword arguments mapping placeholder names to DataFrames

    Returns:
        A lazy DataFrame representing the result of the SQL query

    Raises:
        ValidationError: If a placeholder is used in the query but not passed
            as a keyword argument

    Example: Simple join between two DataFrames
        ```python
        df1 = session.create_dataframe({"id": [1, 2]})
        df2 = session.create_dataframe({"id": [2, 3]})
        result = session.sql(
            "SELECT * FROM {df1} JOIN {df2} USING (id)",
            df1=df1,
            df2=df2
        )
        ```

    Example: Complex query with multiple DataFrames
        ```python
        users = session.create_dataframe({"user_id": [1, 2], "name": ["Alice", "Bob"]})
        orders = session.create_dataframe({"order_id": [1, 2], "user_id": [1, 2]})
        products = session.create_dataframe({"product_id": [1, 2], "name": ["Widget", "Gadget"]})

        result = session.sql(\"\"\"
            SELECT u.name, p.name as product
            FROM {users} u
            JOIN {orders} o ON u.user_id = o.user_id
            JOIN {products} p ON o.product_id = p.product_id
        \"\"\", users=users, orders=orders, products=products)
        ```
    """
    query = query.strip()
    if not query:
        raise ValidationError("SQL query must not be empty.")

    placeholders = set(SQL_PLACEHOLDER_RE.findall(query))
    missing = placeholders - tables.keys()
    if missing:
        raise ValidationError(
            f"Missing DataFrames for placeholders in SQL query: {', '.join(sorted(missing))}. "
            f"Make sure to pass them as keyword arguments, e.g., sql(..., {next(iter(missing))}=df)."
        )

    logical_plans = []
    template_names = []
    input_session_states = []
    for name, table in tables.items():
        if name in placeholders:
            template_names.append(name)
            logical_plans.append(table._logical_plan)
            input_session_states.append(table._session_state)

    DataFrame._ensure_same_session(self._session_state, input_session_states)
    return DataFrame._from_logical_plan(
        SQL.from_session_state(logical_plans, template_names, query, self._session_state),
        self._session_state,
    )
```

### stop

```
stop(skip_usage_summary: bool = False)
```

Stops the session and closes all connections.

Parameters:

- **`skip_usage_summary`**
  (`bool`, default:
  `False`
  )
  –

  Whether to skip printing the usage summary.

Unless `skip_usage_summary` is set, a summary of your session's metrics will print once you stop your session.

Source code in `src/fenic/api/session/session.py`

```
def stop(self, skip_usage_summary: bool = False):
    """Stops the session and closes all connections.

    Args:
        skip_usage_summary: Whether to skip printing the usage summary.

    Unless `skip_usage_summary` is set, a summary of your session's metrics will print once you stop your session.
    """
    self._session_state.stop(skip_usage_summary=skip_usage_summary)
```

### table

```
table(table_name: str) -> DataFrame
```

Returns the specified table as a DataFrame.

Parameters:

- **`table_name`**
  (`str`)
  –

  Name of the table

Returns:

- `DataFrame`
  –

  Table as a DataFrame

Raises:

- `ValueError`
  –

  If the table does not exist

Load an existing table

```
df = session.table("my_table")
```

Source code in `src/fenic/api/session/session.py`

```
def table(self, table_name: str) -> DataFrame:
    """Returns the specified table as a DataFrame.

    Args:
        table_name: Name of the table

    Returns:
        Table as a DataFrame

    Raises:
        ValueError: If the table does not exist

    Example: Load an existing table
        ```python
        df = session.table("my_table")
        ```
    """
    if not self._session_state.catalog.does_table_exist(table_name):
        raise ValueError(f"Table {table_name} does not exist")
    return DataFrame._from_logical_plan(
        TableSource.from_session_state(table_name, self._session_state),
        self._session_state,
    )
```

### view

```
view(view_name: str) -> DataFrame
```

Returns the specified view as a DataFrame.

Parameters:

- **`view_name`**
  (`str`)
  –

  Name of the view

Returns:
DataFrame: Dataframe with the given view

Source code in `src/fenic/api/session/session.py`

```
def view(self, view_name: str) -> DataFrame:
    """Returns the specified view as a DataFrame.

    Args:
        view_name: Name of the view
    Returns:
        DataFrame: Dataframe with the given view
    """
    if not self._session_state.catalog.does_view_exist(view_name):
        raise CatalogError(f"View {view_name} does not exist")

    view_plan = self._session_state.catalog.get_view_plan(view_name)
    validate_view(view_name, view_plan, self._session_state)

    return DataFrame._from_logical_plan(
        view_plan,
        self._session_state,
    )
```

## SessionConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.SessionConfig[SessionConfig]

              click fenic.SessionConfig href "" "fenic.SessionConfig"
```

Configuration for a user session.

This class defines the complete configuration for a user session, including
application settings, model configurations, and optional cloud settings.
It serves as the central configuration object for all language model operations.

Attributes:

- **`app_name`**
  (`str`)
  –

  Name of the application using this session. Defaults to "default_app".
- **`db_path`**
  (`Optional[Path]`)
  –

  Optional path to a local database file for persistent storage.
- **`semantic`**
  (`Optional[SemanticConfig]`)
  –

  Configuration for semantic models (optional).
- **`cloud`**
  (`Optional[CloudConfig]`)
  –

  Optional configuration for cloud execution.
- **`cache`**
  (`Optional[CloudConfig]`)
  –

  Optional configuration for LLM response caching.

Note

The semantic configuration is optional. When not provided, only non-semantic operations
are available. The cloud configuration is optional and only needed for
distributed processing.

Example

Configuring a basic session with a single language model:

```
config = SessionConfig(
    app_name="my_app",
    semantic=SemanticConfig(
        language_models={
            "gpt4": OpenAILanguageModel(
                model_name="gpt-4.1-nano", rpm=100, tpm=100
            )
        }
    ),
)
```

Configuring a session with multiple models and cloud execution:

```
config = SessionConfig(
    app_name="production_app",
    db_path=Path("/path/to/database.db"),
    semantic=SemanticConfig(
        language_models={
            "gpt4": OpenAILanguageModel(
                model_name="gpt-4.1-nano", rpm=100, tpm=100
            ),
            "claude": AnthropicLanguageModel(
                model_name="claude-haiku-4-5",
                rpm=100,
                input_tpm=100,
                output_tpm=100,
            ),
        },
        default_language_model="gpt4",
        embedding_models={
            "openai_embeddings": OpenAIEmbeddingModel(
                model_name="text-embedding-3-small", rpm=100, tpm=100
            )
        },
        default_embedding_model="openai_embeddings",
    ),
    cloud=CloudConfig(size=CloudExecutorSize.MEDIUM),
)
```

Methods:

- **`to_json`**
  –

  Export the session config to a JSON string.

### to_json

```
to_json() -> str
```

Export the session config to a JSON string.

Source code in `src/fenic/api/session/config.py`

```
def to_json(self) -> str:
    """Export the session config to a JSON string."""
    return self.model_dump_json(indent=2)
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
              fenic.StructType[StructType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes.DataType --> fenic.StructType

              click fenic.StructType href "" "fenic.StructType"
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

## SystemToolConfig

```
SystemToolConfig(table_names: list[str], tool_namespace: Optional[str] = None, max_result_rows: int = 100)
```

Configuration for canonical system tools.

fenic can automatically generate a set of canonical tools for operating on one or more fenic tables.

- Schema: list columns/types for any or all tables
- Profile: column statistics (counts, basic numeric analysis [min, max, mean, etc.], contextual information for text columns [average_length, etc.])
- Read: read a selection of rows from a single table. These rows can be paged over, filtered and can use column projections.
- Search Summary: literal or regex search across all text columns in all tables -- returns back dataframe names with result counts. Use `search_mode="literal"` for plain substring search or `search_mode="regex"` for regular expressions.
- Search Content: literal or regex search across a single table, specifying one or more text columns to search across -- returns back rows corresponding to the query. Use `search_mode="literal"` for plain substring search or `search_mode="regex"` for regular expressions.
- Analyze: Write raw SQL to perform complex analysis on one or more tables.

Attributes:

- **`table_names`**
  (`list[str]`)
  –

  List of the fenic table names the tools should be able to access. To allow access to all tables, pass `session.catalog.list_tables()`
- **`tool_namespace`**
  (`Optional[str]`)
  –

  If provided, will prefix the names of the generated tools with this namespace value.
  For example, by default the generated tools will be named `read`, `profile`, etc. With multiple fenic
  MCP servers, these tool names will clash, which can be confusing. In order to disambiguate, the `tool_namespace`
  is prefixed to the tool name (in snake case), so a `tool_namespace` of `fenic` would create the tools `fenic_read`,
  `fenic_profile`, etc.
- **`max_result_rows`**
  (`int`)
  –

  Maximum number of rows to be returned from Read/Analyze tools.

Example:

```
    from fenic import SystemToolConfig
    from fenic.api.mcp.tools import SystemToolConfig
    from fenic.api.mcp.server import create_mcp_server
    from fenic.api.session.session import Session
    session = Session.get_or_create(...)
    df = session.create_dataframe({
        "c1": [1, 2, 3],
        "c2": [4, 5, 6]
    })
    df.write.save_as_table("table1", mode="overwrite")
    session.catalog.set_table_description("table1", "Table 1 Description")
    server = create_mcp_server(session, "Test Server", system_tools=SystemToolConfig(
        table_names=["table1"],
        tool_namespace="Auto",
        max_result_rows=100
    ))
```

Example: Allow generated tools to access all tables in the catalog.

```
    from fenic import SystemToolConfig
    from fenic.api.mcp.tools import SystemToolConfig
    from fenic.api.mcp.server import create_mcp_server
    from fenic.api.session.session import Session
    session = Session.get_or_create(...)
    # Assuming you already have one or more tables saved to the catalog, with descriptions.
    server = create_mcp_server(session, "Test Server", system_tools=SystemToolConfig(
        table_names=session.catalog.list_tables()
        tool_namespace="Auto",
        max_result_rows=100
    ))
```

## ToolParam

Bases: `BaseModel`

```
              flowchart TD
              fenic.ToolParam[ToolParam]

              click fenic.ToolParam href "" "fenic.ToolParam"
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
              fenic.TranscriptType[TranscriptType]
              fenic.core.types.datatypes._LogicalType[_LogicalType]
              fenic.core.types.datatypes.DataType[DataType]

                              fenic.core.types.datatypes._LogicalType --> fenic.TranscriptType
                                fenic.core.types.datatypes.DataType --> fenic.core.types.datatypes._LogicalType

              click fenic.TranscriptType href "" "fenic.TranscriptType"
              click fenic.core.types.datatypes._LogicalType href "" "fenic.core.types.datatypes._LogicalType"
              click fenic.core.types.datatypes.DataType href "" "fenic.core.types.datatypes.DataType"
```

Represents a string containing a transcript in a specific format.

## UserDefinedTool

A tool that has been bound to a specific Parameterized View.

## approx_count_distinct

```
approx_count_distinct(column: ColumnOrName) -> Column
```

Aggregate function: returns an approximate count (HyperLogLog++) of distinct non-null values.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to approximately count distinct values in. Cannot be a StructType column.

Returns:

- `Column`
  –

  A Column expression representing the approximate count-distinct aggregation

Note

Differs from the pyspark implementation in that the relative standard deviation is not configurable.

Approximate distinct count per group

```
# Sample input
df = session.create_dataframe({
    "k": ["a", "a", "b", "b", "b"],
    "v": [1, None, 1, 2, 3],
})

df.group_by(fc.col("k")).agg(
    fc.approx_count_distinct(fc.col("v")).alias("approx_unique_v")
).show()
# Output:
# +---+------------------+
# | k | approx_unique_v  |
# +---+------------------+
# | a |                1 |
# | b |                3 |
# +---+------------------+
```

Nulls are ignored in approximate distinct counts

```
df = session.create_dataframe({"k": ["x", "x"], "v": [None, 3]})
df.group_by(fc.col("k")).agg(fc.approx_count_distinct(fc.col("v")).alias("acd")).show()
# Output:
# +---+-----+
# | k | acd |
# +---+-----+
# | x |   1 |
# +---+-----+
```

Raises:

- `TypeMismatchError`
  –

  If column is a StructType or ArrayType column.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def approx_count_distinct(column: ColumnOrName) -> Column:
    """Aggregate function: returns an approximate count (HyperLogLog++) of distinct non-null values.

    Args:
        column: Column or column name to approximately count distinct values in. Cannot be a StructType column.

    Returns:
        A Column expression representing the approximate count-distinct aggregation

    Note:
        Differs from the pyspark implementation in that the relative standard deviation is not configurable.

    Example: Approximate distinct count per group
        ```python
        # Sample input
        df = session.create_dataframe({
            "k": ["a", "a", "b", "b", "b"],
            "v": [1, None, 1, 2, 3],
        })

        df.group_by(fc.col("k")).agg(
            fc.approx_count_distinct(fc.col("v")).alias("approx_unique_v")
        ).show()
        # Output:
        # +---+------------------+
        # | k | approx_unique_v  |
        # +---+------------------+
        # | a |                1 |
        # | b |                3 |
        # +---+------------------+
        ```

    Example: Nulls are ignored in approximate distinct counts
        ```python
        df = session.create_dataframe({"k": ["x", "x"], "v": [None, 3]})
        df.group_by(fc.col("k")).agg(fc.approx_count_distinct(fc.col("v")).alias("acd")).show()
        # Output:
        # +---+-----+
        # | k | acd |
        # +---+-----+
        # | x |   1 |
        # +---+-----+
        ```

    Raises:
        TypeMismatchError: If column is a StructType or ArrayType<StructType> column.
    """
    return Column._from_logical_expr(
        ApproxCountDistinctExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## array_agg

```
array_agg(column: ColumnOrName) -> Column
```

Alias for collect_list().

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_agg(column: ColumnOrName) -> Column:
    """Alias for collect_list()."""
    return collect_list(column)
```

## asc

```
asc(column: ColumnOrName) -> Column
```

Mark this column for ascending sort order with nulls first.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the ascending ordering to.

Returns:

- `Column`
  –

  A sort expression with ascending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc(column: ColumnOrName) -> Column:
    """Mark this column for ascending sort order with nulls first.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A sort expression with ascending order and nulls first.
    """
    return Column._from_col_or_name(column).asc()
```

## asc_nulls_first

```
asc_nulls_first(column: ColumnOrName) -> Column
```

Alias for asc().

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the ascending ordering to.

Returns:

- `Column`
  –

  A sort expression with ascending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_first(column: ColumnOrName) -> Column:
    """Alias for asc().

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A sort expression with ascending order and nulls first.
    """
    return Column._from_col_or_name(column).asc_nulls_first()
```

## asc_nulls_last

```
asc_nulls_last(column: ColumnOrName) -> Column
```

Mark this column for ascending sort order with nulls last.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the ascending ordering to.

Returns:

- `Column`
  –

  A Column expression representing the column and the ascending sort order with nulls last.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_last(column: ColumnOrName) -> Column:
    """Mark this column for ascending sort order with nulls last.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A Column expression representing the column and the ascending sort order with nulls last.
    """
    return Column._from_col_or_name(column).asc_nulls_last()
```

## async_udf

```
async_udf(f: Optional[Callable[..., Awaitable[Any]]] = None, *, return_type: DataType, max_concurrency: int = 10, timeout_seconds: float = 30, num_retries: int = 0)
```

A decorator for creating async user-defined functions (UDFs) with configurable concurrency and retries.

Async UDFs allow IO-bound operations (API calls, database queries, MCP tool calls)
to be executed concurrently while maintaining DataFrame semantics.

Parameters:

- **`f`**
  (`Optional[Callable[..., Awaitable[Any]]]`, default:
  `None`
  )
  –

  Async function to convert to UDF
- **`return_type`**
  (`DataType`)
  –

  Expected return type of the UDF. Required parameter.
- **`max_concurrency`**
  (`int`, default:
  `10`
  )
  –

  Maximum number of concurrent executions (default: 10)
- **`timeout_seconds`**
  (`float`, default:
  `30`
  )
  –

  Per-item timeout in seconds (default: 30)
- **`num_retries`**
  (`int`, default:
  `0`
  )
  –

  Number of retries for failed items (default: 0)

Basic async UDF

```python
@async_udf(return_type=IntegerType)
async def slow_add(x: int, y: int) -> int:
await asyncio.sleep(1)
return x + y

df = df.select(slow_add(fc.col("x"), fc.col("y")).alias("slow_sum"))

### Or

async def slow_add_fn(x: int, y: int) -> int:
await asyncio.sleep(1)
return x + y

slow_add = async_udf(
slow_add_fn,
return_type=IntegerType
)

```

Example: API call with custom concurrency and retries
`python
@async_udf(
return_type=StructType([
StructField("status", IntegerType),
StructField("data", StringType)
]),
max_concurrency=20,
timeout_seconds=5,
num_retries=2
)
async def fetch_data(id: str) -> dict:
async with aiohttp.ClientSession() as session:
async with session.get(f"https://api.example.com/{id}") as resp:
return {
"status": resp.status,
"data": await resp.text()
}`

Note:
- Individual failures return None instead of raising exceptions
- Async UDFs should not block or do CPU-intensive work, as they
will block execution of other instances of the function call.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def async_udf(
    f: Optional[Callable[..., Awaitable[Any]]] = None,
    *,
    return_type: DataType,
    max_concurrency: int = 10,
    timeout_seconds: float = 30,
    num_retries: int = 0,
):
    """A decorator for creating async user-defined functions (UDFs) with configurable concurrency and retries.

    Async UDFs allow IO-bound operations (API calls, database queries, MCP tool calls)
    to be executed concurrently while maintaining DataFrame semantics.

    Args:
        f: Async function to convert to UDF
        return_type: Expected return type of the UDF. Required parameter.
        max_concurrency: Maximum number of concurrent executions (default: 10)
        timeout_seconds: Per-item timeout in seconds (default: 30)
        num_retries: Number of retries for failed items (default: 0)

    Example: Basic async UDF
        ```python
        @async_udf(return_type=IntegerType)
        async def slow_add(x: int, y: int) -> int:
            await asyncio.sleep(1)
            return x + y

        df = df.select(slow_add(fc.col("x"), fc.col("y")).alias("slow_sum"))

        # Or
        async def slow_add_fn(x: int, y: int) -> int:
            await asyncio.sleep(1)
            return x + y

        slow_add = async_udf(
            slow_add_fn,
            return_type=IntegerType
        )
    ```

    Example: API call with custom concurrency and retries
        ```python
        @async_udf(
            return_type=StructType([
                StructField("status", IntegerType),
                StructField("data", StringType)
            ]),
            max_concurrency=20,
            timeout_seconds=5,
            num_retries=2
        )
        async def fetch_data(id: str) -> dict:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.example.com/{id}") as resp:
                    return {
                        "status": resp.status,
                        "data": await resp.text()
                    }
        ```

    Note:
        - Individual failures return None instead of raising exceptions
        - Async UDFs should not block or do CPU-intensive work, as they
          will block execution of other instances of the function call.
    """

    def _create_async_udf(func: Callable[..., Awaitable[Any]]) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise ValidationError(
                f"@async_udf requires an async function, but found a synchronous "
                f"function {func.__name__!r} of type {type(func)}"
            )

        @wraps(func)
        def _async_udf_wrapper(*cols: ColumnOrName) -> Column:
            col_exprs = [Column._from_col_or_name(c)._logical_expr for c in cols]
            return Column._from_logical_expr(
                AsyncUDFExpr(
                    func,
                    col_exprs,
                    return_type,
                    max_concurrency=max_concurrency,
                    timeout_seconds=timeout_seconds,
                    num_retries=num_retries
                )
            )
        return _async_udf_wrapper

    if _is_logical_type(return_type):
        raise NotImplementedError(f"return_type {return_type} is not supported for async UDFs")

    # Support both @async_udf and async_udf(...) syntax
    if f is None:
        return _create_async_udf
    else:
        return _create_async_udf(f)
```

## avg

```
avg(column: ColumnOrName) -> Column
```

Aggregate function: returns the average (mean) of all values in the specified column. Applies to numeric and embedding types.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the average of

Returns:

- `Column`
  –

  A Column expression representing the average aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def avg(column: ColumnOrName) -> Column:
    """Aggregate function: returns the average (mean) of all values in the specified column. Applies to numeric and embedding types.

    Args:
        column: Column or column name to compute the average of

    Returns:
        A Column expression representing the average aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        AvgExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## coalesce

```
coalesce(*cols: ColumnOrName) -> Column
```

Returns the first non-null value from the given columns for each row.

This function mimics the behavior of SQL's COALESCE function. It evaluates the input columns
in order and returns the first non-null value encountered. If all values are null, returns null.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions or column names to evaluate. Each argument should be a single
  column expression or column name string.

Returns:

- `Column`
  –

  A Column expression containing the first non-null value from the input columns.

Raises:

- `ValidationError`
  –

  If no columns are provided.

coalesce usage

```
df.select(coalesce("col1", "col2", "col3"))
```

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def coalesce(*cols: ColumnOrName) -> Column:
    """Returns the first non-null value from the given columns for each row.

    This function mimics the behavior of SQL's COALESCE function. It evaluates the input columns
    in order and returns the first non-null value encountered. If all values are null, returns null.

    Args:
        *cols: Column expressions or column names to evaluate. Each argument should be a single
            column expression or column name string.

    Returns:
        A Column expression containing the first non-null value from the input columns.

    Raises:
        ValidationError: If no columns are provided.

    Example: coalesce usage
        ```python
        df.select(coalesce("col1", "col2", "col3"))
        ```
    """
    if not cols:
        raise ValidationError("No columns were provided. Please specify at least one column to use with the coalesce method.")

    exprs = [
        Column._from_col_or_name(c)._logical_expr for c in cols
    ]
    return Column._from_logical_expr(CoalesceExpr(exprs))
```

## col

```
col(col_name: str) -> Column
```

Creates a Column expression referencing a column in the DataFrame.

Parameters:

- **`col_name`**
  (`str`)
  –

  Name of the column to reference

Returns:

- `Column`
  –

  A Column expression for the specified column

Raises:

- `TypeError`
  –

  If colName is not a string

Source code in `src/fenic/api/functions/core.py`

```
@validate_call(config=ConfigDict(strict=True))
def col(col_name: str) -> Column:
    """Creates a Column expression referencing a column in the DataFrame.

    Args:
        col_name: Name of the column to reference

    Returns:
        A Column expression for the specified column

    Raises:
        TypeError: If colName is not a string
    """
    return Column._from_column_name(col_name)
```

## collect_list

```
collect_list(column: ColumnOrName) -> Column
```

Aggregate function: collects all values from the specified column into a list.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to collect values from

Returns:

- `Column`
  –

  A Column expression representing the list aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def collect_list(column: ColumnOrName) -> Column:
    """Aggregate function: collects all values from the specified column into a list.

    Args:
        column: Column or column name to collect values from

    Returns:
        A Column expression representing the list aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        ListExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## configure_logging

```
configure_logging(log_level: int = logging.INFO, log_format: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s', log_stream: Optional[TextIO] = None) -> None
```

Configure logging for the library and root logger in interactive environments.

This function ensures that logs from the library's modules appear in output by
setting up a default handler on the root logger *only if* one does not already
exist. This is especially useful in notebooks, scripts, or REPLs where logging
is often unset. It configures the root logger and sets the library's top-level
logger to propagate logs to the root.

If the root logger has no handlers, this function sets up a default configuration
and silences noisy dependencies like 'openai' and 'httpx'.

In more complex applications or when integrating with existing logging
configurations, you might prefer to manage logging setup externally. In such
cases, you may not need to call this function.

Source code in `src/fenic/logging.py`

```
def configure_logging(
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    log_stream: Optional[TextIO] = None,
) -> None:
    """Configure logging for the library and root logger in interactive environments.

    This function ensures that logs from the library's modules appear in output by
    setting up a default handler on the root logger *only if* one does not already
    exist. This is especially useful in notebooks, scripts, or REPLs where logging
    is often unset. It configures the root logger and sets the library's top-level
    logger to propagate logs to the root.

    If the root logger has no handlers, this function sets up a default configuration
    and silences noisy dependencies like 'openai' and 'httpx'.

    In more complex applications or when integrating with existing logging
    configurations, you might prefer to manage logging setup externally. In such
    cases, you may not need to call this function.
    """
    stream = log_stream or sys.stderr
    formatter = logging.Formatter(log_format)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        # Set up root logger only if not already configured
        root_logger.setLevel(log_level)
        root_logger.addHandler(handler)

        # Silence noisy dependencies
        for noisy_logger_name in NOISY_LOGGER_NAMES:
            noisy_logger = logging.getLogger(noisy_logger_name)
            noisy_logger.setLevel(logging.ERROR)

    # Set the library logger level and enable propagation
    library_root_name = __name__.split(".")[0]
    library_logger = logging.getLogger(library_root_name)
    library_logger.setLevel(log_level)
    library_logger.propagate = True
```

## count

```
count(column: ColumnOrName) -> Column
```

Aggregate function: returns the count of non-null values in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to count values in

Returns:

- `Column`
  –

  A Column expression representing the count aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def count(column: ColumnOrName) -> Column:
    """Aggregate function: returns the count of non-null values in the specified column.

    Args:
        column: Column or column name to count values in

    Returns:
        A Column expression representing the count aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    if isinstance(column, str) and column == "*":
        return Column._from_logical_expr(CountExpr(lit("*")._logical_expr))
    return Column._from_logical_expr(
        CountExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## count_distinct

```
count_distinct(*cols: ColumnOrName) -> Column
```

Aggregate function: returns the number of distinct non-null rows across one or more columns.

Behavior: Any row where one or more inputs is null is ignored.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  One or more columns or column names to include in the distinct count.

Returns:

- `Column`
  –

  A Column expression representing the count-distinct aggregation over the provided columns.

Distinct count per group (single column)

```
# Sample input
df = session.create_dataframe({
    "k": ["a", "a", "b", "b"],
    "v": [1, None, 2, 2],
})

df.group_by(fc.col("k")).agg(
    fc.count_distinct(fc.col("v")).alias("num_unique_v")
).show()
# Output:
# +---+--------------+
# | k | num_unique_v |
# +---+--------------+
# | a |            1 |
# | b |            1 |
# +---+--------------+
```

Distinct count across multiple columns (whole DataFrame)

```
# Sample input
df = session.create_dataframe({
    "a": [1, 1, 2, 2, None],
    "b": ["x", "x", "y", "y", "z"],
})

df.agg(
    fc.count_distinct(fc.col("a"), fc.col("b")).alias("num_unique_pairs")
).show()
# Output:
# +------------------+
# | num_unique_pairs |
# +------------------+
# |                2 |
# +------------------+
```

Nulls in any input column are ignored for multi-column distinct

```
df = session.create_dataframe({"a": [1, 1, None], "b": [1, 2, 1]})
df.agg(fc.count_distinct(fc.col("a"), fc.col("b")).alias("cd")).show()
# Output:
# +----+
# | cd |
# +----+
# |  2 |
# +----+
```

Raises:

- `ValidationError`
  –

  If no columns are provided.
- `TypeMismatchError`
  –

  If a column has an unsupported type

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def count_distinct(*cols: ColumnOrName) -> Column:
    """Aggregate function: returns the number of distinct non-null rows across one or more columns.

    Behavior: Any row where one or more inputs is null is ignored.

    Args:
        *cols: One or more columns or column names to include in the distinct count.

    Returns:
        A Column expression representing the count-distinct aggregation over the provided columns.

    Example: Distinct count per group (single column)
        ```python
        # Sample input
        df = session.create_dataframe({
            "k": ["a", "a", "b", "b"],
            "v": [1, None, 2, 2],
        })

        df.group_by(fc.col("k")).agg(
            fc.count_distinct(fc.col("v")).alias("num_unique_v")
        ).show()
        # Output:
        # +---+--------------+
        # | k | num_unique_v |
        # +---+--------------+
        # | a |            1 |
        # | b |            1 |
        # +---+--------------+
        ```

    Example: Distinct count across multiple columns (whole DataFrame)
        ```python
        # Sample input
        df = session.create_dataframe({
            "a": [1, 1, 2, 2, None],
            "b": ["x", "x", "y", "y", "z"],
        })

        df.agg(
            fc.count_distinct(fc.col("a"), fc.col("b")).alias("num_unique_pairs")
        ).show()
        # Output:
        # +------------------+
        # | num_unique_pairs |
        # +------------------+
        # |                2 |
        # +------------------+
        ```

    Example: Nulls in any input column are ignored for multi-column distinct
        ```python
        df = session.create_dataframe({"a": [1, 1, None], "b": [1, 2, 1]})
        df.agg(fc.count_distinct(fc.col("a"), fc.col("b")).alias("cd")).show()
        # Output:
        # +----+
        # | cd |
        # +----+
        # |  2 |
        # +----+
        ```

    Raises:
        ValidationError: If no columns are provided.
        TypeMismatchError: If a column has an unsupported type
    """
    if not cols:
        raise ValidationError("count_distinct requires at least one column")
    exprs = [Column._from_col_or_name(c)._logical_expr for c in cols]
    return Column._from_logical_expr(CountDistinctExpr(exprs))
```

## create_mcp_server

```
create_mcp_server(session: Session, server_name: str, *, user_defined_tools: Optional[List[UserDefinedTool]] = None, system_tools: Optional[SystemToolConfig] = None, concurrency_limit: int = 8) -> FenicMCPServer
```

Create an MCP server from datasets and tools.

Parameters:

- **`session`**
  (`Session`)
  –

  Fenic session used to execute tools.
- **`server_name`**
  (`str`)
  –

  Name of the MCP server.
- **`user_defined_tools`**
  (`Optional[List[UserDefinedTool]]`, default:
  `None`
  )
  –

  User defined tools to register with the MCP server.
- **`system_tools`**
  (`Optional[SystemToolConfig]`, default:
  `None`
  )
  –

  Configuration for automatically created system tools.
- **`concurrency_limit`**
  (`int`, default:
  `8`
  )
  –

  Maximum number of concurrent tool executions.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def create_mcp_server(
    session: Session,
    server_name: str,
    *,
    user_defined_tools: Optional[List[UserDefinedTool]] = None,
    system_tools: Optional[SystemToolConfig] = None,
    concurrency_limit: int = 8,
) -> FenicMCPServer:
    """Create an MCP server from datasets and tools.

    Args:
        session: Fenic session used to execute tools.
        server_name: Name of the MCP server.
        user_defined_tools: User defined tools to register with the MCP server.
        system_tools: Configuration for automatically created system tools.
        concurrency_limit: Maximum number of concurrent tool executions.
    """
    generated_system_tools = []
    user_defined_tools = user_defined_tools or []
    if system_tools:
        generated_system_tools.extend(
            auto_generate_system_tools_from_tables(
                system_tools.table_names,
                session,
                tool_namespace=system_tools.tool_namespace,
                max_result_limit=system_tools.max_result_rows
            )
        )
    if not (user_defined_tools or system_tools):
        raise ConfigurationError("No tools provided. Either provide `user_defined_tools` or set `system_tools` to create system tools for catalog tables.")
    return FenicMCPServer(session._session_state, user_defined_tools, generated_system_tools, server_name, concurrency_limit)
```

## desc

```
desc(column: ColumnOrName) -> Column
```

Mark this column for descending sort order with nulls first.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the descending ordering to.

Returns:

- `Column`
  –

  A sort expression with descending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc(column: ColumnOrName) -> Column:
    """Mark this column for descending sort order with nulls first.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls first.
    """
    return Column._from_col_or_name(column).desc()
```

## desc_nulls_first

```
desc_nulls_first(column: ColumnOrName) -> Column
```

Alias for desc().

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the descending ordering to.

Returns:

- `Column`
  –

  A sort expression with descending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_first(column: ColumnOrName) -> Column:
    """Alias for desc().

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls first.
    """
    return Column._from_col_or_name(column).desc_nulls_first()
```

## desc_nulls_last

```
desc_nulls_last(column: ColumnOrName) -> Column
```

Mark this column for descending sort order with nulls last.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the descending ordering to.

Returns:

- `Column`
  –

  A sort expression with descending order and nulls last.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_last(column: ColumnOrName) -> Column:
    """Mark this column for descending sort order with nulls last.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls last.
    """
    return Column._from_col_or_name(column).desc_nulls_last()
```

## empty

```
empty(data_type: DataType) -> Column
```

Creates a Column expression representing an empty value of the given type.

- If the data type is `ArrayType(...)`, the empty value will be an empty array.
- If the data type is `StructType(...)`, the empty value will be an instance of the struct type with all fields set to `None`.
- For all other data types, the empty value is None (equivalent to calling `null(data_type)`)

This function is useful for creating columns with empty values of a particular type.

Parameters:

- **`data_type`**
  (`DataType`)
  –

  The data type of the empty value

Returns:

- `Column`
  –

  A Column expression representing the empty value

Raises:

- `ValidationError`
  –

  If the data type is not a valid data type

Creating a column with an empty array type

```
# The newly created `b` column will have a value of `[]` for all rows
df.select(fc.col("a"), fc.empty(fc.ArrayType(fc.IntegerType)).alias("b"))
```

Creating a column with an empty struct type

```
# The newly created `b` column will have a value of `{b: None}` for all rows
df.select(fc.col("a"), fc.empty(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("b"))
```

Creating a column with an empty primitive type

```
# The newly created `b` column will have a value of `None` for all rows
df.select(fc.col("a"), fc.empty(fc.IntegerType).alias("b"))
```

Source code in `src/fenic/api/functions/core.py`

```
def empty(data_type: DataType) -> Column:
    """Creates a Column expression representing an empty value of the given type.

    - If the data type is `ArrayType(...)`, the empty value will be an empty array.
    - If the data type is `StructType(...)`, the empty value will be an instance of the struct type with all fields set to `None`.
    - For all other data types, the empty value is None (equivalent to calling `null(data_type)`)

    This function is useful for creating columns with empty values of a particular type.

    Args:
        data_type: The data type of the empty value

    Returns:
        A Column expression representing the empty value

    Raises:
        ValidationError: If the data type is not a valid data type

    Example: Creating a column with an empty array type
        ```python
        # The newly created `b` column will have a value of `[]` for all rows
        df.select(fc.col("a"), fc.empty(fc.ArrayType(fc.IntegerType)).alias("b"))
        ```

    Example: Creating a column with an empty struct type
        ```python
        # The newly created `b` column will have a value of `{b: None}` for all rows
        df.select(fc.col("a"), fc.empty(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("b"))
        ```

    Example: Creating a column with an empty primitive type
        ```python
        # The newly created `b` column will have a value of `None` for all rows
        df.select(fc.col("a"), fc.empty(fc.IntegerType).alias("b"))
        ```
    """
    if isinstance(data_type, ArrayType):
        return Column._from_logical_expr(LiteralExpr([], data_type))
    elif isinstance(data_type, StructType):
        return Column._from_logical_expr(LiteralExpr({}, data_type))
    return null(data_type)
```

## first

```
first(column: ColumnOrName) -> Column
```

Aggregate function: returns the first non-null value in the specified column.

Typically used in aggregations to select the first observed value per group.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name.

Returns:

- `Column`
  –

  Column expression for the first value.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def first(column: ColumnOrName) -> Column:
    """Aggregate function: returns the first non-null value in the specified column.

    Typically used in aggregations to select the first observed value per group.

    Args:
        column: Column or column name.

    Returns:
        Column expression for the first value.
    """
    return Column._from_logical_expr(
        FirstExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## flatten

```
flatten(column: ColumnOrName) -> Column
```

Flattens an array of arrays into a single array (one level deep).

Flattens nested arrays by concatenating all inner arrays into a single array.
Only flattens one level of nesting. Returns null if the input is null.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name containing arrays of arrays.

Returns:

- `Column`
  –

  A Column with flattened arrays (one level deep).

Flattening nested arrays

```
import fenic as fc

df = fc.Session.local().create_dataframe({
    "nested": [[[1, 2], [3, 4]], [[5], [6, 7, 8]], None]
})

result = df.select(fc.flatten("nested").alias("flat"))
# Output:
# ┌──────────────────┐
# │ flat             │
# ├──────────────────┤
# │ [1, 2, 3, 4]     │
# │ [5, 6, 7, 8]     │
# │ null             │
# └──────────────────┘
```

One level only

```
# Deeply nested arrays - only flattens one level
df = fc.Session.local().create_dataframe({
    "deep": [[[[1]], [[2]]], [[[3]]]]
})

result = df.select(fc.flatten("deep"))
# Output: [[[1], [2]], [[3]]]  # Still nested after one level
```

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def flatten(column: ColumnOrName) -> Column:
    """Flattens an array of arrays into a single array (one level deep).

    Flattens nested arrays by concatenating all inner arrays into a single array.
    Only flattens one level of nesting. Returns null if the input is null.

    Args:
        column: Column or column name containing arrays of arrays.

    Returns:
        A Column with flattened arrays (one level deep).

    Example: Flattening nested arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "nested": [[[1, 2], [3, 4]], [[5], [6, 7, 8]], None]
        })

        result = df.select(fc.flatten("nested").alias("flat"))
        # Output:
        # ┌──────────────────┐
        # │ flat             │
        # ├──────────────────┤
        # │ [1, 2, 3, 4]     │
        # │ [5, 6, 7, 8]     │
        # │ null             │
        # └──────────────────┘
        ```

    Example: One level only
        ```python
        # Deeply nested arrays - only flattens one level
        df = fc.Session.local().create_dataframe({
            "deep": [[[[1]], [[2]]], [[[3]]]]
        })

        result = df.select(fc.flatten("deep"))
        # Output: [[[1], [2]], [[3]]]  # Still nested after one level
        ```
    """
    return Column._from_logical_expr(
        FlattenExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## greatest

```
greatest(*cols: ColumnOrName) -> Column
```

Returns the greatest value from the given columns for each row.

This function mimics the behavior of SQL's GREATEST function. It evaluates the input columns
in order and returns the greatest value encountered. If all values are null, returns null.

All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions or column names to evaluate. Each argument should be a single
  column expression or column name string.

Returns:

- `Column`
  –

  A Column expression containing the greatest value from the input columns.

Raises:

- `ValidationError`
  –

  If fewer than two columns are provided.

greatest usage

```
df.select(fc.greatest("col1", "col2", "col3"))
```

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def greatest(*cols: ColumnOrName) -> Column:
    """Returns the greatest value from the given columns for each row.

    This function mimics the behavior of SQL's GREATEST function. It evaluates the input columns
    in order and returns the greatest value encountered. If all values are null, returns null.

    All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

    Args:
        *cols: Column expressions or column names to evaluate. Each argument should be a single
            column expression or column name string.

    Returns:
        A Column expression containing the greatest value from the input columns.

    Raises:
        ValidationError: If fewer than two columns are provided.

    Example: greatest usage
        ```python
        df.select(fc.greatest("col1", "col2", "col3"))
        ```
    """
    if len(cols) < 2:
        raise ValidationError(f"greatest() requires at least 2 columns, got {len(cols)}")

    exprs = [
        Column._from_col_or_name(c)._logical_expr for c in cols
    ]
    return Column._from_logical_expr(GreatestExpr(exprs))
```

## least

```
least(*cols: ColumnOrName) -> Column
```

Returns the least value from the given columns for each row.

This function mimics the behavior of SQL's LEAST function. It evaluates the input columns
in order and returns the least value encountered. If all values are null, returns null.

All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions or column names to evaluate. Each argument should be a single
  column expression or column name string.

Returns:

- `Column`
  –

  A Column expression containing the least value from the input columns.

Raises:

- `ValidationError`
  –

  If fewer than two columns are provided.

least usage

```
df.select(fc.least("col1", "col2", "col3"))
```

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def least(*cols: ColumnOrName) -> Column:
    """Returns the least value from the given columns for each row.

    This function mimics the behavior of SQL's LEAST function. It evaluates the input columns
    in order and returns the least value encountered. If all values are null, returns null.

    All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

    Args:
        *cols: Column expressions or column names to evaluate. Each argument should be a single
            column expression or column name string.

    Returns:
        A Column expression containing the least value from the input columns.

    Raises:
        ValidationError: If fewer than two columns are provided.

    Example: least usage
        ```python
        df.select(fc.least("col1", "col2", "col3"))
        ```
    """
    if len(cols) < 2:
        raise ValidationError(f"least() requires at least 2 columns, got {len(cols)}")

    exprs = [
        Column._from_col_or_name(c)._logical_expr for c in cols
    ]
    return Column._from_logical_expr(LeastExpr(exprs))
```

## lit

```
lit(value: Any) -> Column
```

Creates a Column expression representing a literal value.

Column Data Type must be inferrable from the value

- Cannot be used to create a columm with the literal value `None`. Use `null(data_type)` instead.
- Cannot be used to create a columm with the literal value `[]`. Use `empty(ArrayType(...))` instead.
- Cannot be used to create a columm with the literal value `{}`. Use `empty(StructType(...))` instead.

Parameters:

- **`value`**
  (`Any`)
  –

  The literal value to create a column for

Returns:

- `Column`
  –

  A Column expression representing the literal value

Raises:
ValidationError: If the type of the value cannot be inferred

Source code in `src/fenic/api/functions/core.py`

```
def lit(value: Any) -> Column:
    """Creates a Column expression representing a literal value.

    Column Data Type must be inferrable from the value:
        - Cannot be used to create a columm with the literal value `None`. Use `null(data_type)` instead.
        - Cannot be used to create a columm with the literal value `[]`. Use `empty(ArrayType(...))` instead.
        - Cannot be used to create a columm with the literal value `{}`. Use `empty(StructType(...))` instead.

    Args:
        value: The literal value to create a column for

    Returns:
        A Column expression representing the literal value
    Raises:
        ValidationError: If the type of the value cannot be inferred
    """
    if value is None:
        raise ValidationError("Cannot create a literal with value `None`. Use `null(...)` instead.")
    elif value == []:
        raise ValidationError(f"Cannot create a literal with empty value `{value}` Use `empty(ArrayType(...))` instead.")
    elif value == {}:
        raise ValidationError(f"Cannot create a literal with empty value `{value}` Use `empty(StructType(...))` instead.")
    try:
        inferred_type = infer_dtype_from_pyobj(value)
    except TypeInferenceError as e:
        raise ValidationError(f"`lit` failed to infer type for value `{value}`") from e
    literal_expr = LiteralExpr(value, inferred_type)
    return Column._from_logical_expr(literal_expr)
```

## max

```
max(column: ColumnOrName) -> Column
```

Aggregate function: returns the maximum value in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the maximum of

Returns:

- `Column`
  –

  A Column expression representing the maximum aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def max(column: ColumnOrName) -> Column:
    """Aggregate function: returns the maximum value in the specified column.

    Args:
        column: Column or column name to compute the maximum of

    Returns:
        A Column expression representing the maximum aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        MaxExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## mean

```
mean(column: ColumnOrName) -> Column
```

Aggregate function: returns the mean (average) of all values in the specified column.

Alias for avg().

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the mean of

Returns:

- `Column`
  –

  A Column expression representing the mean aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def mean(column: ColumnOrName) -> Column:
    """Aggregate function: returns the mean (average) of all values in the specified column.

    Alias for avg().

    Args:
        column: Column or column name to compute the mean of

    Returns:
        A Column expression representing the mean aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        AvgExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## min

```
min(column: ColumnOrName) -> Column
```

Aggregate function: returns the minimum value in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the minimum of

Returns:

- `Column`
  –

  A Column expression representing the minimum aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def min(column: ColumnOrName) -> Column:
    """Aggregate function: returns the minimum value in the specified column.

    Args:
        column: Column or column name to compute the minimum of

    Returns:
        A Column expression representing the minimum aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        MinExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## null

```
null(data_type: DataType) -> Column
```

Creates a Column expression representing a null value of the specified data type.

Regardless of the data type, the column will contain a null (None) value.
This function is useful for creating columns with null values of a particular type.

Parameters:

- **`data_type`**
  (`DataType`)
  –

  The data type of the null value

Returns:

- `Column`
  –

  A Column expression representing the null value

Raises:

- `ValidationError`
  –

  If the data type is not a valid data type

Creating a column with a null value of a primitive type

```
# The newly created `b` column will have a value of `None` for all rows
df.select(fc.col("a"), fc.null(fc.IntegerType).alias("b"))
```

Creating a column with a null value of an array/struct type

```
# The newly created `b` and `c` columns will have a value of `None` for all rows
df.select(
    fc.col("a"),
    fc.null(fc.ArrayType(fc.IntegerType)).alias("b"),
    fc.null(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("c"),
)
```

Source code in `src/fenic/api/functions/core.py`

```
def null(data_type: DataType) -> Column:
    """Creates a Column expression representing a null value of the specified data type.

    Regardless of the data type, the column will contain a null (None) value.
    This function is useful for creating columns with null values of a particular type.

    Args:
        data_type: The data type of the null value

    Returns:
        A Column expression representing the null value

    Raises:
        ValidationError: If the data type is not a valid data type

    Example: Creating a column with a null value of a primitive type
        ```python
        # The newly created `b` column will have a value of `None` for all rows
        df.select(fc.col("a"), fc.null(fc.IntegerType).alias("b"))
        ```

    Example: Creating a column with a null value of an array/struct type
        ```python
        # The newly created `b` and `c` columns will have a value of `None` for all rows
        df.select(
            fc.col("a"),
            fc.null(fc.ArrayType(fc.IntegerType)).alias("b"),
            fc.null(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("c"),
        )
        ```

    """
    return Column._from_logical_expr(LiteralExpr(None, data_type))
```

## run_mcp_server_asgi

```
run_mcp_server_asgi(server: FenicMCPServer, *, stateless_http: bool = True, transport: Literal['streamable-http', 'sse'] = 'streamable-http', path: Optional[str] = '/mcp', **kwargs)
```

Run an MCP server as a Starlette ASGI app.

Returns a Starlette ASGI app that can be integrated into any ASGI server.
This is useful for running the MCP server in a production environment, or running the MCP server as part of a larger application.

Parameters:

- **`server`**
  (`FenicMCPServer`)
  –

  MCP server to run.
- **`stateless_http`**
  (`bool`, default:
  `True`
  )
  –

  If True, use stateless HTTP.
- **`transport`**
  (`Literal['streamable-http', 'sse']`, default:
  `'streamable-http'`
  )
  –

  Transport protocol to use (streamable-http, sse).
- **`path`**
  (`Optional[str]`, default:
  `'/mcp'`
  )
  –

  Path to listen on.
- **`kwargs`**
  –

  Additional starlette-specific arguments to pass to FastMCP.

Notes

Additional keyword arguments:
- `middleware`: A list of Starlette `ASGIMiddleware` middleware to apply to the app.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def run_mcp_server_asgi(
    server: FenicMCPServer,
    *,
    stateless_http: bool = True,
    transport: Literal['streamable-http', 'sse'] = 'streamable-http',
    path: Optional[str] = "/mcp",
    **kwargs,
):
    """Run an MCP server as a Starlette ASGI app.

    Returns a Starlette ASGI app that can be integrated into any ASGI server.
    This is useful for running the MCP server in a production environment, or running the MCP server as part of a larger application.

    Args:
        server: MCP server to run.
        stateless_http: If True, use stateless HTTP.
        transport: Transport protocol to use (streamable-http, sse).
        path: Path to listen on.
        kwargs: Additional starlette-specific arguments to pass to FastMCP.

    Notes:
        Additional keyword arguments:
        - `middleware`: A list of Starlette `ASGIMiddleware` middleware to apply to the app.
    """
    return server.http_app(stateless_http=stateless_http, transport=transport, path=path, **kwargs)
```

## run_mcp_server_async

```
run_mcp_server_async(server: FenicMCPServer, *, transport: MCPTransport = 'http', stateless_http: bool = True, port: Optional[int] = None, host: Optional[str] = None, path: Optional[str] = '/mcp', **kwargs)
```

Run an MCP server asynchronously.

Use this when calling from asynchronous code. This does not create a new event loop.

Parameters:

- **`server`**
  (`FenicMCPServer`)
  –

  MCP server to run.
- **`transport`**
  (`MCPTransport`, default:
  `'http'`
  )
  –

  Transport protocol (http, stdio).
- **`stateless_http`**
  (`bool`, default:
  `True`
  )
  –

  If True, use stateless HTTP.
- **`port`**
  (`Optional[int]`, default:
  `None`
  )
  –

  Port to listen on.
- **`host`**
  (`Optional[str]`, default:
  `None`
  )
  –

  Host to listen on.
- **`path`**
  (`Optional[str]`, default:
  `'/mcp'`
  )
  –

  Path to listen on.
- **`kwargs`**
  –

  Additional transport-specific arguments to pass to FastMCP.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
async def run_mcp_server_async(
    server: FenicMCPServer,
    *,
    transport: MCPTransport = "http",
    stateless_http: bool = True,
    port: Optional[int] = None,
    host: Optional[str] = None,
    path: Optional[str] = "/mcp",
    **kwargs,
):
    """Run an MCP server asynchronously.

    Use this when calling from asynchronous code. This does not create a new event loop.

    Args:
        server: MCP server to run.
        transport: Transport protocol (http, stdio).
        stateless_http: If True, use stateless HTTP.
        port: Port to listen on.
        host: Host to listen on.
        path: Path to listen on.
        kwargs: Additional transport-specific arguments to pass to FastMCP.
    """
    await server.run_async(transport=transport, stateless_http=stateless_http, port=port, host=host, path=path, **kwargs)
```

## run_mcp_server_sync

```
run_mcp_server_sync(server: FenicMCPServer, *, transport: MCPTransport = 'http', stateless_http: bool = True, port: Optional[int] = None, host: Optional[str] = None, path: Optional[str] = '/mcp', **kwargs)
```

Run an MCP server synchronously.

Use this when calling from synchronous code. This creates a new event loop and runs the server in it.

Parameters:

- **`server`**
  (`FenicMCPServer`)
  –

  MCP server to run.
- **`transport`**
  (`MCPTransport`, default:
  `'http'`
  )
  –

  Transport protocol (http, stdio).
- **`stateless_http`**
  (`bool`, default:
  `True`
  )
  –

  If True, use stateless HTTP.
- **`port`**
  (`Optional[int]`, default:
  `None`
  )
  –

  Port to listen on.
- **`host`**
  (`Optional[str]`, default:
  `None`
  )
  –

  Host to listen on.
- **`path`**
  (`Optional[str]`, default:
  `'/mcp'`
  )
  –

  Path to listen on.
- **`kwargs`**
  –

  Additional transport-specific arguments to pass to FastMCP.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def run_mcp_server_sync(
    server: FenicMCPServer,
    *,
    transport: MCPTransport = "http",
    stateless_http: bool = True,
    port: Optional[int] = None,
    host: Optional[str] = None,
    path: Optional[str] = "/mcp",
    **kwargs,
):
    """Run an MCP server synchronously.

    Use this when calling from synchronous code. This creates a new event loop and runs the server in it.

    Args:
        server: MCP server to run.
        transport: Transport protocol (http, stdio).
        stateless_http: If True, use stateless HTTP.
        port: Port to listen on.
        host: Host to listen on.
        path: Path to listen on.
        kwargs: Additional transport-specific arguments to pass to FastMCP.
    """
    server.run(transport=transport, stateless_http=stateless_http, port=port, host=host, path=path, **kwargs)
```

## stddev

```
stddev(column: ColumnOrName) -> Column
```

Aggregate function: returns the sample standard deviation of the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name.

Returns:

- `Column`
  –

  Column expression for sample standard deviation.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def stddev(column: ColumnOrName) -> Column:
    """Aggregate function: returns the sample standard deviation of the specified column.

    Args:
        column: Column or column name.

    Returns:
        Column expression for sample standard deviation.
    """
    return Column._from_logical_expr(
        StdDevExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## struct

```
struct(*args: Union[ColumnOrName, List[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column
```

Creates a new struct column from multiple input columns.

Parameters:

- **`*args`**
  (`Union[ColumnOrName, List[ColumnOrName], Tuple[ColumnOrName, ...]]`, default:
  `()`
  )
  –

  Columns or column names to combine into a struct. Can be:

  - Individual arguments
  - Lists of columns/column names
  - Tuples of columns/column names

Returns:

- `Column`
  –

  A Column expression representing a struct containing the input columns

Raises:

- `TypeError`
  –

  If any argument is not a Column, string, or collection of
  Columns/strings

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def struct(
    *args: Union[ColumnOrName, List[ColumnOrName], Tuple[ColumnOrName, ...]]
) -> Column:
    """Creates a new struct column from multiple input columns.

    Args:
        *args: Columns or column names to combine into a struct. Can be:

            - Individual arguments
            - Lists of columns/column names
            - Tuples of columns/column names

    Returns:
        A Column expression representing a struct containing the input columns

    Raises:
        TypeError: If any argument is not a Column, string, or collection of
            Columns/strings
    """
    flattened_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)

    expr_columns = [Column._from_col_or_name(c)._logical_expr for c in flattened_args]

    return Column._from_logical_expr(StructExpr(expr_columns))
```

## sum

```
sum(column: ColumnOrName) -> Column
```

Aggregate function: returns the sum of all values in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the sum of

Returns:

- `Column`
  –

  A Column expression representing the sum aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def sum(column: ColumnOrName) -> Column:
    """Aggregate function: returns the sum of all values in the specified column.

    Args:
        column: Column or column name to compute the sum of

    Returns:
        A Column expression representing the sum aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        SumExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## sum_distinct

```
sum_distinct(column: ColumnOrName) -> Column
```

Aggregate function: returns the sum of distinct numeric values in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the sum of distinct values

Returns:

- `Column`
  –

  A Column expression representing the sum-distinct aggregation

Sum of distinct values per group

```
# Sample input
df = session.create_dataframe({
    "k": ["a", "a", "b", "b"],
    "v": [1, None, 2, 2],
})

# Sum distinct values of column `v` within each group `k`
df.group_by(fc.col("k")).agg(
    fc.sum_distinct(fc.col("v")).alias("sum_distinct_v")
).show()
# Output:
# +---+----------------+
# | k | sum_distinct_v |
# +---+----------------+
# | a |              1 |
# | b |              2 |
# +---+----------------+
```

Nulls are ignored when summing distinct values

```
df = session.create_dataframe({"k": ["x", "x"], "v": [None, 3]})
df.group_by(fc.col("k")).agg(fc.sum_distinct(fc.col("v")).alias("sd")).show()
# Output:
# +---+----+
# | k | sd |
# +---+----+
# | x |  3 |
# +---+----+
```

Raises:

- `TypeMismatchError`
  –

  If column is not a numeric or boolean type

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def sum_distinct(column: ColumnOrName) -> Column:
    """Aggregate function: returns the sum of distinct numeric values in the specified column.

    Args:
        column: Column or column name to compute the sum of distinct values

    Returns:
        A Column expression representing the sum-distinct aggregation

    Example: Sum of distinct values per group
        ```python
        # Sample input
        df = session.create_dataframe({
            "k": ["a", "a", "b", "b"],
            "v": [1, None, 2, 2],
        })

        # Sum distinct values of column `v` within each group `k`
        df.group_by(fc.col("k")).agg(
            fc.sum_distinct(fc.col("v")).alias("sum_distinct_v")
        ).show()
        # Output:
        # +---+----------------+
        # | k | sum_distinct_v |
        # +---+----------------+
        # | a |              1 |
        # | b |              2 |
        # +---+----------------+
        ```

    Example: Nulls are ignored when summing distinct values
        ```python
        df = session.create_dataframe({"k": ["x", "x"], "v": [None, 3]})
        df.group_by(fc.col("k")).agg(fc.sum_distinct(fc.col("v")).alias("sd")).show()
        # Output:
        # +---+----+
        # | k | sd |
        # +---+----+
        # | x |  3 |
        # +---+----+
        ```

    Raises:
        TypeMismatchError: If column is not a numeric or boolean type
    """
    return Column._from_logical_expr(
        SumDistinctExpr(Column._from_col_or_name(column)._logical_expr)
    )
```

## tool_param

```
tool_param(parameter_name: str, data_type: DataType) -> Column
```

Creates an unresolved literal placeholder column with a declared data type.

A placeholder argument for a DataFrame, representing a literal value to be provided at execution time.
If no value is supplied, it defaults to null. Enables parameterized views and macros over fenic DataFrames.

Notes

Supports only Primitive/Object/ArrayLike Types (StringType, IntegerType, FloatType, DoubleType, BooleanType, StructType, ArrayType)

Parameters:

- **`parameter_name`**
  (`str`)
  –

  The name of the parameter to reference.
- **`data_type`**
  (`DataType`)
  –

  The expected data type for the parameter value.

Returns:

- `Column`
  –

  A Column wrapping an UnresolvedLiteralExpr for the given parameter.

A simple tool with one parameter

```python

### Assume we are reading data with a `name` column.

df = session.read.csv(data.csv)
parameterized_df = df.filter(fc.col("name").contains(fc.tool_param('query', StringType)))
...
session.catalog.create_tool(
tool_name="my_tool",
tool_description="A tool that searches the name field",
tool_query=parameterized_df,
result_limit=100,
tool_params=[ToolParam(name="query", description="The name should contain the following value")]
)

A tool with multiple filters

```python

### Assume we are reading data with an `age` column.

df = session.read.csv(users.csv)

### create multiple filters that evaluate to true if a param is not passed.

optional_min = fc.coalesce(fc.col("age") >= tool_param("min_age", IntegerType), fc.lit(True))
optional_max = fc.coalesce(fc.col("age") <= tool_param("max_age", IntegerType), fc.lit(True))
core_filter = df.filter(optional_min & optional_max)
session.catalog.create_tool(
"users_filter",
"Filter users by age",
core_filter,
tool_params=[
ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
]
)

Source code in `src/fenic/api/functions/core.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def tool_param(parameter_name: str, data_type: DataType) -> Column:
    """Creates an unresolved literal placeholder column with a declared data type.

    A placeholder argument for a DataFrame, representing a literal value to be provided at execution time. 
    If no value is supplied, it defaults to null. Enables parameterized views and macros over fenic DataFrames.

    Notes:
        Supports only Primitive/Object/ArrayLike Types (StringType, IntegerType, FloatType, DoubleType, BooleanType, StructType, ArrayType)

    Args:
        parameter_name: The name of the parameter to reference.
        data_type: The expected data type for the parameter value.

    Returns:
        A Column wrapping an UnresolvedLiteralExpr for the given parameter.

    Example: A simple tool with one parameter
        ```python
        # Assume we are reading data with a `name` column.
        df = session.read.csv(data.csv)
        parameterized_df = df.filter(fc.col("name").contains(fc.tool_param('query', StringType)))
        ...
        session.catalog.create_tool(
            tool_name="my_tool",
            tool_description="A tool that searches the name field",
            tool_query=parameterized_df,
            result_limit=100,
            tool_params=[ToolParam(name="query", description="The name should contain the following value")]
        )

    Example: A tool with multiple filters
        ```python
        # Assume we are reading data with an `age` column.
        df = session.read.csv(users.csv)
        # create multiple filters that evaluate to true if a param is not passed.
        optional_min = fc.coalesce(fc.col("age") >= tool_param("min_age", IntegerType), fc.lit(True))
        optional_max = fc.coalesce(fc.col("age") <= tool_param("max_age", IntegerType), fc.lit(True))
        core_filter = df.filter(optional_min & optional_max)
        session.catalog.create_tool(
            "users_filter",
            "Filter users by age",
            core_filter,
            tool_params=[
                ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
                ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
            ]
        )
    """
    if isinstance(data_type, _LogicalType):
        raise ValidationError(f"Cannot use a logical type as a parameter type: {data_type}")

    return Column._from_logical_expr(UnresolvedLiteralExpr(data_type=data_type, parameter_name=parameter_name))
```

## udf

```
udf(f: Optional[Callable] = None, *, return_type: DataType)
```

A decorator or function for creating user-defined functions (UDFs) that can be applied to DataFrame rows.

Warning

UDFs cannot be serialized and are not supported in cloud execution.
User-defined functions contain arbitrary Python code that cannot be transmitted
to remote workers. For cloud compatibility, use built-in fenic functions instead.

When applied, UDFs will:
- Access `StructType` columns as Python dictionaries (`dict[str, Any]`).
- Access `ArrayType` columns as Python lists (`list[Any]`).
- Access primitive types (e.g., `int`, `float`, `str`) as their respective Python types.

Parameters:

- **`f`**
  (`Optional[Callable]`, default:
  `None`
  )
  –

  Python function to convert to UDF
- **`return_type`**
  (`DataType`)
  –

  Expected return type of the UDF. Required parameter.

UDF with primitive types

```
# UDF with primitive types
@udf(return_type=IntegerType)
def add_one(x: int):
    return x + 1

# Or
add_one = udf(lambda x: x + 1, return_type=IntegerType)
```

UDF with nested types

```
# UDF with nested types
@udf(return_type=StructType([StructField("value1", IntegerType), StructField("value2", IntegerType)]))
def example_udf(x: dict[str, int], y: list[int]):
    return {
        "value1": x["value1"] + x["value2"] + y[0],
        "value2": x["value1"] + x["value2"] + y[1],
    }
```

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def udf(f: Optional[Callable] = None, *, return_type: DataType):
    """A decorator or function for creating user-defined functions (UDFs) that can be applied to DataFrame rows.

    Warning:
        UDFs cannot be serialized and are not supported in cloud execution.
        User-defined functions contain arbitrary Python code that cannot be transmitted
        to remote workers. For cloud compatibility, use built-in fenic functions instead.

    When applied, UDFs will:
    - Access `StructType` columns as Python dictionaries (`dict[str, Any]`).
    - Access `ArrayType` columns as Python lists (`list[Any]`).
    - Access primitive types (e.g., `int`, `float`, `str`) as their respective Python types.

    Args:
        f: Python function to convert to UDF

        return_type: Expected return type of the UDF. Required parameter.

    Example: UDF with primitive types
        ```python
        # UDF with primitive types
        @udf(return_type=IntegerType)
        def add_one(x: int):
            return x + 1

        # Or
        add_one = udf(lambda x: x + 1, return_type=IntegerType)
        ```

    Example: UDF with nested types
        ```python
        # UDF with nested types
        @udf(return_type=StructType([StructField("value1", IntegerType), StructField("value2", IntegerType)]))
        def example_udf(x: dict[str, int], y: list[int]):
            return {
                "value1": x["value1"] + x["value2"] + y[0],
                "value2": x["value1"] + x["value2"] + y[1],
            }
        ```
    """

    def _create_udf(func: Callable) -> Callable:
        @wraps(func)
        def _udf_wrapper(*cols: ColumnOrName) -> Column:
            col_exprs = [Column._from_col_or_name(c)._logical_expr for c in cols]
            return Column._from_logical_expr(UDFExpr(func, col_exprs, return_type))

        return _udf_wrapper

    if _is_logical_type(return_type):
        raise NotImplementedError(f"return_type {return_type} is not supported for UDFs")

    if f is not None:
        return _create_udf(f)
    return _create_udf
```

## when

```
when(condition: Column, value: Column) -> Column
```

Evaluates a conditional expression (like if-then).

Evaluates a condition for each row and returns a value when true.
Can be chained with more .when() calls or finished with .otherwise().
All branches must return the same type.

Parameters:

- **`condition`**
  (`Column`)
  –

  Boolean expression to test
- **`value`**
  (`Column`)
  –

  Value to return when condition is True

Returns:

- **`Column`** ( `Column`
  ) –

  A when expression that can be chained with more conditions

Raises:

- `TypeMismatchError`
  –

  If the condition is not a boolean Column expression.

Example

```
# Simple if-then (returns null when false)
df.select(fc.when(col("age") >= 18, fc.lit("adult")))

# If-then-else
df.select(
    fc.when(col("age") >= 18, fc.lit("adult")).otherwise(fc.lit("minor"))
)

# Multiple conditions (if-elif-else)
df.select(
    when(col("score") >= 90, "A")
    .when(col("score") >= 80, "B")
    .when(col("score") >= 70, "C")
    .otherwise("F")
)
```

Note: Without .otherwise(), unmatched rows return null

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def when(condition: Column, value: Column) -> Column:
    """Evaluates a conditional expression (like if-then).

    Evaluates a condition for each row and returns a value when true.
    Can be chained with more .when() calls or finished with .otherwise().
    All branches must return the same type.

    Args:
        condition: Boolean expression to test
        value: Value to return when condition is True

    Returns:
        Column: A when expression that can be chained with more conditions

    Raises:
        TypeMismatchError: If the condition is not a boolean Column expression.

    Example:
        ```python
        # Simple if-then (returns null when false)
        df.select(fc.when(col("age") >= 18, fc.lit("adult")))

        # If-then-else
        df.select(
            fc.when(col("age") >= 18, fc.lit("adult")).otherwise(fc.lit("minor"))
        )

        # Multiple conditions (if-elif-else)
        df.select(
            when(col("score") >= 90, "A")
            .when(col("score") >= 80, "B")
            .when(col("score") >= 70, "C")
            .otherwise("F")
        )
        ```

    Note: Without .otherwise(), unmatched rows return null
    """
    return Column._from_logical_expr(
        WhenExpr(None, condition._logical_expr, value._logical_expr)
    )
```
