# Fenic Architecture Guide

This document provides a comprehensive overview of the fenic codebase architecture for new engineers joining the project.

## Overview

Fenic is a PySpark-inspired DataFrame framework specifically designed for AI and LLM applications. It combines traditional data processing with semantic operators powered by LLMs, enabling efficient batch inference across multiple model providers while maintaining a familiar DataFrame API.

The project is built with a hybrid architecture using Python for the API and logical planning layer, and Rust for performance-critical operations like text parsing and chunking.

## Core Design Principles

### Session-Centric Design
All operations flow through `Session.get_or_create()`. The session manages configuration, execution engine selection (local or cloud), and resource lifecycle. This is similar to SparkSession in PySpark.

### Lazy Evaluation
DataFrame operations build logical plans without immediate execution. Execution is triggered only by actions like `show()`, `collect()`, `to_polars()`, or `count()`. This allows for query optimization before execution.

### Logical vs Physical Separation
The framework separates what to compute (logical plans) from how to compute it (physical plans). Optimizer rules transform logical plans before the backend translates them into physical execution plans.

### Backend Abstraction
Fenic supports multiple execution backends:
- **Local Backend**: Uses Polars for in-memory execution (default)
- **Cloud Backend**: Uses gRPC to execute on typedef cloud infrastructure

The backend is selected via session configuration, allowing the same code to run locally or in the cloud.

## Directory Structure

```
fenic/
├── src/fenic/              # Main Python source code
│   ├── api/                # Public API layer
│   │   ├── dataframe/      # DataFrame, GroupedData, SemanticExtensions
│   │   ├── functions/      # Column functions (col, lit, semantic.*, etc.)
│   │   ├── io/             # DataFrameReader and DataFrameWriter
│   │   ├── session/        # Session and configuration classes
│   │   └── mcp/            # Model Context Protocol server
│   │
│   ├── core/               # Core logical layer
│   │   ├── _logical_plan/  # Logical plan nodes and expressions
│   │   │   ├── plans/      # Plan nodes (Filter, Join, Projection, etc.)
│   │   │   ├── expressions/# Expression nodes (Column, BinaryOp, etc.)
│   │   │   ├── optimizer/  # Optimization rules
│   │   │   └── signatures/ # Function signatures and validation
│   │   ├── types/          # Type system (DataType, Schema, etc.)
│   │   ├── _inference/     # LLM inference abstractions
│   │   └── _serde/         # Serialization for cloud backend
│   │
│   ├── _backends/          # Execution backends
│   │   ├── local/          # Local Polars-based backend
│   │   │   ├── physical_plan/    # Physical plan execution
│   │   │   ├── semantic_operators/ # LLM operator implementations
│   │   │   ├── polars_plugins/   # Custom Polars expressions
│   │   │   ├── transpiler/       # SQL transpilation
│   │   │   ├── catalog.py        # Local catalog management
│   │   │   ├── execution.py      # Execution engine
│   │   │   └── session_state.py  # Local session state
│   │   │
│   │   └── cloud/          # Cloud backend (gRPC-based)
│   │       ├── execution.py      # Cloud execution client
│   │       ├── catalog.py        # Cloud catalog client
│   │       └── session_state.py  # Cloud session state
│   │
│   ├── _inference/         # LLM provider implementations
│   │   ├── openai/         # OpenAI integration
│   │   ├── anthropic/      # Anthropic integration
│   │   ├── google/         # Google (Gemini) integration
│   │   ├── cohere/         # Cohere integration
│   │   ├── openrouter/     # OpenRouter integration
│   │   └── model_client.py # Unified model client interface
│   │
│   ├── _gen/               # Generated protobuf code (for cloud backend)
│   └── scripts/            # CLI scripts (fenic-serve)
│
├── rust/                   # Rust performance extensions
│   └── src/
│       ├── chunking/       # Text chunking with overlap
│       ├── markdown_json/  # Markdown parsing
│       ├── json/           # JSON manipulation (JQ-like)
│       ├── transcript/     # Transcript parsing (SRT, WebVTT)
│       ├── jinja/          # Jinja template rendering
│       ├── regex/          # Regex operations
│       └── dtypes/         # Custom data types
│
├── tests/                  # Test suite
├── examples/               # Example applications
├── docs/                   # Documentation source
├── protos/                 # Protocol buffer definitions
└── tools/                  # Development tools
```

## Architecture Layers

### Layer 1: API Layer (`src/fenic/api/`)

The API layer provides the user-facing interface. Key components:

#### Session (`api/session/`)
- Entry point to the framework
- Manages configuration and backend selection
- Creates DataFrames and provides catalog access
- Methods: `create_dataframe()`, `read.*()`, `table()`, `sql()`

#### DataFrame (`api/dataframe/dataframe.py`)
- Core data structure representing a lazy computation
- Provides PySpark-inspired operations: `select()`, `filter()`, `join()`, `group_by()`, etc.
- Supports method chaining for building complex transformations
- Each operation returns a new DataFrame with an updated logical plan
- Actions trigger execution: `show()`, `collect()`, `to_polars()`, `to_pandas()`, `count()`

#### SemanticExtensions (`api/dataframe/semantic_extensions.py`)
- Accessed via `df.semantic.*` property
- Provides semantic operations powered by LLMs:
  - `with_cluster_labels()`: K-means clustering on embeddings
  - `join()`: Natural language predicate-based joins
  - `sim_join()`: Similarity-based joins using embeddings

#### Functions (`api/functions/`)
- Column-level operations and transformations
- Standard functions: `col()`, `lit()`, `when()`, `coalesce()`, etc.
- Semantic functions via `semantic.*` namespace:
  - `semantic.extract()`: Extract structured data from text
  - `semantic.classify()`: Categorize text with examples
  - `semantic.map()`: Apply natural language transformations
  - `semantic.predicate()`: Create boolean predicates using natural language
  - `semantic.reduce()`: Aggregate grouped data with LLMs
  - `semantic.analyze_sentiment()`: Built-in sentiment analysis
- Special type constructors: `text.*`, `json.*`, `markdown.*`, `embedding.*`

#### IO Layer (`api/io/`)
- `DataFrameReader`: Read data from various sources (CSV, Parquet, JSON, managed tables)
- `DataFrameWriter`: Write DataFrames to different formats

### Layer 2: Core Logical Layer (`src/fenic/core/`)

The core layer defines the logical representation of computations.

#### Logical Plans (`core/_logical_plan/plans/`)

Logical plan nodes represent operations without specifying how to execute them:

- **Source Plans**: `InMemorySource`, `TableSource`, `SQL`
- **Transform Plans**: `Projection`, `Filter`, `Join`, `Sort`, `Explode`, `Unnest`
- **Aggregate Plans**: `Aggregate`, `GroupBy`
- **Semantic Plans**: `SemanticMap`, `SemanticExtract`, `SemanticClassify`, `SemanticJoin`, `SemanticCluster`, `SemanticPredicate`, `SemanticReduce`
- **Sink Plans**: `Show`, `Count`, `Write`
- **Union Plans**: `Union`

Each plan node:
- Contains references to child plan nodes (building a tree)
- Defines a `schema()` method for type inference
- Can be serialized for cloud execution (via protobuf)

#### Expressions (`core/_logical_plan/expressions/`)

Expression nodes represent column-level computations:

- `ColumnRef`: Reference to a column
- `LiteralExpr`: Constant value
- `BinaryOp`: Binary operations (+, -, *, /, ==, !=, <, >, etc.)
- `UnaryOp`: Unary operations (NOT, IS NULL, etc.)
- `FunctionCall`: Built-in function calls
- `CaseWhen`: Conditional expressions
- `Cast`: Type casting
- And many more specialized expressions

#### Optimizer (`core/_logical_plan/optimizer/`)

Transformation rules that optimize logical plans before execution:
- Constant folding
- Filter pushdown opportunities
- Expression simplification
- Dead code elimination

#### Type System (`core/types/`)

Strong typing throughout the framework:

- **Basic Types**: `StringType`, `IntegerType`, `FloatType`, `BooleanType`, `DateType`, `TimestampType`
- **Complex Types**: `ArrayType`, `StructType`, `MapType`
- **Special Types**: `EmbeddingType`, `MarkdownType`, `JsonType`, `TranscriptType`, `HtmlType`, `DocumentPathType`
- **Schema**: Collection of `ColumnField` objects with names and types
- Type validation at plan construction time prevents runtime type errors

### Layer 3: Backend Layer (`src/fenic/_backends/`)

Backends execute logical plans and return results.

#### Local Backend (`_backends/local/`)

Uses Polars for in-memory execution:

**Physical Plan Execution** (`physical_plan/`)
- Translates logical plans to Polars LazyFrame operations
- Handles complex operations like joins, aggregations, window functions
- Manages execution of semantic operators

**Semantic Operators** (`semantic_operators/`)
Each semantic operator has its own implementation:
- `extract.py`: Structured data extraction using LLMs
- `classify.py`: Text classification with few-shot examples
- `map.py`: Natural language transformations
- `predicate.py`: Boolean predicates using natural language
- `join.py`: Semantic joins using LLM predicates
- `sim_join.py`: Similarity-based joins using embeddings
- `cluster.py`: K-means clustering on embeddings
- `reduce.py`: LLM-powered aggregations
- `analyze_sentiment.py`: Sentiment analysis
- `summarize.py`: Text summarization

**Catalog** (`catalog.py`)
- Manages tables and views
- Handles table registration and lookup
- Manages system tables for lineage and metrics

**Execution Engine** (`execution.py`)
- Coordinates plan execution
- Manages caching and persistence
- Tracks query metrics (LLM calls, tokens, costs)

#### Cloud Backend (`_backends/cloud/`)

Executes plans on typedef cloud infrastructure:
- Serializes logical plans to protobuf
- Sends plans via gRPC to cloud execution service
- Receives results and deserializes back to Python
- Provides same API as local backend for transparency

### Layer 4: Inference Layer (`src/fenic/_inference/`)

Manages LLM interactions across multiple providers.

#### Model Client (`model_client.py`)
Unified interface for all LLM providers:
- Batches requests for efficiency
- Handles retries with exponential backoff
- Implements rate limiting strategies
- Tracks token usage and costs
- Supports both streaming and batch modes

#### Provider Implementations
Each provider has its own module:
- `openai/`: OpenAI GPT models
- `anthropic/`: Anthropic Claude models
- `google/`: Google Gemini models (Developer API and Vertex AI)
- `cohere/`: Cohere models (embeddings)
- `openrouter/`: OpenRouter proxy

Provider implementations handle:
- API authentication
- Request formatting
- Response parsing
- Error handling and retries
- Token counting (via tiktoken or provider-specific methods)

### Layer 5: Rust Extensions (`rust/src/`)

Performance-critical operations implemented in Rust and exposed via PyO3:

#### Text Processing
- **Chunking** (`chunking/`): Split text into overlapping chunks for long documents
- **Markdown** (`markdown_json/`): Parse markdown into structured JSON
- **Transcript** (`transcript/`): Parse SRT, WebVTT, and generic transcript formats
- **JSON** (`json/`): JQ-like JSON manipulation

#### Template Rendering
- **Jinja** (`jinja/`): Fast Jinja2-compatible template rendering for semantic operations

#### Utilities
- **Regex** (`regex/`): High-performance regex operations
- **DTypes** (`dtypes/`): Custom Polars data type implementations

These Rust modules are compiled into a Python extension (`_polars_plugins`) using Maturin.

## Data Flow

### Typical Query Execution Flow

1. **User Code**: User writes DataFrame operations
   ```python
   session = Session.get_or_create(SessionConfig(app_name="my_app"))
   df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})
   result = df.filter(col("age") > 25).select("name", "age")
   result.show()
   ```

2. **Logical Plan Construction**: Each operation builds a logical plan
   - `create_dataframe()` → `InMemorySource` plan
   - `.filter()` → `Filter` plan with `BinaryOp(col("age"), ">", 25)` expression
   - `.select()` → `Projection` plan

3. **Optimization**: Optimizer transforms the plan
   - Apply optimization rules (constant folding, etc.)
   - Validate types throughout the plan

4. **Backend Selection**: Session determines execution backend
   - Local: Continue to step 5
   - Cloud: Serialize plan to protobuf, send via gRPC, skip to step 7

5. **Physical Plan Generation** (Local Backend):
   - Translate logical plan to Polars LazyFrame operations
   - Generate execution strategy for semantic operators

6. **Execution** (Local Backend):
   - Execute Polars operations
   - For semantic operators, batch LLM requests
   - Apply rate limiting and retries
   - Track metrics (tokens, costs, latency)

7. **Result Materialization**:
   - Convert result to requested format (Polars, Pandas, Arrow, etc.)
   - Return QueryResult with data and metrics

8. **Display**: Format and print results
   - `.show()` formats as ASCII table
   - `.collect()` returns data with metrics

### Semantic Operation Flow

For semantic operations like `semantic.extract()`:

1. **Plan Node Creation**: Create `SemanticExtract` logical plan node
   - Stores schema, model configuration, template
   - Validates template and schema compatibility

2. **Physical Execution** (Local Backend):
   - Extract rows that need processing
   - Batch rows together (typically 10-100 rows per batch)
   - Render Jinja templates with row data
   - Call LLM provider via `model_client.py`
   - Parse structured outputs (JSON or Pydantic)
   - Handle retries for failed requests
   - Track token usage and costs

3. **Result Integration**:
   - Merge LLM results back into DataFrame
   - Validate output types match schema
   - Return updated DataFrame

## Configuration

### Session Configuration

The `SessionConfig` class controls behavior:

```python
SessionConfig(
    app_name="my_app",                    # Application identifier
    cloud=False,                          # Use cloud backend?
    semantic=SemanticConfig(              # Semantic operation config
        default_language_model=OpenAILanguageModel(model="gpt-4o-mini"),
        default_embedding_model=OpenAIEmbeddingModel(model="text-embedding-3-small"),
        max_concurrent_requests=10,       # LLM concurrency
        rate_limit_strategy="adaptive",   # Rate limiting approach
    ),
)
```

### Model Configuration

Each semantic operation can override the default model:

```python
df.semantic.extract(
    "content",
    schema=MySchema,
    model="gpt-4o-mini",           # Override default
    temperature=0.0,                # Model parameters
    max_tokens=1000,
)
```

## Testing

The test suite is organized by feature area:

- `tests/api/`: API layer tests
- `tests/core/`: Core logic tests (plans, expressions, types)
- `tests/backends/`: Backend-specific tests
- `tests/inference/`: LLM provider integration tests
- `tests/integration/`: End-to-end integration tests

Use pytest markers to control test execution:
- `@pytest.mark.cloud`: Tests requiring cloud backend (excluded by default)
- Run local tests: `just test` or `pytest tests/`
- Run cloud tests: `just test-cloud` or `pytest tests/ -m cloud`

## Development Workflow

### Setup

```bash
# Initial setup (installs dependencies and builds Rust extensions)
just setup

# Or manually:
uv sync                  # Install Python dependencies
just sync-rust           # Build Rust extensions
```

### Testing

```bash
# Run all local tests
just test

# Run specific test file
uv run pytest tests/path/to/test_file.py

# Run specific test
uv run pytest tests/path/to/test_file.py::test_function_name
```

### Linting and Formatting

```bash
# Check and format code
uv run ruff check .
uv run ruff format .
```

### Documentation

```bash
# Preview documentation locally
just preview-docs
```

### Building Rust Extensions

After modifying Rust code:

```bash
just sync-rust
```

This rebuilds the Rust extensions and makes them available to Python.

## Adding New Features

### Adding a New DataFrame Operation

1. Add method to `DataFrame` class in `api/dataframe/dataframe.py`
2. Create corresponding logical plan node in `core/_logical_plan/plans/`
3. Implement schema inference for the plan node
4. Add physical execution logic in `_backends/local/physical_plan/`
5. Add tests in `tests/api/` and `tests/backends/`

### Adding a New Semantic Operation

1. Add method to `SemanticExtensions` class in `api/dataframe/semantic_extensions.py`
2. Or add function to `api/functions/semantic.py` for function-style API
3. Create logical plan node in `core/_logical_plan/plans/`
4. Implement execution in `_backends/local/semantic_operators/`
5. Add examples to documentation
6. Add tests with mocked LLM responses

### Adding a New Column Function

1. Add function to appropriate module in `api/functions/`
2. Create expression node in `core/_logical_plan/expressions/` if needed
3. Add physical execution logic in `_backends/local/physical_plan/`
4. Add signature to `core/_logical_plan/signatures/` for type checking
5. Add tests and documentation

### Adding a New Rust Extension

1. Implement in `rust/src/` with appropriate module structure
2. Expose via PyO3 bindings in `rust/src/lib.rs`
3. Add Python wrapper in appropriate module
4. Build with `just sync-rust`
5. Add tests in both Rust (`rust/`) and Python (`tests/`)

## Key Design Patterns

### Builder Pattern
Logical plans and expressions use builder patterns for construction, allowing fluent APIs and method chaining.

### Factory Methods
DataFrames and plan nodes use factory methods (`_from_logical_plan()`, `from_session_state()`) to ensure proper initialization.

### Visitor Pattern
The `walker.py` module implements visitor pattern for traversing logical plan trees.

### Strategy Pattern
Different backends implement the same execution interface, allowing backend selection at runtime.

### Template Method Pattern
Base classes in semantic operators define common structure while subclasses implement specific LLM interactions.

## Important Conventions

### Code Style
- Use Google-style docstrings
- Follow PEP 8 for Python code
- Use type hints throughout
- Ruff is configured for linting and formatting

### Naming Conventions
- Plan nodes: PascalCase (e.g., `SemanticExtract`)
- Expressions: PascalCase (e.g., `BinaryOp`)
- Functions: snake_case (e.g., `create_dataframe()`)
- Private modules: Prefix with `_` (e.g., `_backends/`)

### API Compatibility
- Public API in `api/` should be stable
- Internal APIs in `core/` and `_backends/` can change
- Provide both snake_case and camelCase for PySpark compatibility

### Error Handling
- Use custom exceptions from `core.error`:
  - `ValidationError`: User input validation
  - `PlanError`: Logical plan issues
  - `ExecutionError`: Runtime execution problems
  - `SessionError`: Session configuration issues
  - `CatalogError`: Catalog operations

### Performance Considerations
- Prefer batch operations over row-by-row processing
- Use Rust extensions for CPU-intensive operations
- Lazy evaluation prevents unnecessary computation
- Cache intermediate results with `.persist()` or `.cache()`

## Resources

### Documentation
- User docs: https://docs.fenic.ai
- API reference: Auto-generated from docstrings
- Examples: `examples/` directory with Jupyter notebooks

### Community
- Discord: https://discord.gg/GdqF3J7huR
- GitHub Issues: https://github.com/typedef-ai/fenic/issues

### Related Tools
- Polars: https://pola.rs - Underlying DataFrame engine
- DuckDB: https://duckdb.org - SQL engine
- PySpark: https://spark.apache.org/docs/latest/api/python/ - API inspiration
- LanceDB: https://lancedb.com - Vector database for embeddings

## Common Patterns

### Reading Data

```python
# From various sources
df = session.read.csv("data.csv")
df = session.read.parquet("data.parquet")
df = session.read.json("data.json")
df = session.table("my_table")  # From catalog
df = session.create_dataframe(data)  # From Python objects
```

### Transformations

```python
# Standard operations
df.select("col1", "col2")
df.filter(col("age") > 25)
df.group_by("category").agg(count("*"))
df.join(other_df, on="id")
df.union(other_df)
df.sort("date")
```

### Semantic Operations

```python
# Extract structured data
df.with_column("parsed", 
    semantic.extract(
        "text_column",
        schema=MyPydanticModel,
        model="gpt-4o-mini"
    )
)

# Classify text
df.with_column("category",
    semantic.classify(
        "feedback",
        classes=["bug", "feature", "question"],
        examples=examples
    )
)

# Filter with natural language
df.filter(semantic.predicate("This {feedback} mentions UI issues"))

# Semantic join
df1.semantic.join(
    df2,
    predicate="{{left_on}} is similar to {{right_on}}",
    left_on=col("description"),
    right_on=col("title")
)
```

### Output Results

```python
# Various output formats
df.show()                # Print to console
polars_df = df.to_polars()
pandas_df = df.to_pandas()
arrow_table = df.to_arrow()
dict_data = df.to_pydict()
list_data = df.to_pylist()

# Get result with metrics
result = df.collect()
print(result.metrics.get_summary())
```

## Troubleshooting

### Common Issues

**LLM Provider Errors**: Set appropriate API key environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

**Rust Extension Build Failures**: Ensure Rust toolchain is installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Type Errors**: Use `df.explain()` to see the logical plan and verify schemas.

**Performance Issues**: Use `.persist()` to cache expensive operations that are reused.

## Next Steps

For new engineers:

1. **Start with Examples**: Run notebooks in `examples/` to understand usage patterns
2. **Read Core Files**: Study key files in this order:
   - `api/session/session.py`
   - `api/dataframe/dataframe.py`
   - `core/_logical_plan/plans/base.py`
   - `_backends/local/execution.py`
3. **Try Simple Changes**: Add a new column function or modify an existing operator
4. **Read Tests**: Tests demonstrate expected behavior and edge cases
5. **Explore Semantic Operators**: Understand how LLM integration works
6. **Join Community**: Ask questions on Discord

Welcome to the fenic project!
