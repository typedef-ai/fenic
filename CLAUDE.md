# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fenic is a PySpark-inspired DataFrame framework specifically designed for AI and LLM applications. It provides semantic operators for LLM-powered data transformations, native support for unstructured data (markdown, transcripts, JSON), and efficient batch inference across multiple model providers (OpenAI, Anthropic, Google).

## Development Commands

### Setup and Dependencies

```bash
# Initial setup (installs dependencies and builds Rust extensions)
just setup

# Sync Python dependencies
just sync

# Build Rust extensions (required after Rust code changes)
just sync-rust

# Sync with cloud extras
just sync-cloud
```

### Testing

```bash
# Run local tests (excludes cloud tests)
just test
# or
just test-local

# Run cloud-related tests
just test-cloud

# Run specific test file
uv run pytest tests/path/to/test_file.py

# Run specific test
uv run pytest tests/path/to/test_file.py::test_function_name
```

### Linting and Formatting

```bash
# Ruff is configured for linting (Google-style docstrings)
# Auto-formats code examples in docstrings
uv run ruff check .
uv run ruff format .
```

### Documentation

```bash
# Preview documentation locally
just preview-docs
```

## Architecture Overview

### Directory Structure

- `src/fenic/`: Main Python source code
  - `api/`: Public API - Session, DataFrame, Functions, IO
  - `core/`: Core logic - logical plans, expressions, types, optimizer
  - `_backends/`: Execution backends (local using Polars, cloud using gRPC)
  - `_inference/`: LLM inference clients for different providers
- `rust/`: Rust source for performance-critical operations (text chunking, markdown/JSON parsing)
- `tests/`: Test suite (pytest-based)
- `examples/`: Example applications demonstrating fenic usage

### Key Architectural Components

1. **Session-Centric Design**: All operations flow through `Session.get_or_create()`. The session manages configuration, execution engine, and resource lifecycle.

2. **Lazy Evaluation**: DataFrame operations build logical plans without immediate execution. Execution is triggered by actions like `show()`, `collect()`, or `count()`.

3. **Logical vs Physical Plans**:
   - Logical plans (`src/fenic/core/_logical_plan/`) define what to compute
   - Physical plans (`src/fenic/_backends/local/physical_plan/`) define how to compute
   - Optimizer rules transform logical plans before execution

4. **Semantic Operators**: First-class support for LLM operations through the `semantic` namespace on DataFrames:
   - `semantic.map()`, `semantic.extract()`, `semantic.classify()`
   - `semantic.join()`, `semantic.group_by()`, `semantic.reduce()`

5. **Backend Abstraction**: Supports both local execution (using Polars) and cloud execution (using gRPC). Backend is selected via session configuration.

6. **Type System**: Strong typing with schema inference throughout. All operations validate types at plan construction time.

### Working with Semantic Operations

Semantic operations are accessed through the DataFrame's `semantic` property:

```python
# Example: Extract structured data from text
df.semantic.extract(
    "content_column",
    schema=MyPydanticModel,
    model="gpt-4o-mini"
)
```

### Adding New Features

1. **New DataFrame Operation**: Add method to `DataFrame` class, create corresponding logical plan node
2. **New Semantic Operation**: Add to `SemanticExtension` class in `api/dataframe/semantic_extension.py`
3. **New Function**: Add to appropriate module in `api/functions/`
4. **New Rust Extension**: Implement in `rust/src/`, expose via PyO3 bindings

### Testing Considerations

- Tests are organized by feature area in `tests/`
- Use `@pytest.mark.cloud` for tests requiring cloud backend
- Mock LLM responses when possible to avoid API calls in tests
- Integration tests in `examples/` demonstrate real-world usage

### Important Notes

- The project uses `uv` as the package manager (not pip or poetry)
- Rust extensions must be rebuilt after changes using `just sync-rust`
- LLM provider API keys must be set as environment variables
- The codebase follows Google-style docstrings
- Ruff is configured for linting with specific rules (see `ruff.toml`)
