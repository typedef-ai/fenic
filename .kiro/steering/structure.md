# Project Structure

## Top-Level Organization

```
fenic/
├── src/fenic/            # Core library source code
├── rust/                 # Rust performance plugins
├── tests/                # Test suite (mirrors src structure)
├── examples/             # Usage examples and demos
├── docs/                 # Documentation source
└── tools/                # Development utilities
```

## Source Code Structure (`src/fenic/`)

### Public API (`src/fenic/api/`)

- `dataframe/` - DataFrame implementation and semantic extensions
- `functions/` - Built-in functions (semantic, text, json, markdown)
- `session/` - Session management and configuration
- `io/` - DataFrameReader and DataFrameWriter
- `types.py` - Public type definitions

### Core Framework (`src/fenic/core/`)

- `_logical_plan/` - Logical plan representation for all operations
  - `expressions/` - Column expressions and functions
  - `plans/` - Plan nodes (filter, select, join, etc.)
  - `signatures/` - Function signature validation
  - `optimizer/` - Query optimization rules
- `types/` - Core type system (DataType, Schema, etc.)
- `_interfaces/` - Backend interfaces (execution, catalog, lineage)

### Execution Backends (`src/fenic/_backends/`)

- `local/` - Local execution using Polars/DuckDB
  - `physical_plan/` - Physical execution plans
  - `semantic_operators/` - LLM-powered operations
  - `transpiler/` - SQL generation
- `cloud/` - Cloud execution via Typedef platform

### LLM Integration (`src/fenic/_inference/`)

- `openai/`, `anthropic/`, `google/` - Provider-specific clients
- `common_openai/` - Shared OpenAI-compatible logic
- `model_catalog.py` - Model registry and configuration

## Rust Components (`rust/`)

Performance-critical operations implemented in Rust:

- `src/dtypes/` - Data type conversions and casting
- `src/json/` - JSON processing with JQ expressions
- `src/markdown_json/` - Markdown parsing and conversion
- `src/transcript/` - Transcript parsing (SRT, generic formats)
- `src/chunking/` - Text chunking with overlap

## Testing Structure (`tests/`)

Tests mirror the source structure:

- `tests/_backends/` - Backend-specific tests
- `tests/_logical_plan/` - Logical plan and optimization tests
- `tests/api/` - Public API tests
- `tests/examples/` - Example validation tests

## Key Architectural Patterns

### Lazy Evaluation

- Operations build logical plans, not immediate execution
- Plans executed only on actions (`.show()`, `.to_pandas()`, etc.)
- Query optimization happens before execution

### Backend Abstraction

- Clean separation between logical plans and physical execution
- Local backend uses Polars/DuckDB, cloud uses Typedef platform
- Semantic operations abstracted across backends

### Type System

- Pydantic-based type validation throughout
- Custom data types for unstructured data (Markdown, Transcript, etc.)
- Schema inference and validation

### Function Registry

- Centralized function signature validation
- Extensible function system for custom operations
- Type checking at plan construction time

## Import Conventions

- Public API imported from `fenic.api`
- Core types from `fenic.core`
- Internal modules use relative imports
- Backend-specific code isolated in `_backends/`
