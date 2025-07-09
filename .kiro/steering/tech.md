# Technology Stack

## Core Technologies

- **Python**: Primary language (3.10-3.12 supported)
- **Rust**: Performance-critical operations via PyO3 plugins
- **Polars**: Local DataFrame execution engine
- **DuckDB**: Local SQL execution and storage
- **LanceDB**: Vector database for embeddings
- **Pydantic**: Type safety and validation

## Build System

- **uv**: Python dependency management and virtual environments
- **maturin**: Rust-Python integration and building
- **just**: Task runner (optional but recommended)

## Key Dependencies

### Core Runtime

- `polars>=1.20.0` - DataFrame operations
- `duckdb>=1.1.3` - SQL execution
- `lancedb>=0.22.0` - Vector operations
- `openai>=1.82.0` - LLM inference
- `tiktoken>=0.9.0` - Token counting
- `sqlglot>=26.25.3` - SQL parsing and transpilation

### Optional Providers

- `anthropic>=0.52.2` - Anthropic Claude models
- `google-genai>=1.21.0` - Google Gemini models
- `fenic-cloud>=0.1.3` - Cloud backend

## Common Commands

### Setup

```bash
# Initial setup
just setup
# or manually:
uv sync
uv run maturin develop --uv
```

### Development

```bash
# Sync Python dependencies
just sync
# or: uv sync

# Build Rust components
just sync-rust
# or: uv run maturin develop --uv

# Build optimized Rust
uv run maturin develop --uv --release
```

### Testing

```bash
# Run local tests (default)
just test
# or: uv run pytest -m "not cloud" tests

# Run cloud tests
just test-cloud
# or: uv run pytest -m cloud tests

# Test with specific model provider
uv run pytest --model-provider=anthropic --model-name='claude-3-5-haiku-latest'

# When running tests with uv, you should pass the `--env-file .env` argument to ensure that we have access to the .env file in the root of the project
# This solves issues with missing API Keys in tests, should you encounter them.
uv run --env-file .env pytest ...
```

### Documentation

```bash
# Preview docs locally
just preview-docs
# or: uv run --group=docs mkdocs serve
```

### Rust Development

```bash
# From rust/ directory - compile only (no Python bindings)
cargo build --no-default-features
cargo test --no-default-features
```

## Environment Variables

Required for LLM providers:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GEMINI_API_KEY="your-google-api-key"
```

### Local Development

For local development, you can use a `.env` file in the project root and pass `--env-file .env` to pytest:

```bash
uv run pytest --env-file .env tests/
```
