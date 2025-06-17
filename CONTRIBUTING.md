# Contributing

## 📁 Directory Structure

The repository is organized as follows:

```bash
fenic/
├── src/fenic/            # Core library
│   ├── api/                  # Public API (DataFrame, Column, functions, session)
│   │   ├── dataframe/        # DataFrame implementation and extensions
│   │   ├── functions/        # Built-in and semantic functions
│   │   ├── session/          # Session management and configuration
│   │   └── types/            # Schema definitions and data types
│   ├── core/                 # Core framework components
│   │   └── _logical_plan/    # Logical plan representation for operators
│   │   ├── types/            # Core types (DataType, Schema, etc)
│   ├── _backends/            # Execution backends
│   │   ├── local/            # Local execution (Polars/DuckDB)
│   │   └── cloud/            # Cloud execution (Typedef)
│   └── _inference/           # LLM inference layer
├── rust/                     # Rust crates for performance-critical operations
├── tests/                    # Test suite mirroring source structure
└── examples/                 # Usage examples and demos
```

---

## 🛠️ Development Setup

Local development requires uv and a Rust toolchain.

> [!OPTIONAL]
> Not required, but recommended: [just](https://just.systems/)

### First-Time Setup

From the root of the repo:

```bash
just setup
# without just
uv sync
uv run maturin develop --uv
```

This will:

- Create a virtual environment
- Build the Rust crate
- Install Python dependencies
- Set up the package in editable mode

> This command also places the built dynamic Rust library inside `src/fenic`.

### Making Changes

- To apply changes made to Python code:

  ```bash
  just sync
  # without just
  uv sync
  ```

- To apply changes made to Rust code:

  ```bash
  just sync-rust
  # without just
  uv run maturin develop --uv
  ```

  Add `--release` or `-r` to build the Rust crate in release mode (better performance).

- To preview changes to the documentation from docstring or other changes:

  ```bash
  just preview-docs
  # without just
  uv run --group docs mkdocs serve
  ```

---

## ✅ Running Tests

Run an individual test file:

```bash
uv run pytest tests/path/to/test_foo.py
```

Run all tests for the **local backend**:

```bash
just test
# or without just
uv run pytest -m "not cloud" tests
```

Run all tests against a different **model provider/model name**:

- OpenAI/gpt-4.1-nano (Default)

```bash
uv run pytest --model-provider=openai --model-name='gpt-4.1-nano'
```

- Anthropic/claude-3-5-haiku-latest

```bash
uv sync --extra=anthropic
uv run pytest --model-provider=anthropic --model-name='claude-3-5-haiku-latest'
```

Run all tests for the **cloud backend**:

```bash
just test-cloud
```

> ⚠️ Note: All tests require a valid OpenAI/Anthropic API key set in the environment variables.

---

## 📓 Running Notebooks (VSCode / Cursor)

To run demo notebooks:

1. Install the **Jupyter** extension.
2. Add the `.venv` path to **Python: Venv Folders** in VSCode settings:
   - Open settings: `Preferences: Open User Settings`
   - Go to Extensions → Python → **Python: Venv Folders**
3. Open a notebook, select the correct kernel from your virtual environment, and run cells.

> Restart the kernel to reflect any code changes made to the `fenic` source.

---

Have questions or want to contribute? Let us know in the [Discord](https://discord.gg/aAvsqRW3)!
