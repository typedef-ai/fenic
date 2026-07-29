# Contributing to Fenic

Canonical HTML: https://docs.fenic.ai/latest/CONTRIBUTING/

Welcome! This guide will help you get set up for local development and testing.

---

## 📁 Directory Overview

```
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

### Requirements

- [`uv`](https://github.com/astral-sh/uv) — manages Python dependencies and environments
- A working **Rust toolchain**

> **Optional but recommended:** [`just`](https://just.systems/) for simpler task running

---

### One-Time Setup

From the project root:

```
just setup
# without just
uv sync
uv run maturin develop --uv
```

This will:

- Create a virtual environment
- Install all Python dev dependencies (including `maturin`)
- Build and install the Rust plugin as an editable Python package

---

### Making Changes

#### Python Code

```
just sync
# or
uv sync
```

#### Rust Code (PyO3 Plugin)

To compile and install the Rust crate with Python bindings into your virtual environment:

```
just sync-rust
# or
uv run maturin develop --uv
```

This builds the Rust crate with Python bindings and makes it available inside the `.venv`.

To **only compile** the Rust crate *without* Python bindings (e.g., for Rust unit tests), run this **from the `rust/` directory**:

```
cargo build --no-default-features

Add `--release` for optimized builds:

```bash
uv run maturin develop --uv --release
```

#### Documentation

To preview changes to the documentation from docstring or other changes:

```
just preview-docs
# without just
uv run --group docs mkdocs serve
```

#### Measuring Install Footprint

To compare the installed size of fenic with different extras, run:

```
uv run --env-file .env python tools/package_size_matrix.py
```

The tool creates temporary `uv` projects that depend on the local checkout, syncs each
project, and reports the resulting `site-packages` and `.venv` sizes along with the
largest installed distributions. By default it measures `core`, `pdf`, `cluster`,
`sim-join`, and `pdf,cluster,sim-join`.

To choose a custom matrix:

```
uv run --env-file .env python tools/package_size_matrix.py \
  --combo core \
  --combo google \
  --combo google,pdf \
  --combo pdf,cluster,sim-join
```

---

## ✅ Running Tests

### Python Tests

Run a specific test file:

```
uv run pytest tests/path/to/test_foo.py
```

Run all tests for the **local backend**:

```
just test
# or without just
uv run pytest -m "not cloud" tests
```

Run all tests against a different **language model provider/model name**:

- OpenAI/gpt-4.1-nano (Default)

```
uv run pytest --language-model-provider=openai --language-model-name='gpt-4.1-nano'
```

- Anthropic/claude-haiku-4-5

```
uv sync --extra=anthropic
uv run pytest --language-model-provider=anthropic --language-model-name='claude-haiku-4-5'
```

- Google/2.5-flash-lite

```
uv sync --extra=google
uv run pytest --language-model-provider=google-developer --language-model-name='gemini-2.5-flash-lite'
```

Run all tests against a different **embeddings model provider/model name**:

- OpenAI/ (Default)

```
uv run pytest --embedding-model-provider=openai --embedding-model-name='text-embedding-3-small'
```

- Google/gemini-embedding-001

```
uv sync --extra=google
uv run pytest --embedding-model-provider=google-developer --embedding-model-name='gemini-embedding-001'
```

Run all tests for the **cloud backend**:

```
just test-cloud
# or
uv sync --extra=cloud
uv run pytest -m cloud tests
```

> ⚠️ Note: All tests require a valid OpenAI/Anthropic API key set in the environment variables.

---

### Rust Tests

From the `rust/` directory:

```
cargo test --no-default-features
```

> Skipping default features avoids Python-specific linking, making it easier to test the Rust library independently of the Python bindings.

---

## 📓 Running Notebooks (VSCode / Cursor)

To run the demo notebooks:

1. Install the **Jupyter** extension in your editor.
2. Add `.venv` to the **Python: Venv Folders** setting in VSCode:
3. Open `Preferences: Open User Settings`
4. Go to Extensions → Python → **Python: Venv Folders**
5. Open a notebook and select the correct Python kernel from the virtual environment.
6. Restart the kernel if you make changes to the `fenic` source code.

---

## 🙋 Need Help?

Have questions or want to contribute? Join us on [Discord](https://discord.gg/GdqF3J7huR)!
