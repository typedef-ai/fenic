# Contributing to Fenic

Welcome! This guide will help you get set up for local development and testing.

---

## 📁 Directory Overview

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

### Requirements

- [`uv`](https://github.com/astral-sh/uv) — manages Python dependencies and environments
- A working **Rust toolchain**

> **Optional but recommended:** [`just`](https://just.systems/) for simpler task running

---

### One-Time Setup

From the project root:

```bash
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

```bash
just sync
# or
uv sync
```

#### Rust Code (PyO3 Plugin)

```bash
just sync-rust
# or
uv run maturin develop --uv
```

Add `--release` for optimized builds:

```bash
uv run maturin develop --uv --release
```

---

## ✅ Running Tests

### Python Tests

Run a specific test file:

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
# or
uv sync --extra=cloud
uv run pytest -m cloud tests
```

---

### Rust Tests

From the `rust/` directory:

```bash
cargo test --no-default-features
```

> Skipping default features avoids Python-specific linking, making it easier to test the Rust library independently of the Python bindings.

---

## 📓 Running Notebooks (VSCode / Cursor)

To run the demo notebooks:

1. Install the **Jupyter** extension in your editor.
2. Add `.venv` to the **Python: Venv Folders** setting in VSCode:
   - Open `Preferences: Open User Settings`
   - Go to Extensions → Python → **Python: Venv Folders**
3. Open a notebook and select the correct Python kernel from the virtual environment.
4. Restart the kernel if you make changes to the `fenic` source code.

---

## 🙋 Need Help?

Have questions or want to contribute? Join us on [Discord](https://discord.gg/GdqF3J7huR)!
