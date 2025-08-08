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

To compile and install the Rust crate with Python bindings into your virtual environment:

```bash
just sync-rust
# or
uv run maturin develop --uv
```

This builds the Rust crate with Python bindings and makes it available inside the `.venv`.

To **only compile** the Rust crate _without_ Python bindings (e.g., for Rust unit tests), run this **from the `rust/` directory**:

````bash
cargo build --no-default-features

Add `--release` for optimized builds:

```bash
uv run maturin develop --uv --release
````

#### Documentation

To preview changes to the documentation from docstring or other changes:

```bash
just preview-docs
# without just
uv run --group docs mkdocs serve
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

Run all tests against a different **language model provider/model name**:

- OpenAI/gpt-4.1-nano (Default)

```bash
uv run pytest --language-model-provider=openai --language-model-name='gpt-4.1-nano'
```

- Anthropic/claude-3-5-haiku-latest

```bash
uv sync --extra=anthropic
uv run pytest --language-model-provider=anthropic --language-model-name='claude-3-5-haiku-latest'
```

- Google/2.0-flash-lite

```bash
uv sync --extra=google
uv run pytest --embedding-model-provider=google-developer --language-model-name='gemini-2.0-flash-lite'
```

Run all tests against a different **embeddings model provider/model name**:

- OpenAI/ (Default)

```bash
uv run pytest --embedding-model-provider=openai --embedding-model-name='text-embedding-3-small'
```

- Google/gemini-embedding-001

```bash
uv sync --extra=google
uv run pytest --embedding-model-provider=google-developer --embedding-model-name='gemini-embedding-001'
```

Run all tests for the **cloud backend**:

```bash
just test-cloud
# or
uv sync --extra=cloud
uv run pytest -m cloud tests
```

> ⚠️ Note: All tests require a valid OpenAI API key set in the environment variables.

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

## Adding model support

### Adding support for a provider

- update the `ModelProvider` enum
- create an instance of the ModelProviderClass (see `OpenAIModelProvider`)
- create an instance of ModelClient for LLM and/or Embedding clients (see `OpenAIBatchChatCompletionsClient` and `OpenAIBatchEmbeddingsClient`)
- create a provider specific instance of `ProfileManager`, if models from this provider will support configuration changes

### Adding a Language or Embedding Models

- Update `model_catalog.py`
- Add your model to the provider’s options
  - for e.g., add to `OpenAILanguageModelName` or `OpenAIEmbeddingModelName`
- Add the model to the catalog object with `_add_model_to_catalog()` with the appropriate `CompletionModelParameters` or `EmbeddingModelParameters`

### Adding provider-specific params using the model Profile

Profile level parameters allow the user to configure their model. You can add fields specific to a provider and subset of models by updating that provider’s profile class.

- A provider `Profile` is defined in the provider's model configuration class (see `OpenAILanguageModel`)
- Add provider specific the model’s `Profile` with the new `Field`
  - for e.g. look at [`config.py`](http://config.py) and see `reasoning_effort` defined in `OpenAILanguageModel.Profile`
- Update the resolved model profile in [`_resolved_session_config.py`](https://github.com/typedef-ai/fenic/pull/135/files#diff-454f845e61c21cebcf1003bf35368aead665489a72376d2ae0b031f3c26293c4)
  - for e.g.: `ResolvedOpenAIModelProfile`

### Adding model-specific support for params

Some params may not be supported or have strict requirements on the input values depending on the model. To add model-specific logic:

- Add a flag to global `CompletionModelParameters` in `model_catalog.py`, then set the flag for each appropriate model added to the catalog (see `_add_model_to_catalog()` calls).
  - this allows models to pass specific flags to the model and profile registry codepaths (for e.g.: `supports_reasoning`)
  - models can be configured to not accept Profiles at all (`supports_profiles`)

- To change or enforce values for a model profile param values based on the model params (for e.g.: to ignore the field for unsupported models), update that model’s `ProfileManager` class
  - use the model_parameters to set the parameters passed to the model client
  - for openai, look at `OpenAICompletionsProfileManager` in `openai_profile_manager.py`

---

## 🙋 Need Help?

Have questions or want to contribute? Join us on [Discord](https://discord.gg/GdqF3J7huR)!
