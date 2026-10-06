# OpenAI-compatible endpoints

Use `fc.OpenAICompatibleLanguageModel` or `fc.OpenAICompatibleEmbeddingModel`
for a user-declared model served by an OpenAI-compatible API. These classes
reuse fenic's OpenAI clients. They do not discover models or capabilities.

## Configure a model

The model name, API base URL, and model parameters are required. Set request
and token rate limits for your endpoint.

```python
import fenic as fc

config = fc.SessionConfig(
    app_name="local-inference",
    semantic=fc.SemanticConfig(
        language_models={
            "local": fc.OpenAICompatibleLanguageModel(
                model_name="qwen3-coder",
                base_url="http://localhost:8000/v1",
                rpm=100,
                tpm=100_000,
                model_parameters=fc.OpenAICompatibleLanguageModel.ModelParameters(
                    context_window_length=32_768,
                    max_output_tokens=4_096,
                ),
            )
        },
        embedding_models={
            "local": fc.OpenAICompatibleEmbeddingModel(
                model_name="bge-m3",
                base_url="http://localhost:8000/v1",
                rpm=100,
                tpm=100_000,
                model_parameters=fc.OpenAICompatibleEmbeddingModel.ModelParameters(
                    output_dimensions=1024,
                    max_input_size=8_192,
                ),
            )
        },
    ),
)
```

The OpenAI SDK still requires `OPENAI_API_KEY`. For a keyless server, set a
placeholder value. Session construction does not call the custom endpoint's
model-list API.

## Parameters and costs

Language models require `context_window_length` and `max_output_tokens`.
Embedding models require `output_dimensions` and `max_input_size`. Fenic
uses these values for planning and batching.

Token costs default to `$0.00`. To report a paid endpoint's costs, declare
`input_token_cost` and, for language models, `output_token_cost` and optionally
`cached_input_token_read_cost`. Each value is USD per token.

Fenic registers these models under `openai-compatible`, separate from `openai`.
A name matching an OpenAI catalog entry retains your declared limits and costs.
It does not change the OpenAI entry. Registration is process-wide, separately
for language and embedding models. Identical redeclarations are accepted.
Different parameters for the same name raise `ConfigurationError`, even when
the endpoint URL differs. Use different names for different model definitions.

## Existing OpenAI configurations

`fc.OpenAILanguageModel` and `fc.OpenAIEmbeddingModel` still require catalog
model names. Use their optional `base_url` for a proxy that forwards those
models. Typos retain the normal catalog validation error.

Token estimates use the existing tiktoken fallback for unknown names. They
may differ from a non-OpenAI model's tokenizer. This feature does not add
tokenizer hooks, backend-specific classes, or capability discovery.
