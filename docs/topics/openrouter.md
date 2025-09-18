# OpenRouter integration

This page describes how Fenic integrates with OpenRouter, how to configure it, and how to work around common issues (especially with structured outputs on non-frontier models).

## What is supported

- **Provider**: `ModelProvider.OPENROUTER`
- **Client**: Uses the OpenAI SDK with OpenRouter `base_url` and headers
- **Dynamic model loading**: Models are fetched from OpenRouter on first use and cached
- **Profiles**: Profile fields map to OpenRouter request parameters via `extra_body`
- **Rate limiting**: Adaptive strategy that respects `X-RateLimit-Reset` and provider-specific 429s

## Setup

### Requirements

- Set `OPENROUTER_API_KEY` in your environment
- _fenic_ automatically configures an OpenAI SDK client to point at OpenRouter (`base_url=https://openrouter.ai/api/v1`)

### Session configuration

You can pick OpenRouter via session configuration.

```python
from _fenic_.api.session.config import SessionConfig

config = SessionConfig(
    language_model=SessionConfig.OpenRouterLanguageModel(
        model_name="openai/gpt-4o",  # Any OpenRouter model id
        profiles={
            "default": SessionConfig.OpenRouterLanguageModel.Profile(
                models=["openai/gpt-4", "openai/gpt-4.1"],  # Fallback models, if the primary model is unavailable
                reasoning_effort="medium",  # For reasoning-capable models
                provider=SessionConfig.OpenRouterLanguageModel.Provider(
                    sort="latency",  # "price" | "throughput" | "latency"
                    only=["OpenAI"],  # Route only to specific providers
                    exclude=["Azure"] # exclude a provider from routing
                    order=["OpenAI", "Azure"] # try providers in a specific order for routing
                    quantizations=["unknown", "fp8", "bf16", "fp16" "fp32"] # only allow providers that offer the model with one of the following quantization levels
                    data_collection="deny" # only allow providers that do not collect/train on prompt data
                    max_prompt_price=5 # max price ($USD) per 1M input tokens
                    max_completion_price=10 # max price ($USD) per 1M output tokens
                ),
            ),
        },
        default_profile="default",
    )
)
```

## Dynamic model loading and the model catalog

- On first use of an OpenRouter model, Fenic fetches available models from your OpenRouter account, translates capabilities and pricing, and caches them.
- Available models depend on the providers you’ve enabled in OpenRouter. If you don’t see a model, ensure it’s available in your OpenRouter dashboard.

## Profile configuration

OpenRouter profiles support the following configuration options:

- **`models`**: List of fallback models to use if the primary model is unavailable
- **`reasoning_effort`**: For reasoning-capable models, set to `"low"`, `"medium"`, or `"high"` for OpenAI models.
- **`reasoning_max_tokens`**: Token budget for reasoning for Gemini/Anthropic models
- \*\*`
- **`provider`**: Provider routing preferences with these options:
  - `sort`: Route by `"price"`, `"throughput"`, or `"latency"`
  - `only`: List of providers to exclusively use (e.g., `["OpenAI", "Anthropic"]`)
  - `exclude`: List of providers to avoid
  - `order`: Specific provider order to try
  - `quantizations`: Allowed model quantizations (e.g., `["fp16", "bf16"]`)
  - `data_collection`: Set to `"allow"` or `"deny"` for data retention preferences
  - `max_prompt_price` / `max_completion_price`: Price limits per 1M tokens

## Structured outputs with OpenRouter

OpenRouter exposes multiple ways to request structured outputs, but support varies by model/provider. Frontier models (Anthropic, OpenAI, Gemini) have consistent support for structured outputs and will generally perform the same
as if one were using the provider's native client. Non-Frontier models can very wildly, depending on the size/quality of the model, the model's capabilities, and the quality of the provider's inference harness. When selecting a model for
a _fenic_ use case involving structured outputs, consider using a frontier or other popular model family (LLama3/4, Mistral, etc.). The more esoteric the choice of model, the less guarunteed your results will be.

1. **Pydantic Structured Outputs (via Response Format)** [Available when the target model lists `structured_outputs` in its `supported_parameters`]
   - This option is generally preferred, the JSON Schema corresponding to the provided Pydantic Model is sent to the model, which constrains its output to JSON and (in theory) coerces the response into the proper format.

2. **Forced Tool Calling (w/JSON Schema)**
   - If the model supports tool calling but not structured outputs, we achieve a similar result by asking the model to return a tool call with a specific json format for the input, which is derived from the provided Pydantic model.

If no providers for the model support either `structured_outputs` or `tools`, and a semantic operation that requires structured output (like `semantic.extract`) is used, _fenic_ will fail fast before starting to send requests.

## Rate limiting

Fenic uses an adaptive strategy for OpenRouter requests:

- Multiplicative backoff on 429s
- Honors `X-RateLimit-Reset` (temporary cooldown window)
- Accepts provider RPM hints when supplied
- Additive RPM increase after consecutive successes (when no hint is active)

## Troubleshooting

### Structured Outputs

#### Error: `No endpoints found that can handle the requested parameters.`

- This is likely occuring because the model configuration or the OpenRouter account attached to the API Key is forcing the use of a single provider or a subset of providers for a given model that do not support `structred_outputs`.
- Try setting `structured_output_strategy` in the model config to `prefer_tools`. This will force _fenic_ to use `Forced Tool Calling` instead of `structured_outputs`, which has broader compatibility.

#### Error `{'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{"error":{"message":"invalid schema for response format: response_format/json_schema: ..."}}\n', 'provider_name': 'Groq'}}` (or similar)

- Examine the error message and determine if a single provider is causing the issue. Try to `exclude` the provider from the model config to see if this resolves the problem.

#### Model is listed as supporting structured outputs, but generation performance is much slower than expected

- Examine in detail the model's overview page in OpenRouter -- somtimes models are listed as supporting `structured_outputs` because a single provider out of 5 supports the functionality, while the other
  4 providers only support `tools`. If this is the case, try setting `structured_output_strategy` in the model config to `prefer_tools`. This will force _fenic_ to use `Forced Tool Calling` instead of `structured_outputs`,
  allowing the request to be load-balanced across all of the model providers.

#### Generated JSON is malformed or inconsistently empty

- Examine the generated error messages closely, if a single provider is a common factor, try adding that provider to the `exclude` list in the provider routing preferences section of the model config.
- Reduce temperature for extraction steps.
- Try setting `structured_output_strategy` in the model config to `prefer_tools`. This will force _fenic_ to use `Forced Tool Calling` instead of `structured_outputs`,
- Try a different model with the same task -- if a different model works more consistently, there is likely some

### Rate Limiting

#### 429 / rate limit exceeded

- _fenic_ will attempt to reduce the rate at which the model is being called if multiple 429s are encountered.
- For some models (typically free models or smaller models with fewer providers, ex. `mistralai/mistral-small-3.2-24b-instruct`) , OpenRouter will throttle at a very low limit, like 100 RPM. _fenic_ will alert the user of this throttling in the logs.
- Persistent 429s usually mean the model is ill-suited for the scale of the use-case -- consider selecting a more popular model that has more capacity.

### General

#### Model Not available

- Ensure that the model indeed exists in the OpenRouter model list.
- Ensure that the OpenRouter account attached to the API Key is not excluding providers in its settings -- if the account excludes the only providers that offer a given model, the model will no longer be available for use.

#### Reasoning not working

- Verify the model supports reasoning by checking its capabilities.
- Use either `reasoning_effort` or `reasoning_max_tokens` in your profile, not both.

#### Bad Provider Behavior

- Use the `provider.only` field to restrict to specific providers, or `provider.exclude` to avoid problematic ones. To persist these settings and avoid needing to set them in the session each time, add them to the [OpenRouter Settings](https://openrouter.ai/settings/preferences) `Allowed Providers` and `Ignored Providers` lists.

## References

- OpenRouter docs: [API reference](https://openrouter.ai/docs/api-reference/chat-completion)
