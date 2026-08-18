# fenic.api.session.config

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/api/session/config/

Session configuration classes for Fenic.

Classes:

- **`AdaptiveTokenEstimationConfig`**
  –

  Tunes adaptive output-token reservation for rate limiting.
- **`AnthropicLanguageModel`**
  –

  Configuration for Anthropic language models.
- **`CloudConfig`**
  –

  Configuration for cloud-based execution.
- **`CloudExecutorSize`**
  –

  Enum defining available cloud executor sizes.
- **`CohereEmbeddingModel`**
  –

  Configuration for Cohere embedding models.
- **`GoogleDeveloperEmbeddingModel`**
  –

  Configuration for Google Developer embedding models.
- **`GoogleDeveloperLanguageModel`**
  –

  Configuration for Gemini models accessible through Google Developer AI Studio.
- **`GoogleVertexEmbeddingModel`**
  –

  Configuration for Google Vertex AI embedding models.
- **`GoogleVertexLanguageModel`**
  –

  Configuration for Google Vertex AI models.
- **`LLMResponseCacheConfig`**
  –

  Configuration for LLM response caching.
- **`OpenAIEmbeddingModel`**
  –

  Configuration for OpenAI embedding models.
- **`OpenAILanguageModel`**
  –

  Configuration for OpenAI language models.
- **`OpenRouterLanguageModel`**
  –

  Configuration for OpenRouter language models.
- **`SemanticConfig`**
  –

  Configuration for semantic language and embedding models.
- **`SessionConfig`**
  –

  Configuration for a user session.

## AdaptiveTokenEstimationConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.AdaptiveTokenEstimationConfig[AdaptiveTokenEstimationConfig]

              click fenic.api.session.config.AdaptiveTokenEstimationConfig href "" "fenic.api.session.config.AdaptiveTokenEstimationConfig"
```

Tunes adaptive output-token reservation for rate limiting.

Output-token reservations are learned from observed usage and clamped to the
request's max_completion_tokens ceiling, then corrected after each response
(settlement). Enabled by default.

Setting `enabled=False` disables adaptive *estimation* — reservations fall
back to the static worst-case ceiling instead of the learned distribution.
Settlement (reconciling the token bucket to actual usage after each response)
is **always on** regardless of this flag. It corrects the bucket in both
directions — refunding the over-reservation (the common case) and debiting
further when a request exceeds its reservation — and neither direction increases
429 risk: a refund only returns capacity the provider never charged, and a debit
only makes the limiter more conservative.

## AnthropicLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.AnthropicLanguageModel[AnthropicLanguageModel]

              click fenic.api.session.config.AnthropicLanguageModel href "" "fenic.api.session.config.AnthropicLanguageModel"
```

Configuration for Anthropic language models.

This class defines the configuration settings for Anthropic language models,
including model selection and separate rate limiting parameters for input and output tokens.

Attributes:

- **`model_name`**
  (`AnthropicLanguageModelName`)
  –

  The name of the Anthropic model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`input_tpm`**
  (`int`)
  –

  Input tokens per minute limit; must be greater than 0.
- **`output_tpm`**
  (`int`)
  –

  Output tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring an Anthropic model with separate input/output rate limits:

```
config = AnthropicLanguageModel(
    model_name="claude-haiku-4-5", rpm=100, input_tpm=100, output_tpm=100
)
```

Configuring an Anthropic model with profiles:

```
config = SessionConfig(
    semantic=SemanticConfig(
        language_models={
            "claude": AnthropicLanguageModel(
                model_name="claude-sonnet-4-6",
                rpm=100,
                input_tpm=100,
                output_tpm=100,
                profiles={
                    "thinking_disabled": AnthropicLanguageModel.Profile(),
                    "fast": AnthropicLanguageModel.Profile(thinking_token_budget=1024),
                    "thorough": AnthropicLanguageModel.Profile(thinking_token_budget=4096)
                },
                default_profile="fast"
            )
        },
        default_language_model="claude"
)

# Using the default "fast" profile for the "claude" model
semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias="claude")

# Using the "thorough" profile for the "claude" model
semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias=ModelAlias(name="claude", profile="thorough"))
```

Classes:

- **`Profile`**
  –

  Anthropic-specific profile configurations.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.AnthropicLanguageModel.Profile[Profile]

              click fenic.api.session.config.AnthropicLanguageModel.Profile href "" "fenic.api.session.config.AnthropicLanguageModel.Profile"
```

Anthropic-specific profile configurations.

This class defines profile configurations for Anthropic models, allowing
different thinking and effort settings to be applied to the same model.

Attributes:

- **`thinking_token_budget`**
  (`Optional[int]`)
  –

  Provide a default thinking budget in tokens. If not provided,
  thinking will be disabled for the profile. The minimum token budget supported by Anthropic is 1024 tokens.
  For Claude models that use adaptive thinking, use `effort` instead.
- **`effort`**
  (`Optional[AnthropicReasoningEffortType]`)
  –

  Provider-native Anthropic effort level. Supported values vary by model:
  low, medium, high, xhigh, and max.
  On adaptive-thinking models the thinking budget shares the request's
  output token window rather than being reserved on top of it, so very
  high effort levels can consume part of the visible completion budget.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Note

If `thinking_token_budget` or adaptive `effort` enables thinking,
`temperature` cannot be customized -- any changes to `temperature`
will be ignored. Effort-only profiles on non-adaptive models
configure Anthropic `output_config` without enabling thinking, so
custom `temperature` remains available when the model supports it.

Example

Configuring a profile with a thinking budget:

```
profile = AnthropicLanguageModel.Profile(thinking_token_budget=2048)
```

Configuring a profile with a large thinking budget:

```
profile = AnthropicLanguageModel.Profile(thinking_token_budget=8192)
```

Configuring a profile with effort:

```
profile = AnthropicLanguageModel.Profile(effort="xhigh")
```

## CloudConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.CloudConfig[CloudConfig]

              click fenic.api.session.config.CloudConfig href "" "fenic.api.session.config.CloudConfig"
```

Configuration for cloud-based execution.

This class defines settings for running operations in a cloud environment,
allowing for scalable and distributed processing of language model operations.

Attributes:

- **`size`**
  (`Optional[CloudExecutorSize]`)
  –

  Size of the cloud executor instance.
  If None, the default size will be used.

Example

Configuring cloud execution with a specific size:

```
config = CloudConfig(size=CloudExecutorSize.MEDIUM)
```

Using default cloud configuration:

```
config = CloudConfig()
```

## CloudExecutorSize

Bases: `str`, `Enum`

```
              flowchart TD
              fenic.api.session.config.CloudExecutorSize[CloudExecutorSize]

              click fenic.api.session.config.CloudExecutorSize href "" "fenic.api.session.config.CloudExecutorSize"
```

Enum defining available cloud executor sizes.

This enum represents the different size options available for cloud-based
execution environments.

Attributes:

- **`SMALL`**
  –

  Small instance size.
- **`MEDIUM`**
  –

  Medium instance size.
- **`LARGE`**
  –

  Large instance size.
- **`XLARGE`**
  –

  Extra large instance size.

## CohereEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.CohereEmbeddingModel[CohereEmbeddingModel]

              click fenic.api.session.config.CohereEmbeddingModel href "" "fenic.api.session.config.CohereEmbeddingModel"
```

Configuration for Cohere embedding models.

This class defines the configuration settings for Cohere embedding models,
including model selection and rate limiting parameters.

Attributes:

- **`model_name`**
  (`CohereEmbeddingModelName`)
  –

  The name of the Cohere model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit for the model.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit for the model.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional dictionary of profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  Default profile name to use if none specified.

Example

Configuring a Cohere embedding model with profiles:

```
cohere_config = CohereEmbeddingModel(
    model_name="embed-v4.0",
    rpm=100,
    tpm=50_000,
    profiles={
        "high_dim": CohereEmbeddingModel.Profile(
            embedding_dimensionality=1536, embedding_task_type="search_document"
        ),
        "classification": CohereEmbeddingModel.Profile(
            embedding_dimensionality=1024, embedding_task_type="classification"
        ),
    },
    default_profile="high_dim",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Cohere embedding models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.CohereEmbeddingModel.Profile[Profile]

              click fenic.api.session.config.CohereEmbeddingModel.Profile href "" "fenic.api.session.config.CohereEmbeddingModel.Profile"
```

Profile configurations for Cohere embedding models.

This class defines profile configurations for Cohere embedding models, allowing
different output dimensionality and task type settings to be applied to the same model.

Attributes:

- **`output_dimensionality`**
  (`Optional[int]`)
  –

  The dimensionality of the embedding created by this model.
  If not provided, the model will use its default dimensionality.
- **`input_type`**
  (`CohereEmbeddingTaskType`)
  –

  The type of input text (search_query, search_document, classification, clustering)

Example

Configuring a profile with custom dimensionality:

```
profile = CohereEmbeddingModel.Profile(output_dimensionality=1536)
```

Configuring a profile with default settings:

```
profile = CohereEmbeddingModel.Profile()
```

## GoogleDeveloperEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleDeveloperEmbeddingModel[GoogleDeveloperEmbeddingModel]

              click fenic.api.session.config.GoogleDeveloperEmbeddingModel href "" "fenic.api.session.config.GoogleDeveloperEmbeddingModel"
```

Configuration for Google Developer embedding models.

This class defines the configuration settings for Google embedding models available in Google Developer AI Studio,
including model selection and rate limiting parameters. These models are accessible using a GOOGLE_API_KEY environment variable.

Attributes:

- **`model_name`**
  (`GoogleDeveloperEmbeddingModelName`)
  –

  The name of the Google Developer embedding model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Developer embedding model with rate limits:

```
config = GoogleDeveloperEmbeddingModelConfig(
    model_name="gemini-embedding-001", rpm=100, tpm=1000
)
```

Configuring a Google Developer embedding model with profiles:

```
config = GoogleDeveloperEmbeddingModelConfig(
    model_name="gemini-embedding-001",
    rpm=100,
    tpm=1000,
    profiles={
        "default": GoogleDeveloperEmbeddingModelConfig.Profile(),
        "high_dim": GoogleDeveloperEmbeddingModelConfig.Profile(
            output_dimensionality=3072
        ),
    },
    default_profile="default",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Developer embedding models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleDeveloperEmbeddingModel.Profile[Profile]

              click fenic.api.session.config.GoogleDeveloperEmbeddingModel.Profile href "" "fenic.api.session.config.GoogleDeveloperEmbeddingModel.Profile"
```

Profile configurations for Google Developer embedding models.

This class defines profile configurations for Google embedding models, allowing
different output dimensionality and task type settings to be applied to the same model.

Attributes:

- **`output_dimensionality`**
  (`Optional[int]`)
  –

  The dimensionality of the embedding created by this model.
  If not provided, the model will use its default dimensionality.
- **`task_type`**
  (`GoogleEmbeddingTaskType`)
  –

  The type of task for the embedding model.

Example

Configuring a profile with custom dimensionality:

```
profile = GoogleDeveloperEmbeddingModelConfig.Profile(
    output_dimensionality=3072
)
```

Configuring a profile with default settings:

```
profile = GoogleDeveloperEmbeddingModelConfig.Profile()
```

## GoogleDeveloperLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleDeveloperLanguageModel[GoogleDeveloperLanguageModel]

              click fenic.api.session.config.GoogleDeveloperLanguageModel href "" "fenic.api.session.config.GoogleDeveloperLanguageModel"
```

Configuration for Gemini models accessible through Google Developer AI Studio.

This class defines the configuration settings for Google Gemini models available in Google Developer AI Studio,
including model selection and rate limiting parameters. These models are accessible using a GOOGLE_API_KEY environment variable.

Attributes:

- **`model_name`**
  (`GoogleDeveloperLanguageModelName`)
  –

  The name of the Google Developer model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Developer model with rate limits:

```
config = GoogleDeveloperLanguageModel(
    model_name="gemini-2.5-flash",
    rpm=100,
    tpm=1000
)
```

Configuring a reasoning Google Developer model with profiles:

```
config = GoogleDeveloperLanguageModel(
    model_name="gemini-2.5-flash",
    rpm=100,
    tpm=1000,
    profiles={
        "thinking_disabled": GoogleDeveloperLanguageModel.Profile(),
        "fast": GoogleDeveloperLanguageModel.Profile(
            thinking_token_budget=1024
        ),
        "thorough": GoogleDeveloperLanguageModel.Profile(
            thinking_token_budget=8192
        ),
    },
    default_profile="fast",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Developer models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleDeveloperLanguageModel.Profile[Profile]

              click fenic.api.session.config.GoogleDeveloperLanguageModel.Profile href "" "fenic.api.session.config.GoogleDeveloperLanguageModel.Profile"
```

Profile configurations for Google Developer models.

This class defines profile configurations for Google Gemini models, allowing
different thinking/reasoning settings to be applied to the same model.

Attributes:

- **`thinking_token_budget`**
  (`Optional[int]`)
  –

  If configuring a reasoning model, provide a thinking budget in tokens.
  If not provided, or if set to 0, thinking will be disabled for the profile (not supported on gemini-2.5-pro).
  To have the model automatically determine a thinking budget based on the complexity of
  the prompt, set this to -1. Note that Gemini models take this as a suggestion -- and not a hard limit.
  It is very possible for the model to generate far more thinking tokens than the suggested budget, and for the
  model to generate reasoning tokens even if thinking is disabled.
  Note: For gemini-3 models, use thinking_level instead.
- **`thinking_level`**
  (`Optional[ThinkingLevelType]`)
  –

  For gemini-3+ models, set the thinking level to high, medium, low, or minimal.
  This parameter is mutually exclusive with thinking_token_budget.
- **`media_resolution`**
  (`Optional[MediaResolutionType]`)
  –

  For gemini-3+ models, set the media resolution for PDF processing.
  Can be "low", "medium", or "high". Affects token cost per page.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Example

Configuring a profile with a fixed thinking budget (gemini-2.5 and earlier):

```
profile = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=4096)
```

Configuring a profile with thinking level (gemini-3+):

```
profile = GoogleDeveloperLanguageModel.Profile(thinking_level="high")
```

## GoogleVertexEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleVertexEmbeddingModel[GoogleVertexEmbeddingModel]

              click fenic.api.session.config.GoogleVertexEmbeddingModel href "" "fenic.api.session.config.GoogleVertexEmbeddingModel"
```

Configuration for Google Vertex AI embedding models.

This class defines the configuration settings for Google embedding models available in Google Vertex AI,
including model selection and rate limiting parameters. These models are accessible using Google Cloud credentials.

Attributes:

- **`model_name`**
  (`GoogleVertexEmbeddingModelName`)
  –

  The name of the Google Vertex embedding model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Vertex embedding model with rate limits:

```
embedding_model = GoogleVertexEmbeddingModel(
    model_name="gemini-embedding-001", rpm=100, tpm=1000
)
```

Configuring a Google Vertex embedding model with profiles:

```
embedding_model = GoogleVertexEmbeddingModel(
    model_name="gemini-embedding-001",
    rpm=100,
    tpm=1000,
    profiles={
        "default": GoogleVertexEmbeddingModel.Profile(),
        "high_dim": GoogleVertexEmbeddingModel.Profile(
            output_dimensionality=3072
        ),
    },
    default_profile="default",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Vertex embedding models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleVertexEmbeddingModel.Profile[Profile]

              click fenic.api.session.config.GoogleVertexEmbeddingModel.Profile href "" "fenic.api.session.config.GoogleVertexEmbeddingModel.Profile"
```

Profile configurations for Google Vertex embedding models.

This class defines profile configurations for Google embedding models, allowing
different output dimensionality and task type settings to be applied to the same model.

Attributes:

- **`output_dimensionality`**
  (`Optional[int]`)
  –

  The dimensionality of the embedding created by this model.
  If not provided, the model will use its default dimensionality.
- **`task_type`**
  (`GoogleEmbeddingTaskType`)
  –

  The type of task for the embedding model.

Example

Configuring a profile with custom dimensionality:

```
profile = GoogleVertexEmbeddingModelConfig.Profile(
    output_dimensionality=3072
)
```

Configuring a profile with default settings:

```
profile = GoogleVertexEmbeddingModelConfig.Profile()
```

## GoogleVertexLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleVertexLanguageModel[GoogleVertexLanguageModel]

              click fenic.api.session.config.GoogleVertexLanguageModel href "" "fenic.api.session.config.GoogleVertexLanguageModel"
```

Configuration for Google Vertex AI models.

This class defines the configuration settings for Google Gemini models available in Google Vertex AI,
including model selection and rate limiting parameters. These models are accessible using Google Cloud credentials.

Attributes:

- **`model_name`**
  (`GoogleVertexLanguageModelName`)
  –

  The name of the Google Vertex model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Example

Configuring a Google Vertex model with rate limits:

```
config = GoogleVertexLanguageModel(
    model_name="gemini-2.5-flash", rpm=100, tpm=1000
)
```

Configuring a reasoning Google Vertex model with profiles:

```
config = GoogleVertexLanguageModel(
    model_name="gemini-2.5-flash",
    rpm=100,
    tpm=1000,
    profiles={
        "thinking_disabled": GoogleVertexLanguageModel.Profile(),
        "fast": GoogleVertexLanguageModel.Profile(thinking_token_budget=1024),
        "thorough": GoogleVertexLanguageModel.Profile(
            thinking_token_budget=8192
        ),
    },
    default_profile="fast",
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for Google Vertex models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.GoogleVertexLanguageModel.Profile[Profile]

              click fenic.api.session.config.GoogleVertexLanguageModel.Profile href "" "fenic.api.session.config.GoogleVertexLanguageModel.Profile"
```

Profile configurations for Google Vertex models.

This class defines profile configurations for Google Gemini models, allowing
different thinking/reasoning settings to be applied to the same underlying model.

Attributes:

- **`thinking_token_budget`**
  (`Optional[int]`)
  –

  If configuring a reasoning model, provide a thinking budget in tokens.
  If not provided, or if set to 0, thinking will be disabled for the profile (not supported on gemini-2.5-pro).
  To have the model automatically determine a thinking budget based on the complexity of
  the prompt, set this to -1. Note that Gemini models take this as a suggestion -- and not a hard limit.
  It is very possible for the model to generate far more thinking tokens than the suggested budget, and for the
  model to generate reasoning tokens even if thinking is disabled.
  Note: For gemini-3 models, use thinking_level instead.
- **`thinking_level`**
  (`Optional[ThinkingLevelType]`)
  –

  For gemini-3+ models, set the thinking level to high, medium, low, or minimal.
  This parameter is mutually exclusive with thinking_token_budget.
- **`media_resolution`**
  (`Optional[MediaResolutionType]`)
  –

  For gemini-3+ models, set the media resolution for PDF processing.
  Can be "low", "medium", or "high". Affects token cost per page.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Example

Configuring a profile with a fixed thinking budget (gemini-2.5 and earlier):

```
profile = GoogleVertexLanguageModel.Profile(thinking_token_budget=4096)
```

Configuring a profile with thinking level (gemini-3+):

```
profile = GoogleVertexLanguageModel.Profile(thinking_level="high")
```

## LLMResponseCacheConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.LLMResponseCacheConfig[LLMResponseCacheConfig]

              click fenic.api.session.config.LLMResponseCacheConfig href "" "fenic.api.session.config.LLMResponseCacheConfig"
```

Configuration for LLM response caching.

LLM response caching stores the results of language model API calls to reduce
costs and improve performance for repeated queries. This is distinct from
DataFrame caching (the `.cache()` operator).

Attributes:

- **`enabled`**
  –

  Whether caching is enabled (default: True).
- **`backend`**
  (`CacheBackend`)
  –

  Cache backend to use (default: LOCAL).
- **`ttl`**
  (`str`)
  –

  Time-to-live duration string (default: "1h").
  Format:  where unit is s/m/h/d.
  Examples: "30s", "15m", "2h", "7d".
  Maximum: 30 days, Minimum: 1 second.
- **`max_size_mb`**
  (`int`)
  –

  Maximum cache size in MB before LRU eviction (default: 128MB).
- **`namespace`**
  (`str`)
  –

  Cache namespace for isolation (default: "default").

Example

Basic configuration within SemanticConfig:

```
config = SessionConfig(
    app_name="my_app",
    semantic=SemanticConfig(
        language_models={
            "gpt": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=1000)
        },
        llm_response_cache=LLMResponseCacheConfig(
            enabled=True,
            ttl="1h",
            max_size_mb=1000,
        )
    )
)
```

Custom TTL and larger cache:

```
llm_response_cache=LLMResponseCacheConfig(
    enabled=True,
    ttl="7d",  # 7 days
    max_size_mb=5000,
)
```

Disabled caching:

```
llm_response_cache=LLMResponseCacheConfig(enabled=False)
```

Methods:

- **`ttl_seconds`**
  –

  Convert TTL string to seconds.
- **`validate_ttl`**
  –

  Validate TTL duration string format.

### ttl_seconds

```
ttl_seconds() -> int
```

Convert TTL string to seconds.

Returns:

- `int`
  –

  TTL duration in seconds.

Raises:

- `ValueError`
  –

  If TTL format is invalid.

Source code in `src/fenic/api/session/config.py`

```
def ttl_seconds(self) -> int:
    """Convert TTL string to seconds.

    Returns:
        TTL duration in seconds.

    Raises:
        ValueError: If TTL format is invalid.
    """
    pattern = r"^(\d+)([smhd])$"
    match = re.match(pattern, self.ttl.lower())

    if not match:
        raise ValueError(f"Invalid TTL format: '{self.ttl}'")

    value, unit = match.groups()
    value = int(value)

    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]
```

### validate_ttl

```
validate_ttl(v: str) -> str
```

Validate TTL duration string format.

Format:  where unit is s/m/h/d
Examples: "30s", "15m", "2h", "7d"

Parameters:

- **`v`**
  (`str`)
  –

  TTL duration string to validate.

Returns:

- `str`
  –

  The validated TTL string.

Raises:

- `ValueError`
  –

  If format is invalid or value is out of range.

Source code in `src/fenic/api/session/config.py`

```
@field_validator("ttl")
@classmethod
def validate_ttl(cls, v: str) -> str:
    """Validate TTL duration string format.

    Format: <number><unit> where unit is s/m/h/d
    Examples: "30s", "15m", "2h", "7d"

    Args:
        v: TTL duration string to validate.

    Returns:
        The validated TTL string.

    Raises:
        ValueError: If format is invalid or value is out of range.
    """
    pattern = r"^(\d+)([smhd])$"
    match = re.match(pattern, v.lower())

    if not match:
        raise ValueError(
            f"Invalid TTL format: '{v}'. "
            "Expected: <number><unit> where unit is s/m/h/d. "
            "Examples: '30m', '2h', '1d'"
        )

    value, unit = match.groups()
    value = int(value)

    # Validate ranges
    if unit == "s" and value < 1:
        raise ValueError("TTL must be at least 1 second")
    if unit == "h" and value > 720:  # 30 days
        raise ValueError("TTL cannot exceed 720 hours (30 days)")
    if unit == "d" and value > 30:
        raise ValueError("TTL cannot exceed 30 days")

    return v
```

## OpenAIEmbeddingModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.OpenAIEmbeddingModel[OpenAIEmbeddingModel]

              click fenic.api.session.config.OpenAIEmbeddingModel href "" "fenic.api.session.config.OpenAIEmbeddingModel"
```

Configuration for OpenAI embedding models.

This class defines the configuration settings for OpenAI embedding models,
including model selection and rate limiting parameters.

Attributes:

- **`model_name`**
  (`OpenAIEmbeddingModelName`)
  –

  The name of the OpenAI embedding model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.

Example

Configuring an OpenAI embedding model with rate limits:

```
config = OpenAIEmbeddingModel(
    model_name="text-embedding-3-small", rpm=100, tpm=100
)
```

## OpenAILanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.OpenAILanguageModel[OpenAILanguageModel]

              click fenic.api.session.config.OpenAILanguageModel href "" "fenic.api.session.config.OpenAILanguageModel"
```

Configuration for OpenAI language models.

This class defines the configuration settings for OpenAI language models,
including model selection and rate limiting parameters.

Attributes:

- **`model_name`**
  (`OpenAILanguageModelName`)
  –

  The name of the OpenAI model to use.
- **`rpm`**
  (`int`)
  –

  Requests per minute limit; must be greater than 0.
- **`tpm`**
  (`int`)
  –

  Tokens per minute limit; must be greater than 0.
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Optional mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The name of the default profile to use if profiles are configured.

Note

When using an o-series or gpt5 reasoning model without specifying a reasoning effort in
a Profile, the `reasoning_effort` will default to `low` (for o-series models) or `minimal`
(for gpt5 models).

Example

Configuring an OpenAI language model with rate limits:

```
config = OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100)
```

Configuring an OpenAI model with profiles:

```
config = OpenAILanguageModel(
    model_name="o4-mini",
    rpm=100,
    tpm=100,
    profiles={
        "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
        "thorough": OpenAILanguageModel.Profile(reasoning_effort="high"),
    },
    default_profile="fast",
)
```

Using a profile in a semantic operation:

```
config = SemanticConfig(
    language_models={
        "o4": OpenAILanguageModel(
            model_name="o4-mini",
            rpm=1_000,
            tpm=1_000_000,
            profiles={
                "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                "thorough": OpenAILanguageModel.Profile(
                    reasoning_effort="high"
                ),
            },
            default_profile="fast",
        )
    },
    default_language_model="o4",
)

# Will use the default "fast" profile for the "o4" model
semantic.map(
    instruction="Construct a formal proof of the {hypothesis}.",
    model_alias="o4",
)

# Will use the "thorough" profile for the "o4" model
semantic.map(
    instruction="Construct a formal proof of the {hypothesis}.",
    model_alias=ModelAlias(name="o4", profile="thorough"),
)
```

Classes:

- **`Profile`**
  –

  OpenAI-specific profile configurations.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.OpenAILanguageModel.Profile[Profile]

              click fenic.api.session.config.OpenAILanguageModel.Profile href "" "fenic.api.session.config.OpenAILanguageModel.Profile"
```

OpenAI-specific profile configurations.

This class defines profile configurations for OpenAI models, allowing a user to reference
the same underlying model in semantic operations with different settings.

Attributes:

- **`reasoning_effort`**
  (`Optional[ReasoningEffort]`)
  –

  Provide a reasoning effort. Only for gpt5 and o-series models.
  Valid values: 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'.
  - For gpt-5.6 models: supports 'xhigh' and 'max'
  - For gpt-5.5 models: defaults to 'medium', supports 'xhigh'
  - For gpt-5.4 models: defaults to 'none' (disabled reasoning), supports 'xhigh'
  - For gpt-5.1 and gpt-5.2 models: defaults to 'none' (disabled reasoning), does NOT support 'minimal' or 'xhigh'
  - For gpt-5 models: defaults to 'minimal', does NOT support 'none'
  - For o-series models: defaults to 'low', does NOT support 'none' or 'minimal'
- **`verbosity`**
  (`Optional[Verbosity]`)
  –

  Provide a verbosity level. Only for gpt5/gpt5.1 models.

Raises:

- `ConfigurationError`
  –

  If a profile is set with parameters that are not supported by the model.

Note

When using an o-series or gpt5 reasoning model with reasoning enabled, the `temperature` cannot be customized.
For gpt-5.1 models with reasoning_effort='none', temperature CAN be customized.

Example

Configuring a profile with medium reasoning effort:

```
profile = OpenAILanguageModel.Profile(reasoning_effort="medium")
```

## OpenRouterLanguageModel

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.OpenRouterLanguageModel[OpenRouterLanguageModel]

              click fenic.api.session.config.OpenRouterLanguageModel href "" "fenic.api.session.config.OpenRouterLanguageModel"
```

Configuration for OpenRouter language models.

This class defines the configuration settings for OpenRouter language models,
including model selection and rate limiting parameters. When fetching available models from OpenRouter, results
will be filtered to only include models from providers that are not in the user’s ignored providers list and are either
in the user’s allowed providers list (if configured) or from any provider (if no allowed providers are specified).

Attributes:

- **`model_name`**
  (`str`)
  –

  `{family}/{model}` identifier (e.g., `anthropic/claude-3-5-sonnet`).
- **`profiles`**
  (`Optional[dict[str, Profile]]`)
  –

  Mapping of profile names to profile configurations.
- **`default_profile`**
  (`Optional[str]`)
  –

  The key in `profiles` to select by default.
- **`structured_output_strategy`**
  (`Optional[StructuredOutputStrategy]`)
  –

  The strategy to use for structured output if a model supports both tool calling and structured outputs.

  - `prefer_tools`: prefer using tools over response format.
  - `prefer_response_format`: prefer using response format over tools.

Requirements

- Set `OPENROUTER_API_KEY` in your environment.

Example:

```
OpenRouterLanguageModel(
    model_name="openai/gpt-oss-20b",
    profiles={
        "default": OpenRouterLanguageModel.Profile(
            provider=OpenRouterLanguageModel.Provider(
                sort="price"  # Routes to the cheapest available provider
            )
        )
    },
)
```

Example:

```
OpenRouterLanguageModel(
    model_name="anthropic/claude-sonnet-4",
    profiles={
        "default": OpenRouterLanguageModel.Profile(
            provider=OpenRouterLanguageModel.Provider(
                only=[
                    "Anthropic"
                ]  # ensures the request will only be routed to Anthropic and not AWS Bedrock or Google Vertex
            )
        )
    },
)
```

Example:

```
OpenRouterLanguageModel(
    model_name="qwen/qwen3-next-80b-a3b-instruct",
    profiles={
        "default": OpenRouterLanguageModel.Profile(
            provider=OpenRouterLanguageModel.Provider(
                sort="throughput", # routes to the provider with the highest overall throughput
                data_collection="deny" # eliminates providers that retain prompt data (would only route to DeepInfra/AtlasCloud, in this example)
                # Eliminate providers that offer an fp8 quantized version of the model, only allowing bf16.
                # Note that many providers have an `unknown` quantization, so you may be excluding more providers than you expect.
                quantizations=["bf16"]
            )
        )
    }
)
```

Classes:

- **`Profile`**
  –

  Profile configurations for OpenRouter language models.
- **`Provider`**
  –

  Provider routing configuration for OpenRouter language models.

### Profile

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.OpenRouterLanguageModel.Profile[Profile]

              click fenic.api.session.config.OpenRouterLanguageModel.Profile href "" "fenic.api.session.config.OpenRouterLanguageModel.Profile"
```

Profile configurations for OpenRouter language models.

Attributes:

- **`models`**
  (`Optional[list[str]]`)
  –

  A list of fallback models to use if the primary model is unavailable.
  ([OpenRouter Documentation](https://openrouter.ai/docs/features/model-routing#the-models-parameter)).
- **`provider`**
  (`Optional[Provider]`)
  –

  Provider routing preferences (include/exclude specific providers, set provider ranking method preference)
  ([OpenRouter Documentation](https://openrouter.ai/docs/features/provider-routing)).
- **`reasoning_effort`**
  (`Optional[OpenRouterReasoningEffort]`)
  –

  OpenRouter reasoning effort configuration (none, minimal, low, medium, high, xhigh, max).
  If the model does support reasoning, but not `reasoning_effort`, a `reasoning_max_tokens` will be calculated
  that is roughly equivalent as a percentage of the model's maximum output size
  ([OpenRouter Documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level))
- **`reasoning_max_tokens`**
  (`Optional[int]`)
  –

  Supported by Anthropic, Gemini, etc., sets a token budget for reasoning
  If the model does support reasoning, but not `reasoning_max_tokens`, a `reasoning_effort_ will be automatically
  calculated based on`reasoning_max_tokens` as a percentage of the model's maximum output size
  ([OpenRouter Documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning))
- **`parsing_engine`**
  (`Optional[ParsingEngine]`)
  –

  The parsing engine to use for processing PDF files. By default, the model's native parsing engine will be used. If the model doesn't support PDF processing and the parsing engine is not provided, an error will be raised. Note: 'mistral-ocr' incurs additional costs.
  ([OpenRouter Documentation](https://openrouter.ai/docs/features/multimodal/pdfs))

### Provider

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.OpenRouterLanguageModel.Provider[Provider]

              click fenic.api.session.config.OpenRouterLanguageModel.Provider href "" "fenic.api.session.config.OpenRouterLanguageModel.Provider"
```

Provider routing configuration for OpenRouter language models.

[Provider Routing Documentation](https://openrouter.ai/docs/features/provider-routing)

Attributes:

- **`order`**
  (`Optional[list[str]]`)
  –

  List of providers to try in order (e.g. ['Anthropic', 'Amazon Bedrock']).
- **`sort`**
  (`Optional[ProviderSort]`)
  –

  Provider routing preference (e.g. 'price', 'throughput', 'latency').
  "price" will route to the cheapest available provider first, progressing through the list of providers in order of price.
  "throughput" will route to the provider with the highest overall recent throughput, progressing through the list of providers in order of throughput.
  "latency" will route to the provider with the lowest overall recent latency, progressing through the list of providers in order of latency.
- **`quantizations`**
  (`Optional[list[ModelQuantization]]`)
  –

  Allowed quantizations. Note: many providers report `unknown`.
- **`data_collection`**
  (`Optional[DataCollection]`)
  –

  Data collection preference. `allow`: allows the use of providers which store prompt data
  non-transiently and may train on it. `deny`: use only providers which do not collect/store prompt data.
- **`only`**
  (`Optional[list[str]]`)
  –

  Only include these providers when performing provider routing.
- **`exclude`**
  (`Optional[list[str]]`)
  –

  Exclude these providers when performing provider routing.
- **`max_prompt_price`**
  (`Optional[float]`)
  –

  Maximum prompt price ($USD per 1M tokens).
- **`max_completion_price`**
  (`Optional[float]`)
  –

  Maximum completion price ($USD per 1M tokens).

## SemanticConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.SemanticConfig[SemanticConfig]

              click fenic.api.session.config.SemanticConfig href "" "fenic.api.session.config.SemanticConfig"
```

Configuration for semantic language and embedding models.

This class defines the configuration for both language models and optional
embedding models used in semantic operations. It ensures that all configured
models are valid and supported by their respective providers.

Attributes:

- **`language_models`**
  (`Optional[dict[str, LanguageModel]]`)
  –

  Mapping of model aliases to language model configurations.
- **`default_language_model`**
  (`Optional[str]`)
  –

  The alias of the default language model to use for semantic operations. Not required
  if only one language model is configured.
- **`embedding_models`**
  (`Optional[dict[str, EmbeddingModel]]`)
  –

  Optional mapping of model aliases to embedding model configurations.
- **`default_embedding_model`**
  (`Optional[str]`)
  –

  The alias of the default embedding model to use for semantic operations.

Note

The embedding model is optional and only required for operations that
need semantic search or embedding capabilities.

Example

Configuring semantic models with a single language model:

```
config = SemanticConfig(
    language_models={
        "gpt4": OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100)
    }
)
```

Configuring semantic models with multiple language models and an embedding model:

```
config = SemanticConfig(
    language_models={
        "gpt4": OpenAILanguageModel(
            model_name="gpt-4.1-nano", rpm=100, tpm=100
        ),
        "claude": AnthropicLanguageModel(
            model_name="claude-haiku-4-5",
            rpm=100,
            input_tpm=100,
            output_tpm=100,
        ),
        "gemini": GoogleDeveloperLanguageModel(
            model_name="gemini-2.5-flash", rpm=100, tpm=1000
        ),
    },
    default_language_model="gpt4",
    embedding_models={
        "openai_embeddings": OpenAIEmbeddingModel(
            model_name="text-embedding-3-small", rpm=100, tpm=100
        )
    },
    default_embedding_model="openai_embeddings",
)
```

Configuring models with profiles:

```
config = SemanticConfig(
    language_models={
        "gpt4": OpenAILanguageModel(
            model_name="gpt-4o-mini",
            rpm=100,
            tpm=100,
            profiles={
                "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                "thorough": OpenAILanguageModel.Profile(
                    reasoning_effort="high"
                ),
            },
            default_profile="fast",
        ),
        "claude": AnthropicLanguageModel(
            model_name="claude-haiku-4-5",
            rpm=100,
            input_tpm=100,
            output_tpm=100,
            profiles={
                "fast": AnthropicLanguageModel.Profile(effort="low"),
                "thorough": AnthropicLanguageModel.Profile(effort="high"),
            },
            default_profile="fast",
        ),
    },
    default_language_model="gpt4",
)
```

Methods:

- **`model_post_init`**
  –

  Post initialization hook to set defaults.
- **`validate_models`**
  –

  Validates that the selected models are supported by the system.

### model_post_init

```
model_post_init(__context) -> None
```

Post initialization hook to set defaults.

This hook runs after the model is initialized and validated.
It sets the default language and embedding models if they are not set
and there is only one model available. For Google models that support
thinking_level, it auto-creates "low" and "high" profiles if no profiles
are configured.

Source code in `src/fenic/api/session/config.py`

```
def model_post_init(self, __context) -> None:
    """Post initialization hook to set defaults.

    This hook runs after the model is initialized and validated.
    It sets the default language and embedding models if they are not set
    and there is only one model available. For Google models that support
    thinking_level, it auto-creates "low" and "high" profiles if no profiles
    are configured.
    """
    if self.language_models:
        # Set default language model if not set and only one model exists
        if self.default_language_model is None and len(self.language_models) == 1:
            self.default_language_model = list(self.language_models.keys())[0]

        # Auto-create profiles for Google models that support thinking_level
        for model_config in self.language_models.values():
            if isinstance(model_config, (GoogleDeveloperLanguageModel, GoogleVertexLanguageModel)):
                model_provider = _get_model_provider_for_model_config(model_config)
                model_params = model_catalog.get_completion_model_parameters(
                    model_provider, model_config.model_name
                )
                if model_params and model_params.supported_thinking_levels and model_config.profiles is None:
                    # Auto-create profiles for each supported thinking level
                    if isinstance(model_config, GoogleDeveloperLanguageModel):
                        model_config.profiles = {
                            level: GoogleDeveloperLanguageModel.Profile(thinking_level=level)
                            for level in model_params.supported_thinking_levels
                        }
                    else:
                        model_config.profiles = {
                            level: GoogleVertexLanguageModel.Profile(thinking_level=level)
                            for level in model_params.supported_thinking_levels
                        }
                    model_config.default_profile = "low"

        # Set default profile for each model if not set and only one profile exists
        for model_config in self.language_models.values():
            if model_config.profiles is not None:
                profile_names = list(model_config.profiles.keys())
                if model_config.default_profile is None and len(profile_names) == 1:
                    model_config.default_profile = profile_names[0]

    # Set default embedding model if not set and only one model exists
    if self.embedding_models:
        if self.default_embedding_model is None and len(self.embedding_models) == 1:
            self.default_embedding_model = list(self.embedding_models.keys())[0]
        # Set default profile for each model if not set and only one preset exists
        for model_config in self.embedding_models.values():
            if (
                hasattr(model_config, "profiles")
                and model_config.profiles is not None
            ):
                preset_names = list(model_config.profiles.keys())
                if model_config.default_profile is None and len(preset_names) == 1:
                    model_config.default_profile = preset_names[0]
```

### validate_models

```
validate_models() -> SemanticConfig
```

Validates that the selected models are supported by the system.

This validator checks that both the language model and embedding model (if provided)
are valid and supported by their respective providers.

Returns:

- `SemanticConfig`
  –

  The validated SemanticConfig instance.

Raises:

- `ConfigurationError`
  –

  If any of the models are not supported.

Source code in `src/fenic/api/session/config.py`

```
@model_validator(mode="after")
def validate_models(self) -> SemanticConfig:
    """Validates that the selected models are supported by the system.

    This validator checks that both the language model and embedding model (if provided)
    are valid and supported by their respective providers.

    Returns:
        The validated SemanticConfig instance.

    Raises:
        ConfigurationError: If any of the models are not supported.
    """
    # Skip validation if no models configured (embedding-only or empty config)
    if not self.language_models and not self.embedding_models:
        return self

    # Validate language models if provided
    if self.language_models:
        available_language_model_aliases = list(self.language_models.keys())
        if self.default_language_model is None and len(self.language_models) > 1:
            raise ConfigurationError(
                f"default_language_model is not set, and multiple language models are configured. Please specify one of: {available_language_model_aliases} as a default_language_model."
            )

        if (
            self.default_language_model is not None
            and self.default_language_model not in self.language_models
        ):
            raise ConfigurationError(
                f"default_language_model {self.default_language_model} is not in configured map of language models. Available models: {available_language_model_aliases} ."
            )

        for model_alias, language_model in self.language_models.items():
            language_model_name = language_model.model_name
            language_model_provider = _get_model_provider_for_model_config(
                language_model
            )

            completion_model_params = model_catalog.get_completion_model_parameters(
                language_model_provider, language_model_name
            )
            if completion_model_params is None:
                raise ConfigurationError(
                    model_catalog.generate_unsupported_completion_model_error_message(
                        language_model_provider, language_model_name
                    )
                )
            if language_model.profiles is not None:
                if not completion_model_params.supports_profiles:
                    raise ConfigurationError(
                        f"Model '{model_alias}' does not support parameter profiles. Please remove the Profile configuration."
                    )
                profile_names = list(language_model.profiles.keys())
                if (
                    language_model.default_profile is None
                    and len(profile_names) > 0
                ):
                    raise ConfigurationError(
                        f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile."
                    )
                if (
                    language_model.default_profile is not None
                    and language_model.default_profile not in profile_names
                ):
                    raise ConfigurationError(
                        f"default_profile {language_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}"
                    )
                for profile_alias, profile in language_model.profiles.items():
                    _validate_language_profile(
                        language_model,
                        model_alias,
                        completion_model_params,
                        profile,
                        profile_alias,
                    )

    if self.embedding_models is not None:
        available_embedding_model_aliases = list(self.embedding_models.keys())
        if self.default_embedding_model is None and len(self.embedding_models) > 1:
            raise ConfigurationError(
                f"default_embedding_model is not set, and multiple embedding models are configured. Please specify one of: {available_embedding_model_aliases} as a default_embedding_model."
            )

        if (
            self.default_embedding_model is not None
            and self.default_embedding_model not in self.embedding_models
        ):
            raise ConfigurationError(
                f"default_embedding_model {self.default_embedding_model} is not in configured map of embedding models. Available models: {available_embedding_model_aliases} ."
            )
        for model_alias, embedding_model in self.embedding_models.items():
            embedding_model_provider = _get_model_provider_for_model_config(
                embedding_model
            )
            embedding_model_name = embedding_model.model_name
            embedding_model_parameters = (
                model_catalog.get_embedding_model_parameters(
                    embedding_model_provider, embedding_model_name
                )
            )
            if embedding_model_parameters is None:
                raise ConfigurationError(
                    model_catalog.generate_unsupported_embedding_model_error_message(
                        embedding_model_provider, embedding_model_name
                    )
                )
            if hasattr(embedding_model, "profiles") and embedding_model.profiles:
                profile_names = list(embedding_model.profiles.keys())
                if (
                    embedding_model.default_profile is None
                    and len(profile_names) > 0
                ):
                    raise ConfigurationError(
                        f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile."
                    )
                if (
                    embedding_model.default_profile is not None
                    and embedding_model.default_profile not in profile_names
                ):
                    raise ConfigurationError(
                        f"default_profile {embedding_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}"
                    )

                for profile_alias, profile in embedding_model.profiles.items():
                    _validate_embedding_profile(
                        embedding_model_parameters,
                        model_alias,
                        profile_alias,
                        profile,
                    )

    return self
```

## SessionConfig

Bases: `BaseModel`

```
              flowchart TD
              fenic.api.session.config.SessionConfig[SessionConfig]

              click fenic.api.session.config.SessionConfig href "" "fenic.api.session.config.SessionConfig"
```

Configuration for a user session.

This class defines the complete configuration for a user session, including
application settings, model configurations, and optional cloud settings.
It serves as the central configuration object for all language model operations.

Attributes:

- **`app_name`**
  (`str`)
  –

  Name of the application using this session. Defaults to "default_app".
- **`db_path`**
  (`Optional[Path]`)
  –

  Optional path to a local database file for persistent storage.
- **`semantic`**
  (`Optional[SemanticConfig]`)
  –

  Configuration for semantic models (optional).
- **`cloud`**
  (`Optional[CloudConfig]`)
  –

  Optional configuration for cloud execution.
- **`cache`**
  (`Optional[CloudConfig]`)
  –

  Optional configuration for LLM response caching.

Note

The semantic configuration is optional. When not provided, only non-semantic operations
are available. The cloud configuration is optional and only needed for
distributed processing.

Example

Configuring a basic session with a single language model:

```
config = SessionConfig(
    app_name="my_app",
    semantic=SemanticConfig(
        language_models={
            "gpt4": OpenAILanguageModel(
                model_name="gpt-4.1-nano", rpm=100, tpm=100
            )
        }
    ),
)
```

Configuring a session with multiple models and cloud execution:

```
config = SessionConfig(
    app_name="production_app",
    db_path=Path("/path/to/database.db"),
    semantic=SemanticConfig(
        language_models={
            "gpt4": OpenAILanguageModel(
                model_name="gpt-4.1-nano", rpm=100, tpm=100
            ),
            "claude": AnthropicLanguageModel(
                model_name="claude-haiku-4-5",
                rpm=100,
                input_tpm=100,
                output_tpm=100,
            ),
        },
        default_language_model="gpt4",
        embedding_models={
            "openai_embeddings": OpenAIEmbeddingModel(
                model_name="text-embedding-3-small", rpm=100, tpm=100
            )
        },
        default_embedding_model="openai_embeddings",
    ),
    cloud=CloudConfig(size=CloudExecutorSize.MEDIUM),
)
```

Methods:

- **`to_json`**
  –

  Export the session config to a JSON string.

### to_json

```
to_json() -> str
```

Export the session config to a JSON string.

Source code in `src/fenic/api/session/config.py`

```
def to_json(self) -> str:
    """Export the session config to a JSON string."""
    return self.model_dump_json(indent=2)
```
