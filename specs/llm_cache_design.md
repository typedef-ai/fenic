# LLM Response Cache Design Specification

**Version:** 3.0 Final
**Status:** Implementation Ready
**Last Updated:** 2025-01-24

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Implementation](#implementation)
6. [Integration](#integration)
7. [Testing](#testing)

---

## Overview

### Purpose

Provide persistent caching of LLM responses in Fenic's batch processing pipeline to:

- **Save money**: Avoid duplicate API calls (~$0.15-$15 per 1M tokens)
- **Speed up iteration**: Cached responses return instantly
- **Enable fault recovery**: Resume failed batches without reprocessing
- **Provide analytics**: SQL-queryable cache for cost tracking

### Key Design Decisions

1. **Short TTL with duration strings**: Default 1 hour (not 30 days) - perfect for batch jobs
2. **Normalized + JSON storage**: No pickle security risks, SQL-queryable
3. **SQLite only**: Simple, sufficient, no Redis complexity
4. **Graceful degradation**: Cache errors never break pipelines

### Success Metrics

- **Cost Savings**: Avoid duplicate LLM calls
- **Hit Rate**: Target 50%+ for batch processing
- **Zero Failures**: Cache errors logged, never raised
- **Fast**: <10ms cache lookups for batches of 100

---

## Quick Start

### Enable Caching

```python
from fenic.api import Session
from fenic.api.session.config import (
    LLMResponseCacheConfig,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)

config = SessionConfig(
    app_name="my_batch_job",
    semantic=SemanticConfig(
        language_models={
            "default": OpenAILanguageModel(
                model_name="gpt-4o-mini",
                rpm=100,
                tpm=1000,
            )
        },
        default_language_model="default",
        llm_response_cache=LLMResponseCacheConfig(
            ttl="30m",  # 30 minutes
            max_size_mb=5000,
            namespace="batch-job",
        ),
    ),
)

session = Session.get_or_create(config)
# All LLM calls now cached automatically!
```

### Query Cache Analytics

```sql
-- Cost savings by model
SELECT
    model,
    SUM(total_tokens) / 1000000.0 * 0.150 as saved_usd,
    SUM(access_count) as cache_hits
FROM llm_responses
GROUP BY model;

-- Most expensive cached responses
SELECT cache_key, model, total_tokens, access_count
FROM llm_responses
ORDER BY total_tokens DESC
LIMIT 10;
```

---

## Architecture

### Design Principles

1. **Protocol-based**: Easy to swap implementations
2. **Thread-safe**: Handle concurrent access from ModelClient's asyncio loop
3. **Graceful degradation**: Cache failures never break pipelines
4. **Batch-optimized**: Efficient bulk lookups/stores
5. **Observable**: Rich statistics and logging

### Cache Flow

```markdown
Request Batch → Pull Cache hits for entire batch
For Each Request:
Check Cache → Hit? Return cached : Call API → Store in cache → Return
↓
Miss
↓
ModelClient.\_submit_batch_requests()
↓
API Call via make_single_request()
↓
ModelClient.\_handle_response()
↓
Store successful response in cache
```

### Storage Schema (SQLite)

```sql
CREATE TABLE llm_responses (
    -- Primary key
    cache_key TEXT NOT NULL,
    namespace TEXT NOT NULL,

    -- Core queryable fields (normalized)
    model TEXT NOT NULL,
    completion TEXT NOT NULL,
    cached_at TIMESTAMP NOT NULL,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,

    -- Token usage (normalized for analytics)
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER DEFAULT 0,
    thinking_tokens INTEGER DEFAULT 0,

    -- Complex data (JSON bytes)
    logprobs_data BLOB,

    -- Schema version for migration
    response_version INTEGER DEFAULT 1,

    PRIMARY KEY (cache_key, namespace)
);

-- Indices for common queries
CREATE INDEX idx_cached_at ON llm_responses(namespace, cached_at);
CREATE INDEX idx_last_accessed ON llm_responses(namespace, last_accessed);
CREATE INDEX idx_model ON llm_responses(model);
CREATE INDEX idx_token_usage ON llm_responses(namespace, total_tokens);
```

**Why Normalized Schema?**

- ✅ SQL-queryable for analytics
- ✅ No pickle security risks
- ✅ Compact storage (no repeated field names)
- ✅ Easy to debug (human-readable)

### Cache Key Generation

The cache key is produced by the cache backend itself. `SQLiteLLMCache`
implements `compute_key(request, model, profile_hash)` and always hashes the
following normalized payload (language completion requests only; embedding
requests use the same fingerprinting logic for deduplication but are not yet
persisted in the cache):

- Model name
- Full encoded message payload (system, user, examples, tool calls)
- Max completion tokens and temperature
- Structured output fingerprint (if supplied)
- Requested top-logprobs
- Profile metadata: both the `request.model_profile` name and (critically) the
  serialized profile contents via `profile_hash`

```python
key_data = {
    "model": model,
    "messages": request.messages.encode().hex(),
    "max_tokens": request.max_completion_tokens,
    "temperature": request.temperature,
    "model_profile": request.model_profile,
    "profile_hash": profile_hash,
    "top_logprobs": request.top_logprobs,
}
if request.structured_output:
    key_data["structured_output"] = (
        request.structured_output.schema_fingerprint
    )
serialized = json.dumps(key_data, sort_keys=True).encode("utf-8")
return hashlib.sha256(serialized).hexdigest()
```

`profile_hash` is produced in `ModelClient` via the `ProfileHashMixin`. Each
provider resolves the requested profile, serializes the dataclass with
`dataclasses.asdict`, and feeds it through SHA-256. This guarantees that two
profiles with the same name but different parameters will never collide in the
cache (and ensures the deduplication keys for embeddings also evolve when
profiles change, even though we defer storing their vectors until we add schema
support).

---

## Configuration

### LLMResponseCacheConfig

The config object lives in `src/fenic/api/session/config.py` under the semantic
section of `SessionConfig`. Key behaviors:

- `backend`, `ttl`, `max_size_mb`, and `namespace` are the main knobs; omitting
  the config disables caching entirely.
- TTL strings must match `<number><unit>` (s/m/h/d) with guardrails
  (≥1s, ≤720h, ≤30d). Invalid entries raise during validation.
- `max_size_mb` is clamped to `(0, 100_000]` to protect from runaway local DBs.
- `ttl_seconds()` converts the duration into an integer for backends.

### SessionConfig Integration

`SessionConfig.semantic.llm_response_cache` wires the cache into the session
pipeline. When present, `Session.get_or_create()` initializes the configured
backend (currently SQLite) and exposes it through the session state so every
`DataFrame.semantic` action transparently participates in caching.

---

## Implementation

### Protocol Interface

`src/fenic/_inference/cache/protocol.py` defines three lightweight types:

- `CachedResponse` mirrors `FenicCompletionsResponse` plus metadata (timestamps,
  access counts, token accounting) and exposes `to_fenic_response()`.
- `CacheStats` aggregates hits, misses, stores, errors, and basic size info so
  backends can expose telemetry without leaking implementation details.
- `LLMResponseCache` is a thin `Protocol` that mandates `get`, `get_batch`,
  `set`, `set_batch`, `delete`, `clear`, `stats`, and `close`. All methods must
  swallow backend failures and default to safe fallbacks (e.g., empty dict on
  batch get), ensuring cache outages never break user pipelines.

### SQLite Implementation

`SQLiteLLMCache` (`src/fenic/_inference/cache/sqlite_cache.py`) provides the
reference backend. Highlights:

- **Storage layout**: single normalized table (`llm_responses`) keyed by
  `(cache_key, namespace)` with JSON blobs for logprobs and integer columns for
  token analytics. Indices cover `cached_at`, `last_accessed`, `model`, and
  token usage.
- **Concurrency**: uses a small connection pool (default 3) with WAL mode,
  `synchronous=NORMAL`, and in-memory temp storage to keep lookups <10 ms while
  still being thread-safe.
- **TTL + eviction**: every `get`/`get_batch` filters by `cached_at >
now - ttl`, increments access counters, and lazily updates `last_accessed`.
  `_maybe_evict` trims ~10 % of least-recently-used rows when the DB exceeds
  `max_size_mb`. Currently, this is done lazily when new items are fetched from
  the cache -- as a future add on, we can add logic that sits in the background
  and performs cache cleanup.
- **Error handling**: all operations catch exceptions, increment `errors`, and
  return safe defaults (e.g., empty dict). Corrupted databases are deleted and
  rebuilt automatically on init.
- **Stats**: `stats()` queries SQLite metadata for entry counts and file size,
  then combines it with the in-memory hit/miss/store counters maintained under a
  lock.

---

## Integration

### ModelClient Changes

Caching responsibilities inside `ModelClient` are intentionally compact:

- **Fingerprints everywhere**: `_build_request_key()` calls
  `generate_request_fingerprint()` for every request (completion or embedding).
  The value is stored on `QueueItem.request_fingerprint`, ensuring deduplication
  and caching share the same key without re-hashing later.
- **Batch lookups**: `_submit_batch_requests()` precomputes fingerprints, runs a
  single `cache.get_batch()` for completion fingerprints, and immediately
  short-circuits futures that hit. Embeddings are never handed to the cache but
  still dedup thanks to shared fingerprints.
- **Writes on success**: `_handle_response()` only calls `cache.set()` for
  successful completion responses. Any caching error is logged and ignored so
  the main future still resolves.
- **Cache optionality**: because fingerprints live entirely within
  `ModelClient`, disabling the cache still deduplicates identical requests—the
  cache simply doesn't see the `get_batch`/`set` calls.

---

## Testing

### Test Suite Overview

We split validation across focused layers to keep coverage high without redundant
end-to-end scenarios:

- `tests/_inference/cache/test_cache_config.py` ensures the declarative
  configuration (`LLMResponseCacheConfig`) accepts only valid TTL strings,
  namespaces, and size limits.
- `tests/_inference/cache/test_sqlite_cache.py` covers the storage engine in
  isolation (set/get TTL behavior, TTL expiry, batch ops, connection pooling,
  WAL handling, corruption recovery). These tests operate directly on
  `SQLiteLLMCache` and do not involve `ModelClient`.
- `tests/_inference/cache/test_request_fingerprint.py` validates the
  deterministic key builder across prompts, models, temperatures, structured
  output schemas, profile hashes, etc., ensuring cache/dedup fingerprints remain
  stable.
- `tests/_inference/test_model_client_cache_behavior.py` exercises the model
  client orchestration layer with lightweight fakes: completion requests are
  shown to deduplicate and reuse cached responses, embedding requests are proven
  to deduplicate without touching the cache, and profile hash mutations generate
  distinct cache keys.
- `tests/_inference/cache/test_cache_end_to_end_semantic.py` now acts as a smoke
  test verifying semantic operators produce cache hits on repeated execution.
  Detailed profile-hash scenarios are covered in the unit suite so this file
  can remain lean and fast.

---

### Migration Guide

```python
# Before: No caching
session = Session.get_or_create(SessionConfig(app_name="my_app"))

# After: With caching
session = Session.get_or_create(
    SessionConfig(
        app_name="my_app",
        semantic=SemanticConfig(
            language_models={...},
            default_language_model="default",
            llm_response_cache=LLMResponseCacheConfig(ttl="1h"),
        ),
    )
)
```

### Monitoring

Track these metrics:

- **Hit rate**: Target 50%+
- **Cache size**: Monitor growth
- **Error rate**: Should be <0.1%
- **Latency**: Cache lookups <10ms

Query analytics:

```sql
-- Daily cache performance
SELECT
    DATE(cached_at) as date,
    COUNT(*) as new_entries,
    SUM(access_count) as total_hits
FROM llm_responses
GROUP BY DATE(cached_at)
ORDER BY date DESC;
```

---

## Summary

This design provides a **production-ready, secure, and performant** caching solution that:

- ✅ Saves money by avoiding duplicate LLM calls
- ✅ Speeds up batch processing workflows
- ✅ Enables fault recovery and iteration
- ✅ Provides analytics via SQL queries
- ✅ Maintains security (no pickle)
- ✅ Never breaks pipelines (graceful degradation)
