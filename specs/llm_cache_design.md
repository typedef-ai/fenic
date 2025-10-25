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
8. [Deployment](#deployment)

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
from fenic.api.session.config import SessionConfig, CacheConfig

# Default: 1 hour TTL
config = SessionConfig(
    app_name="my_batch_job",
    cache=CacheConfig(enabled=True)
)

# Custom TTL
config = SessionConfig(
    app_name="my_app",
    cache=CacheConfig(
        enabled=True,
        ttl="30m",  # 30 minutes
        max_size_mb=5000
    )
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
Request → Check Cache → Hit? Return cached : Call API → Store in cache → Return
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

```python
import hashlib
import json
from fenic._inference.types import FenicCompletionsRequest

class CacheKeyGenerator:
    """Generates deterministic cache keys from LLM requests."""

    @staticmethod
    def compute_key(request: FenicCompletionsRequest, model: str) -> str:
        """Compute SHA-256 hash of request parameters.

        Includes: model, messages, max_tokens, temperature, structured_output,
                  model_profile, top_logprobs

        Returns:
            64-character hex string
        """
        key_data = {
            "model": model,
            "messages": request.messages.encode().hex(),
            "max_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "model_profile": request.model_profile,
            "top_logprobs": request.top_logprobs,
        }

        if request.structured_output:
            key_data["structured_output"] = json.dumps(
                request.structured_output.schema,
                sort_keys=True,
                separators=(',', ':')
            )

        serialized = json.dumps(key_data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()
```

---

## Configuration

### CacheConfig

````python
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class CacheBackend(str, Enum):
    """Cache backend implementations."""
    SQLITE = "sqlite"
    MEMORY = "memory"
    DISABLED = "disabled"

class CacheConfig(BaseModel):
    """Configuration for LLM response caching.

    Attributes:
        enabled: Whether caching is enabled (default: True)
        backend: Cache backend to use (default: SQLITE)
        ttl: Time-to-live duration string (default: "1h")
            Examples: "30m", "2h", "7d"
        max_size_mb: Maximum cache size before LRU eviction (default: 1000)
        namespace: Cache namespace for isolation (default: "default")

    Note:
        The cache database is automatically stored alongside the session's DuckDB
        database with the name `_{app_name}_llm_cache.db`. The location is determined
        by the session's `db_path` configuration (defaults to current directory).
        The underscore prefix indicates it's a system database.

    Example:
        ```python
        # Default (1 hour TTL)
        cache = CacheConfig(enabled=True)

        # Custom TTL
        cache = CacheConfig(
            enabled=True,
            ttl="30m",  # 30 minutes
            max_size_mb=5000
        )

        # Long-running project (7 day TTL)
        cache = CacheConfig(
            enabled=True,
            ttl="7d"
        )
        ```
    """

    enabled: bool = Field(default=True)
    backend: CacheBackend = Field(default=CacheBackend.SQLITE)
    ttl: str = Field(default="1h")
    max_size_mb: int = Field(default=1000, gt=0, le=100000)
    namespace: str = Field(default="default")

    @field_validator("ttl")
    @classmethod
    def validate_ttl(cls, v: str) -> str:
        """Validate TTL duration string format.

        Format: <number><unit> where unit is s/m/h/d
        Examples: "30s", "15m", "2h", "7d"

        Raises:
            ValueError: If format is invalid
        """
        import re

        pattern = r'^(\d+)([smhd])$'
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
        if unit == 's' and value < 1:
            raise ValueError("TTL must be at least 1 second")
        if unit == 'h' and value > 720:  # 30 days
            raise ValueError("TTL cannot exceed 720 hours")
        if unit == 'd' and value > 30:
            raise ValueError("TTL cannot exceed 30 days")

        return v

    def ttl_seconds(self) -> int:
        """Convert TTL string to seconds."""
        import re

        pattern = r'^(\d+)([smhd])$'
        match = re.match(pattern, self.ttl.lower())

        if not match:
            raise ValueError(f"Invalid TTL format: '{self.ttl}'")

        value, unit = match.groups()
        value = int(value)

        multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        return value * multipliers[unit]
````

### SessionConfig Integration

```python
# In src/fenic/api/session/config.py

class SessionConfig(BaseModel):
    """Configuration for a user session."""

    app_name: str = "default_app"
    db_path: Optional[Path] = None
    semantic: Optional[SemanticConfig] = None
    cloud: Optional[CloudConfig] = None
    cache: Optional[CacheConfig] = None  # NEW
```

---

## Implementation

### Protocol Interface

```python
from typing import Protocol, Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CachedResponse:
    """Cached LLM response with metadata."""
    completion: str
    model: str
    cached_at: datetime
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cached_tokens: int = 0
    thinking_tokens: int = 0
    logprobs: Optional[list] = None
    access_count: int = 0

    def to_fenic_response(self) -> FenicCompletionsResponse:
        """Convert to FenicCompletionsResponse."""
        from fenic._inference.types import ResponseUsage

        usage = None
        if self.prompt_tokens is not None:
            usage = ResponseUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.total_tokens or 0,
                cached_tokens=self.cached_tokens,
                thinking_tokens=self.thinking_tokens,
            )

        return FenicCompletionsResponse(
            completion=self.completion,
            logprobs=self.logprobs,
            usage=usage,
        )

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int
    misses: int
    stores: int
    errors: int
    hit_rate: float
    total_entries: int = 0
    size_bytes: int = 0

class LLMResponseCache(Protocol):
    """Protocol for LLM response caching.

    All implementations must be thread-safe.
    """

    def get(self, cache_key: str) -> Optional[CachedResponse]:
        """Retrieve cached response. Returns None if not found or expired."""
        ...

    def get_batch(self, cache_keys: List[str]) -> Dict[str, Optional[CachedResponse]]:
        """Retrieve multiple responses. Returns dict with all keys."""
        ...

    def set(
        self,
        cache_key: str,
        response: FenicCompletionsResponse,
        model: str,
    ) -> bool:
        """Store response. Returns True if successful."""
        ...

    def set_batch(
        self,
        entries: List[tuple[str, FenicCompletionsResponse, str]]
    ) -> int:
        """Store multiple responses. Returns count of successful stores."""
        ...

    def delete(self, cache_key: str) -> bool:
        """Delete entry. Returns True if found and deleted."""
        ...

    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        ...

    def stats(self) -> CacheStats:
        """Get performance statistics."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...
```

### SQLite Implementation

```python
import logging
import sqlite3
import threading
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SQLiteLLMCache:
    """SQLite-backed LLM response cache with normalized storage.

    Thread-safe implementation using thread-local connections and WAL mode.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        ttl_seconds: int = 3600,
        max_size_mb: int = 1000,
        namespace: str = "default",
    ):
        if db_path is None:
            cache_dir = Path.home() / ".fenic"
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "llm_cache.db")

        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.namespace = namespace

        # Thread-local connections
        self._local = threading.local()

        # Statistics
        self._stats_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._errors = 0

        self._init_db()

        logger.info(
            f"Initialized SQLite cache at {self.db_path} "
            f"(ttl={ttl_seconds}s, max_size={max_size_mb}MB, namespace={namespace})"
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
                isolation_level="DEFERRED"
            )
            self._local.conn.row_factory = sqlite3.Row

            # Enable WAL mode for concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")  # 64MB
            self._local.conn.execute("PRAGMA temp_store=MEMORY")

        return self._local.conn

    def _init_db(self):
        """Initialize normalized schema."""
        conn = self._get_connection()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_responses (
                cache_key TEXT NOT NULL,
                namespace TEXT NOT NULL,
                model TEXT NOT NULL,
                completion TEXT NOT NULL,
                cached_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cached_tokens INTEGER DEFAULT 0,
                thinking_tokens INTEGER DEFAULT 0,
                logprobs_data BLOB,
                response_version INTEGER DEFAULT 1,
                PRIMARY KEY (cache_key, namespace)
            )
        """)

        # Create indices
        for idx, cols in [
            ("idx_cached_at", "(namespace, cached_at)"),
            ("idx_last_accessed", "(namespace, last_accessed)"),
            ("idx_model", "(model)"),
            ("idx_token_usage", "(namespace, total_tokens)"),
        ]:
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx}
                ON llm_responses{cols}
            """)

        conn.commit()

    def get(self, cache_key: str) -> Optional[CachedResponse]:
        """Retrieve cached response."""
        try:
            conn = self._get_connection()
            cutoff = datetime.now() - timedelta(seconds=self.ttl_seconds)

            cursor = conn.execute("""
                SELECT
                    completion, model, cached_at,
                    prompt_tokens, completion_tokens, total_tokens,
                    cached_tokens, thinking_tokens,
                    logprobs_data, access_count
                FROM llm_responses
                WHERE cache_key = ? AND namespace = ? AND cached_at > ?
            """, (cache_key, self.namespace, cutoff))

            row = cursor.fetchone()

            if row:
                # Update access stats
                conn.execute("""
                    UPDATE llm_responses
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE cache_key = ? AND namespace = ?
                """, (datetime.now(), cache_key, self.namespace))
                conn.commit()

                # Deserialize logprobs
                logprobs = None
                if row['logprobs_data']:
                    logprobs = json.loads(row['logprobs_data'].decode('utf-8'))

                with self._stats_lock:
                    self._hits += 1

                return CachedResponse(
                    completion=row['completion'],
                    model=row['model'],
                    cached_at=datetime.fromisoformat(row['cached_at']),
                    prompt_tokens=row['prompt_tokens'],
                    completion_tokens=row['completion_tokens'],
                    total_tokens=row['total_tokens'],
                    cached_tokens=row['cached_tokens'] or 0,
                    thinking_tokens=row['thinking_tokens'] or 0,
                    logprobs=logprobs,
                    access_count=row['access_count'] + 1
                )
            else:
                with self._stats_lock:
                    self._misses += 1
                return None

        except Exception as e:
            with self._stats_lock:
                self._errors += 1
            logger.warning(f"Cache get error for key {cache_key[:8]}...: {e}")
            return None

    def get_batch(self, cache_keys: List[str]) -> Dict[str, Optional[CachedResponse]]:
        """Retrieve multiple cached responses."""
        result = {}

        if not cache_keys:
            return result

        try:
            conn = self._get_connection()
            cutoff = datetime.now() - timedelta(seconds=self.ttl_seconds)

            placeholders = ','.join('?' * len(cache_keys))
            cursor = conn.execute(f"""
                SELECT
                    cache_key, completion, model, cached_at,
                    prompt_tokens, completion_tokens, total_tokens,
                    cached_tokens, thinking_tokens,
                    logprobs_data, access_count
                FROM llm_responses
                WHERE cache_key IN ({placeholders})
                  AND namespace = ?
                  AND cached_at > ?
            """, (*cache_keys, self.namespace, cutoff))

            found_keys = set()

            for row in cursor:
                key = row['cache_key']
                found_keys.add(key)

                logprobs = None
                if row['logprobs_data']:
                    logprobs = json.loads(row['logprobs_data'].decode('utf-8'))

                result[key] = CachedResponse(
                    completion=row['completion'],
                    model=row['model'],
                    cached_at=datetime.fromisoformat(row['cached_at']),
                    prompt_tokens=row['prompt_tokens'],
                    completion_tokens=row['completion_tokens'],
                    total_tokens=row['total_tokens'],
                    cached_tokens=row['cached_tokens'] or 0,
                    thinking_tokens=row['thinking_tokens'] or 0,
                    logprobs=logprobs,
                    access_count=row['access_count'] + 1
                )

            # Update access stats
            if found_keys:
                now = datetime.now()
                placeholders = ','.join('?' * len(found_keys))
                conn.execute(f"""
                    UPDATE llm_responses
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE cache_key IN ({placeholders})
                      AND namespace = ?
                """, (now, *found_keys, self.namespace))
                conn.commit()

            # Add None for missing keys
            for key in cache_keys:
                if key not in result:
                    result[key] = None

            with self._stats_lock:
                self._hits += len(found_keys)
                self._misses += len(cache_keys) - len(found_keys)

        except Exception as e:
            with self._stats_lock:
                self._errors += 1
            logger.warning(f"Cache get_batch error: {e}")
            result = {key: None for key in cache_keys}

        return result

    def set(
        self,
        cache_key: str,
        response: FenicCompletionsResponse,
        model: str,
    ) -> bool:
        """Store response in cache."""
        try:
            conn = self._get_connection()
            now = datetime.now()

            # Extract normalized fields
            prompt_tokens = response.usage.prompt_tokens if response.usage else None
            completion_tokens = response.usage.completion_tokens if response.usage else None
            total_tokens = response.usage.total_tokens if response.usage else None
            cached_tokens = response.usage.cached_tokens if response.usage else 0
            thinking_tokens = response.usage.thinking_tokens if response.usage else 0

            # Serialize logprobs as JSON
            logprobs_data = None
            if response.logprobs:
                logprobs_data = json.dumps(response.logprobs).encode('utf-8')

            conn.execute("""
                INSERT OR REPLACE INTO llm_responses
                (cache_key, namespace, model, completion, cached_at, last_accessed,
                 prompt_tokens, completion_tokens, total_tokens, cached_tokens, thinking_tokens,
                 logprobs_data, response_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                cache_key, self.namespace, model, response.completion, now, now,
                prompt_tokens, completion_tokens, total_tokens, cached_tokens, thinking_tokens,
                logprobs_data
            ))

            conn.commit()

            with self._stats_lock:
                self._stores += 1

            self._maybe_evict()

            return True

        except Exception as e:
            with self._stats_lock:
                self._errors += 1
            logger.warning(f"Cache set error for key {cache_key[:8]}...: {e}")
            return False

    def set_batch(
        self,
        entries: List[tuple[str, FenicCompletionsResponse, str]]
    ) -> int:
        """Store multiple responses."""
        stored = 0

        if not entries:
            return 0

        try:
            conn = self._get_connection()
            now = datetime.now()

            for cache_key, response, model in entries:
                try:
                    prompt_tokens = response.usage.prompt_tokens if response.usage else None
                    completion_tokens = response.usage.completion_tokens if response.usage else None
                    total_tokens = response.usage.total_tokens if response.usage else None
                    cached_tokens = response.usage.cached_tokens if response.usage else 0
                    thinking_tokens = response.usage.thinking_tokens if response.usage else 0

                    logprobs_data = None
                    if response.logprobs:
                        logprobs_data = json.dumps(response.logprobs).encode('utf-8')

                    conn.execute("""
                        INSERT OR REPLACE INTO llm_responses
                        (cache_key, namespace, model, completion, cached_at, last_accessed,
                         prompt_tokens, completion_tokens, total_tokens, cached_tokens, thinking_tokens,
                         logprobs_data, response_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (
                        cache_key, self.namespace, model, response.completion, now, now,
                        prompt_tokens, completion_tokens, total_tokens, cached_tokens, thinking_tokens,
                        logprobs_data
                    ))

                    stored += 1
                except Exception as e:
                    logger.warning(f"Error storing cache entry {cache_key[:8]}...: {e}")

            conn.commit()

            with self._stats_lock:
                self._stores += stored

            self._maybe_evict()

        except Exception as e:
            with self._stats_lock:
                self._errors += 1
            logger.warning(f"Cache set_batch error: {e}")

        return stored

    def delete(self, cache_key: str) -> bool:
        """Delete cached entry."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                DELETE FROM llm_responses
                WHERE cache_key = ? AND namespace = ?
            """, (cache_key, self.namespace))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def clear(self) -> int:
        """Clear all entries in namespace."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                DELETE FROM llm_responses WHERE namespace = ?
            """, (self.namespace,))
            conn.commit()

            cleared = cursor.rowcount
            logger.info(f"Cleared {cleared} entries from namespace '{self.namespace}'")
            return cleared
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0

    def stats(self) -> CacheStats:
        """Get performance statistics."""
        try:
            conn = self._get_connection()

            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM llm_responses WHERE namespace = ?
            """, (self.namespace,))
            total_entries = cursor.fetchone()['count']

            cursor = conn.execute("""
                SELECT page_count * page_size as size
                FROM pragma_page_count(), pragma_page_size()
            """)
            size_bytes = cursor.fetchone()['size']

            with self._stats_lock:
                total = self._hits + self._misses
                hit_rate = self._hits / total if total > 0 else 0.0

                return CacheStats(
                    hits=self._hits,
                    misses=self._misses,
                    stores=self._stores,
                    errors=self._errors,
                    hit_rate=hit_rate,
                    total_entries=total_entries,
                    size_bytes=size_bytes
                )
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return CacheStats(
                hits=0, misses=0, stores=0, errors=0, hit_rate=0.0,
                total_entries=0, size_bytes=0
            )

    def _maybe_evict(self):
        """Evict LRU entries if cache exceeds max size."""
        try:
            conn = self._get_connection()

            cursor = conn.execute("""
                SELECT page_count * page_size as size
                FROM pragma_page_count(), pragma_page_size()
            """)
            size_bytes = cursor.fetchone()['size']
            size_mb = size_bytes / (1024 * 1024)

            if size_mb > self.max_size_mb:
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM llm_responses WHERE namespace = ?
                """, (self.namespace,))
                total_count = cursor.fetchone()['count']
                evict_count = max(1, total_count // 10)  # Evict 10%

                conn.execute("""
                    DELETE FROM llm_responses
                    WHERE cache_key IN (
                        SELECT cache_key FROM llm_responses
                        WHERE namespace = ?
                        ORDER BY last_accessed ASC
                        LIMIT ?
                    )
                """, (self.namespace, evict_count))
                conn.commit()

                logger.info(
                    f"Evicted {evict_count} LRU entries "
                    f"(size: {size_mb:.1f}MB > {self.max_size_mb}MB)"
                )

        except Exception as e:
            logger.warning(f"Cache eviction error: {e}")

    def close(self):
        """Close database connection."""
        try:
            if hasattr(self._local, 'conn'):
                self._local.conn.close()
                del self._local.conn
        except Exception as e:
            logger.warning(f"Error closing cache: {e}")
```

---

## Integration

### ModelClient Changes

#### 1. Add Cache to ModelClient

```python
# In src/fenic/_inference/model_client.py

class ModelClient(Generic[RequestT, ResponseT], ABC):
    def __init__(
        self,
        model: str,
        model_provider: ModelProvider,
        model_provider_class: ModelProviderClass,
        rate_limit_strategy: RateLimitStrategy,
        token_counter: TokenCounter,
        queue_size: int = 100,
        initial_backoff_seconds: float = 1,
        backoff_factor: float = 2,
        max_backoffs: int = 10,
        cache: Optional[LLMResponseCache] = None,  # NEW
    ):
        # ... existing init ...
        self.cache = cache  # NEW

        if self.cache:
            logger.info(f"LLM response caching enabled for model {model}")
```

#### 2. Update QueueItem

```python
@dataclass
class QueueItem(Generic[RequestT]):
    thread_id: int
    request: RequestT
    future: Future
    estimated_tokens: TokenEstimate
    batch_id: str
    cache_key: Optional[str] = None  # NEW
```

#### 3. Cache Lookup in \_submit_batch_requests

```python
def _submit_batch_requests(
    self,
    requests: List[Optional[RequestT]],
    batch_id: str
) -> tuple[List[Future], int, TokenEstimate]:
    request_futures: List[Future] = []
    current_thread_id = threading.get_ident()
    unique_futures: Dict[Any, Future] = {}
    num_unique_requests = 0
    total_token_estimate = TokenEstimate()

    # NEW: Batch cache lookup
    cache_lookups = []
    cache_key_to_idx = {}

    if self.cache is not None:
        for idx, request in enumerate(requests):
            if request is not None:
                cache_key = CacheKeyGenerator.compute_key(request, self.model)
                cache_lookups.append(cache_key)
                cache_key_to_idx[cache_key] = idx

        cached_responses = self.cache.get_batch(cache_lookups)
        cache_hits = sum(1 for v in cached_responses.values() if v is not None)

        if cache_hits > 0:
            logger.info(
                f"Batch {batch_id}: {cache_hits}/{len(cache_lookups)} cache hits "
                f"({cache_hits/len(cache_lookups):.1%})"
            )
    else:
        cached_responses = {}

    with tqdm(
        total=len(requests),
        desc=f"Submitting requests (batch: {batch_id}, model: {self.model})",
        unit="req",
    ) as pbar:
        for idx, request in enumerate(requests):
            self._maybe_raise_thread_exception()

            if request is None:
                req_future = Future()
                request_futures.append(req_future)
                req_future.set_result(None)
                pbar.update(1)
                continue

            # NEW: Check cache
            cache_key = CacheKeyGenerator.compute_key(request, self.model)
            cached = cached_responses.get(cache_key)

            if cached is not None:
                # Cache hit
                req_future = Future()
                request_futures.append(req_future)
                req_future.set_result(cached.to_fenic_response())
                pbar.update(1)
                continue

            # Cache miss - normal processing
            req_future, estimated_tokens = self._get_or_create_request_future(
                unique_futures, request
            )
            request_futures.append(req_future)

            if estimated_tokens is not None:
                num_unique_requests += 1
                total_token_estimate += estimated_tokens
                queue_item = QueueItem(
                    thread_id=current_thread_id,
                    request=request,
                    future=req_future,
                    estimated_tokens=estimated_tokens,
                    batch_id=batch_id,
                    cache_key=cache_key,  # NEW
                )
                enqueue_future = asyncio.run_coroutine_threadsafe(
                    self._enqueue_request(queue_item),
                    self._event_loop,
                )
                enqueue_future.result()

            pbar.update(1)

    return request_futures, num_unique_requests, total_token_estimate
```

#### 4. Cache Storage in \_handle_response

```python
async def _handle_response(
    self,
    queue_item: QueueItem[RequestT],
    maybe_response: Union[None, ResponseT, TransientException, FatalException],
):
    if isinstance(maybe_response, TransientException):
        # Retry logic (unchanged)
        if self.num_backoffs >= self.max_backoffs:
            self._register_thread_exception(queue_item, ...)
        else:
            await self.retry_queue.put(queue_item)

    elif isinstance(maybe_response, FatalException):
        # Error handling (unchanged)
        self._register_thread_exception(queue_item, maybe_response.exception)

    else:
        # NEW: Cache successful response
        if self.cache and hasattr(queue_item, 'cache_key') and queue_item.cache_key:
            try:
                self.cache.set(
                    queue_item.cache_key,
                    maybe_response,
                    self.model,
                )
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")

        # Set result (unchanged)
        if not queue_item.future.done():
            queue_item.future.set_result(maybe_response)
```

---

## Testing

### Unit Tests

```python
# tests/unit/test_cache_config.py

import pytest
from fenic.api.session.config import CacheConfig

class TestCacheConfig:
    def test_default_ttl(self):
        config = CacheConfig(enabled=True)
        assert config.ttl == "1h"
        assert config.ttl_seconds() == 3600

    def test_duration_parsing(self):
        cases = [
            ("30s", 30),
            ("15m", 900),
            ("2h", 7200),
            ("7d", 604800),
        ]
        for ttl_str, expected_seconds in cases:
            config = CacheConfig(ttl=ttl_str)
            assert config.ttl_seconds() == expected_seconds

    def test_invalid_ttl_format(self):
        with pytest.raises(ValueError, match="Invalid TTL format"):
            CacheConfig(ttl="invalid")

        with pytest.raises(ValueError, match="Invalid TTL format"):
            CacheConfig(ttl="1x")

    def test_ttl_range_validation(self):
        with pytest.raises(ValueError, match="cannot exceed 30 days"):
            CacheConfig(ttl="31d")

        with pytest.raises(ValueError, match="cannot exceed 720 hours"):
            CacheConfig(ttl="721h")


# tests/unit/test_sqlite_cache.py

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from fenic._inference.cache.sqlite_cache import SQLiteLLMCache
from fenic._inference.types import FenicCompletionsResponse, ResponseUsage

class TestSQLiteCache:
    @pytest.fixture
    def temp_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        cache = SQLiteLLMCache(
            db_path=db_path,
            ttl_seconds=3600,
            max_size_mb=100,
            namespace="test"
        )

        yield cache

        cache.close()
        Path(db_path).unlink(missing_ok=True)

    def test_set_and_get(self, temp_cache):
        response = FenicCompletionsResponse(
            completion="Hello!",
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
        )

        # Set
        success = temp_cache.set("test_key", response, "gpt-4o-mini")
        assert success

        # Get
        cached = temp_cache.get("test_key")
        assert cached is not None
        assert cached.completion == "Hello!"
        assert cached.model == "gpt-4o-mini"
        assert cached.total_tokens == 15

    def test_cache_miss(self, temp_cache):
        cached = temp_cache.get("nonexistent")
        assert cached is None

    def test_ttl_expiration(self, temp_cache):
        response = FenicCompletionsResponse(completion="Test", logprobs=None)
        temp_cache.set("test_key", response, "gpt-4o-mini")

        # Manually expire
        conn = temp_cache._get_connection()
        old_date = datetime.now() - timedelta(hours=2)
        conn.execute("""
            UPDATE llm_responses
            SET cached_at = ?
            WHERE cache_key = ?
        """, (old_date, "test_key"))
        conn.commit()

        # Should be expired
        cached = temp_cache.get("test_key")
        assert cached is None

    def test_batch_operations(self, temp_cache):
        responses = [
            FenicCompletionsResponse(completion=f"Response {i}", logprobs=None)
            for i in range(10)
        ]

        # Batch set
        entries = [
            (f"key_{i}", responses[i], "gpt-4o-mini")
            for i in range(10)
        ]
        stored = temp_cache.set_batch(entries)
        assert stored == 10

        # Batch get
        keys = [f"key_{i}" for i in range(10)]
        results = temp_cache.get_batch(keys)

        assert len(results) == 10
        for i, key in enumerate(keys):
            assert results[key] is not None
            assert results[key].completion == f"Response {i}"

    def test_statistics(self, temp_cache):
        response = FenicCompletionsResponse(completion="Test", logprobs=None)

        temp_cache.set("key1", response, "gpt-4o-mini")
        temp_cache.get("key1")  # Hit
        temp_cache.get("key1")  # Hit
        temp_cache.get("key2")  # Miss

        stats = temp_cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.stores == 1
        assert stats.hit_rate == 2/3
```

### Integration Tests

```python
# tests/integration/test_model_client_cache.py

import pytest
from unittest.mock import Mock, patch

@pytest.mark.integration
class TestModelClientCacheIntegration:
    def test_cache_hit_skips_api_call(self):
        """Verify cache hits don't make API calls."""
        # TODO: Full integration test
        pass

    def test_cache_miss_makes_api_call(self):
        """Verify cache misses result in API calls."""
        # TODO: Full integration test
        pass

    def test_successful_response_cached(self):
        """Verify successful responses are stored."""
        # TODO: Full integration test
        pass
```

---

## Deployment

### Phase 1: Core Implementation (Week 1)

**Day 1-2**: Cache infrastructure

- [ ] Implement `CacheConfig` with validation
- [ ] Implement `CacheKeyGenerator`
- [ ] Implement `SQLiteLLMCache`
- [ ] Unit tests

**Day 3-4**: ModelClient integration

- [ ] Update `ModelClient.__init__` to accept cache
- [ ] Add cache lookup in `_submit_batch_requests`
- [ ] Add cache storage in `_handle_response`
- [ ] Integration tests

**Day 5**: Documentation & polish

- [ ] API documentation
- [ ] Usage examples
- [ ] Performance benchmarks

### Phase 2: Beta Testing (Week 2)

- [ ] Deploy to development environment
- [ ] Monitor metrics (hit rate, error rate, latency)
- [ ] Gather user feedback
- [ ] Iterate based on findings

### Phase 3: Production (Week 3)

- [ ] Enable for all users (opt-in)
- [ ] Monitor production metrics
- [ ] Document best practices
- [ ] Consider enabling by default

### Migration Guide

```python
# Before: No caching
session = Session.get_or_create(SessionConfig(app_name="my_app"))

# After: With caching
session = Session.get_or_create(
    SessionConfig(
        app_name="my_app",
        cache=CacheConfig(enabled=True, ttl="1h")
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
