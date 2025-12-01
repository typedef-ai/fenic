"""SQLite-backed LLM response cache implementation."""

import json
import logging
import queue
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from fenic._inference.cache.protocol import CachedResponse, CacheStats, LLMResponseCache
from fenic._inference.types import FenicCompletionsResponse

logger = logging.getLogger(__name__)
SQLITE_MAX_CONNECTIONS_DEFAULT = 3

class SQLiteLLMCache(LLMResponseCache):
    """SQLite-backed LLM response cache with normalized storage.

    Thread-safe implementation using thread-local connections and WAL mode.
    Stores responses in a normalized schema with JSON serialization for complex
    data.

    Attributes:
        db_path: Path to SQLite database file.
        ttl_seconds: Time-to-live in seconds for cached entries.
        max_size_mb: Maximum cache size in MB before LRU eviction.
        namespace: Cache namespace for isolation.

    Example:
        Basic usage:

        ```python
        from fenic._inference.cache.sqlite_cache import SQLiteLLMCache
        from fenic._inference.types import FenicCompletionsResponse

        cache = SQLiteLLMCache(
            db_path="/path/to/cache.db",
            ttl_seconds=3600,  # 1 hour
            max_size_mb=1000,
            namespace="my_project",
        )

        # Store a response
        response = FenicCompletionsResponse(completion="Hello!", logprobs=None)
        cache.set("key123", response, "gpt-4o-mini")

        # Retrieve it
        cached = cache.get("key123")
        if cached:
            print(cached.completion)  # "Hello!"

        # Get stats
        stats = cache.stats()
        print(f"Hit rate: {stats.hit_rate:.1%}")

        # Clean up
        cache.close()
        ```

        Batch operations:

        ```python
        # Batch get
        keys = ["key1", "key2", "key3"]
        results = cache.get_batch(keys)

        # Batch set
        entries = [
            ("key1", response1, "gpt-4o-mini"),
            ("key2", response2, "gpt-4o-mini"),
        ]
        count = cache.set_batch(entries)
        ```
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        ttl_seconds: int = 3600,
        max_size_mb: int = 1000,
        namespace: str = "default",
        max_connections: int = SQLITE_MAX_CONNECTIONS_DEFAULT,
    ):
        """Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database. If None, uses ~/.fenic/llm_cache.db.
            ttl_seconds: Time-to-live for cached entries (default: 3600 = 1 hour).
            max_size_mb: Maximum cache size before LRU eviction (default: 1000 MB).
            namespace: Cache namespace for isolation (default: "default").
            max_connections: Number of connections in the pool (default: 3).
                Note: SQLite in WAL mode allows only one writer at a time, so having
                too many connections can increase contention. 2-3 connections is optimal
                for the write-heavy cache workload.
        """
        if db_path is None:
            cache_dir = Path.home() / ".fenic"
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "llm_cache.db")

        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.max_size_mb = max_size_mb
        self.namespace = namespace
        self.max_connections = max_connections

        # Connection pool
        self._pool = queue.Queue(maxsize=max_connections)
        self._pool_lock = threading.Lock()
        self._initialized_connections = 0
        self._closed = False

        # Statistics
        self._stats_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._errors = 0

        # Try to initialize, delete if corrupted
        try:
            self._init_db()
        except sqlite3.DatabaseError as e:
            logger.warning(f"Cache DB corrupted ({e}), deleting and recreating")
            self._handle_corruption()
            self._init_db()

        logger.info(
            f"Initialized SQLite cache at {self.db_path} "
            f"(ttl={ttl_seconds}s, max_size={max_size_mb}MB, namespace={namespace}, pool_size={max_connections})"
        )

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper settings.

        Returns:
            Configured SQLite connection with WAL mode enabled.
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
            isolation_level="DEFERRED",
        )
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB
        conn.execute("PRAGMA temp_store=MEMORY")

        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool.

        Returns:
            A database connection from the pool.

        Raises:
            ValueError: If cache has been closed.
        """
        if self._closed:
            raise ValueError("Cache has been closed")

        # First try to get an existing connection from the pool (non-blocking)
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            pass

        # No connection available, check if we can create a new one
        with self._pool_lock:
            if self._initialized_connections < self.max_connections:
                conn = self._create_connection()
                self._initialized_connections += 1
                return conn

        # Can't create more connections, wait for one to be released
        return self._pool.get(block=True)

    def release_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool.

        Args:
            conn: Connection to return to the pool.
        """
        if conn and not self._closed:
            self._pool.put(conn)

    def _handle_corruption(self):
        """Delete corrupted cache and all related files."""
        logger.warning(f"Deleting corrupted cache at {self.db_path}")
        try:
            Path(self.db_path).unlink(missing_ok=True)
            Path(f"{self.db_path}-wal").unlink(missing_ok=True)
            Path(f"{self.db_path}-shm").unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")

    def _init_db(self):
        """Initialize normalized schema."""
        conn = self.get_connection()
        try:
            conn.execute(
                """
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
            """
            )

            # Create indices
            for idx, cols in [
                ("idx_cached_at", "(namespace, cached_at)"),
                ("idx_last_accessed", "(namespace, last_accessed)"),
                ("idx_model", "(model)"),
                ("idx_token_usage", "(namespace, total_tokens)"),
            ]:
                conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {idx}
                    ON llm_responses{cols}
                """
                )

            conn.commit()
        finally:
            self.release_connection(conn)

    def get(self, cache_key: str) -> Optional[CachedResponse]:
        """Retrieve cached response.

        Args:
            cache_key: Unique cache key.

        Returns:
            CachedResponse if found and not expired, None otherwise.
        """
        conn = self.get_connection()
        try:
            cutoff = datetime.now() - timedelta(seconds=self.ttl_seconds)

            cursor = conn.execute(
                """
                SELECT
                    completion, model, cached_at,
                    prompt_tokens, completion_tokens, total_tokens,
                    cached_tokens, thinking_tokens,
                    logprobs_data, access_count
                FROM llm_responses
                WHERE cache_key = ? AND namespace = ? AND cached_at > ?
            """,
                (cache_key, self.namespace, cutoff),
            )

            row = cursor.fetchone()

            if row:
                # Update access stats
                conn.execute(
                    """
                    UPDATE llm_responses
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE cache_key = ? AND namespace = ?
                """,
                    (datetime.now(), cache_key, self.namespace),
                )
                conn.commit()

                # Deserialize logprobs
                logprobs = None
                if row["logprobs_data"]:
                    logprobs = json.loads(row["logprobs_data"].decode("utf-8"))

                with self._stats_lock:
                    self._hits += 1

                return CachedResponse(
                    completion=row["completion"],
                    model=row["model"],
                    cached_at=datetime.fromisoformat(row["cached_at"]),
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                    total_tokens=row["total_tokens"],
                    cached_tokens=row["cached_tokens"] or 0,
                    thinking_tokens=row["thinking_tokens"] or 0,
                    logprobs=logprobs,
                    access_count=row["access_count"] + 1,
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
        finally:
            self.release_connection(conn)

    def get_batch(self, cache_keys: List[str]) -> Dict[str, CachedResponse]:
        """Retrieve multiple cached responses.

        Args:
            cache_keys: List of cache keys to retrieve.

        Returns:
            Dictionary mapping cache keys to responses (only includes hits).
        """
        result = {}

        if not cache_keys:
            return result

        conn = self.get_connection()
        try:
            cutoff = datetime.now() - timedelta(seconds=self.ttl_seconds)

            placeholders = ",".join("?" * len(cache_keys))
            cursor = conn.execute(
                f"""
                SELECT
                    cache_key, completion, model, cached_at,
                    prompt_tokens, completion_tokens, total_tokens,
                    cached_tokens, thinking_tokens,
                    logprobs_data, access_count
                FROM llm_responses
                WHERE cache_key IN ({placeholders})
                  AND namespace = ?
                  AND cached_at > ?
            """, # nosec: B608 this is a placeholder string we control
                (*cache_keys, self.namespace, cutoff),
            )

            found_keys = set()

            for row in cursor:
                key = row["cache_key"]
                found_keys.add(key)

                logprobs = None
                if row["logprobs_data"]:
                    logprobs = json.loads(row["logprobs_data"].decode("utf-8"))

                result[key] = CachedResponse(
                    completion=row["completion"],
                    model=row["model"],
                    cached_at=datetime.fromisoformat(row["cached_at"]),
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                    total_tokens=row["total_tokens"],
                    cached_tokens=row["cached_tokens"] or 0,
                    thinking_tokens=row["thinking_tokens"] or 0,
                    logprobs=logprobs,
                    access_count=row["access_count"] + 1,
                )

            # Update access stats
            if found_keys:
                now = datetime.now()
                placeholders = ",".join("?" * len(found_keys))
                conn.execute(
                    f"""
                    UPDATE llm_responses
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE cache_key IN ({placeholders})
                      AND namespace = ?
                """, # nosec: B608 this is a placeholder string we control
                    (now, *found_keys, self.namespace),
                )
                conn.commit()

            with self._stats_lock:
                self._hits += len(found_keys)
                self._misses += len(cache_keys) - len(found_keys)

        except Exception as e:
            with self._stats_lock:
                self._errors += 1
            logger.warning(f"Cache get_batch error: {e}")
        finally:
            self.release_connection(conn)

        return result

    def set(
        self,
        cache_key: str,
        response: FenicCompletionsResponse,
        model: str,
    ) -> bool:
        """Store response in cache.

        Args:
            cache_key: Unique cache key.
            response: The response to cache.
            model: The model that generated this response.

        Returns:
            True if stored successfully, False otherwise.
        """
        conn = self.get_connection()
        try:
            now = datetime.now()

            # Extract normalized fields
            prompt_tokens = response.usage.prompt_tokens if response.usage else None
            completion_tokens = (
                response.usage.completion_tokens if response.usage else None
            )
            total_tokens = response.usage.total_tokens if response.usage else None
            cached_tokens = response.usage.cached_tokens if response.usage else 0
            thinking_tokens = response.usage.thinking_tokens if response.usage else 0

            # Serialize logprobs as JSON
            logprobs_data = None
            if response.logprobs:
                logprobs_data = json.dumps(response.logprobs).encode("utf-8")

            conn.execute(
                """
                INSERT OR REPLACE INTO llm_responses
                (cache_key, namespace, model, completion, cached_at, last_accessed,
                 prompt_tokens, completion_tokens, total_tokens, cached_tokens, thinking_tokens,
                 logprobs_data, response_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
                (
                    cache_key,
                    self.namespace,
                    model,
                    response.completion,
                    now,
                    now,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cached_tokens,
                    thinking_tokens,
                    logprobs_data,
                ),
            )

            conn.commit()

            with self._stats_lock:
                self._stores += 1

            self._maybe_evict(conn)

            return True

        except Exception as e:
            with self._stats_lock:
                self._errors += 1
            logger.warning(f"Cache set error for key {cache_key[:8]}...: {e}")
            return False
        finally:
            self.release_connection(conn)

    def set_batch(
        self, entries: List[tuple[str, FenicCompletionsResponse, str]]
    ) -> int:
        """Store multiple responses.

        Args:
            entries: List of (cache_key, response, model) tuples.

        Returns:
            Count of successfully stored entries.
        """
        stored = 0

        if not entries:
            return 0

        conn = self.get_connection()
        try:
            now = datetime.now()

            for cache_key, response, model in entries:
                if cache_key:
                    try:
                        prompt_tokens = (
                            response.usage.prompt_tokens if response.usage else None
                        )
                        completion_tokens = (
                            response.usage.completion_tokens if response.usage else None
                        )
                        total_tokens = (
                            response.usage.total_tokens if response.usage else None
                        )
                        cached_tokens = (
                            response.usage.cached_tokens if response.usage else 0
                        )
                        thinking_tokens = (
                            response.usage.thinking_tokens if response.usage else 0
                        )

                        logprobs_data = None
                        if response.logprobs:
                            logprobs_data = json.dumps(response.logprobs).encode("utf-8")

                        conn.execute(
                            """
                            INSERT OR REPLACE INTO llm_responses
                            (cache_key, namespace, model, completion, cached_at, last_accessed,
                            prompt_tokens, completion_tokens, total_tokens, cached_tokens, thinking_tokens,
                            logprobs_data, response_version)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                            (
                                cache_key,
                                self.namespace,
                                model,
                                response.completion,
                                now,
                                now,
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                                cached_tokens,
                                thinking_tokens,
                                logprobs_data,
                            ),
                        )

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
        finally:
            self.release_connection(conn)

        return stored


    def clear(self) -> int:
        """Clear all entries in namespace.

        Returns:
            Number of entries cleared.
        """
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                DELETE FROM llm_responses WHERE namespace = ?
            """,
                (self.namespace,),
            )
            conn.commit()

            cleared = cursor.rowcount
            logger.info(f"Cleared {cleared} entries from namespace '{self.namespace}'")
            return cleared
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")
            return 0
        finally:
            self.release_connection(conn)

    def stats(self) -> CacheStats:
        """Get performance statistics.

        Returns:
            CacheStats with current metrics.
        """
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM llm_responses WHERE namespace = ?
            """,
                (self.namespace,),
            )
            total_entries = cursor.fetchone()["count"]

            cursor = conn.execute(
                """
                SELECT page_count * page_size as size
                FROM pragma_page_count(), pragma_page_size()
            """
            )
            size_bytes = cursor.fetchone()["size"]

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
                    size_bytes=size_bytes,
                )
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return CacheStats(
                hits=0,
                misses=0,
                stores=0,
                errors=0,
                hit_rate=0.0,
                total_entries=0,
                size_bytes=0,
            )
        finally:
            self.release_connection(conn)

    def _maybe_evict(self, conn: sqlite3.Connection):
        """Evict LRU entries if cache exceeds max size.

        Args:
            conn: Database connection to use (reuses caller's connection to avoid deadlock).
        """
        try:
            cursor = conn.execute(
                """
                SELECT page_count * page_size as size
                FROM pragma_page_count(), pragma_page_size()
            """
            )
            size_bytes = cursor.fetchone()["size"]
            size_mb = size_bytes / (1024 * 1024)

            if size_mb > self.max_size_mb:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as count FROM llm_responses WHERE namespace = ?
                """,
                    (self.namespace,),
                )
                total_count = cursor.fetchone()["count"]
                evict_count = max(1, total_count // 10)  # Evict 10%

                conn.execute(
                    """
                    DELETE FROM llm_responses
                    WHERE cache_key IN (
                        SELECT cache_key FROM llm_responses
                        WHERE namespace = ?
                        ORDER BY last_accessed ASC
                        LIMIT ?
                    )
                """,
                    (self.namespace, evict_count),
                )
                conn.commit()

                logger.info(
                    f"Evicted {evict_count} LRU entries "
                    f"(size: {size_mb:.1f}MB > {self.max_size_mb}MB)"
                )

        except Exception as e:
            logger.warning(f"Cache eviction error: {e}")

    def close(self):
        """Close all pooled connections and checkpoint WAL.

        Simple approach: close all connections, try to checkpoint WAL.
        If something fails, just log it - the cache is expendable.
        """
        self._closed = True

        # Close all pooled connections
        closed_count = 0
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
                closed_count += 1
            except (queue.Empty, Exception) as e:
                logger.debug(f"Error during pool cleanup: {e}")
                pass

        logger.debug(f"Closed {closed_count} pooled connections")

        # Try to checkpoint WAL (don't care if it fails)
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
            logger.debug(f"WAL checkpoint completed for cache at {self.db_path}")
        except Exception as e:
            logger.debug(f"WAL checkpoint failed (not critical): {e}")
