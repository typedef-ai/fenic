"""Unit tests for SQLiteLLMCache."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from fenic._inference.cache.sqlite_cache import SQLiteLLMCache
from fenic._inference.types import FenicCompletionsResponse, ResponseUsage


class TestSQLiteLLMCache:
    """Test suite for SQLiteLLMCache."""

    @pytest.fixture
    def temp_cache(self):
        """Create a temporary cache for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        cache = SQLiteLLMCache(
            db_path=db_path,
            ttl_seconds=3600,
            max_size_mb=100,
            namespace="test",
        )

        yield cache

        cache.close()
        Path(db_path).unlink(missing_ok=True)

    def test_set_and_get(self, temp_cache: SQLiteLLMCache):
        """Test basic set and get operations."""
        response = FenicCompletionsResponse(
            completion="Hello!",
            logprobs=None,
            usage=ResponseUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
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
        assert cached.prompt_tokens == 10
        assert cached.completion_tokens == 5

    def test_cache_miss(self, temp_cache: SQLiteLLMCache):
        """Test that non-existent keys return None."""
        cached = temp_cache.get("nonexistent")
        assert cached is None

    def test_ttl_expiration(self, temp_cache: SQLiteLLMCache):
        """Test that expired entries are not returned."""
        response = FenicCompletionsResponse(
            completion="Test", logprobs=None, usage=None
        )
        temp_cache.set("test_key", response, "gpt-4o-mini")

        # Manually expire by setting cached_at to 2 hours ago
        conn = temp_cache.get_connection()
        try:
            old_date = datetime.now() - timedelta(hours=2)
            conn.execute(
                """
                UPDATE llm_responses
                SET cached_at = ?
                WHERE cache_key = ?
            """,
                (old_date, "test_key"),
            )
            conn.commit()
        finally:
            temp_cache.release_connection(conn)

        # Should be expired
        cached = temp_cache.get("test_key")
        assert cached is None

    def test_access_count(self, temp_cache: SQLiteLLMCache):
        """Test that access count is incremented on get."""
        response = FenicCompletionsResponse(
            completion="Test", logprobs=None, usage=None
        )
        temp_cache.set("test_key", response, "gpt-4o-mini")

        # First access
        cached1 = temp_cache.get("test_key")
        assert cached1 is not None
        assert cached1.access_count == 1

        # Second access (note: we need to fetch from DB again to see updated count)
        cached2 = temp_cache.get("test_key")
        assert cached2 is not None
        assert cached2.access_count == 2

    def test_batch_get(self, temp_cache: SQLiteLLMCache):
        """Test batch get operations."""
        responses = [
            FenicCompletionsResponse(
                completion=f"Response {i}", logprobs=None, usage=None
            )
            for i in range(10)
        ]

        # Store responses
        for i, response in enumerate(responses):
            temp_cache.set(f"key_{i}", response, "gpt-4o-mini")

        # Batch get
        keys = [f"key_{i}" for i in range(10)]
        results = temp_cache.get_batch(keys)

        assert len(results) == 10
        for i, key in enumerate(keys):
            assert results[key] is not None
            assert results[key].completion == f"Response {i}"

    def test_batch_get_with_missing_keys(self, temp_cache: SQLiteLLMCache):
        """Test batch get with some missing keys."""
        # Store only some keys
        for i in range(5):
            response = FenicCompletionsResponse(
                completion=f"Response {i}", logprobs=None, usage=None
            )
            temp_cache.set(f"key_{i}", response, "gpt-4o-mini")

        # Request more keys than stored
        keys = [f"key_{i}" for i in range(10)]
        results = temp_cache.get_batch(keys)

        # Only the 5 cached keys should be returned
        assert len(results) == 5
        for i in range(5):
            assert results[f"key_{i}"].completion == f"Response {i}"
        # Missing keys should not be in results
        for i in range(5, 10):
            assert f"key_{i}" not in results

    def test_batch_set(self, temp_cache: SQLiteLLMCache):
        """Test batch set operations."""
        responses = [
            FenicCompletionsResponse(
                completion=f"Response {i}", logprobs=None, usage=None
            )
            for i in range(10)
        ]

        # Batch set
        entries = [(f"key_{i}", responses[i], "gpt-4o-mini") for i in range(10)]
        stored = temp_cache.set_batch(entries)
        assert stored == 10

        # Verify all stored
        for i in range(10):
            cached = temp_cache.get(f"key_{i}")
            assert cached is not None
            assert cached.completion == f"Response {i}"

    def test_batch_set_skips_null_keys(self, temp_cache: SQLiteLLMCache):
        """Ensure batch set ignores entries with null cache keys."""
        responses = [
            FenicCompletionsResponse(
                completion=f"Response {i}", logprobs=None, usage=None
            )
            for i in range(3)
        ]

        entries = [
            ("valid_key_0", responses[0], "gpt-4o-mini"),
            (None, responses[1], "gpt-4o-mini"),
            ("valid_key_1", responses[2], "gpt-4o-mini"),
        ]

        stored = temp_cache.set_batch(entries)
        assert stored == 2

        for key in ("valid_key_0", "valid_key_1"):
            cached = temp_cache.get(key)
            assert cached is not None
            assert cached.completion.startswith("Response")

        conn = temp_cache.get_connection()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM llm_responses WHERE namespace = ?",
                (temp_cache.namespace,),
            ).fetchone()[0]
        finally:
            temp_cache.release_connection(conn)

        assert count == 2

    def test_clear(self, temp_cache: SQLiteLLMCache):
        """Test clear operation."""
        # Store multiple entries
        for i in range(10):
            response = FenicCompletionsResponse(
                completion=f"Response {i}", logprobs=None, usage=None
            )
            temp_cache.set(f"key_{i}", response, "gpt-4o-mini")

        # Clear all
        cleared = temp_cache.clear()
        assert cleared == 10

        # Verify all gone
        for i in range(10):
            assert temp_cache.get(f"key_{i}") is None

    def test_namespace_isolation(self):
        """Test that different namespaces are isolated."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            cache1 = SQLiteLLMCache(db_path=db_path, ttl_seconds=3600, namespace="ns1")
            cache2 = SQLiteLLMCache(db_path=db_path, ttl_seconds=3600, namespace="ns2")

            response = FenicCompletionsResponse(
                completion="Test", logprobs=None, usage=None
            )

            # Store in namespace 1
            cache1.set("key", response, "gpt-4o-mini")

            # Should be in namespace 1
            assert cache1.get("key") is not None

            # Should NOT be in namespace 2
            assert cache2.get("key") is None

            cache1.close()
            cache2.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_statistics(self, temp_cache: SQLiteLLMCache):
        """Test statistics tracking."""
        response = FenicCompletionsResponse(
            completion="Test", logprobs=None, usage=None
        )

        temp_cache.set("key1", response, "gpt-4o-mini")
        temp_cache.get("key1")  # Hit
        temp_cache.get("key1")  # Hit
        temp_cache.get("key2")  # Miss

        stats = temp_cache.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.stores == 1
        assert stats.hit_rate == 2 / 3
        assert stats.total_entries == 1

    def test_with_usage_info(self, temp_cache: SQLiteLLMCache):
        """Test caching responses with full usage information."""
        response = FenicCompletionsResponse(
            completion="Test response",
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cached_tokens=20,
                thinking_tokens=10,
            ),
        )

        temp_cache.set("test_key", response, "gpt-4o-mini")
        cached = temp_cache.get("test_key")

        assert cached is not None
        assert cached.prompt_tokens == 100
        assert cached.completion_tokens == 50
        assert cached.total_tokens == 150
        assert cached.cached_tokens == 20
        assert cached.thinking_tokens == 10

    def test_with_logprobs(self, temp_cache: SQLiteLLMCache):
        """Test caching responses with logprobs."""
        # Mock logprobs data (simplified version)
        logprobs_data = [
            {"token": "Hello", "logprob": -0.5},
            {"token": "world", "logprob": -0.3},
        ]

        response = FenicCompletionsResponse(
            completion="Hello world",
            logprobs=logprobs_data,
            usage=None,
        )

        temp_cache.set("test_key", response, "gpt-4o-mini")
        cached = temp_cache.get("test_key")

        assert cached is not None
        assert cached.logprobs == logprobs_data

    def test_to_fenic_response(self, temp_cache: SQLiteLLMCache):
        """Test conversion from CachedResponse to FenicCompletionsResponse."""
        original = FenicCompletionsResponse(
            completion="Test",
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cached_tokens=0,
                thinking_tokens=0,
            ),
        )

        temp_cache.set("test_key", original, "gpt-4o-mini")
        cached = temp_cache.get("test_key")

        assert cached is not None
        restored = cached.to_fenic_response()

        assert restored.completion == original.completion
        assert restored.usage.prompt_tokens == original.usage.prompt_tokens
        assert restored.usage.completion_tokens == original.usage.completion_tokens
        assert restored.usage.total_tokens == original.usage.total_tokens

    def test_update_existing_entry(self, temp_cache: SQLiteLLMCache):
        """Test that setting same key updates the entry."""
        response1 = FenicCompletionsResponse(
            completion="First", logprobs=None, usage=None
        )
        response2 = FenicCompletionsResponse(
            completion="Second", logprobs=None, usage=None
        )

        temp_cache.set("test_key", response1, "gpt-4o-mini")
        cached1 = temp_cache.get("test_key")
        assert cached1.completion == "First"

        # Update
        temp_cache.set("test_key", response2, "gpt-4o-mini")
        cached2 = temp_cache.get("test_key")
        assert cached2.completion == "Second"

    def test_empty_batch_operations(self, temp_cache: SQLiteLLMCache):
        """Test batch operations with empty lists."""
        results = temp_cache.get_batch([])
        assert results == {}

        stored = temp_cache.set_batch([])
        assert stored == 0

    def test_default_db_path(self):
        """Test that default db_path is created correctly."""
        cache = SQLiteLLMCache(ttl_seconds=3600, namespace="test")

        expected_path = Path.home() / ".fenic" / "llm_cache.db"
        assert cache.db_path == str(expected_path)

        cache.close()

    def test_connection_pool_initialization(self):
        """Test that connection pool is initialized with correct size."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            cache = SQLiteLLMCache(
                db_path=db_path,
                ttl_seconds=3600,
                namespace="test",
            )

            # One connection is created during initialization for schema setup
            assert cache._initialized_connections == 1

            # Get multiple connections simultaneously - should create them as needed
            conn1 = cache.get_connection()  # Gets the one from pool
            conn2 = cache.get_connection()  # Creates a new one
            conn3 = cache.get_connection()  # Creates another new one
            assert cache._initialized_connections == 3

            cache.release_connection(conn1)
            cache.release_connection(conn2)
            cache.release_connection(conn3)

            cache.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_connection_pool_concurrency(self, temp_cache: SQLiteLLMCache):
        """Test that connection pool handles concurrent requests."""
        import threading

        results = []
        errors = []

        def worker(thread_id):
            try:
                # Each thread does some cache operations
                response = FenicCompletionsResponse(
                    completion=f"Response from thread {thread_id}",
                    logprobs=None,
                    usage=None,
                )

                for i in range(5):
                    key = f"thread_{thread_id}_key_{i}"
                    temp_cache.set(key, response, "gpt-4o-mini")
                    cached = temp_cache.get(key)
                    assert cached is not None
                    results.append((thread_id, i))
            except Exception as e:
                errors.append((thread_id, e))

        # Run 10 threads concurrently
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Check all operations succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 50  # 10 threads * 5 operations each

    def test_wal_checkpoint_on_close(self):
        """Test that WAL is checkpointed when cache is closed."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            cache = SQLiteLLMCache(db_path=db_path, ttl_seconds=3600, namespace="test")

            # Add some data to create WAL file
            for i in range(10):
                response = FenicCompletionsResponse(
                    completion=f"Response {i}", logprobs=None, usage=None
                )
                cache.set(f"key_{i}", response, "gpt-4o-mini")

            # Close cache (should checkpoint WAL)
            cache.close()

            # Check file sizes
            db_file = Path(db_path)
            wal_file = Path(f"{db_path}-wal")
            Path(f"{db_path}-shm")

            # Main DB should exist and have data
            assert db_file.exists()
            assert db_file.stat().st_size > 0

            # WAL file should either not exist or be empty after checkpoint
            if wal_file.exists():
                # If WAL exists, it should be very small (header only)
                assert wal_file.stat().st_size < 1000

            # Verify data is persisted by opening DB directly
            import sqlite3

            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM llm_responses")
            count = cursor.fetchone()[0]
            conn.close()

            assert count == 10

        finally:
            for path in [db_path, f"{db_path}-wal", f"{db_path}-shm"]:
                Path(path).unlink(missing_ok=True)

    def test_corruption_handling_on_init(self):
        """Test that corrupted DB is deleted and recreated."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create a corrupted database file
            with open(db_path, "wb") as f:
                f.write(b"This is not a valid SQLite database")

            # Should delete and recreate
            cache = SQLiteLLMCache(db_path=db_path, ttl_seconds=3600, namespace="test")

            # Should be able to use it normally
            response = FenicCompletionsResponse(
                completion="Test", logprobs=None, usage=None
            )
            success = cache.set("test_key", response, "gpt-4o-mini")
            assert success

            cached = cache.get("test_key")
            assert cached is not None

            cache.close()

        finally:
            for path in [db_path, f"{db_path}-wal", f"{db_path}-shm"]:
                Path(path).unlink(missing_ok=True)

    def test_use_after_close(self, temp_cache: SQLiteLLMCache):
        """Test that operations fail after cache is closed."""
        response = FenicCompletionsResponse(
            completion="Test", logprobs=None, usage=None
        )

        # Set before closing
        temp_cache.set("test_key", response, "gpt-4o-mini")

        # Close cache
        temp_cache.close()

        # Operations should raise ValueError
        with pytest.raises(ValueError, match="has been closed"):
            temp_cache.get("test_key")

        with pytest.raises(ValueError, match="has been closed"):
            temp_cache.set("another_key", response, "gpt-4o-mini")

    def test_close_idempotency(self, temp_cache: SQLiteLLMCache):
        """Test that calling close multiple times is safe."""
        response = FenicCompletionsResponse(
            completion="Test", logprobs=None, usage=None
        )
        temp_cache.set("test_key", response, "gpt-4o-mini")

        # Should not raise errors
        temp_cache.close()
        temp_cache.close()
        temp_cache.close()
