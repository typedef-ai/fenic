"""Tests for async UDF functionality."""

import asyncio
import time
from typing import Dict, Optional

import pydantic
import pytest

import fenic as fc
from fenic.api.session import Session
from fenic.core.error import ExecutionError, ValidationError
from fenic.core.types import IntegerType, JsonType, StringType, StructField, StructType


class TestAsyncUDF:
    """Test cases for async UDF functionality."""

    def test_basic_async_udf(self, local_session: Session):
        """Test basic async UDF with simple addition."""

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=3,
            timeout_seconds=5,
            num_retries=0
        )
        async def async_add(x: int, y: int) -> int:
            await asyncio.sleep(0.1)
            return x + y

        data = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ]

        df = local_session.create_dataframe(data)
        result = df.select(
            fc.col("a"),
            fc.col("b"),
            async_add(fc.col("a"), fc.col("b")).alias("sum")
        ).to_pylist()

        expected = [
            {"a": 1, "b": 2, "sum": 3},
            {"a": 3, "b": 4, "sum": 7},
            {"a": 5, "b": 6, "sum": 11},
        ]

        assert result == expected

    def test_async_udf_concurrency(self, local_session: Session):
        """Test that async UDF respects concurrency limits."""
        call_times = []

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=2,  # Only 2 concurrent calls
            timeout_seconds=5,
            num_retries=0
        )
        async def track_concurrency(x: int) -> int:
            start = time.time()
            call_times.append(("start", start, x))
            await asyncio.sleep(0.2)  # Each call takes 0.2 seconds
            end = time.time()
            call_times.append(("end", end, x))
            return x * 2

        # With 5 items and max_concurrency=2, should take ~0.6 seconds
        # (3 batches: 2, 2, 1)
        data = [{"x": i} for i in range(1, 6)]
        df = local_session.create_dataframe(data)

        start_time = time.time()
        result = df.select(
            fc.col("x"),
            track_concurrency(fc.col("x")).alias("doubled")
        ).to_pylist()
        elapsed = time.time() - start_time

        # Verify results
        expected = [{"x": i, "doubled": i * 2} for i in range(1, 6)]
        assert result == expected

        # Verify timing - should take at least 0.5 seconds (3 batches * 0.2s)
        assert elapsed >= 0.5, f"Expected at least 0.5s, got {elapsed:.2f}s"
        assert elapsed < 1.0, f"Expected less than 1.0s, got {elapsed:.2f}s"

    def test_async_udf_with_failures(self, local_session: Session):
        """Test async UDF handling of individual failures."""

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=3,
            timeout_seconds=1,
            num_retries=0
        )
        async def failing_func(x: int) -> int:
            if x > 3:
                raise ValueError(f"Value {x} is too large!")
            await asyncio.sleep(0.05)
            return x * 10

        data = [{"x": i} for i in range(1, 6)]
        df = local_session.create_dataframe(data)

        result = df.select(
            fc.col("x"),
            failing_func(fc.col("x")).alias("result")
        ).to_pylist()

        # Values > 3 should return None due to failure
        expected = [
            {"x": 1, "result": 10},
            {"x": 2, "result": 20},
            {"x": 3, "result": 30},
            {"x": 4, "result": None},
            {"x": 5, "result": None},
        ]

        assert result == expected

    def test_async_udf_with_retries(self, local_session: Session):
        """Test async UDF retry logic."""
        attempt_counts = {}

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=2,
            timeout_seconds=5,
            num_retries=2  # Retry up to 2 times
        )
        async def flaky_func(x: int) -> int:
            if x not in attempt_counts:
                attempt_counts[x] = 0
            attempt_counts[x] += 1

            # Fail first attempt for x > 2
            if x > 2 and attempt_counts[x] == 1:
                raise ValueError(f"Simulated failure for {x}")

            await asyncio.sleep(0.02)
            return x * 100

        data = [{"x": i} for i in range(1, 5)]
        df = local_session.create_dataframe(data)

        result = df.select(
            fc.col("x"),
            flaky_func(fc.col("x")).alias("result")
        ).to_pylist()

        expected = [
            {"x": 1, "result": 100},
            {"x": 2, "result": 200},
            {"x": 3, "result": 300},  # Should succeed on retry
            {"x": 4, "result": 400},  # Should succeed on retry
        ]

        assert result == expected

        # Verify retry attempts
        assert attempt_counts[1] == 1  # No retry needed
        assert attempt_counts[2] == 1  # No retry needed
        assert attempt_counts[3] == 2  # One retry
        assert attempt_counts[4] == 2  # One retry

    def test_async_udf_with_timeout(self, local_session: Session):
        """Test async UDF timeout handling."""

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=3,
            timeout_seconds=0.15,  # 150ms timeout
            num_retries=0
        )
        async def slow_func(x: int) -> int:
            if x > 2:
                await asyncio.sleep(0.3)  # Will timeout
            else:
                await asyncio.sleep(0.05)  # Won't timeout
            return x * 1000

        data = [{"x": i} for i in range(1, 5)]
        df = local_session.create_dataframe(data)

        result = df.select(
            fc.col("x"),
            slow_func(fc.col("x")).alias("result")
        ).to_pylist()

        # x > 2 should timeout and return None
        expected = [
            {"x": 1, "result": 1000},
            {"x": 2, "result": 2000},
            {"x": 3, "result": None},  # Timeout
            {"x": 4, "result": None},  # Timeout
        ]

        assert result == expected

    def test_async_udf_with_struct_return(self, local_session: Session):
        """Test async UDF returning struct types."""

        @fc.async_udf(
            return_type=StructType([
                StructField("original", IntegerType),
                StructField("squared", IntegerType),
                StructField("status", StringType)
            ]),
            max_concurrency=3,
            timeout_seconds=5,
            num_retries=0
        )
        async def compute_stats(x: int) -> Dict:
            await asyncio.sleep(0.02)
            return {
                "original": x,
                "squared": x * x,
                "status": "even" if x % 2 == 0 else "odd"
            }

        data = [{"x": i} for i in range(1, 4)]
        df = local_session.create_dataframe(data)

        result = df.select(
            fc.col("x"),
            compute_stats(fc.col("x")).alias("stats")
        ).to_pylist()

        expected = [
            {"x": 1, "stats": {"original": 1, "squared": 1, "status": "odd"}},
            {"x": 2, "stats": {"original": 2, "squared": 4, "status": "even"}},
            {"x": 3, "stats": {"original": 3, "squared": 9, "status": "odd"}},
        ]

        assert result == expected

    def test_multiple_async_udfs(self, local_session: Session):
        """Test multiple async UDFs in the same query."""

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=2,
            timeout_seconds=5,
            num_retries=0
        )
        async def double(x: int) -> int:
            await asyncio.sleep(0.02)
            return x * 2

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=2,
            timeout_seconds=5,
            num_retries=0
        )
        async def triple(x: int) -> int:
            await asyncio.sleep(0.02)
            return x * 3

        data = [{"x": i} for i in range(1, 4)]
        df = local_session.create_dataframe(data)

        result = df.select(
            fc.col("x"),
            double(fc.col("x")).alias("doubled"),
            triple(fc.col("x")).alias("tripled")
        ).to_pylist()

        expected = [
            {"x": 1, "doubled": 2, "tripled": 3},
            {"x": 2, "doubled": 4, "tripled": 6},
            {"x": 3, "doubled": 6, "tripled": 9},
        ]

        assert result == expected

    def test_async_udf_with_none_handling(self, local_session: Session):
        """Test async UDF handling of None inputs."""

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=3,
            timeout_seconds=5,
            num_retries=0
        )
        async def handle_none(x: Optional[int]) -> Optional[int]:
            if x is None:
                return None
            await asyncio.sleep(0.02)
            return x + 100

        data = [
            {"x": 1},
            {"x": None},
            {"x": 3},
            {"x": None},
            {"x": 5},
        ]

        df = local_session.create_dataframe(data)
        result = df.select(
            fc.col("x"),
            handle_none(fc.col("x")).alias("result")
        ).to_pylist()

        expected = [
            {"x": 1, "result": 101},
            {"x": None, "result": None},
            {"x": 3, "result": 103},
            {"x": None, "result": None},
            {"x": 5, "result": 105},
        ]

        assert result == expected

    def test_async_udf_single_argument(self, local_session: Session):
        """Test async UDF with single argument."""

        @fc.async_udf(
            return_type=StringType,
            max_concurrency=2,
            timeout_seconds=5,
            num_retries=0
        )
        async def format_number(x: int) -> str:
            await asyncio.sleep(0.02)
            return f"Number: {x}"

        data = [{"x": i} for i in range(1, 4)]
        df = local_session.create_dataframe(data)

        result = df.select(
            fc.col("x"),
            format_number(fc.col("x")).alias("formatted")
        ).to_pylist()

        expected = [
            {"x": 1, "formatted": "Number: 1"},
            {"x": 2, "formatted": "Number: 2"},
            {"x": 3, "formatted": "Number: 3"},
        ]

        assert result == expected

    def test_async_udf_without_decorator(self, local_session: Session):
        """Test @async_udf without decorator syntax."""
        async def simple_func(x: int) -> int:
            return x * 2
        async_udf = fc.async_udf(simple_func, return_type=IntegerType)
        df = local_session.create_dataframe([{"x": i} for i in range(1, 4)])
        result = df.select(fc.col("x"), async_udf(fc.col("x")).alias("result")).to_pylist()
        expected = [{"x": 1, "result": 2}, {"x": 2, "result": 4}, {"x": 3, "result": 6}]
        assert result == expected

    def test_async_udf_concurrency_performance_comparison(self, local_session: Session):
        """Test that higher concurrency is faster than serial execution."""

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=1,  # Serial execution
            timeout_seconds=10,
            num_retries=0
        )
        async def slow_serial_func(x: int) -> int:
            await asyncio.sleep(0.1)  # 100ms per call
            return x * 10

        @fc.async_udf(
            return_type=IntegerType,
            max_concurrency=5,  # Concurrent execution
            timeout_seconds=10,
            num_retries=0
        )
        async def slow_concurrent_func(x: int) -> int:
            await asyncio.sleep(0.1)  # Same 100ms per call
            return x * 10

        # Test data - 10 items
        data = [{"x": i} for i in range(1, 11)]
        df = local_session.create_dataframe(data)

        # Test serial execution (concurrency=1)
        start_time = time.time()
        serial_result = df.select(
            fc.col("x"),
            slow_serial_func(fc.col("x")).alias("result")
        ).to_pylist()
        serial_elapsed = time.time() - start_time

        # Test concurrent execution (concurrency=5)
        start_time = time.time()
        concurrent_result = df.select(
            fc.col("x"),
            slow_concurrent_func(fc.col("x")).alias("result")
        ).to_pylist()
        concurrent_elapsed = time.time() - start_time

        # Verify results are the same
        expected = [{"x": i, "result": i * 10} for i in range(1, 11)]
        assert serial_result == expected
        assert concurrent_result == expected

        # Verify timing expectations:
        # Serial: ~1.0 seconds (10 items × 0.1s each)
        # Concurrent: ~0.2 seconds (2 batches × 0.1s with concurrency=5)

        # Serial should take at least 0.9 seconds (allowing some overhead)
        assert serial_elapsed >= 0.9, f"Serial execution too fast: {serial_elapsed:.2f}s (expected >= 0.9s)"

        # Concurrent should take less than 0.5 seconds
        assert concurrent_elapsed < 0.5, f"Concurrent execution too slow: {concurrent_elapsed:.2f}s (expected < 0.5s)"

    def test_async_udf_without_return_type_errors(self):
        """Test @async_udf without return type errors."""

        # This should work but needs explicit return_type
        with pytest.raises(pydantic.ValidationError):
            @fc.async_udf
            async def simple_func(x: int) -> int:
                return x * 2


    def test_async_udf_without_async_errors(self):
        """Test @async_udf without async errors."""

        with pytest.raises(ValidationError):
            @fc.async_udf(return_type=IntegerType)
            def simple_func(x: int) -> int:
                return x * 2

    def test_async_udf_with_wrong_return_type_errors(self, local_session: Session):
        """Test @async_udf with wrong return type errors."""

        with pytest.raises(ExecutionError):
            df = local_session.create_dataframe([{"x": i} for i in range(1, 4)])
            @fc.async_udf(return_type=IntegerType)
            async def simple_func(x: int) -> str:
                return "hello"
            df.select(fc.col("x"), simple_func(fc.col("x")).alias("result")).to_pylist()

    def test_async_udf_with_logical_type_errors(self):
        """Test @async_udf with logical type errors."""

        with pytest.raises(NotImplementedError):
            @fc.async_udf(return_type=JsonType)
            async def simple_func(x: int) -> int:
                return x * 2
