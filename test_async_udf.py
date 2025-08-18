"""Quick test of async UDF functionality."""

import asyncio
import time

import fenic as fc
from fenic.core.types import IntegerType
from fenic import Session, SessionConfig

@fc.async_udf(
    return_type=IntegerType,
    max_concurrency=3,
    timeout_seconds=5,
    num_retries=1
)
async def slow_add(x: int, y: int) -> int:
    """Simple async function that adds two numbers after a delay."""
    await asyncio.sleep(0.5)
    return x + y


def main():
    # Create a session and test data
    session = Session.get_or_create(
        SessionConfig(
            app_name="test_async_udf",
        )
    )

    data = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
        {"a": 7, "b": 8},
        {"a": 9, "b": 10},
    ]

    df = session.create_dataframe(data)

    print("Original DataFrame:")
    df.show()

    # Apply async UDF
    start_time = time.time()
    result = df.select(
        fc.col("a"),
        fc.col("b"),
        slow_add(fc.col("a"), fc.col("b")).alias("sum")
    )

    print("\nDataFrame with async UDF result:")
    result.show()

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.2f} seconds")
    print(f"With max_concurrency=3, 5 items with 0.5s delay each should take ~1s")

    # Test with failure
    @fc.async_udf(
        return_type=IntegerType,
        max_concurrency=2,
        timeout_seconds=1,
        num_retries=0
    )
    async def failing_func(x: int) -> int:
        if x > 5:
            raise ValueError(f"Value {x} is too large!")
        await asyncio.sleep(0.1)
        return x * 2

    df2 = session.create_dataframe([{"x": i} for i in range(1, 10)])

    print("\n\nTesting with failures (values > 5 will fail):")
    result2 = df2.select(
        fc.col("x"),
        failing_func(fc.col("x")).alias("doubled")
    )
    result2.show()
    print("Note: Failed items return None")


if __name__ == "__main__":
    main()
