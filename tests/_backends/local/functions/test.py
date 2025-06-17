#!/usr/bin/env python3
"""
Quick test to check Polars behavior when left_on and right_on use the same column name.
"""

import polars as pl


def test_same_name_left_right_on():
    """Test if Polars treats left_on="id", right_on="id" same as on="id"."""

    df1 = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })

    df2 = pl.DataFrame({
        "id": [1, 2, 4],
        "age": [25, 30, 35]
    })

    print("Test DataFrames:")
    print("df1:", list(df1.columns))
    print("df2:", list(df2.columns))
    print()

    # Test 1: Using on="id"
    result_on = df1.join(df2, on="id")
    print("Using on='id':")
    print("Schema:", list(result_on.columns))
    print("Shape:", result_on.shape)
    print()

    # Test 2: Using left_on="id", right_on="id"
    result_left_right = df1.join(df2, left_on="id", right_on="id")
    print("Using left_on='id', right_on='id':")
    print("Schema:", list(result_left_right.columns))
    print("Shape:", result_left_right.shape)
    print()

    # Compare
    schemas_match = list(result_on.columns) == list(result_left_right.columns)
    print(f"Schemas match: {schemas_match}")

    if schemas_match:
        print("✅ Polars treats left_on='id', right_on='id' same as on='id'")
    else:
        print("❌ Polars treats them differently:")
        print(f"  on='id' schema: {list(result_on.columns)}")
        print(f"  left_on/right_on schema: {list(result_left_right.columns)}")

if __name__ == "__main__":
    test_same_name_left_right_on()
