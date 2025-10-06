from typing import Optional, Union

import polars as pl


def apply_ingestion_coercions(df: pl.DataFrame, coerce_array: bool) -> pl.DataFrame:
    """Apply type coercions to normalize data types during ingestion.

    This is intended for ingestion from external systems (e.g., DuckDB, Parquet)
    that may produce types unsupported or inconsistently handled by Fenic.

    We only use Array type (pl.List) for embeddings.

    Coercion rules:
    - `Array` and `List` types are recursively coerced to ensure their inner types
      are normalized.
    - `Struct` types are coerced field-by-field to apply the same normalization logic.

    Args:
        df: The input Polars DataFrame containing possibly nonstandard or
            backend-specific types.

    Returns:
        A new Polars DataFrame with all coercions applied to conform to Fenic-compatible types.
    """
    expressions = []
    for col_name in df.columns:
        dtype = df[col_name].dtype
        target_dtype = _build_target_dtype(dtype, coerce_array)

        if target_dtype != dtype:
            expressions.append(pl.col(col_name).cast(target_dtype))
        else:
            expressions.append(pl.col(col_name))

    return df.select(expressions)


def _build_target_dtype(dtype: pl.DataType, coerce_array: bool) -> pl.DataType:
    if isinstance(dtype, pl.Array) and coerce_array:
        return pl.List(_build_target_dtype(dtype.inner, coerce_array))
    elif isinstance(dtype, pl.List):
        return pl.List(_build_target_dtype(dtype.inner, coerce_array))
    elif isinstance(dtype, pl.Datetime):
        # DuckDB always uses UTC as its session timezone, so we set UTC here.
        return pl.Datetime(time_unit="us", time_zone="UTC")
    elif isinstance(dtype, pl.Struct):
        new_fields = [
            pl.Field(field.name, _build_target_dtype(field.dtype, coerce_array))
            for field in dtype.fields
        ]
        return pl.Struct(new_fields)
    return dtype


# =============================================================================
# Semantic Join-related utilities
# =============================================================================

def normalize_column_before_join(
    df: pl.DataFrame,
    col: Union[str, pl.Expr],
    alias: str
) -> tuple[pl.DataFrame, Optional[str]]:
    """Normalize a column for join operations by applying a consistent alias.

    This method handles both existing columns (string names) and derived
    expressions. Derived expressions are computed and added to the DataFrame,
    while existing columns are simply renamed.

    Args:
        df: DataFrame to normalize
        col: Column specification - either a column name (str) or expression (pl.Expr)
        alias: Target alias for the normalized column

    Returns:
        Tuple of:
        - Modified DataFrame with normalized column
        - Original column name if col was a string, None if it was an expression
    """
    if isinstance(col, pl.Expr):
        # Add derived column with alias
        return df.with_columns(col.alias(alias)), None
    else:
        # Rename existing column
        return df.rename({col: alias}), col

def restore_column_after_join(
    df: pl.DataFrame,
    original_name: Optional[str],
    alias: str
) -> pl.DataFrame:
    """Restore column to original state after join operation.

    This reverses the normalization from normalize_column_before_join:
    - Renames aliased columns back to their original names
    - Drops temporary columns that were created from expressions

    Args:
        df: DataFrame to restore
        original_name: Original column name (None if column was derived)
        alias: Current alias of the column

    Returns:
        DataFrame with restored column state
    """
    if original_name:
        # Restore original column name
        return df.rename({alias: original_name})
    else:
        # Drop temporary derived column
        return df.drop(alias)
