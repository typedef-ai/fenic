import polars as pl


def apply_ingestion_coercions(df: pl.DataFrame) -> pl.DataFrame:
    """Apply type coercions to normalize data types during ingestion.

    This is intended for ingestion from external systems (e.g., DuckDB, Parquet)
    that may produce types unsupported or inconsistently handled by Fenic.

    Coercion rules:
    - `Date` and `Datetime` are cast to `String` to preserve formatting and avoid
      incompatibilities with backends that lack full date/time support.
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
        target_dtype = _build_target_dtype(dtype)

        if target_dtype != dtype:
            expressions.append(pl.col(col_name).cast(target_dtype))
        else:
            expressions.append(pl.col(col_name))

    return df.select(expressions)


def _build_target_dtype(dtype: pl.DataType) -> pl.DataType:
    if dtype in [pl.Date, pl.Datetime]:
        return pl.String
    elif isinstance(dtype, pl.Array):
        return pl.List(_build_target_dtype(dtype.inner))
    elif isinstance(dtype, pl.List):
        return pl.List(_build_target_dtype(dtype.inner))
    elif isinstance(dtype, pl.Struct):
        new_fields = [
            pl.Field(field.name, _build_target_dtype(field.dtype))
            for field in dtype.fields
        ]
        return pl.Struct(new_fields)
    return dtype
