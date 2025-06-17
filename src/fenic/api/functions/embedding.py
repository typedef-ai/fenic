"""Embedding functions."""

from typing import List, Union

import numpy as np
from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan.expressions import (
    EmbeddingNormalizeExpr,
    EmbeddingSimilarityExpr,
)
from fenic.core.error import ValidationError
from fenic.core.types import SemanticSimilarityMetric


def normalize(column: ColumnOrName) -> Column:
    """Normalize embedding vectors to unit length.

    Args:
        column: Column containing embedding vectors.

    Returns:
        Column: A column of normalized embedding vectors with the same embedding type.

    Notes:
        - Normalizes each embedding vector to have unit length (L2 norm = 1)
        - Preserves the original embedding model in the type
        - Null values are preserved as null
        - Zero vectors become NaN after normalization

    Example: Normalize embeddings for dot product similarity
        ```python
        # Normalize embeddings for dot product similarity comparisons
        df.select(
            embedding.normalize(col("embeddings")).alias("unit_embeddings")
        )
        ```

    Example: Compare normalized embeddings using dot product
        ```python
        # Compare normalized embeddings using dot product (equivalent to cosine similarity)
        normalized_df = df.select(embedding.normalize(col("embeddings")).alias("norm_emb"))
        query = [0.6, 0.8]  # Already normalized
        normalized_df.select(
            embedding.compute_similarity(col("norm_emb"), query, metric="dot").alias("dot_product_sim")
        )
        ```
    """
    column_expr = Column._from_col_or_name(column)._logical_expr
    return Column._from_logical_expr(EmbeddingNormalizeExpr(column_expr))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def compute_similarity(
    column: ColumnOrName,
    other: Union[ColumnOrName, List[float], np.ndarray],
    metric: SemanticSimilarityMetric = "cosine"
) -> Column:
    """Compute similarity between embedding vectors using specified metric.

    Args:
        column: Column containing embedding vectors.

        other: Either:

            - Another column containing embedding vectors for pairwise similarity
            - A query vector (list of floats or numpy array) for similarity with each embedding

        metric: The similarity metric to use. Options:

            - `cosine`: Cosine similarity (range: -1 to 1, higher is more similar)
            - `dot`: Dot product similarity (raw inner product)
            - `l2`: L2 (Euclidean) distance (lower is more similar)

    Returns:
        Column: A column of float values representing similarity scores.

    Raises:
        ValidationError: If query vector contains NaN values or has invalid dimensions.

    Notes:
        - Cosine similarity normalizes vectors internally, so pre-normalization is not required
        - Dot product does not normalize, useful when vectors are already normalized
        - L2 distance measures the straight-line distance between vectors
        - When using two columns, dimensions must match between embeddings

    Example: Compute dot product with a query vector
        ```python
        # Compute dot product with a query vector
        query = [0.1, 0.2, 0.3]
        df.select(
            embedding.compute_similarity(col("embeddings"), query).alias("similarity")
        )
        ```

    Example: Compute cosine similarity with a query vector
        ```python
        query = [0.6, ... 0.8]  # Already normalized
        df.select(
            embedding.compute_similarity(col("embeddings"), query, metric="cosine").alias("cosine_sim")
        )
        ```

    Example: Compute pairwise dot products between columns
        ```python
        # Compute L2 distance between two columns of embeddings
        df.select(
            embedding.compute_similarity(col("embeddings1"), col("embeddings2"), metric="l2").alias("distance")
        )
        ```

    Example: Using numpy array as query vector
        ```python
        # Use numpy array as query vector
        import numpy as np
        query = np.array([0.1, 0.2, 0.3])
        df.select(embedding.compute_similarity("embeddings", query))
        ```
    """
    column_expr = Column._from_col_or_name(column)._logical_expr

    # Check if other is a column
    if isinstance(other, ColumnOrName):
        other_expr = Column._from_col_or_name(other)._logical_expr
        return Column._from_logical_expr(
            EmbeddingSimilarityExpr(column_expr, other_expr, metric)
        )

    # Otherwise it's a query vector
    if isinstance(other, list):
        query_array = np.array(other, dtype=np.float32)
    else:
        query_array = other.astype(np.float32)

    # Check for NaNs
    if np.any(np.isnan(query_array)):
        raise ValidationError("Query vector cannot contain NaN values")

    return Column._from_logical_expr(
        EmbeddingSimilarityExpr(column_expr, query_array, metric)
    )
