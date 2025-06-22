import logging
from typing import Optional, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
from lance.util import KMeans

from fenic._backends.local.semantic_operators.utils import (
    filter_invalid_embeddings_expr,
)

logger = logging.getLogger(__name__)


class Cluster:
    def __init__(
        self,
        input: pl.DataFrame,
        embedding_column_name: str,
        num_centroids: int,
        centroid_dimensions: Optional[int],
        num_iter: int = 20,
    ):
        self.input = input
        self.embedding_column_name = embedding_column_name
        input_height = input.height
        if num_centroids > input_height:
            logger.warning(
                f"`num_centroids` was set to {num_centroids}, but the input DataFrame only contains {input_height} rows. "
                f"Reducing `num_centroids` to {input_height} to match the available number of rows."
            )
        self.num_centroids = min(num_centroids, input_height)
        self.num_iter = num_iter
        self.centroid_dimensions = centroid_dimensions

    def execute(self) -> pl.DataFrame:
        """Perform semantic clustering on the DataFrame.

        Returns:
            pl.DataFrame: The DataFrame with cluster assignments and centroids - adds "_cluster_id" and "_cluster_centroid" columns
        """
        cluster_ids, centroids = self._cluster_by_column()
        res =  self.input.with_columns(
            pl.Series(cluster_ids).alias("_cluster_id")
        )
        if self.centroid_dimensions is not None:
            res = res.with_columns(
                pl.from_arrow(pa.array(centroids, type=pa.list_(pa.float32(), self.centroid_dimensions))).alias("_cluster_centroid")
            )
        return res

    def _cluster_by_column(
        self,
    ) -> Tuple[list[int | None], list[np.ndarray | None] | None]:
        """Returns cluster IDs and centroids for each row using kmeans clustering.

        Returns:
            tuple: A tuple of (cluster_ids, centroids) where:
                - cluster_ids: list[int | None] - cluster ID for each row, None for invalid embeddings
                - centroids: list[list[float] | None] - centroid embedding for each row, None for invalid embeddings
        """
        df = self.input
        valid_mask = df.select(filter_invalid_embeddings_expr(self.embedding_column_name)).to_series()
        valid_df = df.filter(valid_mask)

        if valid_df.is_empty():
            return [None] * df.height, [None] * df.height

        # Perform clustering on valid embeddings
        embeddings = np.stack(valid_df[self.embedding_column_name])
        kmeans = KMeans(k=self.num_centroids, max_iters=self.num_iter)
        kmeans.fit(embeddings)
        predicted = kmeans.predict(embeddings).tolist()

        # Get centroids - they should be in order corresponding to cluster IDs
        cluster_centroids = np.stack(kmeans.centroids.to_numpy(zero_copy_only=False))

        # Build full results with None for invalid rows
        cluster_ids = [None] * df.height
        centroids = [None] * df.height if self.centroid_dimensions is not None else None
        valid_indices = valid_mask.to_numpy().nonzero()[0]

        for idx, cluster_id in zip(valid_indices, predicted, strict=True):
            cluster_ids[idx] = cluster_id
            if self.centroid_dimensions is not None:
                centroids[idx] = cluster_centroids[cluster_id]

        return cluster_ids, centroids
