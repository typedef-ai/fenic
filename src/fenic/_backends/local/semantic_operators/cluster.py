import logging
from typing import Optional

import numpy as np
import polars as pl
import pyarrow as pa
from sklearn.cluster import KMeans

from fenic._backends.local.semantic_operators.utils import (
    filter_invalid_embeddings_expr,
)
from fenic.core._logical_plan.plans import CentroidInfo

logger = logging.getLogger(__name__)


class Cluster:
    def __init__(
        self,
        input: pl.DataFrame,
        embedding_column_name: str,
        num_clusters: int,
        max_iter: int,
        num_init: int,
        label_column: str,
        centroid_info: Optional[CentroidInfo],
    ):
        self.input = input
        self.embedding_column_name = embedding_column_name
        input_height = input.height
        if num_clusters > input_height:
            logger.warning(
                f"`num_clusters` was set to {num_clusters}, but the input DataFrame only contains {input_height} rows. "
                f"Reducing `num_clusters` to {input_height} to match the available number of rows."
            )
        self.num_clusters = min(num_clusters, input_height)
        self.max_iter = max_iter
        self.num_init = num_init
        self.label_column = label_column
        self.centroid_info = centroid_info

    def execute(self) -> pl.DataFrame:
        df = self.input
        valid_mask = df.select(filter_invalid_embeddings_expr(self.embedding_column_name)).to_series()
        valid_df = df.filter(valid_mask)

        cluster_ids = [None] * df.height
        valid_indices = valid_mask.to_numpy().nonzero()[0]

        centroids = None
        if not valid_df.is_empty():
            embeddings = np.stack(valid_df[self.embedding_column_name])

            # Using sklearn KMeans with k-means++ initialization (default)
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                max_iter=self.max_iter,
                init='k-means++',  # This is the default, but being explicit
                n_init=self.num_init,  # Number of times to run k-means with different centroid seeds
                random_state=42  # For reproducibility
            )

            predicted = kmeans.fit_predict(embeddings)
            cluster_centroids = kmeans.cluster_centers_

            if self.centroid_info is not None:
                centroids = [None] * df.height

            for idx, cluster_id in zip(valid_indices, predicted, strict=True):
                cluster_ids[idx] = cluster_id
                if centroids is not None:
                    centroids[idx] = cluster_centroids[cluster_id]

        res = df.with_columns(pl.Series(cluster_ids).alias(self.label_column))

        if self.centroid_info is not None:
            res = res.with_columns(
                pl.from_arrow(
                    pa.array(centroids, type=pa.list_(pa.float32(), self.centroid_info.num_dimensions))
                ).alias(self.centroid_info.centroid_column)
            )

        return res
