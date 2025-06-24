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
        label_column: str,
        centroid_info: Optional[Tuple[str, int]],
        num_iter: int = 50,
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
            kmeans = KMeans(k=self.num_centroids, max_iters=self.num_iter)
            kmeans.fit(embeddings)
            predicted = kmeans.predict(embeddings).tolist()
            cluster_centroids = kmeans.centroids.to_numpy(zero_copy_only=False)

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
                    pa.array(centroids, type=pa.list_(pa.float32(), self.centroid_info[1]))
                ).alias(self.centroid_info[0])
            )

        return res
