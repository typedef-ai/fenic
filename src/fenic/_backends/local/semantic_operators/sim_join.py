import os
import tempfile
import uuid
from typing import TYPE_CHECKING

import polars as pl

from fenic._backends.local.semantic_operators.utils import (
    filter_invalid_embeddings_expr,
)
from fenic._constants import VECTOR_INDEX_DIR
from fenic._optional_dependencies import import_optional_dependency
from fenic.core.types.enums import SemanticSimilarityMetric

if TYPE_CHECKING:
    from lancedb.db import DBConnection, Table

# LanceDB column names
DISTANCE_COL_NAME = "_distance"
# IMPORTANT: Lance expects a column named "vector" in the table.
VECTOR_COL_NAME = "vector"

# TODO(rohitrastogi): Make these guids so they don't collide with any column names in a user dataframe.
LEFT_ON_COL_NAME = "__left_on__"
RIGHT_ON_COL_NAME = "__right_on__"
LEFT_ID_COL_NAME = "__left_id__"
RIGHT_ID_COL_NAME = "__right_id__"
MATCH_RESULT_COL_NAME = "__match_result__"
DEFAULT_LEFT_BATCH_SIZE = 1_024

class SimJoin:
    def __init__(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        k: int,
        similarity_metric: SemanticSimilarityMetric,
        left_batch_size: int = DEFAULT_LEFT_BATCH_SIZE,
        include_left_on: bool = True,
        include_right_on: bool = True,
    ):
        self.left = left.with_row_index(LEFT_ID_COL_NAME)
        self.right = right.with_row_index(RIGHT_ID_COL_NAME)
        self.k = k
        self.similarity_metric = similarity_metric
        if left_batch_size <= 0:
            raise ValueError("left_batch_size must be positive")
        self.left_batch_size = left_batch_size
        self.include_left_on = include_left_on
        self.include_right_on = include_right_on

    def execute(self) -> pl.DataFrame:
        """Perform semantic similarity join on the DataFrame using vector embeddings.

        Args:
            left (pl.DataFrame): Left DataFrame with embeddings in `left_on`.
            right (pl.DataFrame): Right DataFrame with embeddings in `right_on`.
            K (int): Number of nearest neighbors to retrieve from `right` for each row in `left`.
        """
        left = self.left.filter(filter_invalid_embeddings_expr(LEFT_ON_COL_NAME))
        right = self.right.filter(filter_invalid_embeddings_expr(RIGHT_ON_COL_NAME))

        if left.is_empty() or right.is_empty():
            return self._empty_result_with_schema(left, right)

        matches_df = self._batch_similarity_search(left, right)
        left_result = left if self.include_left_on else left.drop(LEFT_ON_COL_NAME)
        right_result = right if self.include_right_on else right.drop(RIGHT_ON_COL_NAME)

        result = (
            matches_df.join(left_result, on=LEFT_ID_COL_NAME, how="inner")
            .join(right_result, on=RIGHT_ID_COL_NAME, how="inner")
            .drop([LEFT_ID_COL_NAME, RIGHT_ID_COL_NAME])
        )
        # Reorder columns to have similarity score last
        cols = [col for col in result.columns if col != DISTANCE_COL_NAME]
        cols.append(DISTANCE_COL_NAME)
        result = result.select(cols)
        return result

    def _batch_similarity_search(
        self, left: pl.DataFrame, right: pl.DataFrame
    ) -> pl.DataFrame:
        os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
        table_name = uuid.uuid4().hex
        lancedb = import_optional_dependency(
            "lancedb",
            extra="sim-join",
            feature="semantic similarity joins",
        )
        with tempfile.TemporaryDirectory(
            prefix="sim_join_", dir=VECTOR_INDEX_DIR
        ) as lance_table_dir:
            db: DBConnection = lancedb.connect(lance_table_dir)
            tbl: Table = db.create_table(
                table_name,
                right.select(RIGHT_ON_COL_NAME, RIGHT_ID_COL_NAME).rename(
                    {RIGHT_ON_COL_NAME: VECTOR_COL_NAME}
                ),
            )
            if len(right) > 5000:
                tbl.create_index(metric=self.similarity_metric)

            # The final N×k result remains materialized by contract, but each
            # search/explode transform is bounded to a narrow left-side slice.
            match_chunks = [
                self._search_left_batch(
                    left.slice(offset, self.left_batch_size), tbl
                )
                for offset in range(0, len(left), self.left_batch_size)
            ]
            return pl.concat(match_chunks)

    def _search_left_batch(self, left_batch: pl.DataFrame, table: "Table") -> pl.DataFrame:
        """Search one bounded left-side slice and return narrow match rows."""

        def search_vectors(left_embedding, left_id):
            results = (
                table.search(left_embedding)
                .distance_type(self.similarity_metric)
                .limit(self.k)
                .to_list()
            )
            return [
                {
                    LEFT_ID_COL_NAME: left_id,
                    RIGHT_ID_COL_NAME: result[RIGHT_ID_COL_NAME],
                    DISTANCE_COL_NAME: result[DISTANCE_COL_NAME],
                }
                for result in results
            ]

        # LanceDB does not support parallel vector-batch searches. Keep these
        # per-row searches local to this slice so the transient explode is bounded.
        return (
            left_batch.select(
                pl.struct([pl.col(LEFT_ON_COL_NAME), pl.col(LEFT_ID_COL_NAME)])
                .map_elements(
                    lambda value: search_vectors(
                        value[LEFT_ON_COL_NAME], value[LEFT_ID_COL_NAME]
                    ),
                    return_dtype=pl.List(
                        pl.Struct(
                            {
                                LEFT_ID_COL_NAME: pl.Int32,
                                RIGHT_ID_COL_NAME: pl.Int32,
                                DISTANCE_COL_NAME: pl.Float64,
                            }
                        )
                    ),
                )
                .alias(MATCH_RESULT_COL_NAME)
            )
            .explode(MATCH_RESULT_COL_NAME)
            .unnest(MATCH_RESULT_COL_NAME)
        )

    def _empty_result_with_schema(
        self, left: pl.DataFrame, right: pl.DataFrame
    ) -> pl.DataFrame:
        extra_cols = [
            (DISTANCE_COL_NAME, pl.Float64),
        ]

        # Drop the ID columns after join
        left_schema = [
            (name, dtype) for name, dtype in left.schema.items() if name != LEFT_ID_COL_NAME
        ]
        right_schema = [
            (name, dtype) for name, dtype in right.schema.items() if name != RIGHT_ID_COL_NAME
        ]
        if not self.include_left_on:
            left_schema = [
                (name, dtype) for name, dtype in left_schema if name != LEFT_ON_COL_NAME
            ]
        if not self.include_right_on:
            right_schema = [
                (name, dtype)
                for name, dtype in right_schema
                if name != RIGHT_ON_COL_NAME
            ]

        schema = left_schema + right_schema + extra_cols

        return pl.DataFrame(
            {name: pl.Series(name, [], dtype=dtype) for name, dtype in schema}
        )
