import uuid

import lancedb
import polars as pl
from lancedb.db import DBConnection, Table

from fenic._backends.local.semantic_operators.utils import (
    filter_invalid_embeddings_expr,
)
from fenic._constants import VECTOR_INDEX_DIR
from fenic.core.types.enums import SemanticSimilarityMetric

# LanceDB column names
DISTANCE_COL_NAME = "_distance"
# IMPORTANT: Lance expects a column named "vector" in the table.
VECTOR_COL_NAME = "vector"

LEFT_ON_COL_NAME = "_left_on"
RIGHT_ON_COL_NAME = "_right_on"


class SimJoin:
    def __init__(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        k: int,
        similarity_metric: SemanticSimilarityMetric,
    ):
        self.left = left.with_row_index("_left_id")
        self.right = right.with_row_index("_right_id")
        self.k = k
        self.similarity_metric = similarity_metric

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

        result = (
            matches_df.join(left, on="_left_id", how="inner")
            .join(right, on="_right_id", how="inner")
            .drop(["_left_id", "_right_id"])
        )
        # Reorder columns to have similarity score last
        cols = [col for col in result.columns if col != DISTANCE_COL_NAME]
        cols.append(DISTANCE_COL_NAME)
        result = result.select(cols)
        return result

    def _batch_similarity_search(
        self, left: pl.DataFrame, right: pl.DataFrame
    ) -> pl.DataFrame:
        guid = uuid.uuid4().hex
        lance_table_dir = f"{VECTOR_INDEX_DIR}/{guid}"
        db: DBConnection = lancedb.connect(lance_table_dir)
        tbl: Table = db.create_table(
            guid,
            right.select(RIGHT_ON_COL_NAME, "_right_id").rename(
                {RIGHT_ON_COL_NAME: VECTOR_COL_NAME}
            ),
        )
        if len(right) > 5000:
            tbl.create_index(metric=self.similarity_metric)

        # Define UDF to perform search for each row
        def search_vectors(left_embedding, left_id):
            results = tbl.search(left_embedding).distance_type(self.similarity_metric).limit(self.k).to_list()

            # Create list of structs with search results
            matches = []
            for result in results:
                matches.append(
                    {
                        "_left_id": left_id,
                        "_right_id": result["_right_id"],
                        DISTANCE_COL_NAME: result[DISTANCE_COL_NAME],
                    }
                )
            return matches

        # TODO(rohitrastogi): Do some experiments to see if sending concurrent requests to LanceDB
        # using a thread pool is faster than sending requests sequentially. Vector search is CPU bound and LanceDB
        # releases the GIL, so I'm not sure there will be any performance gains.
        # FYI, LanceDB doesn't support batch vector search. If you pass a batch of vectors, it doesn't
        # actually search in parallel.
        return (
            left.select(
                pl.struct([pl.col(LEFT_ON_COL_NAME), pl.col("_left_id")])
                .map_elements(
                    lambda x: search_vectors(x[LEFT_ON_COL_NAME], x["_left_id"]),
                    return_dtype=pl.List(
                        pl.Struct(
                            {
                                "_left_id": pl.Int32,
                                "_right_id": pl.Int32,
                                DISTANCE_COL_NAME: pl.Float64,
                            }
                        )
                    ),
                )
                .alias("_matches")
            )
            .explode("_matches")
            .unnest("_matches")
        )

    def _empty_result_with_schema(
        self, left: pl.DataFrame, right: pl.DataFrame
    ) -> pl.DataFrame:
        extra_cols = [
            (DISTANCE_COL_NAME, pl.Float64),
        ]

        # Drop the ID columns after join
        left_schema = [
            (name, dtype) for name, dtype in left.schema.items() if name != "_left_id"
        ]
        right_schema = [
            (name, dtype) for name, dtype in right.schema.items() if name != "_right_id"
        ]

        schema = left_schema + right_schema + extra_cols

        return pl.DataFrame(
            {name: pl.Series(name, [], dtype=dtype) for name, dtype in schema}
        )
