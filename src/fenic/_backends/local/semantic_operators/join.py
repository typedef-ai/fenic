import logging
import re
from typing import List, Optional

import polars as pl

from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic._backends.local.semantic_operators.utils import (
    uppercase_instruction_placeholder,
)
from fenic._constants import (
    EXAMPLE_LEFT_KEY,
    EXAMPLE_RIGHT_KEY,
)
from fenic._inference.language_model import LanguageModel
from fenic.core.types import JoinExampleCollection, PredicateExampleCollection

logger = logging.getLogger(__name__)


class Join:
    def __init__(
        self,
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        left_on: str,
        right_on: str,
        join_instruction: str,
        model: LanguageModel,
        temperature: float,
        examples: Optional[JoinExampleCollection] = None,
    ):
        self.left_df = left_df.with_row_index("_left_id")
        self.right_df = right_df.with_row_index("_right_id")
        join_instruction = re.sub(r":(left|right)", "", join_instruction)
        self.join_instruction = uppercase_instruction_placeholder(join_instruction)
        self.left_on = left_on
        self.right_on = right_on
        self.examples = examples
        self.temperature = temperature
        self.model = model

    def execute(self) -> pl.DataFrame:
        join_inputs = self._build_join_pairs_df()
        if join_inputs is None:
            return self._empty_result_with_schema(self.left_df, self.right_df)
        semantic_predicate = Predicate(
            input=join_inputs.select([self.left_on, self.right_on]),
            user_instruction=self.join_instruction,
            examples=self._convert_examples(),
            temperature=self.temperature,
            model=self.model,
        )
        results = semantic_predicate.execute()
        return self._postprocess(join_inputs, results)

    def _build_join_pairs_df(self) -> pl.DataFrame | None:
        if self.left_df.is_empty() or self.right_df.is_empty():
            return None

        left_documents = self.left_df.select([self.left_on, "_left_id"]).filter(
            pl.col(self.left_on).is_not_null()
        )
        right_documents = self.right_df.select([self.right_on, "_right_id"]).filter(
            pl.col(self.right_on).is_not_null()
        )

        return left_documents.join(right_documents, how="cross")

    def _convert_examples(self) -> PredicateExampleCollection:
        if not self.examples:
            return []

        examples_df = self.examples.to_polars()
        examples_df = examples_df.rename(
            {
                EXAMPLE_LEFT_KEY: self.left_on,
                EXAMPLE_RIGHT_KEY: self.right_on,
            }
        )
        return PredicateExampleCollection.from_polars(examples_df)

    def _postprocess(
        self, join_pairs_df: pl.DataFrame, results: List[Optional[bool]]
    ) -> pl.DataFrame:
        """Use predicate results to construct the final joined dataframe."""
        # Add results as a column to join_pairs_df
        join_pairs_with_results = join_pairs_df.with_columns(
            pl.Series("match_result", results)
        )
        filtered_results = join_pairs_with_results.filter(
            pl.col("match_result")
        ).select(["_left_id", "_right_id"])

        joined_df = (
            filtered_results.join(self.left_df, on="_left_id", how="inner")
            .join(self.right_df, on="_right_id", how="inner")
            .drop(["_left_id", "_right_id"])
        )
        return joined_df

    def _empty_result_with_schema(
        self, left: pl.DataFrame, right: pl.DataFrame
    ) -> pl.DataFrame:
        left_schema = [
            (name, dtype) for name, dtype in left.schema.items() if name != "_left_id"
        ]
        right_schema = [
            (name, dtype) for name, dtype in right.schema.items() if name != "_right_id"
        ]

        schema = left_schema + right_schema

        # Build empty DataFrame from schema
        return pl.DataFrame(
            {name: pl.Series(name, [], dtype=dtype) for name, dtype in schema}
        )
