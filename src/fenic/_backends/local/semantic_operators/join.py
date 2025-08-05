import logging
from typing import Optional

import polars as pl

import fenic._backends.local.polars_plugins  # noqa: F401
from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic._constants import (
    LEFT_ON_KEY,
    RIGHT_ON_KEY,
)
from fenic._inference.language_model import LanguageModel
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core.types import JoinExampleCollection, PredicateExampleCollection

logger = logging.getLogger(__name__)

# TODO(rohitrastogi): Make this a guid so it doesn't collide with any column names in a user dataframe.
RENDERED_INSTRUCTION_KEY = "__rendered_instruction__"
MATCH_RESULT_KEY = "__match_result__"
LEFT_ID_KEY = "__left_id__"
RIGHT_ID_KEY = "__right_id__"

class Join:
    def __init__(
        self,
        left_df: pl.DataFrame,
        right_df: pl.DataFrame,
        jinja_template: str,
        strict: bool,
        model: LanguageModel,
        temperature: float,
        examples: Optional[JoinExampleCollection] = None,
        model_alias: Optional[ResolvedModelAlias] = None,
    ):
        self.left_df = left_df.with_row_index(LEFT_ID_KEY)
        self.right_df = right_df.with_row_index(RIGHT_ID_KEY)
        self.jinja_template = jinja_template
        self.strict = strict
        self.examples = examples
        self.temperature = temperature
        self.model = model
        self.model_alias = model_alias

    def execute(self) -> pl.DataFrame:
        join_inputs = self._build_join_pairs_df()
        if join_inputs is None:
            return self._empty_result_with_schema(self.left_df, self.right_df)
        semantic_predicate = Predicate(
            input=join_inputs[RENDERED_INSTRUCTION_KEY],
            jinja_template=self.jinja_template,
            examples=self._convert_examples(),
            temperature=self.temperature,
            model=self.model,
            model_alias=self.model_alias,
        )
        results = semantic_predicate.execute()
        return self._postprocess(join_inputs, results)

    def _build_join_pairs_df(self) -> pl.DataFrame | None:
        if self.left_df.is_empty() or self.right_df.is_empty():
            return None
        left_documents = self.left_df.select([LEFT_ON_KEY, LEFT_ID_KEY])
        right_documents = self.right_df.select([RIGHT_ON_KEY, RIGHT_ID_KEY])
        if self.strict:
            left_documents = left_documents.filter(
                pl.col(LEFT_ON_KEY).is_not_null()
            )
            right_documents = right_documents.filter(
                pl.col(RIGHT_ON_KEY).is_not_null()
            )

        joined_df = left_documents.join(right_documents, how="cross")
        render_expr = pl.struct([pl.col(LEFT_ON_KEY), pl.col(RIGHT_ON_KEY)]).jinja.render(
            template=self.jinja_template,
            strict=self.strict,
        )
        return joined_df.with_columns(render_expr.alias(RENDERED_INSTRUCTION_KEY)).drop([LEFT_ON_KEY, RIGHT_ON_KEY])

    def _convert_examples(self) -> PredicateExampleCollection:
        if not self.examples:
            return []

        examples_df = self.examples.to_polars()
        return PredicateExampleCollection.from_polars(examples_df)

    def _postprocess(
        self, join_pairs_df: pl.DataFrame, results: pl.Series
    ) -> pl.DataFrame:
        """Use predicate results to construct the final joined dataframe."""
        # Add results as a column to join_pairs_df
        join_pairs_with_results = join_pairs_df.with_columns(pl.Series(MATCH_RESULT_KEY, results))
        filtered_results = join_pairs_with_results.filter(
            pl.col(MATCH_RESULT_KEY)
        ).select([LEFT_ID_KEY, RIGHT_ID_KEY])

        joined_df = (
            filtered_results.join(self.left_df, on=LEFT_ID_KEY, how="inner")
            .join(self.right_df, on=RIGHT_ID_KEY, how="inner")
            .drop([LEFT_ID_KEY, RIGHT_ID_KEY])
        )
        return joined_df

    def _empty_result_with_schema(
        self, left: pl.DataFrame, right: pl.DataFrame
    ) -> pl.DataFrame:
        left_schema = [
            (name, dtype) for name, dtype in left.schema.items() if name != LEFT_ID_KEY
        ]
        right_schema = [
            (name, dtype) for name, dtype in right.schema.items() if name != RIGHT_ID_KEY
        ]

        schema = left_schema + right_schema

        # Build empty DataFrame from schema
        return pl.DataFrame(
            {name: pl.Series(name, [], dtype=dtype) for name, dtype in schema}
        )
