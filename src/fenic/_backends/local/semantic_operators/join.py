from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
from fenic.core.error import ExecutionError, InternalError
from fenic.core.types import JoinExampleCollection, PredicateExampleCollection

# TODO(rohitrastogi): Make this a guid so it doesn't collide with any column names in a user dataframe.
RENDERED_INSTRUCTION_KEY = "__rendered_instruction__"
MATCH_RESULT_KEY = "__match_result__"
LEFT_ID_KEY = "__left_id__"
RIGHT_ID_KEY = "__right_id__"
DEFAULT_PAIR_BLOCK_SIZE = 1_024
DEFAULT_BLOCK_TOKEN_BUDGET = 32_768


class Join:
    """Evaluate one pair tile at a time, overlapping its token sub-blocks.

    The pair cap (1,024 by default) also caps join-owned provider work, even
    when the client's RPM/look-ahead window is larger. Token sub-blocks share
    the existing provider limiter without a settlement barrier between them.
    Context preflight checks rendered user prompts only, not system text,
    examples, request framing, or reserved output tokens. It does not guarantee
    that the full request fits the model context.
    """

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
        pair_block_size: int = DEFAULT_PAIR_BLOCK_SIZE,
        block_token_budget: int = DEFAULT_BLOCK_TOKEN_BUDGET,
    ):
        self.left_df = left_df.with_row_index(LEFT_ID_KEY)
        self.right_df = right_df.with_row_index(RIGHT_ID_KEY)
        self.jinja_template = jinja_template
        self.strict = strict
        self.examples = examples
        self.temperature = temperature
        self.model = model
        self.model_alias = model_alias
        if pair_block_size <= 0:
            raise InternalError("pair_block_size must be positive")
        if block_token_budget <= 0:
            raise InternalError("block_token_budget must be positive")
        self.pair_block_size = pair_block_size
        self.block_token_budget = block_token_budget

    def execute(self) -> pl.DataFrame:
        join_documents = self._join_documents()
        if join_documents is None:
            return self._empty_result_with_schema(self.left_df, self.right_df)

        left_documents, right_documents = join_documents
        survivor_chunks = []
        examples = self._convert_examples()
        for join_pairs in self._iter_join_pair_blocks(left_documents, right_documents):
            survivor_chunks.extend(self._execute_pair_tile(join_pairs, examples))
            # Drop the previous rendered tile before constructing the next one.
            del join_pairs

        if not survivor_chunks:
            return self._empty_result_with_schema(self.left_df, self.right_df)

        survivor_pairs = pl.concat(survivor_chunks)
        self._assert_unique_survivor_pairs(survivor_pairs)
        return self._postprocess(survivor_pairs)

    def _execute_pair_tile(
        self, join_pairs: pl.DataFrame, examples: PredicateExampleCollection
    ) -> list[pl.DataFrame]:
        blocks = list(self._split_block_by_token_budget(join_pairs))
        # A finite tile is the hard slot ceiling, independent of max(B, RPM).
        if sum(len(block) for block in blocks) > self.pair_block_size:
            raise InternalError("semantic.join pair tile exceeds cap")
        evaluate = partial(self._evaluate_token_block, examples=examples)
        if len(blocks) == 1:
            survivors = [evaluate(blocks[0])]
        else:
            # One worker per sub-block avoids a token-dependent concurrency cap.
            # All owned work drains before another tile can be built, even on error.
            with ThreadPoolExecutor(max_workers=len(blocks)) as executor:
                survivors = list(executor.map(evaluate, blocks))
        return [survivor for survivor in survivors if not survivor.is_empty()]

    def _evaluate_token_block(
        self, join_pairs: pl.DataFrame, examples: PredicateExampleCollection
    ) -> pl.DataFrame:
        predicate = Predicate(
            input=join_pairs[RENDERED_INSTRUCTION_KEY],
            jinja_template=self.jinja_template,
            examples=examples,
            temperature=self.temperature,
            model=self.model,
            model_alias=self.model_alias,
        )
        return self._select_survivors(join_pairs, predicate.execute())

    def _join_documents(self) -> tuple[pl.DataFrame, pl.DataFrame] | None:
        if self.left_df.is_empty() or self.right_df.is_empty():
            return None
        left_documents = self.left_df.select([LEFT_ON_KEY, LEFT_ID_KEY])
        right_documents = self.right_df.select([RIGHT_ON_KEY, RIGHT_ID_KEY])
        if self.strict:
            left_documents = left_documents.filter(pl.col(LEFT_ON_KEY).is_not_null())
            right_documents = right_documents.filter(pl.col(RIGHT_ON_KEY).is_not_null())

        if left_documents.is_empty() or right_documents.is_empty():
            return None
        return left_documents, right_documents

    def _iter_join_pair_blocks(
        self, left_documents: pl.DataFrame, right_documents: pl.DataFrame
    ) -> Iterator[pl.DataFrame]:
        right_block_size = min(len(right_documents), self.pair_block_size)
        left_block_size = max(1, self.pair_block_size // right_block_size)
        for left_offset in range(0, len(left_documents), left_block_size):
            left_block = left_documents.slice(left_offset, left_block_size)
            for right_offset in range(0, len(right_documents), right_block_size):
                right_block = right_documents.slice(right_offset, right_block_size)
                yield self._build_join_pair_block(left_block, right_block)

    def _build_join_pair_block(
        self, left_documents: pl.DataFrame, right_documents: pl.DataFrame
    ) -> pl.DataFrame:
        joined_df = left_documents.join(right_documents, how="cross")
        if len(joined_df) > self.pair_block_size:
            raise InternalError(
                "semantic.join pair block exceeds cap "
                f"({len(joined_df)} > {self.pair_block_size})"
            )
        render_expr = pl.struct(
            [pl.col(LEFT_ON_KEY), pl.col(RIGHT_ON_KEY)]
        ).jinja.render(
            template=self.jinja_template,
            strict=self.strict,
        )
        return joined_df.with_columns(render_expr.alias(RENDERED_INSTRUCTION_KEY)).drop(
            [LEFT_ON_KEY, RIGHT_ON_KEY]
        )

    def _split_block_by_token_budget(
        self, join_pairs: pl.DataFrame
    ) -> Iterator[pl.DataFrame]:
        context_limit = self.model.model_parameters.context_window_length
        prompt_tokens = 0
        for prompt in join_pairs[RENDERED_INSTRUCTION_KEY]:
            pair_tokens = self.model.count_tokens(prompt)
            if pair_tokens > context_limit:
                raise ExecutionError(
                    "semantic.join rendered prompt is too large "
                    f"({pair_tokens} tokens) and exceeds the model context limit "
                    f"({context_limit} tokens). Reduce the join inputs or use a "
                    "smaller prompt."
                )
            prompt_tokens += pair_tokens
        yield from self._split_validated_block(join_pairs, prompt_tokens)

    def _split_validated_block(
        self, join_pairs: pl.DataFrame, prompt_tokens: int
    ) -> Iterator[pl.DataFrame]:
        if len(join_pairs) == 1 or prompt_tokens <= self.block_token_budget:
            yield join_pairs
            return

        split_at = len(join_pairs) // 2
        for block in (join_pairs.slice(0, split_at), join_pairs.slice(split_at)):
            tokens = sum(
                self.model.count_tokens(prompt)
                for prompt in block[RENDERED_INSTRUCTION_KEY]
            )
            yield from self._split_validated_block(block, tokens)

    def _convert_examples(self) -> PredicateExampleCollection:
        if not self.examples:
            return []

        examples_df = self.examples.to_polars()
        return PredicateExampleCollection.from_polars(examples_df)

    def _select_survivors(
        self, join_pairs: pl.DataFrame, results: pl.Series
    ) -> pl.DataFrame:
        return (
            join_pairs.with_columns(
                pl.Series(MATCH_RESULT_KEY, results, dtype=pl.Boolean)
            )
            .filter(pl.col(MATCH_RESULT_KEY))
            .select([LEFT_ID_KEY, RIGHT_ID_KEY])
        )

    def _assert_unique_survivor_pairs(self, survivor_pairs: pl.DataFrame) -> None:
        if survivor_pairs.select(
            pl.struct([LEFT_ID_KEY, RIGHT_ID_KEY]).is_duplicated().any()
        ).item():
            raise InternalError("semantic.join produced duplicate survivor pairs")

    def _postprocess(self, survivor_pairs: pl.DataFrame) -> pl.DataFrame:
        """Materialize wide rows only for predicate-surviving ID pairs."""
        return (
            survivor_pairs.join(self.left_df, on=LEFT_ID_KEY, how="inner")
            .join(self.right_df, on=RIGHT_ID_KEY, how="inner")
            .drop([LEFT_ID_KEY, RIGHT_ID_KEY])
        )

    def _empty_result_with_schema(
        self, left: pl.DataFrame, right: pl.DataFrame
    ) -> pl.DataFrame:
        left_schema = [
            (name, dtype) for name, dtype in left.schema.items() if name != LEFT_ID_KEY
        ]
        right_schema = [
            (name, dtype)
            for name, dtype in right.schema.items()
            if name != RIGHT_ID_KEY
        ]

        schema = left_schema + right_schema

        # Build empty DataFrame from schema
        return pl.DataFrame(
            {name: pl.Series(name, [], dtype=dtype) for name, dtype in schema}
        )
