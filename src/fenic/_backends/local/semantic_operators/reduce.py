import logging
import math
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
from typing import Optional

import polars as pl
import jinja2

from fenic._constants import PREFIX_TOKENS_PER_MESSAGE
from fenic._inference.language_model import LanguageModel
from fenic._inference.types import LMRequestMessages
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias

logger = logging.getLogger(__name__)

CONTEXT_WINDOW_REDUCTION_FACTOR = 0.7

class Reduce:
    """Hierarchical document reduction for handling context window limitations.

    This class implements a greedy left-to-right packing algorithm that:
    1. Processes documents in order, preserving temporal/logical sequence
    2. Packs as many documents as possible into each LLM call
    3. Hierarchically reduces results until a single summary remains

    The left-to-right processing ensures that document order (e.g., temporal sorting)
    is preserved throughout the reduction hierarchy. For example, with monthly reports
    sorted by date, early batches will naturally group Q1 months, creating coherent
    temporal summaries at each level.
    """

    SYSTEM_PROMPT = dedent("""\
        Follow the user's instruction exactly and generate only the requested output.

        Requirements:
        1. Follow the instruction exactly as written
        2. Output only what is requested - no explanations, no prefixes, no metadata
        3. Be concise and direct
        4. Do not add formatting or structure unless explicitly requested""")

    USER_MESSAGE_TEMPLATE = jinja2.Template(dedent("""\
        {{user_instruction}}

        {% for doc in docs %}
        <document{{loop.index}}>
        {{doc}}
        </document{{loop.index}}>
        {%- if not loop.last %}

        {% endif -%}
        {% endfor %}"""))

    SYSTEM_MESSAGE = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    def __init__(
            self,
            input: pl.Series,
            user_instruction: str,
            model: LanguageModel,
            max_tokens: int,
            temperature: float,
            model_alias: Optional[ResolvedModelAlias] = None,
    ):
        self.input = input
        self.user_instruction = user_instruction
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_profile = model_alias.profile if model_alias else None
        self.prefix_tokens = (
            self.model.count_tokens([self.SYSTEM_MESSAGE])
            + PREFIX_TOKENS_PER_MESSAGE
        )

    def execute(self) -> pl.Series:
        """Execute parallel reduction across all groups."""
        with ThreadPoolExecutor(max_workers=min(10, len(self.input))) as executor:
            results = list(
                executor.map(lambda x: self._reduce_group(*x), enumerate(self.input))
            )
        return pl.Series(results)

    def _reduce_group(self, group_index: int, group: pl.Series) -> str | None:
        """Reduces a single group of documents hierarchically until a single output is obtained.

        The reduction preserves document order through greedy left-to-right packing:
        - Level 0: Original documents in order
        - Level 1+: Summaries from previous level, maintaining sequence

        This ensures temporal/logical coherence when documents are pre-sorted.

        Args:
            group_index: The index of the current group being processed.
            group: A Polars Series containing the documents in the current group.

        Returns:
            The final synthesized output for the group.
        """
        operation_name = f"semantic.reduce(group={group_index})"
        docs = group.to_list()
        tree_level = 0

        # Continue until we have a single summary
        # tree_level == 0 check ensures we process even single-document groups
        while len(docs) > 1 or tree_level == 0:
            logger.debug(
                f"Tree level {tree_level}: processing {len(docs)} documents for group {group_index}"
            )

            messages_batch = self._build_request_messages_batch(docs, tree_level)
            if not messages_batch:
                logger.warning(f"No valid documents to process in group {group_index}")
                return None

            responses = self.model.get_completions(
                messages=messages_batch,
                operation_name=operation_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model_profile=self.model_profile,
            )

            # Extract completions for next level
            reduced_docs = [response.completion for response in responses]
            docs = reduced_docs
            tree_level += 1

        return docs[0]

    def _build_request_messages_batch(
        self, docs: list[str], tree_level: int
    ) -> list[LMRequestMessages] | None:
        """Creates batches of requests using greedy left-to-right packing.

        This method preserves document order while maximizing context utilization:
        1. Process documents left-to-right
        2. Pack documents greedily until context limit reached
        3. Start new batch and continue

        This ensures that related documents (e.g., consecutive time periods)
        are likely to be summarized together.

        Args:
            docs: A list of documents (raw or synthesized) to be batched.
            tree_level: The current level of the aggregation tree (0 for leaf nodes).

        Returns:
            A list of LMRequestMessages, where each represents a batch of documents
            packed greedily from left to right.
        """
        user_message_tokens = (
            self.model.count_tokens(self.user_instruction) + self.prefix_tokens
        )

        # Calculate available tokens for documents
        # Using safety factor to avoid edge cases with token counting
        theoretical_max_input_tokens = (
            self.model.max_context_window_length -
            self.model.model_parameters.max_output_tokens
        )
        if theoretical_max_input_tokens <= 0:
            theoretical_max_input_tokens = self.model.max_context_window_length

        max_input_tokens = math.floor(
            theoretical_max_input_tokens * CONTEXT_WINDOW_REDUCTION_FACTOR
        )

        messages_batch: list[LMRequestMessages] = []
        request_docs: list[str] = []
        request_tokens = 0

        for doc in docs:
            if not doc:
                logger.debug(f"Skipping empty document at tree level {tree_level}")
                continue

            doc_tokens = self.model.count_tokens(doc)

            # Check if single document exceeds context
            if user_message_tokens + doc_tokens > max_input_tokens:
                raise ValueError(
                    f"semantic.reduce document is too large ({user_message_tokens + doc_tokens} tokens) "
                    f"and exceeds the maximum allowed size ({max_input_tokens} tokens) "
                    f"for the chosen model's context window. "
                    f"Please reduce the document size by either: "
                    f"1) Summarizing the content before processing, or "
                    f"2) Breaking it into smaller chunks using methods like "
                    f"text.recursive_character_chunk() or text.token_chunk()."
                )

            # Check if adding this doc would exceed context limit
            if request_docs and (user_message_tokens + request_tokens + doc_tokens > max_input_tokens):
                # Flush current batch and start new one
                messages_batch.append(
                    self._build_request_messages(request_docs)
                )
                logger.debug(
                    f"Tree level {tree_level}: created batch with {len(request_docs)} documents "
                    f"({request_tokens} tokens)"
                )
                request_docs = [doc]
                request_tokens = doc_tokens
            else:
                # Add to current batch
                request_docs.append(doc)
                request_tokens += doc_tokens

        # Handle final batch
        if request_docs:
            messages_batch.append(
                self._build_request_messages(request_docs)
            )
            logger.debug(
                f"Tree level {tree_level}: created final batch with {len(request_docs)} documents "
                f"({request_tokens} tokens)"
            )

        return messages_batch if messages_batch else None

    def _build_request_messages(
        self, docs: list[str]
    ) -> LMRequestMessages:
        """Builds a user message for the LLM.

        Args:
            docs: A list of documents to include in the prompt.

        Returns:
            A LMRequestMessages ready to be sent to the LLM.
        """
        return LMRequestMessages(
            system=self.SYSTEM_PROMPT,
            user=self.USER_MESSAGE_TEMPLATE.render(
                docs=docs,
                user_instruction=self.user_instruction
            ),
            examples=[],
        )
