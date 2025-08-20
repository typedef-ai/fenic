import logging
import re
from typing import Callable

import fenic as fc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FenicAPIDocQuerySearch:
    """Search for queries to the Fenic API.
    Supports both keyword and regex search.
    """

    @classmethod
    def _is_valid_regex(cls, query: str) -> bool:
        """Heuristic check to see if the query is a regex."""
        try:
            re.compile(query)
            return True
        except re.error:
            return False

    @classmethod
    def _search_api_docs_regex(cls, df: fc.DataFrame, query: str) -> fc.DataFrame:
        """Search API documentation using regex."""
        return df.filter(
            fc.col("name").rlike(f"(?i){query}")
            | fc.col("qualified_name").rlike(f"(?i){query}")
            | (
                fc.col("docstring").is_not_null()
                & fc.col("docstring").rlike(f"(?i){query}")
            )
            | (
                fc.col("annotation").is_not_null()
                & fc.col("annotation").rlike(f"(?i){query}")
            )
            | (
                fc.col("returns").is_not_null()
                & fc.col("returns").rlike(f"(?i){query}")
            )
        )

    @classmethod
    def _search_learnings_regex(cls, df: fc.DataFrame, query: str) -> fc.DataFrame:
        """Search learnings using regex."""
        return df.filter(
            fc.col("question").rlike(f"(?i){query}")
            | fc.col("answer").rlike(f"(?i){query}")
            | fc.array_contains(fc.col("keywords"), query)
        )

    @classmethod
    def _search_learnings_keyword(cls, df: fc.DataFrame, term: str) -> fc.DataFrame:
        """Search learnings using keyword."""
        return df.filter(
            fc.col("question").contains(term)
            | fc.col("answer").contains(term)
            | fc.array_contains(fc.col("keywords"), term)
        )

    @classmethod
    def _search_terms(
        cls,
        df: fc.DataFrame,
        query: str,
        search_func: Callable[[fc.DataFrame, str], fc.DataFrame],
    ) -> fc.DataFrame:
        """Search using multiple terms."""
        # First search the query as a whole.
        result_df = search_func(df, query)
        logger.debug(f"result_df - {query}: {result_df.count()}")

        # look for each individual term as well.
        terms = query.lower().split()
        terms_data_frames = []
        for term in terms:
            terms_data_frames.append(search_func(df, term))
        result_df = result_df.union(terms_data_frames[0])
        for df in terms_data_frames[1:]:
            result_df = result_df.union(df)

        logger.debug(f"learnings results: {result_df.to_pydict()}")

        return result_df

    @classmethod
    def search_learnings(cls, session: fc.Session, query: str) -> fc.DataFrame:
        """Search learnings using keyword."""
        if session.catalog.does_table_exist("learnings"):
            try:
                learnings_df = session.table("learnings")

                logger.debug(f"Searching learnings with regex: {query}")
                learnings_search = cls._search_terms(
                    learnings_df, query, cls._search_learnings_regex
                )

                # Add relevance scoring for learnings
                learnings_scored = learnings_search.select(
                    "question",
                    "answer",
                    "learning_type",
                    "keywords",
                    "related_functions",
                    fc.when(fc.col("question").rlike(f"(?i){query}"), fc.lit(10))
                    .otherwise(fc.lit(0))
                    .alias("question_score"),
                    fc.when(fc.col("answer").rlike(f"(?i){query}"), fc.lit(5))
                    .otherwise(fc.lit(0))
                    .alias("answer_score"),
                    fc.when(fc.array_contains(fc.col("keywords"), query), fc.lit(3))
                    .otherwise(fc.lit(0))
                    .alias("keywords_score"),
                )

                # Calculate total score with correction boost
                learnings_scored = learnings_scored.select(
                    "*",
                    (
                        fc.col("question_score")
                        + fc.col("answer_score")
                        + fc.col("keywords_score")
                    ).alias("base_score"),
                ).select(
                    "*",
                    fc.when(
                        fc.col("learning_type") == "correction",
                        fc.col("base_score") * 1.5,
                    )
                    .otherwise(fc.col("base_score"))
                    .alias("score"),
                )

                # Sort and limit learnings (max 7 results)
                return learnings_scored.order_by(fc.col("score").desc()).limit(7)
            except Exception as e:
                logger.error(f"Warning: Learnings search failed: {e}")
                return None

    @classmethod
    def search_api_docs(cls, session: fc.Session, query: str) -> fc.DataFrame:
        # Search API documentation
        df = session.table("api_df")

        # Filter only public API elements
        df = df.filter(
            (fc.col("is_public")) & (~fc.col("qualified_name").contains("._"))
        )

        if not cls._is_valid_regex(query):
            raise ValueError("Invalid regex query")
        logger.debug(f"Searching API docs with regex: {query}")
        search_df = cls._search_api_docs_regex(df, query)

        # Add relevance scoring
        search_df = search_df.select(
            "type",
            "name",
            "qualified_name",
            "docstring",
            fc.when(fc.col("name").rlike(f"(?i){query}"), fc.lit(10))
            .otherwise(fc.lit(0))
            .alias("name_score"),
            fc.when(fc.col("qualified_name").rlike(f"(?i){query}"), fc.lit(5))
            .otherwise(fc.lit(0))
            .alias("path_score"),
        )

        # Calculate total score and sort
        search_df = search_df.select(
            "*", (fc.col("name_score") + fc.col("path_score")).alias("score")
        )

        return search_df