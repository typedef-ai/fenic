"""Semantic extensions for DataFrames providing clustering and semantic join operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, get_args

from fenic.core.error import ValidationError
from fenic.core.types import (
    JoinExampleCollection,
)

if TYPE_CHECKING:
    from fenic.api.dataframe import DataFrame

import fenic.core._utils.misc as utils
from fenic.api.column import Column, ColumnOrName
from fenic.api.functions import col
from fenic.core._logical_plan.expressions import LiteralExpr
from fenic.core._logical_plan.plans import (
    SemanticCluster,
    SemanticJoin,
    SemanticSimilarityJoin,
)
from fenic.core.types.enums import SemanticSimilarityMetric


class SemanticExtensions:
    """A namespace for semantic dataframe operators."""

    def __init__(self, df: DataFrame):
        """Initialize semantic extensions.

        Args:
            df: The DataFrame to extend with semantic operations.
        """
        self._df = df

    def with_cluster_labels(
        self,
        by: ColumnOrName,
        num_clusters: int,
        max_iter: int = 300,
        num_init: int = 1,
        label_column: str = "cluster_label",
        centroid_column: Optional[str] = None,
    ) -> DataFrame:
        """Cluster rows using K-means and add cluster metadata columns.

        This method clusters rows based on the given embedding column or expression using K-means.
        It adds a new column with cluster assignments, and optionally includes the centroid embedding
        for each assigned cluster.

        Args:
            by: Column or expression producing embeddings to cluster (e.g., `embed(col("text"))`).
            num_clusters: Number of clusters to compute (must be > 0).
            max_iter: Maximum iterations for a single run of the k-means algorithm. The algorithm stops when it either converges or reaches this limit.
            num_init: Number of independent runs of k-means with different centroid seeds. The best result is selected.
            label_column: Name of the output column for cluster IDs. Default is "cluster_label".
            centroid_column: If provided, adds a column with this name containing the centroid embedding
                            for each row's assigned cluster.

        Returns:
            A DataFrame with all original columns plus:
            - `<label_column>`: integer cluster assignment (0 to num_clusters - 1)
            - `<centroid_column>`: cluster centroid embedding, if specified

        Raises:
            ValidationError: If num_clusters is not a positive integer
            ValidationError: If max_iter is not a positive integer
            ValidationError: If num_init is not a positive integer
            ValidationError: If label_column is not a non-empty string
            ValidationError: If centroid_column is not a non-empty string
            TypeMismatchError: If the column is not an EmbeddingType

        Example: Basic clustering
            ```python
            # Cluster customer feedback and add cluster metadata
            clustered_df = df.semantic.with_cluster_labels("feedback_embeddings", num_clusters=5)

            # Then use regular operations to analyze clusters
            clustered_df.group_by("cluster_label").agg(count("*"), avg("rating"))
            ```

        Example: Filter outliers using centroids
            ```python
            # Cluster and filter out rows far from their centroid
            clustered_df = df.semantic.with_cluster_labels("embeddings", num_clusters=3, num_init=10, centroid_column="cluster_centroid")
            clean_df = clustered_df.filter(
                embedding.compute_similarity("embeddings", "cluster_centroid", metric="cosine") > 0.7
            )
            ```
        """
        # Validate num_clusters
        if not isinstance(num_clusters, int) or num_clusters <= 0:
            raise ValidationError("`num_clusters` must be a positive integer.")

        # Validate max_iter
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValidationError("`max_iter` must be a positive integer.")

        # Validate num_init
        if not isinstance(num_init, int) or num_init <= 0:
            raise ValidationError("`num_init` must be a positive integer.")

        # Validate clustering target
        if not isinstance(by, ColumnOrName):
            raise ValidationError(
                f"Invalid cluster by: expected a column name (str) or Column object, got {type(by).__name__}."
            )

        # Validate label_column
        if not isinstance(label_column, str) or not label_column:
            raise ValidationError("`label_column` must be a non-empty string.")

        # Validate centroid_column if provided
        if centroid_column is not None:
            if not isinstance(centroid_column, str) or not centroid_column:
                raise ValidationError("`centroid_column` must be a non-empty string if provided.")

        # Check that the expression isn't a literal
        by_expr = Column._from_col_or_name(by)._logical_expr
        if isinstance(by_expr, LiteralExpr):
            raise ValidationError(
                f"Invalid cluster by: Cannot cluster by a literal value: {by_expr}."
            )

        return self._df._from_logical_plan(
            SemanticCluster.from_session_state(
                self._df._logical_plan,
                by_expr,
                num_clusters=num_clusters,
                max_iter=max_iter,
                num_init=num_init,
                label_column=label_column,
                centroid_column=centroid_column,
                session_state=self._df._session_state,
            ),
            self._df._session_state,
        )

    def join(
        self,
        other: DataFrame,
        join_instruction: str,
        examples: Optional[JoinExampleCollection] = None,
        model_alias: Optional[str] = None,
    ) -> DataFrame:
        """Performs a semantic join between two DataFrames using a natural language predicate.

        That evaluates to either true or false for each potential row pair.

        The join works by:
        1. Evaluating the provided join_instruction as a boolean predicate for each possible pair of rows
        2. Including ONLY the row pairs where the predicate evaluates to True in the result set
        3. Excluding all row pairs where the predicate evaluates to False

        The instruction must reference **exactly two columns**, one from each DataFrame,
        using the `:left` and `:right` suffixes to indicate column origin.

        This is useful when row pairing decisions require complex reasoning based on a custom predicate rather than simple equality or similarity matching.

        Args:
            other: The DataFrame to join with.
            join_instruction: A natural language description of how to match values.

                - Must include one placeholder from the left DataFrame (e.g. `{resume_summary:left}`)
                and one from the right (e.g. `{job_description:right}`).
                - This instruction is evaluated as a boolean predicate - pairs where it's `True` are included,
                pairs where it's `False` are excluded.
            examples: Optional JoinExampleCollection containing labeled pairs (`left`, `right`, `output`)
                to guide the semantic join behavior.
            model_alias: Optional alias for the language model to use for the mapping. If None, will use the language model configured as the default.

        Returns:
            DataFrame: A new DataFrame containing only the row pairs where the join_instruction
                      predicate evaluates to True.

        Raises:
            TypeError: If `other` is not a DataFrame or `join_instruction` is not a string.
            ValueError: If the instruction format is invalid or references invalid columns.

        Example: Basic semantic join
            ```python
            # Match job listings with candidate resumes based on title/skills
            # Only includes pairs where the predicate evaluates to True
            df_jobs.semantic.join(df_resumes,
                join_instruction="Given a candidate's resume_summary: {resume_summary:left} and a job description: {job_description:right}, does the candidate have the appropriate skills for the job?"
            )
            ```

        Example: Semantic join with examples
            ```python
            # Improve join quality with examples
            examples = JoinExampleCollection()
            examples.create_example(JoinExample(
                left="5 years experience building backend services in Python using asyncio, FastAPI, and PostgreSQL",
                right="Senior Software Engineer - Backend",
                output=True))  # This pair WILL be included in similar cases
            examples.create_example(JoinExample(
                left="5 years experience with growth strategy, private equity due diligence, and M&A",
                right="Product Manager - Hardware",
                output=False))  # This pair will NOT be included in similar cases
            df_jobs.semantic.join(df_resumes,
                join_instruction="Given a candidate's resume_summary: {resume_summary:left} and a job description: {job_description:right}, does the candidate have the appropriate skills for the job?",
                examples=examples)
            ```
        """
        from fenic.api.dataframe.dataframe import DataFrame

        if not isinstance(other, DataFrame):
            raise TypeError(f"other argument must be a DataFrame, got {type(other)}")

        if not isinstance(join_instruction, str):
            raise TypeError(
                f"join_instruction argument must be a string, got {type(join_instruction)}"
            )
        join_columns = utils.parse_instruction(join_instruction)
        if len(join_columns) != 2:
            raise ValueError(
                f"join_instruction must contain exactly two columns, got {len(join_columns)}"
            )
        left_on = None
        right_on = None
        for join_col in join_columns:
            if join_col.endswith(":left"):
                if left_on is not None:
                    raise ValueError(
                        "join_instruction cannot contain multiple :left columns"
                    )
                left_on = col(join_col.split(":")[0])
            elif join_col.endswith(":right"):
                if right_on is not None:
                    raise ValueError(
                        "join_instruction cannot contain multiple :right columns"
                    )
                right_on = col(join_col.split(":")[0])
            else:
                raise ValueError(
                    f"Column '{join_col}' must end with either :left or :right"
                )

        if left_on is None or right_on is None:
            raise ValueError(
                "join_instruction must contain exactly one :left and one :right column"
            )

        DataFrame._ensure_same_session(self._df._session_state, [other._session_state])
        return self._df._from_logical_plan(
            SemanticJoin.from_session_state(
                left=self._df._logical_plan,
                right=other._logical_plan,
                left_on=left_on._logical_expr,
                right_on=right_on._logical_expr,
                join_instruction=join_instruction,
                model_alias=model_alias,
                examples=examples,
                session_state=self._df._session_state,
            ),
            self._df._session_state,
        )

    def sim_join(
        self,
        other: DataFrame,
        left_on: ColumnOrName,
        right_on: ColumnOrName,
        k: int = 1,
        similarity_metric: SemanticSimilarityMetric = "cosine",
        similarity_score_column: Optional[str] = None,
    ) -> DataFrame:
        """Performs a semantic similarity join between two DataFrames using embedding expressions.

        For each row in the left DataFrame, returns the top `k` most semantically similar rows
        from the right DataFrame based on the specified similarity metric.

        Args:
            other: The right-hand DataFrame to join with.
            left_on: Expression or column representing embeddings in the left DataFrame.
            right_on: Expression or column representing embeddings in the right DataFrame.
            k: Number of most similar matches to return per row.
            similarity_metric: Similarity metric to use: "l2", "cosine", or "dot".
            similarity_score_column: If set, adds a column with this name containing similarity scores.
                If None, the scores are omitted.

        Returns:
            A DataFrame containing one row for each of the top-k matches per row in the left DataFrame.
            The result includes all columns from both DataFrames, optionally augmented with a similarity score column
            if `similarity_score_column` is provided.

        Raises:
            ValidationError: If `k` is not positive or if the columns are invalid.
            ValidationError: If `similarity_metric` is not one of "l2", "cosine", "dot"

        Example: Match queries to FAQ entries
            ```python
            # Match customer queries to FAQ entries
            df_queries.semantic.sim_join(
                df_faqs,
                left_on=embeddings(col("query_text")),
                right_on=embeddings(col("faq_question")),
                k=1
            )
            ```

        Example: Link headlines to articles
            ```python
            # Link news headlines to full articles
            df_headlines.semantic.sim_join(
                df_articles,
                left_on=embeddings(col("headline")),
                right_on=embeddings(col("content")),
                k=3,
                return_similarity_scores=True
            )
            ```

        Example: Find similar job postings
            ```python
            # Find similar job postings across two sources
            df_linkedin.semantic.sim_join(
                df_indeed,
                left_on=embeddings(col("job_title")),
                right_on=embeddings(col("job_description")),
                k=2
            )
            ```
        """
        from fenic.api.dataframe.dataframe import DataFrame

        if not isinstance(right_on, ColumnOrName):
            raise ValidationError(
                f"The `right_on` argument must be a `Column` or a string representing a column name, "
                f"but got `{type(right_on).__name__}` instead."
            )
        if not isinstance(other, DataFrame):
            raise ValidationError(
                            f"The `other` argument to `sim_join()` must be a DataFrame`, but got `{type(other).__name__}`."
                        )
        if not (isinstance(k, int) and k > 0):
            raise ValidationError(
                f"The parameter `k` must be a positive integer, but received `{k}`."
            )
        args = get_args(SemanticSimilarityMetric)
        if similarity_metric not in args:
            raise ValidationError(
                f"The `similarity_metric` argument must be one of {args}, but got `{similarity_metric}`."
            )

        def _validate_column(column: ColumnOrName, name: str):
            if column is None:
                raise ValidationError(f"The `{name}` argument must not be None.")
            if not isinstance(column, ColumnOrName):
                raise ValidationError(
                    f"The `{name}` argument must be a `Column` or a string representing a column name, "
                    f"but got `{type(column).__name__}` instead."
                )

        _validate_column(left_on, "left_on")
        _validate_column(right_on, "right_on")

        DataFrame._ensure_same_session(self._df._session_state, [other._session_state])
        return self._df._from_logical_plan(
            SemanticSimilarityJoin.from_session_state(
                self._df._logical_plan,
                other._logical_plan,
                Column._from_col_or_name(left_on)._logical_expr,
                Column._from_col_or_name(right_on)._logical_expr,
                k,
                similarity_metric,
                similarity_score_column,
                self._df._session_state,
            ),
            self._df._session_state,
        )

    # Spark aliases
    simJoin = sim_join
