"""Semantic extensions for DataFrames providing clustering and semantic join operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, get_args

from fenic._inference.request_utils import validate_timeout
from fenic.core.error import ValidationError
from fenic.core.types import (
    JoinExampleCollection,
)
from fenic.core.types.semantic import ModelAlias, _resolve_model_alias

if TYPE_CHECKING:
    from fenic.api.dataframe import DataFrame

from fenic.api.column import Column, ColumnOrName
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
        request_timeout: Optional[float] = None,
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
            request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

        Returns:
            A DataFrame with all original columns plus:
            - `<label_column>`: integer cluster assignment (0 to num_clusters - 1)
            - `<centroid_column>`: cluster centroid embedding, if specified

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
        # Validate request_timeout
        request_timeout = validate_timeout(request_timeout)

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
                request_timeout=request_timeout,
            ),
            self._df._session_state,
        )

    def join(
        self,
        other: DataFrame,
        predicate: str,
        left_on: Column,
        right_on: Column,
        strict: bool = True,
        examples: Optional[JoinExampleCollection] = None,
        model_alias: Optional[Union[str, ModelAlias]] = None,
        request_timeout: Optional[float] = None,
    ) -> DataFrame:
        """Performs a semantic join between two DataFrames using a natural language predicate.

        This method evaluates a boolean predicate for each potential row pair between the two DataFrames,
        including only those pairs where the predicate evaluates to True.

        The join process:
        1. For each row in the left DataFrame, evaluates the predicate in the jinja template against each row in the right DataFrame
        2. Includes row pairs where the predicate returns True
        3. Excludes row pairs where the predicate returns False
        4. Returns a new DataFrame containing all columns from both DataFrames for the matched pairs

        The jinja template must use exactly two column placeholders:
        - One from the left DataFrame: `{{ left_on }}`
        - One from the right DataFrame: `{{ right_on }}`

        Args:
            other: The DataFrame to join with.
            predicate: A Jinja2 template containing the natural language predicate.
                Must include placeholders for exactly one column from each DataFrame.
                The template is evaluated as a boolean - True includes the pair, False excludes it.
            left_on: The column from the left DataFrame (self) to use in the join predicate.
            right_on: The column from the right DataFrame (other) to use in the join predicate.
            strict: If True, when either the left_on or right_on column has a None value for a row pair,
                    that pair is automatically excluded from the join (predicate is not evaluated).
                    If False, None values are rendered according to Jinja2's null rendering behavior.
                    Default is True.
            examples: Optional JoinExampleCollection containing labeled examples to guide the join.
                Each example should have:
                - left: Sample value from the left column
                - right: Sample value from the right column
                - output: Boolean indicating whether this pair should be joined (True) or not (False)
            model_alias: Optional alias for the language model to use. If None, uses the default model.
            request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

        Returns:
            DataFrame: A new DataFrame containing matched row pairs with all columns from both DataFrames.

        Example: Basic semantic join
            ```python
            # Match job listings with candidate resumes based on title/skills
            # Only includes pairs where the predicate evaluates to True
            df_jobs.semantic.join(df_resumes,
                predicate=dedent('''\
                    Job Description: {{left_on}}
                    Candidate Background: {{right_on}}
                    The candidate is qualified for the job.'''),
                left_on=col("job_description"),
                right_on=col("work_experience"),
                examples=examples
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
            df_jobs.semantic.join(
                other=df_resumes,
                predicate=dedent('''\
                    Job Description: {{left_on}}
                    Candidate Background: {{right_on}}
                    The candidate is qualified for the job.'''),
                left_on=col("job_description"),
                right_on=col("work_experience"),
                examples=examples
            )
            ```
        """
        from fenic.api.dataframe.dataframe import DataFrame

        if not isinstance(other, DataFrame):
            raise ValidationError(f"other argument must be a DataFrame, got {type(other)}")

        if not isinstance(predicate, str):
            raise ValidationError(
                f"The `predicate` argument to `semantic.join` must be a string, got {type(predicate)}"
            )
        if not isinstance(left_on, Column):
            raise ValidationError(f"`left_on` argument must be a Column, got {type(left_on)} instead.")
        if not isinstance(right_on, Column):
            raise ValidationError(f"`right_on` argument must be a Column, got {type(right_on)} instead.")
        if examples is not None and not isinstance(examples, JoinExampleCollection):
            raise ValidationError(f"`examples` argument must be a JoinExampleCollection, got {type(examples)} instead.")
        if model_alias is not None and not isinstance(model_alias, (str, ModelAlias)):
            raise ValidationError(f"`model_alias` argument must be a string or ModelAlias, got {type(model_alias)} instead.")

        # Validate request_timeout
        request_timeout = validate_timeout(request_timeout)

        resolved_model_alias = _resolve_model_alias(model_alias)
        DataFrame._ensure_same_session(self._df._session_state, [other._session_state])

        return self._df._from_logical_plan(
            SemanticJoin.from_session_state(
                left=self._df._logical_plan,
                right=other._logical_plan,
                left_on=left_on._logical_expr,
                right_on=right_on._logical_expr,
                jinja_template=predicate,
                strict=strict,
                model_alias=resolved_model_alias,
                examples=examples,
                request_timeout=request_timeout,
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
        request_timeout: Optional[float] = None,
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
            request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

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

        # Validate request_timeout
        request_timeout = validate_timeout(request_timeout)
        DataFrame._ensure_same_session(self._df._session_state, [other._session_state])
        return self._df._from_logical_plan(
            SemanticSimilarityJoin.from_session_state(
                self._df._logical_plan,
                other._logical_plan,
                Column._from_col_or_name(left_on)._logical_expr,
                Column._from_col_or_name(right_on)._logical_expr,
                k,
                similarity_metric=similarity_metric,
                similarity_score_column=similarity_score_column,
                session_state=self._df._session_state,
                request_timeout=request_timeout,
            ),
            self._df._session_state,
        )

    # Spark aliases
    simJoin = sim_join
