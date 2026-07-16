import fenic as fc
import structlog

logger = structlog.get_logger(__name__)


def search_api_docs(
    session: fc.Session, query: str, types: list[str] | None = None
) -> fc.DataFrame:
    # Search API documentation
    df = session.table("api_df")

    # Filter only public API elements
    df = df.filter(
        (fc.col("is_public")) & (~fc.col("qualified_name").rlike(r"(^|\.)_"))
    )

    # Optional filter by API element types (e.g., function, class, method)
    if types:
        # Use Column.is_in per Fenic API for membership filtering
        df = df.filter(fc.col("type").is_in(types))

    logger.debug(f"Searching API docs with regex: {query}")
    search_df = _search_api_docs_regex(df, query)

    # Add relevance scoring across common fields
    search_df = search_df.select(
        "type",
        "name",
        "qualified_name",
        "docstring",
        fc.when(fc.col("name").rlike(f"(?i){query}"), fc.lit(12))
        .otherwise(fc.lit(0))
        .alias("name_score"),
        fc.when(fc.col("qualified_name").rlike(f"(?i){query}"), fc.lit(6))
        .otherwise(fc.lit(0))
        .alias("path_score"),
        fc.when(
            fc.col("docstring").is_not_null()
            & fc.col("docstring").rlike(f"(?i){query}"),
            fc.lit(4),
        )
        .otherwise(fc.lit(0))
        .alias("doc_score"),
        fc.when(
            fc.col("annotation").is_not_null()
            & fc.col("annotation").rlike(f"(?i){query}"),
            fc.lit(2),
        )
        .otherwise(fc.lit(0))
        .alias("annotation_score"),
        fc.when(
            fc.col("returns").is_not_null() & fc.col("returns").rlike(f"(?i){query}"),
            fc.lit(2),
        )
        .otherwise(fc.lit(0))
        .alias("returns_score"),
    )

    # Calculate total score
    search_df = search_df.select(
        "*",
        (
            fc.col("name_score")
            + fc.col("path_score")
            + fc.col("doc_score")
            + fc.col("annotation_score")
            + fc.col("returns_score")
        ).alias("score"),
    )

    return search_df


def _search_api_docs_regex(df: fc.DataFrame, query: str) -> fc.DataFrame:
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
        | (fc.col("returns").is_not_null() & fc.col("returns").rlike(f"(?i){query}"))
    )


def get_entity_by_qualified_name(
    session: fc.Session, qualified_name: str
) -> fc.DataFrame:
    """Fetch a single API entity by its qualified name (exact match)."""
    df = session.table("api_df")
    df = df.filter((fc.col("is_public")) & (fc.col("qualified_name") == qualified_name))
    # Return key fields commonly used by consumers
    return df.select(
        "type",
        "name",
        "qualified_name",
        "docstring",
        "annotation",
        "returns",
    )
