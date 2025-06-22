"""Query module for semantic operations on DataFrames."""

from fenic.api.catalog import Catalog
from fenic.api.column import Column, ColumnOrName
from fenic.api.dataframe import DataFrame, GroupedData, SemanticExtensions
from fenic.api.functions import (
    array,
    array_agg,
    array_contains,
    array_size,
    asc,
    asc_nulls_first,
    asc_nulls_last,
    avg,
    coalesce,
    col,
    collect_list,
    count,
    desc,
    desc_nulls_first,
    desc_nulls_last,
    embedding,
    first,
    json,
    lit,
    markdown,
    max,
    mean,
    min,
    semantic,
    struct,
    sum,
    text,
    udf,
    when,
)
from fenic.api.io import DataFrameReader, DataFrameWriter
from fenic.api.lineage import Lineage
from fenic.api.session import (
    AnthropicModelConfig,
    GoogleGLAModelConfig,
    OpenAIModelConfig,
    SemanticConfig,
    Session,
    SessionConfig,
)

__all__ = [
    # Session
    "Session",
    "SessionConfig",
    "OpenAIModelConfig",
    "AnthropicModelConfig",
    "GoogleGLAModelConfig",
    "SemanticConfig",
    # IO
    "DataFrameReader",
    "DataFrameWriter",
    # DataFrame
    "DataFrame",
    "GroupedData",
    "SemanticExtensions",
    # Column
    "Column",
    "ColumnOrName",
    # Catalog
    "Catalog",
    # Functions
    "semantic",
    "text",
    "json",
    "markdown",
    "embedding",
    "array",
    "array_agg",
    "array_contains",
    "array_size",
    "asc",
    "asc_nulls_first",
    "asc_nulls_last",
    "avg",
    "coalesce",
    "collect_list",
    "count",
    "desc",
    "desc_nulls_first",
    "desc_nulls_last",
    "first",
    "max",
    "mean",
    "min",
    "struct",
    "sum",
    "udf",
    "when",
    "col",
    "lit",
    # Lineage
    "Lineage",
]
