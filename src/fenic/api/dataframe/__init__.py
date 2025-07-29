"""DataFrame API for Fenic - provides DataFrame and grouped data operations."""

from fenic.api.dataframe.dataframe import DataFrame
from fenic.api.dataframe.grouped_data import GroupedData
from fenic.api.dataframe.semantic_extensions import (
    SemanticExtensions,
)

__all__ = ["DataFrame", "GroupedData", "SemanticExtensions"]
