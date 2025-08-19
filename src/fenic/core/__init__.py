"""Core module for Fenic."""

from fenic.core.metrics import (
    LMMetrics,
    OperatorMetrics,
    QueryMetrics,
    RMMetrics,
)
from fenic.core.types import (
    ArrayType,
    BooleanType,
    BranchSide,
    ClassDefinition,
    ClassifyExample,
    ClassifyExampleCollection,
    ColumnField,
    DataLike,
    DataLikeType,
    DataType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    FuzzySimilarityMethod,
    HtmlType,
    IntegerType,
    JoinExample,
    JoinExampleCollection,
    JsonType,
    KeyPoints,
    MapExample,
    MapExampleCollection,
    MarkdownType,
    Paragraph,
    PredicateExample,
    PredicateExampleCollection,
    QueryResult,
    Schema,
    SemanticSimilarityMetric,
    StringType,
    StructField,
    StructType,
    TranscriptType,
)

# Re-export MCP generator (optional dependency)
try:  # pragma: no cover - import-guard
    from fenic.core.mcp.generator import MCPGenerator, create_mcp_server_from_views
except Exception:  # pragma: no cover - if fastmcp missing
    MCPGenerator = None  # type: ignore
    create_mcp_server_from_views = None  # type: ignore

__all__ = [
    # Types
    "ArrayType",
    "BooleanType",
    "BranchSide",
    "DataType",
    "DocumentPathType",
    "DoubleType",
    "EmbeddingType",
    "FloatType",
    "HtmlType",
    "IntegerType",
    "JsonType",
    "MarkdownType",
    "StringType",
    "StructField",
    "StructType",
    "TranscriptType",
    "ColumnField",
    "Schema",
    "ClassDefinition",
    "ClassifyExample",
    "ClassifyExampleCollection",
    "JoinExample",
    "JoinExampleCollection",
    "MapExample",
    "MapExampleCollection",
    "PredicateExample",
    "PredicateExampleCollection",
    "SemanticSimilarityMetric",
    "KeyPoints",
    "Paragraph",
    "FuzzySimilarityMethod",
    # Metrics
    "QueryMetrics",
    "LMMetrics",
    "RMMetrics",
    "OperatorMetrics",
    # QueryResult
    "DataLike",
    "DataLikeType",
    "QueryResult",
    # MCP (optional)
    "MCPGenerator",
    "create_mcp_server_from_views",
]
