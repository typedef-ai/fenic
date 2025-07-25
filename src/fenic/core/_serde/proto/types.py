"""Protobuf type imports with Proto suffix for use in serialization.

This module imports all generated protobuf classes with a 'Proto' suffix
to avoid naming conflicts with the Python classes they serialize.
"""

from __future__ import annotations

from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ClassifyExample as ClassifyExampleProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ClassifyExampleCollection as ClassifyExampleCollectionProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    JoinExample as JoinExampleProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    JoinExampleCollection as JoinExampleCollectionProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    KeyPoints as KeyPointsProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    MapExample as MapExampleProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    MapExampleCollection as MapExampleCollectionProto,
)

# Complex type protobuf classes
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    NumpyArray as NumpyArrayProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    Paragraph as ParagraphProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    PredicateExample as PredicateExampleProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    PredicateExampleCollection as PredicateExampleCollectionProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    PydanticModelType as PydanticModelTypeProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ResolvedClassDefinition as ResolvedClassDefinitionProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ResolvedModelAlias as ResolvedModelAliasProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarArray as ScalarArrayProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarStruct as ScalarStructProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarStructField as ScalarStructFieldProto,
)
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarValue as ScalarValueProto,
)

# DataType protobuf classes
from fenic.gen.protos.logical_plan.v1.complex_types_pb2 import (
    SummarizationFormat as SummarizationFormatProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    ArrayType as ArrayTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    BooleanType as BooleanTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    DataType as DataTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    DocumentPathType as DocumentPathTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    DoubleType as DoubleTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    EmbeddingType as EmbeddingTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    FloatType as FloatTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    HTMLType as HTMLTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    IntegerType as IntegerTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    JSONType as JSONTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    MarkdownType as MarkdownTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    StringType as StringTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    StructField as StructFieldProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    StructType as StructTypeProto,
)
from fenic.gen.protos.logical_plan.v1.datatypes_pb2 import (
    TranscriptType as TranscriptTypeProto,
)
from fenic.gen.protos.logical_plan.v1.enums_pb2 import (
    ChunkCharacterSet as ChunkCharacterSetProto,
)
from fenic.gen.protos.logical_plan.v1.enums_pb2 import (
    ChunkLengthFunction as ChunkLengthFunctionProto,
)
from fenic.gen.protos.logical_plan.v1.enums_pb2 import (
    Operator as OperatorProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    AliasExpr as AliasExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    AnalyzeSentimentExpr as AnalyzeSentimentExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ArithmeticExpr as ArithmeticExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayContainsExpr as ArrayContainsExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayExpr as ArrayExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayJoinExpr as ArrayJoinExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayLengthExpr as ArrayLengthExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    AvgExpr as AvgExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    BooleanExpr as BooleanExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ByteLengthExpr as ByteLengthExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    CastExpr as CastExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    CoalesceExpr as CoalesceExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ColumnExpr as ColumnExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ConcatExpr as ConcatExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ContainsAnyExpr as ContainsAnyExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ContainsExpr as ContainsExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    CountExpr as CountExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    CountTokensExpr as CountTokensExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    # Embedding expressions
    EmbeddingNormalizeExpr as EmbeddingNormalizeExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    EmbeddingsExpr as EmbeddingsExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    EmbeddingSimilarityExpr as EmbeddingSimilarityExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    EndsWithExpr as EndsWithExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    EqualityComparisonExpr as EqualityComparisonExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    FirstExpr as FirstExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    FuzzyRatioExpr as FuzzyRatioExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    FuzzyTokenSetRatioExpr as FuzzyTokenSetRatioExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    FuzzyTokenSortRatioExpr as FuzzyTokenSortRatioExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ILikeExpr as ILikeExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    IndexExpr as IndexExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    InExpr as InExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    IsNullExpr as IsNullExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    JinjaExpr as JinjaExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    # JSON expressions
    JqExpr as JqExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    JsonContainsExpr as JsonContainsExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    JsonTypeExpr as JsonTypeExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    LikeExpr as LikeExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ListExpr as ListExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    LiteralExpr as LiteralExprProto,
)

# Base/Basic Expressions
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    LogicalExpr as LogicalExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    MaxExpr as MaxExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    # Markdown Expressions
    MdExtractHeaderChunks as MdExtractHeaderChunksProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    MdGenerateTocExpr as MdGenerateTocExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    MdGetCodeBlocksExpr as MdGetCodeBlocksExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    MdToJsonExpr as MdToJsonExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    MinExpr as MinExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    NotExpr as NotExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    NumericComparisonExpr as NumericComparisonExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    OtherwiseExpr as OtherwiseExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    RecursiveTextChunkExpr as RecursiveTextChunkExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    RecursiveTextChunkExprConfiguration as RecursiveTextChunkExprConfigurationProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    RegexpSplitExpr as RegexpSplitExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    ReplaceExpr as ReplaceExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    RLikeExpr as RLikeExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticClassifyExpr as SemanticClassifyExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticExtractExpr as SemanticExtractExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    # Semantic expressions
    SemanticMapExpr as SemanticMapExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticPredExpr as SemanticPredExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticReduceExpr as SemanticReduceExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticSummarizeExpr as SemanticSummarizeExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SortExpr as SortExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SplitPartExpr as SplitPartExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    StartsWithExpr as StartsWithExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    StdDevExpr as StdDevExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    StringCasingExpr as StringCasingExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    StripCharsExpr as StripCharsExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    StrLengthExpr as StrLengthExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    StructExpr as StructExprProto,
)

# Aggregate expressions
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    SumExpr as SumExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    TextChunkExpr as TextChunkExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    TextChunkExprConfiguration as TextChunkExprConfigurationProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    # Text expressions
    TextractExpr as TextractExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    TsParseExpr as TsParseExprProto,
)
from fenic.gen.protos.logical_plan.v1.expressions_pb2 import (
    # Case expressions
    WhenExpr as WhenExprProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    SQL as SQLProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Aggregate as AggregateProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    ColumnField as ColumnFieldProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    DropDuplicates as DropDuplicatesProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Explode as ExplodeProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    FenicSchema as FenicSchemaProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    # Sink plans
    FileSink as FileSinkProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    FileSource as FileSourceProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Filter as FilterProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    # Source plans
    InMemorySource as InMemorySourceProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Join as JoinProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Limit as LimitProto,
)

# Plan protobuf classes
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    LogicalPlan as LogicalPlanProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    # Transform plans
    Projection as ProjectionProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    SemanticCluster as SemanticClusterProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import SemanticJoin as SemanticJoinProto
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    SemanticSimilarityJoin as SemanticSimilarityJoinProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Sort as SortProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    TableSink as TableSinkProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    TableSource as TableSourceProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Union as UnionProto,
)
from fenic.gen.protos.logical_plan.v1.plans_pb2 import (
    Unnest as UnnestProto,
)

# Export all protobuf classes for easy importing
__all__ = [
    # DataType classes
    "DataTypeProto",
    "StringTypeProto",
    "IntegerTypeProto",
    "FloatTypeProto",
    "DoubleTypeProto",
    "BooleanTypeProto",
    "ArrayTypeProto",
    "StructTypeProto",
    "StructFieldProto",
    "EmbeddingTypeProto",
    "TranscriptTypeProto",
    "DocumentPathTypeProto",
    "MarkdownTypeProto",
    "HTMLTypeProto",
    "JSONTypeProto",
    # Enum classes
    "OperatorProto",
    "ChunkLengthFunctionProto",
    "ChunkCharacterSetProto",
    # Complex type classes
    "NumpyArrayProto",
    "PydanticModelTypeProto",
    "KeyPointsProto",
    "ParagraphProto",
    "SummarizationFormatProto",
    "MapExampleProto",
    "MapExampleCollectionProto",
    "ClassifyExampleProto",
    "ClassifyExampleCollectionProto",
    "PredicateExampleProto",
    "PredicateExampleCollectionProto",
    "JoinExampleProto",
    "JoinExampleCollectionProto",
    "ResolvedClassDefinitionProto",
    "ResolvedModelAliasProto",
    # Expression classes
    "LogicalExprProto",
    "ColumnExprProto",
    "LiteralExprProto",
    "ScalarValueProto",
    "ScalarArrayProto",
    "ScalarStructProto",
    "ScalarStructFieldProto",
    "AliasExprProto",
    "SortExprProto",
    "IndexExprProto",
    "ArrayExprProto",
    "StructExprProto",
    "CastExprProto",
    "NotExprProto",
    "CoalesceExprProto",
    "InExprProto",
    "IsNullExprProto",
    "ArrayLengthExprProto",
    "ArrayContainsExprProto",
    # Binary Exprs
    "ArithmeticExprProto",
    "BooleanExprProto",
    "NumericComparisonExprProto",
    "EqualityComparisonExprProto",
    # Semantic expression classes
    "SemanticMapExprProto",
    "SemanticExtractExprProto",
    "SemanticPredExprProto",
    "SemanticReduceExprProto",
    "SemanticClassifyExprProto",
    "AnalyzeSentimentExprProto",
    "EmbeddingsExprProto",
    "SemanticSummarizeExprProto",
    # Embedding expression classes
    "EmbeddingNormalizeExprProto",
    "EmbeddingSimilarityExprProto",
    # Text expression classes
    "RecursiveTextChunkExprConfigurationProto",
    "TextChunkExprConfigurationProto",
    "TextractExprProto",
    "TextChunkExprProto",
    "RecursiveTextChunkExprProto",
    "CountTokensExprProto",
    "ConcatExprProto",
    "ArrayJoinExprProto",
    "ContainsExprProto",
    "ContainsAnyExprProto",
    "RLikeExprProto",
    "LikeExprProto",
    "ILikeExprProto",
    "TsParseExprProto",
    "StartsWithExprProto",
    "EndsWithExprProto",
    "RegexpSplitExprProto",
    "SplitPartExprProto",
    "StringCasingExprProto",
    "StripCharsExprProto",
    "ReplaceExprProto",
    "StrLengthExprProto",
    "ByteLengthExprProto",
    "JinjaExprProto",
    "FuzzyRatioExprProto",
    "FuzzyTokenSortRatioExprProto",
    "FuzzyTokenSetRatioExprProto",
    # JSON expression classes
    "JqExprProto",
    "JsonTypeExprProto",
    "JsonContainsExprProto",
    # Markdown expression classes
    "MdToJsonExprProto",
    "MdGetCodeBlocksExprProto",
    "MdGenerateTocExprProto",
    "MdExtractHeaderChunksProto",
    # Case expression classes
    "WhenExprProto",
    "OtherwiseExprProto",
    # Aggregate expression classes
    "SumExprProto",
    "AvgExprProto",
    "CountExprProto",
    "MaxExprProto",
    "MinExprProto",
    "FirstExprProto",
    "ListExprProto",
    "StdDevExprProto",
    # Plan classes
    "LogicalPlanProto",
    "FenicSchemaProto",
    "ColumnFieldProto",
    # Source plan classes
    "InMemorySourceProto",
    "FileSourceProto",
    "TableSourceProto",
    # Transform plan classes
    "ProjectionProto",
    "FilterProto",
    "JoinProto",
    "AggregateProto",
    "UnionProto",
    "LimitProto",
    "ExplodeProto",
    "DropDuplicatesProto",
    "SortProto",
    "UnnestProto",
    "SQLProto",
    "SemanticClusterProto",
    "SemanticJoinProto",
    "SemanticSimilarityJoinProto",
    # Sink plan classes
    "FileSinkProto",
    "TableSinkProto",
]
