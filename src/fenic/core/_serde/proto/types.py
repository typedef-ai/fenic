"""Protobuf type imports with Proto suffix for use in serialization.

This module imports all generated protobuf classes with a 'Proto' suffix
to avoid naming conflicts with the Python classes they serialize.
"""

from __future__ import annotations

from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ClassifyExample as ClassifyExampleProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ClassifyExampleCollection as ClassifyExampleCollectionProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    JoinExample as JoinExampleProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    JoinExampleCollection as JoinExampleCollectionProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    KeyPoints as KeyPointsProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    MapExample as MapExampleProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    MapExampleCollection as MapExampleCollectionProto,
)

# Complex type protobuf classes
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    NumpyArray as NumpyArrayProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    Paragraph as ParagraphProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    PredicateExample as PredicateExampleProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    PredicateExampleCollection as PredicateExampleCollectionProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ResolvedClassDefinition as ResolvedClassDefinitionProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ResolvedModelAlias as ResolvedModelAliasProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ResolvedResponseFormat as ResolvedResponseFormatProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarArray as ScalarArrayProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarStruct as ScalarStructProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarStructField as ScalarStructFieldProto,
)
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    ScalarValue as ScalarValueProto,
)

# DataType protobuf classes
from fenic._gen.protos.logical_plan.v1.complex_types_pb2 import (
    SummarizationFormat as SummarizationFormatProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    ArrayType as ArrayTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    BooleanType as BooleanTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    DataType as DataTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    DateType as DateTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    DocumentPathType as DocumentPathTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    DoubleType as DoubleTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    EmbeddingType as EmbeddingTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    FloatType as FloatTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    HTMLType as HTMLTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    IntegerType as IntegerTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    JSONType as JSONTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    MarkdownType as MarkdownTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    StringType as StringTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    StructField as StructFieldProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    StructType as StructTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    TimestampType as TimestampTypeProto,
)
from fenic._gen.protos.logical_plan.v1.datatypes_pb2 import (
    TranscriptType as TranscriptTypeProto,
)
from fenic._gen.protos.logical_plan.v1.enums_pb2 import (
    ChunkCharacterSet as ChunkCharacterSetProto,
)
from fenic._gen.protos.logical_plan.v1.enums_pb2 import (
    ChunkLengthFunction as ChunkLengthFunctionProto,
)
from fenic._gen.protos.logical_plan.v1.enums_pb2 import (
    DocContentType as DocContentTypeProto,
)
from fenic._gen.protos.logical_plan.v1.enums_pb2 import (
    FuzzySimilarityMethod as FuzzySimilarityMethodProto,
)
from fenic._gen.protos.logical_plan.v1.enums_pb2 import (
    Operator as OperatorProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    AliasExpr as AliasExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    AnalyzeSentimentExpr as AnalyzeSentimentExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ApproxCountDistinctExpr as ApproxCountDistinctExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArithmeticExpr as ArithmeticExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayCompactExpr as ArrayCompactExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayContainsExpr as ArrayContainsExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayDistinctExpr as ArrayDistinctExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayExceptExpr as ArrayExceptExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayExpr as ArrayExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayIntersectExpr as ArrayIntersectExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayJoinExpr as ArrayJoinExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayLengthExpr as ArrayLengthExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayMaxExpr as ArrayMaxExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayMinExpr as ArrayMinExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayRemoveExpr as ArrayRemoveExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayRepeatExpr as ArrayRepeatExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayReverseExpr as ArrayReverseExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArraySliceExpr as ArraySliceExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArraySortExpr as ArraySortExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArraysOverlapExpr as ArraysOverlapExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ArrayUnionExpr as ArrayUnionExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    AvgExpr as AvgExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    BooleanExpr as BooleanExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ByteLengthExpr as ByteLengthExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    CastExpr as CastExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    CoalesceExpr as CoalesceExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ColumnExpr as ColumnExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ConcatExpr as ConcatExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ContainsAnyExpr as ContainsAnyExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ContainsExpr as ContainsExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    CountDistinctExpr as CountDistinctExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    CountExpr as CountExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    CountTokensExpr as CountTokensExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    DateAddExpr as DateAddExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    DateDiffExpr as DateDiffExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    DateFormatExpr as DateFormatExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    DateTruncExpr as DateTruncExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    DayExpr as DayExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ElementAtExpr as ElementAtExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    # Embedding expressions
    EmbeddingNormalizeExpr as EmbeddingNormalizeExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    EmbeddingsExpr as EmbeddingsExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    EmbeddingSimilarityExpr as EmbeddingSimilarityExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    EndsWithExpr as EndsWithExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    EqualityComparisonExpr as EqualityComparisonExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    FirstExpr as FirstExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    FlattenExpr as FlattenExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    FromUTCTimestampExpr as FromUTCTimestampExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    FuzzyRatioExpr as FuzzyRatioExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    FuzzyTokenSetRatioExpr as FuzzyTokenSetRatioExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    FuzzyTokenSortRatioExpr as FuzzyTokenSortRatioExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    GreatestExpr as GreatestExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    HourExpr as HourExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ILikeExpr as ILikeExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    IndexExpr as IndexExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    InExpr as InExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    IsNullExpr as IsNullExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    JinjaExpr as JinjaExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    # JSON expressions
    JqExpr as JqExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    JsonContainsExpr as JsonContainsExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    JsonTypeExpr as JsonTypeExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    LeastExpr as LeastExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    LikeExpr as LikeExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ListExpr as ListExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    LiteralExpr as LiteralExprProto,
)

# Base/Basic Expressions
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    LogicalExpr as LogicalExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MaxExpr as MaxExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    # Markdown Expressions
    MdExtractHeaderChunks as MdExtractHeaderChunksProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MdGenerateTocExpr as MdGenerateTocExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MdGetCodeBlocksExpr as MdGetCodeBlocksExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MdToJsonExpr as MdToJsonExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MilliSecondExpr as MilliSecondExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MinExpr as MinExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MinuteExpr as MinuteExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    MonthExpr as MonthExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    NotExpr as NotExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    NowExpr as NowExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    NumericComparisonExpr as NumericComparisonExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    OtherwiseExpr as OtherwiseExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RecursiveTextChunkExpr as RecursiveTextChunkExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RegexpCountExpr as RegexpCountExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RegexpExtractAllExpr as RegexpExtractAllExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RegexpExtractExpr as RegexpExtractExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RegexpInstrExpr as RegexpInstrExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RegexpSplitExpr as RegexpSplitExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RegexpSubstrExpr as RegexpSubstrExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ReplaceExpr as ReplaceExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    RLikeExpr as RLikeExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SecondExpr as SecondExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticClassifyExpr as SemanticClassifyExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticExtractExpr as SemanticExtractExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    # Semantic expressions
    SemanticMapExpr as SemanticMapExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticParsePDFExpr as SemanticParsePDFExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticPredExpr as SemanticPredExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticReduceExpr as SemanticReduceExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SemanticSummarizeExpr as SemanticSummarizeExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SeriesLiteralExpr as SeriesLiteralExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SortExpr as SortExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SplitPartExpr as SplitPartExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    StartsWithExpr as StartsWithExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    StdDevExpr as StdDevExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    StringCasingExpr as StringCasingExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    StripCharsExpr as StripCharsExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    StrLengthExpr as StrLengthExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    StructExpr as StructExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SumDistinctExpr as SumDistinctExprProto,
)

# Aggregate expressions
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    SumExpr as SumExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    TextChunkExpr as TextChunkExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    # Text expressions
    TextractExpr as TextractExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    TimestampAddExpr as TimestampAddExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    TimestampDiffExpr as TimestampDiffExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ToDateExpr as ToDateExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ToTimestampExpr as ToTimestampExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    ToUTCTimestampExpr as ToUTCTimestampExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    TsParseExpr as TsParseExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    UnresolvedLiteralExpr as UnresolvedLiteralExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    # Case expressions
    WhenExpr as WhenExprProto,
)
from fenic._gen.protos.logical_plan.v1.expressions_pb2 import (
    YearExpr as YearExprProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    SQL as SQLProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Aggregate as AggregateProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    CacheInfo as CacheInfoProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    ColumnField as ColumnFieldProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    DocSource as DocSourceProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    DropDuplicates as DropDuplicatesProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Explode as ExplodeProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    ExplodeWithIndex as ExplodeWithIndexProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    FenicSchema as FenicSchemaProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    # Sink plans
    FileSink as FileSinkProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    FileSource as FileSourceProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Filter as FilterProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    # Source plans
    InMemorySource as InMemorySourceProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Join as JoinProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Limit as LimitProto,
)

# Plan protobuf classes
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    LogicalPlan as LogicalPlanProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    # Transform plans
    Projection as ProjectionProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    SemanticCluster as SemanticClusterProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    SemanticJoin as SemanticJoinProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    SemanticSimilarityJoin as SemanticSimilarityJoinProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Sort as SortProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    TableSink as TableSinkProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    TableSource as TableSourceProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Union as UnionProto,
)
from fenic._gen.protos.logical_plan.v1.plans_pb2 import (
    Unnest as UnnestProto,
)
from fenic._gen.protos.logical_plan.v1.tools_pb2 import (
    ToolDefinition as ToolDefinitionProto,
)
from fenic._gen.protos.logical_plan.v1.tools_pb2 import (
    ToolParameter as ToolParameterProto,
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
    "DateTypeProto",
    "TimestampTypeProto",
    # Enum classes
    "OperatorProto",
    "ChunkLengthFunctionProto",
    "ChunkCharacterSetProto",
    "DocContentTypeProto",
    # Complex type classes
    "NumpyArrayProto",
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
    "ResolvedResponseFormatProto",
    "UnresolvedLiteralExprProto",
    # Expression classes
    "LogicalExprProto",
    "ColumnExprProto",
    "LiteralExprProto",
    "SeriesLiteralExprProto",
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
    "ArrayDistinctExprProto",
    "ArrayMaxExprProto",
    "ArrayMinExprProto",
    "ArraySortExprProto",
    "ArrayReverseExprProto",
    "ArrayRemoveExprProto",
    "ArrayUnionExprProto",
    "ArrayIntersectExprProto",
    "ArrayExceptExprProto",
    "ArrayCompactExprProto",
    "ArrayRepeatExprProto",
    "FlattenExprProto",
    "ArraySliceExprProto",
    "ElementAtExprProto",
    "ArraysOverlapExprProto",
    "GreatestExprProto",
    "LeastExprProto",
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
    "SemanticParsePDFExprProto",
    "AnalyzeSentimentExprProto",
    "EmbeddingsExprProto",
    "SemanticSummarizeExprProto",
    # Embedding expression classes
    "EmbeddingNormalizeExprProto",
    "EmbeddingSimilarityExprProto",
    # Text expression classes
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
    "RegexpCountExprProto",
    "RegexpExtractAllExprProto",
    "RegexpExtractExprProto",
    "RegexpInstrExprProto",
    "RegexpSplitExprProto",
    "RegexpSubstrExprProto",
    "SplitPartExprProto",
    "StringCasingExprProto",
    "StripCharsExprProto",
    "ReplaceExprProto",
    "StrLengthExprProto",
    "ByteLengthExprProto",
    "JinjaExprProto",
    "FuzzyRatioExprProto",
    "FuzzySimilarityMethodProto",
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
    "CountDistinctExprProto",
    "ApproxCountDistinctExprProto",
    "SumDistinctExprProto",
    # Plan classes
    "LogicalPlanProto",
    "FenicSchemaProto",
    "ColumnFieldProto",
    "CacheInfoProto",
    # Source plan classes
    "InMemorySourceProto",
    "FileSourceProto",
    "TableSourceProto",
    "DocSourceProto",
    # Transform plan classes
    "ProjectionProto",
    "FilterProto",
    "JoinProto",
    "AggregateProto",
    "UnionProto",
    "LimitProto",
    "ExplodeProto",
    "ExplodeWithIndexProto",
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
    # Tools
    "ToolParameterProto",
    "ToolDefinitionProto",
    # Date time related classes
    "YearExprProto",
    "MonthExprProto",
    "DayExprProto",
    "HourExprProto",
    "MinuteExprProto",
    "SecondExprProto",
    "MilliSecondExprProto",
    "DateAddExprProto",
    "TimestampAddExprProto",
    "DateTruncExprProto",
    "NowExprProto",
    "ToDateExprProto",
    "ToTimestampExprProto",
    "DateFormatExprProto",
    "DateDiffExprProto",
    "TimestampDiffExprProto",
    "ToUTCTimestampExprProto",
    "FromUTCTimestampExprProto",
]
