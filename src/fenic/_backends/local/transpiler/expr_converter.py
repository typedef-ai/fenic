from __future__ import annotations

from datetime import timezone
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from fenic.core._logical_plan.expressions.basic import UnresolvedLiteralExpr

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

import datetime
import json
import logging

import numpy as np
import polars as pl
import pyarrow as pa

import fenic._backends.local.polars_plugins  # noqa: F401
from fenic._backends.local.async_udf_stream import AsyncUDFSyncStream
from fenic._backends.local.async_utils import EventLoopManager
from fenic._backends.local.semantic_operators import (
    AnalyzeSentiment,
)
from fenic._backends.local.semantic_operators import Classify as SemanticClassify
from fenic._backends.local.semantic_operators import Extract as SemanticExtract
from fenic._backends.local.semantic_operators import Map as SemanticMap
from fenic._backends.local.semantic_operators import ParsePDF as SemanticParsePDF
from fenic._backends.local.semantic_operators import Predicate as SemanticPredicate
from fenic._backends.local.semantic_operators import Reduce as SemanticReduce
from fenic._backends.local.semantic_operators import Summarize as SemanticSummarize
from fenic._backends.local.semantic_operators.reduce import (
    DATA_COLUMN_NAME,
    SORT_KEY_COLUMN_NAME,
)
from fenic._backends.local.template import TemplateFormatReader
from fenic._backends.schema_serde import serialize_data_type
from fenic.core._logical_plan.expressions import (
    AliasExpr,
    AnalyzeSentimentExpr,
    ApproxCountDistinctExpr,
    ArithmeticExpr,
    ArrayCompactExpr,
    ArrayContainsExpr,
    ArrayDistinctExpr,
    ArrayExceptExpr,
    ArrayExpr,
    ArrayIntersectExpr,
    ArrayJoinExpr,
    ArrayLengthExpr,
    ArrayMaxExpr,
    ArrayMinExpr,
    ArrayRemoveExpr,
    ArrayRepeatExpr,
    ArrayReverseExpr,
    ArraySliceExpr,
    ArraySortExpr,
    ArraysOverlapExpr,
    ArrayUnionExpr,
    AsyncUDFExpr,
    AvgExpr,
    BooleanExpr,
    ByteLengthExpr,
    CastExpr,
    ChunkLengthFunction,
    CoalesceExpr,
    ColumnExpr,
    ConcatExpr,
    ContainsAnyExpr,
    ContainsExpr,
    CountDistinctExpr,
    CountExpr,
    CountTokensExpr,
    DateAddExpr,
    DateDiffExpr,
    DateFormatExpr,
    DateTruncExpr,
    DayExpr,
    ElementAtExpr,
    EmbeddingNormalizeExpr,
    EmbeddingsExpr,
    EmbeddingSimilarityExpr,
    EndsWithExpr,
    EqualityComparisonExpr,
    FirstExpr,
    FlattenExpr,
    FromUTCTimestampExpr,
    FuzzyRatioExpr,
    FuzzyTokenSetRatioExpr,
    FuzzyTokenSortRatioExpr,
    GreatestExpr,
    HourExpr,
    ILikeExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    JinjaExpr,
    JqExpr,
    JsonContainsExpr,
    JsonTypeExpr,
    LeastExpr,
    LikeExpr,
    ListExpr,
    LiteralExpr,
    LogicalExpr,
    MaxExpr,
    MdExtractHeaderChunks,
    MdGenerateTocExpr,
    MdGetCodeBlocksExpr,
    MdToJsonExpr,
    MilliSecondExpr,
    MinExpr,
    MinuteExpr,
    MonthExpr,
    NotExpr,
    NowExpr,
    NumericComparisonExpr,
    Operator,
    OtherwiseExpr,
    RecursiveTextChunkExpr,
    RegexpCountExpr,
    RegexpExtractAllExpr,
    RegexpExtractExpr,
    RegexpInstrExpr,
    RegexpSplitExpr,
    RegexpSubstrExpr,
    ReplaceExpr,
    RLikeExpr,
    SecondExpr,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticParsePDFExpr,
    SemanticPredExpr,
    SemanticReduceExpr,
    SemanticSummarizeExpr,
    SeriesLiteralExpr,
    SortExpr,
    SplitPartExpr,
    StartsWithExpr,
    StdDevExpr,
    StringCasingExpr,
    StripCharsExpr,
    StrLengthExpr,
    StructExpr,
    SumDistinctExpr,
    SumExpr,
    TextChunkExpr,
    TextractExpr,
    TimestampAddExpr,
    TimestampDiffExpr,
    ToDateExpr,
    ToTimestampExpr,
    ToUTCTimestampExpr,
    TsParseExpr,
    UDFExpr,
    WhenExpr,
    YearExpr,
)
from fenic.core._logical_plan.expressions.base import AggregateExpr
from fenic.core._utils.schema import (
    convert_custom_dtype_to_polars,
)
from fenic.core._utils.type_inference import infer_dtype_from_pyobj
from fenic.core.error import InternalError
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DataType,
    DateType,
    EmbeddingType,
    IntegerType,
    JsonType,
    StringType,
    StructField,
    StructType,
    _PrimitiveType,
    _TimestampType,
)
from fenic.core.types.enums import FuzzySimilarityMethod

logger = logging.getLogger(__name__)

class ExprConverter:
    def __init__(self, session_state: LocalSessionState):
        self.session_state = session_state

    def convert(
        self, logical: LogicalExpr,
        with_alias: bool = True,
    ) -> pl.Expr:
        """Convert a logical expression to a Polars physical expression with aliasing."""
        result = self._convert_expr(logical)
        if isinstance(logical, AliasExpr) or isinstance(logical, ColumnExpr) or not with_alias:
            return result
        return self._with_alias(result, logical)


    @singledispatchmethod
    def _convert_expr(self, logical: LogicalExpr) -> pl.Expr:
        """Convert a logical expression to a Polars physical expression without aliasing."""
        raise NotImplementedError(f"Conversion not implemented for {type(logical)}")


    def _with_alias(self, expr: pl.Expr, logical: Any) -> pl.Expr:
        """Add an alias to a Polars expression based on the string representation of the logical expression."""
        return expr.alias(str(logical))

    @_convert_expr.register
    def _convert_column_expr(self, logical: ColumnExpr) -> pl.Expr:
        return pl.col(logical.name)


    @_convert_expr.register
    def _convert_literal_expr(self, logical: LiteralExpr) -> pl.Expr:
        def _literal_to_polars_expr(value: Any, data_type: DataType) -> pl.Expr:
            if value is None:
                return pl.lit(None, dtype=convert_custom_dtype_to_polars(data_type))

            if isinstance(data_type, _TimestampType):
                return pl.lit(_pytype_datetime_to_utc(value), dtype=pl.Datetime(time_zone="UTC"))

            if isinstance(data_type, _PrimitiveType):
                return pl.lit(value, dtype=convert_custom_dtype_to_polars(data_type))

            if isinstance(data_type, ArrayType):
                elems = [_literal_to_polars_expr(v, data_type.element_type) for v in value]
                if not elems:
                    return pl.lit([], dtype=convert_custom_dtype_to_polars(data_type))
                return pl.concat_list(elems)

            if isinstance(data_type, StructType):
                fields = [
                    _literal_to_polars_expr(value.get(field.name), field.data_type).alias(
                        field.name
                    )
                    for field in data_type.struct_fields
                ]
                return pl.struct(fields)

            raise ValueError(f"Unsupported data type {data_type} for literal conversion")

        return _literal_to_polars_expr(logical.literal, logical.data_type)

    @_convert_expr.register
    def _convert_series_literal_expr(self, logical: SeriesLiteralExpr) -> pl.Expr:
        """Convert a SeriesLiteralExpr to a Polars expression.

        This wraps the Series in pl.lit(). If the series is a single value, we return a literal of the single value so
        that Polars can handle broadcasting. We've already checked upstream that the series is not empty.
        We check at execution time that the series length matches the height of the DataFrame.
        """
        if len(logical.series) == 1:
            # If the Series has length 1, just return a literal of the single value.
            return pl.lit(logical.series[0])
        return pl.lit(logical.series)

    @_convert_expr.register
    def _convert_unresolved_literal_expr(self, logical: UnresolvedLiteralExpr) -> pl.Expr:
        raise InternalError(
            f"An unresolved literal expression was found in the plan: {logical}, which was not expected. "
            "All unresolved literal expressions should have been resolved by the binder."
        )


    @_convert_expr.register
    def _convert_alias_expr(self, logical: AliasExpr) -> pl.Expr:
        base_expr = self._convert_expr(logical.expr)
        return base_expr.alias(logical.name)


    @_convert_expr.register
    def _convert_arithmetic_expr(self, logical: ArithmeticExpr) -> pl.Expr:
        left = self._convert_expr(logical.left)
        right = self._convert_expr(logical.right)

        op_handlers = {
            Operator.PLUS: lambda left, right: left + right,
            Operator.MINUS: lambda left, right: left - right,
            Operator.MULTIPLY: lambda left, right: left * right,
            Operator.DIVIDE: lambda left, right: left / right,
        }

        if logical.op in op_handlers:
            return op_handlers[logical.op](left, right)
        else:
            raise NotImplementedError(f"Unsupported arithmetic operator: {logical.op}")


    @_convert_expr.register
    def _convert_sort_expr(self, logical: SortExpr) -> pl.Expr:
        raise ValueError(
            "asc and desc() expressions can only be used in sort or order_by operations."
        )


    def _handle_comparison_expr(self, logical) -> pl.Expr:
        left = self._convert_expr(logical.left)
        right = self._convert_expr(logical.right)

        op_handlers = {
            Operator.EQ: lambda left, right: left == right,
            Operator.NOT_EQ: lambda left, right: left != right,
            Operator.GT: lambda left, right: left > right,
            Operator.GTEQ: lambda left, right: left >= right,
            Operator.LT: lambda left, right: left < right,
            Operator.LTEQ: lambda left, right: left <= right,
            Operator.AND: lambda left, right: left & right,
            Operator.OR: lambda left, right: left | right,
        }

        if logical.op in op_handlers:
            return op_handlers[logical.op](left, right)
        else:
            raise NotImplementedError(f"Unsupported comparison operator: {logical.op}")


    @_convert_expr.register(BooleanExpr)
    @_convert_expr.register(EqualityComparisonExpr)
    @_convert_expr.register(NumericComparisonExpr)
    def _convert_comparison_expr(self, logical) -> pl.Expr:
        result = self._handle_comparison_expr(logical)
        return result


    def _convert_avg_expr(self, logical: AvgExpr) -> pl.Expr:
        """Convert AvgExpr, handling embeddings specially."""
        converted_expr = self._convert_expr(logical.expr)

        # Check if we're averaging embeddings
        if isinstance(logical.input_type, EmbeddingType):
            def embedding_avg(series: pl.Series, embedding_dim: int) -> pl.Series:
                # TODO(rohitrastogi): Benchmark processing each group concurrently using a threadpool.
                # NumPy's mean() is already multi-threaded via C bindings, so additional threading may
                # not be faster. Test with realistic embedding sizes and group counts.
                result = []
                for emb_list in series.to_list():
                    if not emb_list:
                        result.append(None)
                        continue

                    filtered = [emb for emb in emb_list if emb is not None]
                    if not filtered:
                        result.append(None)
                    else:
                        mean_emb = np.mean(filtered, axis=0).astype(np.float32)
                        result.append(mean_emb)

                arrow_type = pa.list_(pa.float32(), embedding_dim)
                return pl.from_arrow(pa.array(result, type=arrow_type))

            return converted_expr.map_batches(
                lambda batch: embedding_avg(batch, logical.input_type.dimensions),
                return_dtype=pl.Array(pl.Float32, logical.input_type.dimensions),
                agg_list=True,
                returns_scalar=True
            )
        else:
            return converted_expr.mean()

    @_convert_expr.register(AggregateExpr)
    def _convert_aggregate_expr(self, logical: AggregateExpr) -> pl.Expr:
        # Special handling for AvgExpr
        if isinstance(logical, AvgExpr):
            return self._convert_avg_expr(logical)

        agg_handlers = {
            SumExpr: lambda expr: self._convert_expr(
                expr.expr
            ).sum(),
            SumDistinctExpr: lambda expr: (
                lambda base: base.unique().sum()
            )(self._convert_expr(expr.expr)),
            MinExpr: lambda expr: self._convert_expr(
                expr.expr,
            ).min(),
            MaxExpr: lambda expr: self._convert_expr(
                expr.expr,
            ).max(),
            CountExpr: lambda expr: (
                pl.len()
                if isinstance(expr.expr, LiteralExpr)
                else self._convert_expr(expr.expr).count()
            ),
            CountDistinctExpr: self._convert_count_distinct_expr,
            ApproxCountDistinctExpr: lambda expr: (
                # Match PySpark semantics: ignore nulls
                self._convert_expr(expr.expr).drop_nulls().approx_n_unique()
            ),
            ListExpr: lambda expr: self._convert_expr(
                expr.expr
            ),
            FirstExpr: lambda expr: self._convert_expr(
                expr.expr
            ).first(),
            StdDevExpr: lambda expr: self._convert_expr(
                expr.expr
            ).std(),
        }

        for expr_type, handler in agg_handlers.items():
            if isinstance(logical, expr_type):
                return handler(logical)

        if isinstance(logical, SemanticReduceExpr):
            group_context_names = list(logical.group_context_exprs.keys()) if logical.group_context_exprs else None
            polars_exprs = [self._convert_expr(logical.input_expr).alias(DATA_COLUMN_NAME)]
            for name, expr in logical.group_context_exprs.items():
                polars_exprs.append(self._convert_expr(expr).alias(name))
            descending: List[bool] = []
            nulls_last: List[bool] = []
            for i, order_by_expr in enumerate(logical.order_by_exprs):
                polars_exprs.append(self._convert_expr(order_by_expr.expr).alias(SORT_KEY_COLUMN_NAME + f"_{i}"))
                descending.append(not order_by_expr.ascending)
                nulls_last.append(order_by_expr.nulls_last)
            struct = pl.struct(polars_exprs)

            # sem_reduce_fn takes a Series of Series (each inner Series is a group of values)
            # and returns a Series of strings, one per group.
            # The Series of Series is created by the map_batches call below.
            def sem_reduce_fn(batch: pl.Series) -> pl.Series:
                return SemanticReduce(
                    input=batch,
                    user_instruction=logical.instruction,
                    model=self.session_state.get_language_model(logical.model_alias),
                    max_tokens=logical.max_tokens,
                    temperature=logical.temperature,
                    model_alias=logical.model_alias,
                    group_context_names=group_context_names,
                    descending=descending,
                    nulls_last=nulls_last,
                ).execute()
            return struct.map_batches(
                sem_reduce_fn, return_dtype=pl.Utf8, agg_list=True, returns_scalar=True
            )

        raise NotImplementedError(f"Unsupported aggregate function: {type(logical)}")

    def _convert_count_distinct_expr(self, logical: CountDistinctExpr) -> pl.Expr:
        """Convert CountDistinctExpr to a Polars expression matching PySpark semantics.

        PySpark semantics: rows where any of the input columns is null are ignored
        prior to performing the distinct count across the row tuple.
        """
        converted_inputs = [self._convert_expr(e) for e in logical.exprs]
        struct_expr = pl.struct(converted_inputs)
        # Compute a null mask directly from input expressions (avoid struct field introspection)
        any_null = pl.any_horizontal([expr.is_null() for expr in converted_inputs])
        return struct_expr.filter(~any_null).n_unique()


    @_convert_expr.register(UDFExpr)
    def _convert_udf_expr(self, logical: UDFExpr) -> pl.Expr:
        struct = pl.struct(
            [self._convert_expr(arg) for arg in logical.args]
        )
        converted_udf = _convert_udf_to_map_elements(
            logical.func
        )
        return struct.map_elements(
            converted_udf,
            return_dtype=convert_custom_dtype_to_polars(logical.return_type),
        )

    @_convert_expr.register(AsyncUDFExpr)
    def _convert_async_udf_expr(self, logical: AsyncUDFExpr) -> pl.Expr:
        # Create struct from input columns
        input_struct = pl.struct([self._convert_expr(arg) for arg in logical.args])

        # Apply async function via map_batches
        def execute_async_udf(batch: pl.Series) -> pl.Series:
            # Extract struct as an iterable of dicts [{col1: val1, col2: val2}, ...]
            items = ([row[name] for name in batch.struct.fields] for row in batch)

            # Use context manager for automatic loop lifecycle management
            with EventLoopManager().loop_context() as loop:
                async_udf = AsyncUDFSyncStream(
                    lambda item: logical.func(*item),
                    loop=loop,
                    max_concurrency=logical.max_concurrency,
                    timeout=logical.timeout_seconds,
                    num_retries=logical.num_retries
                )

                def results_generator():
                    for result in async_udf.call(items):
                        if isinstance(result, Exception):
                            yield None
                        else:
                            # Runtime type checking using Fenic's existing type inference
                            if result:
                                inferred_type = infer_dtype_from_pyobj(result)
                                if inferred_type != logical.return_type:
                                    raise TypeError(f"Expected {logical.return_type}, got {inferred_type} in async UDF")
                            yield result

                return pl.Series(results_generator(), dtype=convert_custom_dtype_to_polars(logical.return_type))

        return input_struct.map_batches(execute_async_udf)

    @_convert_expr.register(StructExpr)
    def _convert_struct_expr(self, logical: StructExpr) -> pl.Expr:
        return pl.struct(
            [self._convert_expr(child) for child in logical.children()]
        )


    @_convert_expr.register(ArrayExpr)
    def _convert_array_expr(self, logical: ArrayExpr) -> pl.Expr:
        return pl.concat_list(
            [self._convert_expr(child) for child in logical.children()]
        )


    @_convert_expr.register(IndexExpr)
    def _convert_index_expr(self, logical: IndexExpr) -> pl.Expr:
        base_expr = self._convert_expr(logical.expr)
        index_expr = self._convert_expr(logical.index)
        if logical.input_type == "array":
            return base_expr.list.get(index_expr, null_on_oob=True)
        elif logical.input_type == "struct":
            return base_expr.struct.field(logical.index.literal)
        else:
            raise InternalError(f"Unsupported index key type: {logical.input_type}")


    @_convert_expr.register(TsParseExpr)
    def _convert_ts_parse_expr(self, logical: TsParseExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        return physical_expr.transcript.parse(logical.format)


    @_convert_expr.register(TextractExpr)
    def _convert_textract_expr(self, logical: TextractExpr) -> pl.Expr:
        col_expr = self._convert_expr(logical.input_expr)
        struct_expr = pl.struct([col_expr])

        def extract_fields(row: Any) -> Dict[str, any]:
            text = str(row[str(logical.input_expr)])
            if not text:
                return {col: None for col in logical.parsed_template.columns}
            reader = TemplateFormatReader(logical.parsed_template, text)
            result_dict = reader.parse() or {
                col: None for col in logical.parsed_template.columns
            }
            return {
                col: result_dict.get(col, None) for col in logical.parsed_template.columns
            }

        return_struct_type = logical.parsed_template.to_struct_schema()

        return struct_expr.map_elements(
            extract_fields,
            return_dtype=convert_custom_dtype_to_polars(return_struct_type),
        )

    @_convert_expr.register(JinjaExpr)
    def _convert_jinja_expr(self, logical: JinjaExpr) -> pl.Expr:
        # Convert all input expressions
        column_exprs = [self._convert_expr(expr) for expr in logical.exprs]

        # Create struct of all inputs
        struct_expr = pl.struct(column_exprs)

        # Call the Jinja plugin
        return struct_expr.jinja.render(
            template=logical.template,
            strict=logical.strict,
        )

    @_convert_expr.register(SemanticMapExpr)
    def _convert_semantic_map_expr(self, logical: SemanticMapExpr) -> pl.Expr:
        def sem_map_fn(batch: pl.Series) -> pl.Series:
            return SemanticMap(
                input=batch,
                jinja_template=logical.template,
                model=self.session_state.get_language_model(logical.model_alias),
                examples=logical.examples,
                max_tokens=logical.max_tokens,
                temperature=logical.temperature,
                response_format=logical.response_format,
                model_alias=logical.model_alias,
                request_timeout=logical.request_timeout,
            ).execute()

        column_exprs = [self._convert_expr(expr) for expr in logical.exprs]
        struct_expr = pl.struct(column_exprs)
        jinja_expr = struct_expr.jinja.render(
            template=logical.template,
            strict=logical.strict,
        )

        if logical.response_format:
            return jinja_expr.map_batches(
                sem_map_fn,
                return_dtype=convert_custom_dtype_to_polars(logical.response_format.struct_type),
            )
        return jinja_expr.map_batches(
            sem_map_fn,
            return_dtype=pl.String
        )

    @_convert_expr.register(RecursiveTextChunkExpr)
    def _convert_text_chunk_expr(self, logical: RecursiveTextChunkExpr) -> pl.Expr:
        text_col = self._convert_expr(logical.input_expr)
        kwargs = dict(logical.chunking_configuration)
        return text_col.chunking.recursive(**kwargs)


    @_convert_expr.register(CountTokensExpr)
    def _count_tokens(self, logical: CountTokensExpr) -> pl.Expr:
        text_col = self._convert_expr(logical.input_expr)
        return text_col.tokenization.count_tokens()


    @_convert_expr.register(TextChunkExpr)
    def _convert_token_chunk_expr(self, logical: TextChunkExpr) -> pl.Expr:
        import tiktoken

        model = "cl100k_base"
        encoding = tiktoken.get_encoding(model)
        config = logical.chunking_configuration
        chunk_overlap = round(
            config.chunk_overlap_percentage * (config.desired_chunk_size / 100.0)
        )
        window_size = config.desired_chunk_size - chunk_overlap

        def token_chunk_udf(val: Any) -> List[str]:
            if val is None:
                return []
            tokens = encoding.encode(val)
            chunks = [
                tokens[i : i + config.desired_chunk_size]
                for i in range(0, len(tokens), window_size)
            ]
            return [encoding.decode(chunk) for chunk in chunks]

        def word_chunk_udf(val: Any) -> List[str]:
            if val is None:
                return []
            words = val.split()
            step = config.desired_chunk_size - chunk_overlap
            chunks = [
                words[i : i + config.desired_chunk_size] for i in range(0, len(words), step)
            ]
            return [" ".join(chunk) for chunk in chunks]

        def character_chunk_udf(val: Any) -> List[str]:
            if val is None:
                return []
            characters = list(val)
            step = config.desired_chunk_size - chunk_overlap
            chunks = [
                characters[i : i + config.desired_chunk_size]
                for i in range(0, len(characters), step)
            ]
            return ["".join(chunk) for chunk in chunks]

        chunk_udfs = {
            ChunkLengthFunction.TOKEN: token_chunk_udf,
            ChunkLengthFunction.WORD: word_chunk_udf,
            ChunkLengthFunction.CHARACTER: character_chunk_udf,
        }

        output_dtype = convert_custom_dtype_to_polars(ArrayType(element_type=StringType))
        return self._convert_expr(
            logical.input_expr
        ).map_elements(
            chunk_udfs[config.chunk_length_function_name], return_dtype=output_dtype
        )


    @_convert_expr.register(SemanticExtractExpr)
    def _convert_semantic_extract_expr(self, logical: SemanticExtractExpr) -> pl.Expr:
        def sem_ext_fn(batch: pl.Series) -> pl.Series:
            return SemanticExtract(
                input=batch,
                response_format=logical.response_format,
                model=self.session_state.get_language_model(logical.model_alias),
                max_output_tokens=logical.max_tokens,
                temperature=logical.temperature,
                model_alias=logical.model_alias,
                request_timeout=logical.request_timeout,
            ).execute()

        return self._convert_expr(logical.expr).map_batches(
            sem_ext_fn,
            return_dtype=convert_custom_dtype_to_polars(
                logical.response_format.struct_type
            ),
        )


    @_convert_expr.register(SemanticPredExpr)
    def _convert_semantic_pred_expr(self, logical: SemanticPredExpr) -> pl.Expr:
        def sem_pred_fn(batch: pl.Series) -> pl.Series:
            return SemanticPredicate(
                input=batch,
                jinja_template=logical.template,
                model=self.session_state.get_language_model(logical.model_alias),
                examples=logical.examples,
                temperature=logical.temperature,
                model_alias=logical.model_alias,
                request_timeout=logical.request_timeout,
            ).execute()

        column_exprs = [self._convert_expr(expr) for expr in logical.exprs]
        struct_expr = pl.struct(column_exprs)
        jinja_expr = struct_expr.jinja.render(
            template=logical.template,
            strict=logical.strict,
        )

        return jinja_expr.map_batches(sem_pred_fn, return_dtype=pl.Boolean)


    @_convert_expr.register(SemanticClassifyExpr)
    def _convert_semantic_classify_expr(self, logical: SemanticClassifyExpr) -> pl.Expr:
        def sem_classify_fn(batch: pl.Series) -> pl.Series:
            return SemanticClassify(
                input=batch,
                classes=logical.classes,
                model=self.session_state.get_language_model(logical.model_alias),
                temperature=logical.temperature,
                examples=logical.examples,
                model_alias=logical.model_alias,
                request_timeout=logical.request_timeout,
            ).execute()

        return self._convert_expr(logical.expr).map_batches(
            sem_classify_fn, return_dtype=pl.Utf8
        )


    @_convert_expr.register(AnalyzeSentimentExpr)
    def _convert_semantic_analyze_sentiment_expr(self, logical: AnalyzeSentimentExpr) -> pl.Expr:
        def sem_sentiment_fn(batch: pl.Series) -> pl.Series:
            return AnalyzeSentiment(
                input=batch,
                model=self.session_state.get_language_model(logical.model_alias),
                temperature=logical.temperature,
                model_alias=logical.model_alias,
                request_timeout=logical.request_timeout,
            ).execute()

        return self._convert_expr(logical.expr).map_batches(
            sem_sentiment_fn, return_dtype=pl.Utf8
        )

    @_convert_expr.register(SemanticSummarizeExpr)
    def _convert_semantic_summarize_expr(self,
        logical: SemanticSummarizeExpr
    ) -> pl.Expr:
        def sem_summarize_fn(batch: pl.Series) -> pl.Series:
            return SemanticSummarize(
                input=batch,
                format=logical.format,
                temperature=logical.temperature,
                model=self.session_state.get_language_model(logical.model_alias),
                request_timeout=logical.request_timeout,
            ).execute()

        return self._convert_expr(logical.expr).map_batches(
            sem_summarize_fn, return_dtype=pl.Utf8
        )

    @_convert_expr.register(SemanticParsePDFExpr)
    def _convert_parse_pdf_expr(self, logical: SemanticParsePDFExpr) -> pl.Expr:
        def parse_pdf_fn(batch: pl.Series) -> pl.Series:
            return SemanticParsePDF(
                input=batch,
                model=self.session_state.get_language_model(logical.model_alias),
                page_separator=logical.page_separator,
                describe_images=logical.describe_images,
                model_alias=logical.model_alias,
                max_output_tokens=logical.max_output_tokens,
                request_timeout=logical.request_timeout,
            ).execute()

        return self._convert_expr(logical.expr).map_batches(
            parse_pdf_fn, return_dtype=pl.Utf8
        )

    @_convert_expr.register(ArrayJoinExpr)
    def _convert_array_join_expr(self, logical: ArrayJoinExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.join(
            logical.delimiter
        )


    @_convert_expr.register(ContainsExpr)
    def _convert_contains_expr(self, logical: ContainsExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        substr_expr = self._convert_expr(logical.substr)
        return physical_expr.str.contains(pattern=substr_expr, literal=True)


    @_convert_expr.register(ContainsAnyExpr)
    def _convert_contains_any_expr(self, logical: ContainsAnyExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        return physical_expr.str.contains_any(
            patterns=logical.substrs, ascii_case_insensitive=logical.case_insensitive
        )


    # Group like/regex expressions together
    def _handle_regex_like_expr(self, logical) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        pattern = self._convert_expr(logical.pattern)
        case_insensitive = True if isinstance(logical, ILikeExpr) else False

        strict = False
        if isinstance(logical.pattern, LiteralExpr) and logical.pattern.data_type == StringType:
            # If the pattern is a "string" literal, then make sure that strict is True.
            strict = True

        if isinstance(logical, ILikeExpr) or isinstance(logical, LikeExpr):
            # Convert the LIKE pattern to a regex pattern
            pattern = _like_to_regex(pattern, case_insensitive)

        return physical_expr.str.contains(pattern=pattern, literal=False, strict=strict)

    @_convert_expr.register(RLikeExpr)
    @_convert_expr.register(LikeExpr)
    @_convert_expr.register(ILikeExpr)
    def _convert_like_expr(self, logical) -> pl.Expr:
        return self._handle_regex_like_expr(logical)


    @_convert_expr.register(StartsWithExpr)
    def _convert_starts_with_expr(self, logical: StartsWithExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        substr_expr = self._convert_expr(logical.substr)
        return physical_expr.str.starts_with(prefix=substr_expr)


    @_convert_expr.register(EndsWithExpr)
    def _convert_ends_with_expr(self, logical: EndsWithExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        substr_expr = self._convert_expr(logical.substr)
        return physical_expr.str.ends_with(suffix=substr_expr)


    @_convert_expr.register(EmbeddingsExpr)
    def _convert_embeddings_expr(self, logical: EmbeddingsExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        if logical.dimensions is None:
            raise InternalError("Embedding dimensions not set for embeddings expression")

        embedding_model = self.session_state.get_embedding_model(logical.model_alias)
        def embeddings_fn(batch: pl.Series) -> pl.Series:
            return pl.from_arrow(embedding_model.get_embeddings(batch, logical.model_alias))

        return physical_expr.map_batches(embeddings_fn, return_dtype=pl.Array(pl.Float32, logical.dimensions))


    @_convert_expr.register(SplitPartExpr)
    def _convert_split_part_expr(self, logical: SplitPartExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        part_number_expr = self._convert_expr(logical.part_number)
        delimiter_expr = self._convert_expr(logical.delimiter)

        split_expr = physical_expr.str.split(delimiter_expr)

        # Convert from 1-based to 0-based indexing for positive numbers
        part_expr = (
            pl.when(part_number_expr > 0)
            .then(part_number_expr - 1)
            .otherwise(part_number_expr)
        )

        # Get the part and handle out of range with empty string
        return (
            split_expr.list.get(part_expr, null_on_oob=True)
            .fill_null("")
        )


    @_convert_expr.register(ArrayContainsExpr)
    def _convert_array_contains_expr(self, logical: ArrayContainsExpr) -> pl.Expr:
        array_expr = self._convert_expr(logical.expr)
        element_expr = self._convert_expr(logical.other)

        return array_expr.list.contains(element_expr)


    @_convert_expr.register(RegexpSplitExpr)
    def _convert_regexp_split_expr(self, logical: RegexpSplitExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        delimiter = "|||SPLIT|||"
        if logical.limit > 0:
            split_ready = physical_expr.str.replace(
                pattern=logical.pattern,
                value=delimiter,
                literal=False,
                n=logical.limit - 1,
            )
            return split_ready.str.split(by=delimiter)
        else:
            split_ready = physical_expr.str.replace_all(
                pattern=logical.pattern, value=delimiter, literal=False
            )
            return split_ready.str.split(by=delimiter)


    @_convert_expr.register(RegexpCountExpr)
    def _convert_regexp_count_expr(self, logical: RegexpCountExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        pattern_expr = self._convert_expr(logical.pattern)

        # str.count_matches accepts expressions directly
        return physical_expr.str.count_matches(pattern_expr, literal=False)


    @_convert_expr.register(RegexpExtractExpr)
    def _convert_regexp_extract_expr(self, logical: RegexpExtractExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        pattern_expr = self._convert_expr(logical.pattern)

        # str.extract accepts pattern as expr, but group_index must be int
        return physical_expr.str.extract(pattern_expr, logical.idx).fill_null("")


    @_convert_expr.register(RegexpExtractAllExpr)
    def _convert_regexp_extract_all_expr(self, logical: RegexpExtractAllExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        pattern_expr = self._convert_expr(logical.pattern)

        # For PySpark compatibility: extract all matches of a specific group
        if isinstance(logical.idx, LiteralExpr) and isinstance(logical.pattern, LiteralExpr):
            group_idx = int(logical.idx.literal)
            pattern = logical.pattern.literal

            if group_idx == 0:
                # Group 0: extract entire match using extract_all (fast path)
                return physical_expr.str.extract_all(pattern)

        # For column expressions, use the Rust plugin for PySpark-compatible behavior
        idx_expr = self._convert_expr(logical.idx)
        return physical_expr.regexp.extract_all(pattern_expr, idx_expr)


    @_convert_expr.register(RegexpInstrExpr)
    def _convert_regexp_instr_expr(self, logical: RegexpInstrExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        pattern_expr = self._convert_expr(logical.pattern)

        # If idx is a literal, use native Polars operations (faster)
        if isinstance(logical.idx, LiteralExpr):
            # Extract the match at the specified group index
            match_expr = physical_expr.str.extract(pattern_expr, int(logical.idx.literal))
            # Find position: use str.find on the match within the original string
            # Add 1 to convert from 0-based to 1-based indexing (PySpark compatibility)
            # Return 0 if no match, null if input is null
            return pl.when(physical_expr.is_null()).then(None).otherwise(
                pl.when(match_expr.is_null()).then(pl.lit(0)).otherwise(
                    physical_expr.str.find(match_expr, literal=True) + 1
                )
            )

        idx_expr = self._convert_expr(logical.idx)
        return physical_expr.regexp.instr(pattern_expr, idx_expr)


    @_convert_expr.register(RegexpSubstrExpr)
    def _convert_regexp_substr_expr(self, logical: RegexpSubstrExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        pattern_expr = self._convert_expr(logical.pattern)

        # str.extract accepts pattern as expr, extract group 0 (entire match)
        return physical_expr.str.extract(pattern_expr, 0)


    @_convert_expr.register(StripCharsExpr)
    def _convert_strip_chars_expr(self, logical: StripCharsExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        chars_expr = self._convert_expr(logical.chars) if logical.chars else None

        strip_methods = {
            "both": physical_expr.str.strip_chars,
            "left": physical_expr.str.strip_chars_start,
            "right": physical_expr.str.strip_chars_end,
        }

        if logical.side in strip_methods:
            return strip_methods[logical.side](characters=chars_expr)
        else:
            raise NotImplementedError(f"Unsupported side: {logical.side}")


    @_convert_expr.register(StringCasingExpr)
    def _convert_string_casing_expr(self, logical: StringCasingExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)

        case_methods = {
            "upper": physical_expr.str.to_uppercase,
            "lower": physical_expr.str.to_lowercase,
            "title": physical_expr.str.to_titlecase,
        }

        if logical.case in case_methods:
            return case_methods[logical.case]()
        else:
            raise NotImplementedError(f"Unsupported case: {logical.case}")


    @_convert_expr.register(ReplaceExpr)
    def _convert_replace_expr(self, logical: ReplaceExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        is_search_column = isinstance(logical.search, ColumnExpr)
        physical_search_expr = self._convert_expr(logical.search)
        physical_replacement_expr = self._convert_expr(logical.replacement)

        if not is_search_column:
            return physical_expr.str.replace_all(
                pattern=physical_search_expr,
                value=physical_replacement_expr,
                literal=logical.literal,
            )
        else:
            # https://github.com/pola-rs/polars/issues/14367
            # Polars doesn't currently support replace with a expression, so we need to use replace_all and over as a workaround
            return physical_expr.str.replace_all(
                pattern=physical_search_expr.first(),
                value=physical_replacement_expr,
                literal=logical.literal,
            ).over(physical_search_expr)

    @_convert_expr.register(StrLengthExpr)
    def _convert_str_length_expr(self, logical: StrLengthExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        return physical_expr.str.len_chars()


    @_convert_expr.register(ByteLengthExpr)
    def _convert_byte_length_expr(self, logical: ByteLengthExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        return physical_expr.str.len_bytes()


    @_convert_expr.register(ConcatExpr)
    def _convert_concat_expr(self, logical: ConcatExpr) -> pl.Expr:
        return pl.concat_str(
            [
                self._convert_expr(expr)
                for expr in logical.exprs
            ],
            separator="",
        )


    @_convert_expr.register(CoalesceExpr)
    def _convert_coalesce_expr(self, logical: CoalesceExpr) -> pl.Expr:
        return pl.coalesce(
            [
                self._convert_expr(expr)
                for expr in logical.exprs
            ],
        )


    @_convert_expr.register(IsNullExpr)
    def _convert_is_null_expr(self, logical: IsNullExpr) -> pl.Expr:
        if logical.is_null:
            return self._convert_expr(logical.expr).is_null()
        else:
            return self._convert_expr(logical.expr).is_not_null()


    @_convert_expr.register(ArrayLengthExpr)
    def _convert_array_length_expr(self, logical: ArrayLengthExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.len()


    @_convert_expr.register(ArrayDistinctExpr)
    def _convert_array_distinct_expr(self, logical: ArrayDistinctExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.unique()

    @_convert_expr.register(ArrayMaxExpr)
    def _convert_array_max_expr(self, logical: ArrayMaxExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.max()

    @_convert_expr.register(ArrayMinExpr)
    def _convert_array_min_expr(self, logical: ArrayMinExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.min()

    @_convert_expr.register(ArraySortExpr)
    def _convert_array_sort_expr(self, logical: ArraySortExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.sort()

    @_convert_expr.register(ArrayReverseExpr)
    def _convert_array_reverse_expr(self, logical: ArrayReverseExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.reverse()

    @_convert_expr.register(ArrayRemoveExpr)
    def _convert_array_remove_expr(self, logical: ArrayRemoveExpr) -> pl.Expr:
        array_expr = self._convert_expr(logical.expr)
        element_expr = self._convert_expr(logical.element)
        # Filter out elements that match the target element
        # Need to handle the case where element_expr might be a column reference
        return array_expr.list.eval(pl.element().filter(pl.element() != element_expr.first()))

    @_convert_expr.register(ArrayUnionExpr)
    def _convert_array_union_expr(self, logical: ArrayUnionExpr) -> pl.Expr:
        left_expr = self._convert_expr(logical.left)
        right_expr = self._convert_expr(logical.right)
        # PySpark array_union returns distinct elements from both arrays
        return left_expr.list.set_union(right_expr)

    @_convert_expr.register(ArrayIntersectExpr)
    def _convert_array_intersect_expr(self, logical: ArrayIntersectExpr) -> pl.Expr:
        left_expr = self._convert_expr(logical.left)
        right_expr = self._convert_expr(logical.right)
        # PySpark array_intersect returns distinct elements in both arrays
        return left_expr.list.set_intersection(right_expr)

    @_convert_expr.register(ArrayExceptExpr)
    def _convert_array_except_expr(self, logical: ArrayExceptExpr) -> pl.Expr:
        left_expr = self._convert_expr(logical.left)
        right_expr = self._convert_expr(logical.right)
        # PySpark array_except returns distinct elements in left but not in right
        return left_expr.list.set_difference(right_expr)

    @_convert_expr.register(ArrayCompactExpr)
    def _convert_array_compact_expr(self, logical: ArrayCompactExpr) -> pl.Expr:
        # PySpark array_compact removes null values from array
        return self._convert_expr(logical.expr).list.drop_nulls()

    @_convert_expr.register(ArrayRepeatExpr)
    def _convert_array_repeat_expr(self, logical: ArrayRepeatExpr) -> pl.Expr:
        element_expr = self._convert_expr(logical.element)
        count_expr = self._convert_expr(logical.count)

        def _repeat_elements(s: pl.Series) -> pl.Series:
            """Repeat elements based on count values in struct series."""
            elem_series = s.struct.field("_elem")
            count_series = s.struct.field("_count")

            return pl.Series([
                [elem] * count if count is not None else None
                for elem, count in zip(elem_series, count_series, strict=True)
            ], dtype=pl.List(elem_series.dtype))

        return pl.struct([element_expr.alias("_elem"), count_expr.alias("_count")]).map_batches(
            _repeat_elements
        )

    @_convert_expr.register(FlattenExpr)
    def _convert_flatten_expr(self, logical: FlattenExpr) -> pl.Expr:
        # PySpark flatten flattens one level of nesting (array of arrays -> array)
        # Use list.eval to explode inner arrays and collect them
        return self._convert_expr(logical.expr).list.eval(pl.element().explode())

    @_convert_expr.register(ArraySliceExpr)
    def _convert_array_slice_expr(self, logical: ArraySliceExpr) -> pl.Expr:
        array_expr = self._convert_expr(logical.expr)
        start_expr = self._convert_expr(logical.start)
        length_expr = self._convert_expr(logical.length)
        # PySpark slice uses 1-based indexing for start position
        # Positive indices: convert 1-based to 0-based (subtract 1)
        # Negative indices: keep as-is (both PySpark and Polars use -1 for last element)
        adjusted_start = pl.when(start_expr > 0).then(start_expr - 1).otherwise(start_expr)
        return array_expr.list.slice(adjusted_start, length_expr)

    @_convert_expr.register(ElementAtExpr)
    def _convert_element_at_expr(self, logical: ElementAtExpr) -> pl.Expr:
        array_expr = self._convert_expr(logical.expr)
        index_expr = self._convert_expr(logical.index)
        # PySpark element_at uses 1-based indexing:
        #   - Positive: 1 is first element (convert to 0-based by subtracting 1)
        #   - Negative: -1 is last element (same in Polars, no conversion needed)
        adjusted_index = pl.when(index_expr > 0).then(index_expr - 1).otherwise(index_expr)
        return array_expr.list.get(adjusted_index)

    @_convert_expr.register(ArraysOverlapExpr)
    def _convert_arrays_overlap_expr(self, logical: ArraysOverlapExpr) -> pl.Expr:
        left_expr = self._convert_expr(logical.left)
        right_expr = self._convert_expr(logical.right)
        # PySpark arrays_overlap returns true if arrays have at least one common element
        # Check if intersection is non-empty
        intersection = left_expr.list.set_intersection(right_expr)
        return intersection.list.len() > 0


    @_convert_expr.register(CastExpr)
    def _convert_cast_expr(self, logical: CastExpr) -> pl.Expr:
        if not logical.source_type:
            raise InternalError("Source type not set for cast expression")
        source_dtype = json.dumps(serialize_data_type(logical.source_type))
        dest_dtype = json.dumps(serialize_data_type(logical.dest_type))
        return self._convert_expr(logical.expr).dtypes.cast(
            source_dtype, dest_dtype
        )

    @_convert_expr.register(NotExpr)
    def _convert_not_expr(self, logical: NotExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).not_()


    @_convert_expr.register(WhenExpr)
    def _convert_when_expr(self, logical: WhenExpr) -> pl.Expr:
        if isinstance(logical.expr, WhenExpr):
            # Evaluate the final when expression
            return (
                self._convert_when_expr(logical.expr)
                .when(self._convert_expr(logical.condition))
                .then(
                    self._convert_expr(logical.value).alias(
                        str(logical)
                    )
                )
            )
        else:
            # head of condition chain
            return pl.when(
                self._convert_expr(logical.condition)
            ).then(self._convert_expr(logical.value))


    @_convert_expr.register(OtherwiseExpr)
    def _convert_otherwise_expr(self, logical: OtherwiseExpr) -> pl.Expr:
        return self._convert_when_expr(logical.expr).otherwise(
            self._convert_expr(logical.value).alias(
                str(logical)
            )
        )


    @_convert_expr.register(InExpr)
    def _convert_in_expr(self, logical: InExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).is_in(
            self._convert_expr(logical.other)
        )

    @_convert_expr.register(JqExpr)
    def _convert_jq_expr(self, logical: JqExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).json.jq(logical.query)

    @_convert_expr.register(JsonTypeExpr)
    def _convert_json_type_expr(self, logical: JsonTypeExpr) -> pl.Expr:
        source_dtype = json.dumps(serialize_data_type(JsonType))
        dest_dtype = json.dumps(
            serialize_data_type(StructType([StructField("result", StringType)]))
        )

        return (
            self._convert_expr(logical.expr)
            .json.jq(logical.jq_query)
            .list.get(0)
            .dtypes.cast(source_dtype, dest_dtype)
            .struct.field("result")
        )


    @_convert_expr.register(JsonContainsExpr)
    def _convert_json_contains_expr(self, logical: JsonContainsExpr) -> pl.Expr:
        source_dtype = json.dumps(serialize_data_type(JsonType))
        dest_dtype = json.dumps(
            serialize_data_type(StructType([StructField("result", BooleanType)]))
        )

        return (
            self._convert_expr(logical.expr)
            .json.jq(logical.jq_query)
            .list.get(0)
            .dtypes.cast(source_dtype, dest_dtype)
            .struct.field("result")
        )

    @_convert_expr.register(MdToJsonExpr)
    def _convert_md_to_json_expr(self, logical: MdToJsonExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).markdown.to_json()

    @_convert_expr.register(MdGetCodeBlocksExpr)
    def _convert_md_get_code_blocks_expr(self, logical: MdGetCodeBlocksExpr) -> pl.Expr:
        source_dtype = json.dumps(serialize_data_type(JsonType))
        dest_dtype = json.dumps(
            serialize_data_type(
                ArrayType(element_type=
                    StructType([
                        StructField("language", StringType),
                        StructField("code", StringType),
                    ])
                )
            )
        )

        return (
            self._convert_expr(logical.expr)
            .markdown.to_json()
            .json.jq(logical.jq_query)
            .list.get(0)
            .dtypes.cast(source_dtype, dest_dtype)
        )

    @_convert_expr.register(MdGenerateTocExpr)
    def _convert_md_generate_toc_expr(self, logical: MdGenerateTocExpr) -> pl.Expr:
        # Cast the JSON result to extract the toc field
        source_dtype = json.dumps(serialize_data_type(JsonType))
        dest_dtype = json.dumps(
            serialize_data_type(StructType([StructField("toc", StringType)]))
        )

        return (
            self._convert_expr(logical.expr)
            .markdown.to_json()
            .json.jq(logical.jq_query)
            .list.get(0)
            .dtypes.cast(source_dtype, dest_dtype)
            .struct.field("toc")
        )


    @_convert_expr.register(MdExtractHeaderChunks)
    def _convert_md_chunk_by_headings_expr(self, logical: MdExtractHeaderChunks) -> pl.Expr:
        source_dtype = json.dumps(serialize_data_type(JsonType))
        dest_dtype = json.dumps(
            serialize_data_type(
                ArrayType(element_type=
                    StructType([
                        StructField("heading", StringType),
                        StructField("level", IntegerType),
                        StructField("content", StringType),
                        StructField("parent_heading", StringType),
                        StructField("full_path", StringType),
                    ])
                )
            )
        )

        return (
            self._convert_expr(logical.expr)
            .markdown.to_json()
            .json.jq(logical.jq_query)
            .list.get(0)
            .dtypes.cast(source_dtype, dest_dtype)
        )


    @_convert_expr.register(EmbeddingNormalizeExpr)
    def _convert_embedding_normalize_expr(self, logical: EmbeddingNormalizeExpr) -> pl.Expr:
        if logical.dimensions is None:
            raise InternalError("EmbeddingNormalizeExpr dimensions not set")

        def normalize_fn(batch: pl.Series) -> pl.Series:
            mask = ~batch.is_null().to_numpy()
            if not np.any(mask):
                return pl.Series([None] * len(batch))

            embeddings = batch.to_numpy()  # shape (N, D)

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = np.divide(embeddings, norms, where=norms != 0)
            # Zero vectors become NaN after normalization - expand condition to match dimensions
            zero_mask = (norms == 0)  # shape (N, 1)
            normalized = np.where(zero_mask, np.nan, normalized)

            # Fast path: no nulls in input
            if np.all(mask):
                return pl.Series(normalized)

            # Need to use pyarrow to create a nullable fixed size list array, otherwise polars will not be able to handle nulls
            # for its array type.
            nested_values = []
            for i in range(len(batch)):
                if mask[i]:
                    nested_values.append(normalized[i])
                else:
                    nested_values.append(None)

            return pl.from_arrow(pa.array(nested_values, type=pa.list_(pa.float32(), logical.dimensions)))

        return self._convert_expr(logical.expr).map_batches(
            normalize_fn, return_dtype=pl.Array(pl.Float32, logical.dimensions)
        )


    @_convert_expr.register(EmbeddingSimilarityExpr)
    def _convert_embedding_similarity_expr(self, logical: EmbeddingSimilarityExpr) -> pl.Expr:
        if isinstance(logical.other, LogicalExpr):
            # Case 1: Column vs Column similarity
            def pairwise_similarity_fn(batch: pl.Series) -> pl.Series:
                fields = batch.struct.fields
                embeddings1 = batch.struct.field(fields[0])
                embeddings2 = batch.struct.field(fields[1])

                mask1 = ~embeddings1.is_null().to_numpy()
                mask2 = ~embeddings2.is_null().to_numpy()
                combined_mask = mask1 & mask2
                combined_mask = mask1 & mask2

                if not np.any(combined_mask):
                    return pl.Series([None] * len(batch))

                similarities = _calculate_similarity_numpy(
                    embeddings1.to_numpy(), embeddings2.to_numpy(), logical.metric
                )

                # Fast path: no nulls in input. embeddings1 and embeddings2 ~should be zero copy numpy arrays.
                if np.all(combined_mask):
                    return pl.Series(similarities)

                result = []
                for i in range(len(similarities)):
                    if combined_mask[i]:
                        result.append(similarities[i])
                    else:
                        result.append(None)
                return pl.Series(result)

            expr1 = self._convert_expr(logical.expr)
            expr2 = self._convert_expr(logical.other)
            return pl.struct([expr1, expr2]).map_batches(
                pairwise_similarity_fn, return_dtype=pl.Float32
            )
        else:
            # Case 2: Column vs single query vector similarity
            query_vector = logical.other # This is already a numpy array

            def similarity_fn(batch: pl.Series) -> pl.Series:
                mask = ~batch.is_null().to_numpy()

                if not np.any(mask):
                    return pl.Series([None] * len(batch))

                similarities = _calculate_similarity_numpy(
                    batch.to_numpy(), query_vector, logical.metric
                )

                if np.all(mask):
                    return pl.Series(similarities)

                result = []
                for i in range(len(similarities)):
                    if mask[i]:
                        result.append(similarities[i])
                    else:
                        result.append(None)
                return pl.Series(result)

            return self._convert_expr(logical.expr).map_batches(
                similarity_fn, return_dtype=pl.Float32
            )

    @_convert_expr.register(FuzzyRatioExpr)
    def _convert_fuzzy_similarity_expr(self, logical: FuzzyRatioExpr) -> pl.Expr:
        left_expr = self._convert_expr(logical.expr)
        right_expr = self._convert_expr(logical.other)

        return _convert_fuzzy_similarity_method_to_expr(left_expr, right_expr, logical.method)

    @_convert_expr.register(FuzzyTokenSortRatioExpr)
    def _convert_fuzzy_token_sort_ratio_expr(self, logical: FuzzyTokenSortRatioExpr) -> pl.Expr:
        left_tokens = _tokenize_for_fuzzy_similarity(self._convert_expr(logical.expr))
        right_tokens = _tokenize_for_fuzzy_similarity(self._convert_expr(logical.other))

        left_expr = left_tokens.list.sort().list.join(" ")
        right_expr = right_tokens.list.sort().list.join(" ")

        return _convert_fuzzy_similarity_method_to_expr(left_expr, right_expr, logical.method)

    @_convert_expr.register(FuzzyTokenSetRatioExpr)
    def _convert_fuzzy_token_set_ratio_expr(self, logical: FuzzyTokenSetRatioExpr) -> pl.Expr:
        # Tokenize and normalize
        left_tokens = _tokenize_for_fuzzy_similarity(self._convert_expr(logical.expr))
        right_tokens = _tokenize_for_fuzzy_similarity(self._convert_expr(logical.other))

        # Get unique tokens and sort
        left_set = left_tokens.list.unique().list.sort()
        right_set = right_tokens.list.unique().list.sort()

        # Get intersection and differences
        intersection = left_set.list.set_intersection(right_set)
        diff_left = left_set.list.set_difference(right_set)
        diff_right = right_set.list.set_difference(left_set)

        # Create strings for comparison
        intersection_str = intersection.list.sort().list.join(" ")
        diff_left_str = diff_left.list.sort().list.join(" ")
        diff_right_str = diff_right.list.sort().list.join(" ")
        left_set_str = left_set.list.join(" ")  # Already sorted
        right_set_str = right_set.list.join(" ")  # Already sorted

        # Three comparisons matching canonical implementation:
        # 1. diff_left vs diff_right
        # 2. intersection vs left_set (intersection + diff_left)
        # 3. intersection vs right_set (intersection + diff_right)
        ratio1 = _convert_fuzzy_similarity_method_to_expr(diff_left_str, diff_right_str, logical.method)
        ratio2 = _convert_fuzzy_similarity_method_to_expr(intersection_str, left_set_str, logical.method)
        ratio3 = _convert_fuzzy_similarity_method_to_expr(intersection_str, right_set_str, logical.method)

        # Return the maximum
        return pl.max_horizontal([ratio1, ratio2, ratio3])

    @_convert_expr.register(GreatestExpr)
    def _convert_greatest_expr(self, logical: GreatestExpr) -> pl.Expr:
        return pl.max_horizontal([self._convert_expr(expr) for expr in logical.exprs])

    @_convert_expr.register(LeastExpr)
    def _convert_least_expr(self, logical: LeastExpr) -> pl.Expr:
        return pl.min_horizontal([self._convert_expr(expr) for expr in logical.exprs])

    @_convert_expr.register(YearExpr)
    def _convert_year_expr(self, logical: YearExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).dt.year()

    @_convert_expr.register(MonthExpr)
    def _convert_month_expr(self, logical: MonthExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).dt.month()

    @_convert_expr.register(DayExpr)
    def _convert_day_expr(self, logical: DayExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).dt.day()

    @_convert_expr.register(HourExpr)
    def _convert_hour_expr(self, logical: HourExpr) -> pl.Expr:
        converted_expr = self._convert_expr(logical.expr)
        if logical.temporal_type == DateType:
            return _get_zero_ts_part_from_date(converted_expr)
        return converted_expr.dt.hour()

    @_convert_expr.register(MinuteExpr)
    def _convert_minute_expr(self, logical: MinuteExpr) -> pl.Expr:
        converted_expr = self._convert_expr(logical.expr)
        if logical.temporal_type == DateType:
            return _get_zero_ts_part_from_date(converted_expr)
        return converted_expr.dt.minute()

    @_convert_expr.register(SecondExpr)
    def _convert_second_expr(self, logical: SecondExpr) -> pl.Expr:
        converted_expr = self._convert_expr(logical.expr)
        if logical.temporal_type == DateType:
            return _get_zero_ts_part_from_date(converted_expr)
        return converted_expr.dt.second()

    @_convert_expr.register(MilliSecondExpr)
    def _convert_milli_second_expr(self, logical: MilliSecondExpr) -> pl.Expr:
        converted_expr = self._convert_expr(logical.expr)
        if logical.temporal_type == DateType:
            return _get_zero_ts_part_from_date(converted_expr)
        return converted_expr.dt.millisecond()

    @_convert_expr.register(ToDateExpr)
    def _convert_to_date_expr(self, logical: ToDateExpr) -> pl.Expr:
        """Convert a string into a date."""
        # logical.format should be a chrono format.
        return self._convert_expr(logical.expr).str.strptime(dtype=pl.Date, format=logical.format)

    @_convert_expr.register(ToTimestampExpr)
    def _convert_to_timestamp_expr(self, logical: ToTimestampExpr) -> pl.Expr:
        """Convert a string into a timestamp."""
        # logical.format should be a chrono format.
        result = self._convert_expr(logical.expr).str.strptime(dtype=pl.Datetime, format=logical.format)
        if _format_has_tz_tokens(logical.format):
            return result.dt.convert_time_zone("UTC")
        else:
            return result.dt.replace_time_zone("UTC")

    @_convert_expr.register(NowExpr)
    def _convert_now_expr(self, logical: NowExpr) -> pl.Expr:
        """Convert a string into a timestamp."""
        return pl.lit(datetime.datetime.now(tz=datetime.timezone.utc), dtype=pl.Date if logical.as_date else pl.Datetime)

    @_convert_expr.register(DateTruncExpr)
    def _convert_date_trunc_expr(self, logical: DateTruncExpr) -> pl.Expr:
        # Note: truncating a date to 1h is a no-op in this case as the date is already truncated to the hour.
        return self._convert_expr(logical.expr).dt.truncate(logical.unit)

    @_convert_expr.register(DateAddExpr)
    def _convert_date_add_expr(self, logical: DateAddExpr) -> pl.Expr:
        if logical.sub:
            # Must substact the days from the date/timestamp column.
            return self._convert_expr(logical.expr) - pl.duration(days=self._convert_expr(logical.days))
        return self._convert_expr(logical.expr) + pl.duration(days=self._convert_expr(logical.days))

    @_convert_expr.register(TimestampAddExpr)
    def _convert_timestamp_add_expr(self, logical: TimestampAddExpr) -> pl.Expr:
        quantity_expr = self._convert_expr(logical.quantity)
        return self._convert_expr(logical.expr).dt.offset_by(pl.format("{}{}", quantity_expr, pl.lit(logical.unit)))

    @_convert_expr.register(DateFormatExpr)
    def _convert_date_format_expr(self, logical: DateFormatExpr) -> pl.Expr:
        # logical.format should be a chrono format.
        return self._convert_expr(logical.expr).dt.to_string(format=logical.format)

    @_convert_expr.register(DateDiffExpr)
    def _convert_date_diff_expr(self, logical: DateDiffExpr) -> pl.Expr:
        end_expr = self._convert_expr(logical.end).cast(pl.Datetime)
        start_expr = self._convert_expr(logical.start).cast(pl.Datetime)
        return (end_expr - start_expr).dt.total_days()

    @_convert_expr.register(TimestampDiffExpr)
    def _convert_timestamp_diff_expr(self, logical: TimestampDiffExpr) -> pl.Expr:
        """Convert a timestamp diff expression to a Polars expression.

        We convert the start and end expressions to datetime before performing the calculation.
        UTC before performing the calculation.
        """
        end_expr = self._convert_expr(logical.end).cast(pl.Datetime)
        start_expr = self._convert_expr(logical.start).cast(pl.Datetime)
        duration_expr = end_expr - start_expr
        if logical.unit == "year":
            return (duration_expr.dt.total_days() / 365).cast(pl.Int64)
        elif logical.unit == "month":
            return (duration_expr.dt.total_days() / 30).cast(pl.Int64)
        elif logical.unit == "day":
            return duration_expr.dt.total_days()
        elif logical.unit == "hour":
            return duration_expr.dt.total_hours()
        elif logical.unit == "minute":
            return duration_expr.dt.total_minutes()
        elif logical.unit == "second":
            return duration_expr.dt.total_seconds()
        elif logical.unit == "millisecond":
            return duration_expr.dt.total_milliseconds()
        else:
            raise InternalError(f"Unknown timestamp diff unit: {logical.unit}. Invalid state.")

    @_convert_expr.register(ToUTCTimestampExpr)
    def _convert_to_utc_timestamp_expr(self, logical: ToUTCTimestampExpr) -> pl.Expr:
        return (self._convert_expr(logical.expr)
            .dt.replace_time_zone(logical.timezone)
            .dt.convert_time_zone("UTC"))

    @_convert_expr.register(FromUTCTimestampExpr)
    def _convert_from_utc_timestamp_expr(self, logical: FromUTCTimestampExpr) -> pl.Expr:
        return (self._convert_expr(logical.expr)
            .dt.convert_time_zone(logical.timezone)
            .dt.replace_time_zone("UTC"))


def _convert_fuzzy_similarity_method_to_expr(expr: pl.Expr, other: pl.Expr, method: FuzzySimilarityMethod) -> pl.Expr:
    if method == "indel":
        return expr.fuzz.normalized_indel_similarity(other)
    if method == "levenshtein":
        return expr.fuzz.normalized_levenshtein_similarity(other)
    elif method == "damerau_levenshtein":
        return expr.fuzz.normalized_damerau_levenshtein_similarity(other)
    elif method == "jaro_winkler":
        return expr.fuzz.normalized_jarowinkler_similarity(other)
    elif method == "jaro":
        return expr.fuzz.normalized_jaro_similarity(other)
    elif method == "hamming":
        return expr.fuzz.normalized_hamming_similarity(other)
    else:
        raise InternalError(f"Unknown fuzzy similarity method: {method}. Invalid state.")

def _tokenize_for_fuzzy_similarity(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(r"\s+", " ").str.strip_chars().str.split(" ")

def _calculate_similarity_numpy(
    embeddings: np.ndarray, query: np.ndarray, metric: str
) -> np.ndarray:
    if metric == "dot":
        return np.sum(embeddings * query, axis=1)

    if metric == "cosine":
        dots = np.sum(embeddings * query, axis=1)
        norms_embeddings = np.linalg.norm(embeddings, axis=1)
        norms_query = np.linalg.norm(query, axis=-1)
        denom = norms_embeddings * norms_query

        return np.divide(dots, denom, out=np.full_like(dots, np.nan), where=denom != 0)

    if metric == "l2":
        return np.linalg.norm(embeddings - query, axis=1)

    raise InternalError(f"Unknown similarity metric: {metric}. Invalid state.")

def _convert_udf_to_map_elements(udf: Callable) -> Callable:
    """Converts a scalar-based UDF into one that works with `map_elements` in Polars.

    Args:
        udf: The original UDF that takes scalar arguments.
        columns: List of column names that the UDF should operate on.

    Returns:
        A function that operates on a row (Struct) and applies the UDF to the unwrapped values.
    """

    def adapted_udf(row):
        column_values = list(row.values())
        # Unpack the row (Struct) by its column names
        return udf(*column_values)

    return adapted_udf

def _like_to_regex(pattern_expr: pl.Expr, case_insensitive: bool = False) -> pl.Expr:
    """Convert a LIKE pattern to a regex pattern."""
    # Escape regex metacharacters except SQL wildcards % and _
    # Character class: [](){}^$.|+\ and the backslash itself
    meta_class = r"([\\\[\]\(\)\{\}\^\$\.\|\+])"

    regex_expr = (
        pattern_expr
        # Escape each meta character by prefixing with a backslash
        .str.replace_all(meta_class, r"\\\1")
        # Translate SQL wildcards to regex
        .str.replace_all("%", ".*", literal=True)
        .str.replace_all("_", ".",  literal=True)
    )

    if case_insensitive:
        return (pl.lit("(?i)") + regex_expr)

    return regex_expr

def _format_has_tz_tokens(fmt: str) -> bool:
    """Check if a format string has any timezone tokens."""
    tz_tokens = {"V", "z", "O", "X", "Z"}
    return set(fmt) & tz_tokens

def _get_zero_ts_part_from_date(expr: pl.Expr) -> pl.Expr:
    """Get the zero part of a date expression."""
    return (expr.dt.year() * pl.lit(0, dtype=pl.Int32))

def _pytype_datetime_to_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        # Naive datetime, assume it's UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Aware datetime, convert to UTC
        return dt.astimezone(timezone.utc)
