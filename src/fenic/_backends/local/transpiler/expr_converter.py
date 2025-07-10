from __future__ import annotations

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

import json

import numpy as np
import polars as pl
import pyarrow as pa

import fenic._backends.local.polars_plugins  # noqa: F401
from fenic._backends.local.semantic_operators import (
    AnalyzeSentiment,
)
from fenic._backends.local.semantic_operators import Classify as SemanticClassify
from fenic._backends.local.semantic_operators import Extract as SemanticExtract
from fenic._backends.local.semantic_operators import Map as SemanticMap
from fenic._backends.local.semantic_operators import Predicate as SemanticPredicate
from fenic._backends.local.semantic_operators import Reduce as SemanticReduce
from fenic._backends.local.semantic_operators import Summarize as SemanticSummarize
from fenic._backends.local.template import TemplateFormatReader
from fenic._backends.schema_serde import serialize_data_type
from fenic.core._logical_plan.expressions import (
    AggregateExpr,
    AliasExpr,
    AnalyzeSentimentExpr,
    ArithmeticExpr,
    ArrayContainsExpr,
    ArrayExpr,
    ArrayJoinExpr,
    ArrayLengthExpr,
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
    CountExpr,
    CountTokensExpr,
    EmbeddingNormalizeExpr,
    EmbeddingsExpr,
    EmbeddingSimilarityExpr,
    EndsWithExpr,
    EqualityComparisonExpr,
    FirstExpr,
    ILikeExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    JqExpr,
    JsonContainsExpr,
    JsonTypeExpr,
    LikeExpr,
    ListExpr,
    LiteralExpr,
    LogicalExpr,
    MaxExpr,
    MdExtractHeaderChunks,
    MdGenerateTocExpr,
    MdGetCodeBlocksExpr,
    MdToJsonExpr,
    MinExpr,
    NotExpr,
    NumericComparisonExpr,
    Operator,
    OtherwiseExpr,
    RecursiveTextChunkExpr,
    RegexpSplitExpr,
    ReplaceExpr,
    RLikeExpr,
    SemanticClassifyExpr,
    SemanticExtractExpr,
    SemanticMapExpr,
    SemanticPredExpr,
    SemanticReduceExpr,
    SemanticSummarizeExpr,
    SortExpr,
    SplitPartExpr,
    StartsWithExpr,
    StdDevExpr,
    StringCasingExpr,
    StripCharsExpr,
    StrLengthExpr,
    StructExpr,
    SumExpr,
    TextChunkExpr,
    TextractExpr,
    TsParseExpr,
    UDFExpr,
    WhenExpr,
)
from fenic.core._utils.extract import convert_extract_schema_to_pydantic_type
from fenic.core._utils.schema import (
    convert_custom_dtype_to_polars,
    convert_pydantic_type_to_custom_struct_type,
)
from fenic.core.error import InternalError
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DataType,
    EmbeddingType,
    IntegerType,
    JsonType,
    StringType,
    StructField,
    StructType,
    _PrimitiveType,
)
from fenic.core.types.extract_schema import ExtractSchema


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

            if isinstance(data_type, _PrimitiveType):
                return pl.lit(value, dtype=convert_custom_dtype_to_polars(data_type))

            if isinstance(data_type, ArrayType):
                elems = [_literal_to_polars_expr(v, data_type.element_type) for v in value]
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

            def sem_reduce_fn(batch: pl.Series) -> str:
                return SemanticReduce(
                    input=batch,
                    user_instruction=logical.instruction,
                    model=self.session_state.get_language_model(logical.model_alias),
                    max_tokens=logical.max_tokens,
                    temperature=logical.temperature,
                ).execute()

            struct = pl.struct(
                [
                    self._convert_expr(expr)
                    for expr in logical.exprs
                ]
            )
            return struct.map_batches(
                sem_reduce_fn, return_dtype=pl.Utf8, agg_list=True, returns_scalar=True
            )

        raise NotImplementedError(f"Unsupported aggregate function: {type(logical)}")


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


    @_convert_expr.register(StructExpr)
    def _convert_struct_expr(self, logical: StructExpr) -> pl.Expr:
        return pl.struct(
            [self._convert_expr(arg) for arg in logical.args]
        )


    @_convert_expr.register(ArrayExpr)
    def _convert_array_expr(self, logical: ArrayExpr) -> pl.Expr:
        return pl.concat_list(
            [self._convert_expr(arg) for arg in logical.args]
        )


    @_convert_expr.register(IndexExpr)
    def _convert_index_expr(self, logical: IndexExpr) -> pl.Expr:
        base_expr = self._convert_expr(logical.expr)
        if isinstance(logical.index, int):
            return base_expr.list.get(logical.index, null_on_oob=True)
        elif isinstance(logical.index, str):
            return base_expr.struct.field(str(logical.index))
        else:
            raise NotImplementedError(f"Unsupported index key type: {type(logical.index)}")


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


    @_convert_expr.register(SemanticMapExpr)
    def _convert_semantic_map_expr(self, logical: SemanticMapExpr) -> pl.Expr:
        def sem_map_fn(batch: pl.Series) -> pl.Series:
            expanded_df = pl.DataFrame(
                {field: batch.struct.field(field) for field in batch.struct.fields}
            )

            return SemanticMap(
                input=expanded_df,
                user_instruction=logical.instruction,
                model=self.session_state.get_language_model(logical.model_alias),
                examples=logical.examples,
                max_tokens=logical.max_tokens,
                temperature=logical.temperature,
                response_format=logical.response_format,
            ).execute()

        struct = pl.struct(
            [
                self._convert_expr(expr)
                for expr in logical.exprs
            ]
        )
        return struct.map_batches(sem_map_fn, return_dtype=pl.String)


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
        config = logical.chunk_configuration
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
        pydantic_model = (
                convert_extract_schema_to_pydantic_type(logical.schema)
                if isinstance(logical.schema, ExtractSchema)
                else logical.schema
            )
        def sem_ext_fn(batch: pl.Series) -> pl.Series:
            return SemanticExtract(
                input=batch,
                schema=pydantic_model,
                model=self.session_state.get_language_model(logical.model_alias),
                max_output_tokens=logical.max_tokens,
                temperature=logical.temperature,
            ).execute()

        return self._convert_expr(logical.expr).map_batches(
            sem_ext_fn,
            return_dtype=convert_custom_dtype_to_polars(
                convert_pydantic_type_to_custom_struct_type(pydantic_model)
            ),
        )


    @_convert_expr.register(SemanticPredExpr)
    def _convert_semantic_pred_expr(self, logical: SemanticPredExpr) -> pl.Expr:
        def sem_predicate_fn(batch: pl.Series) -> pl.Series:
            expanded_df = pl.DataFrame(
                {field: batch.struct.field(field) for field in batch.struct.fields}
            )
            return SemanticPredicate(
                input=expanded_df,
                user_instruction=logical.instruction,
                model=self.session_state.get_language_model(logical.model_alias),
                temperature=logical.temperature,
                examples=logical.examples,
            ).execute()

        struct = pl.struct(
            [
                self._convert_expr(expr)
                for expr in logical.exprs
            ]
        )
        return struct.map_batches(sem_predicate_fn, return_dtype=pl.Boolean)


    @_convert_expr.register(SemanticClassifyExpr)
    def _convert_semantic_classify_expr(self, logical: SemanticClassifyExpr) -> pl.Expr:
        def sem_classify_fn(batch: pl.Series) -> pl.Series:
            labels_enum = (
                SemanticClassifyExpr.transform_labels_list_into_enum(logical.labels)
                if isinstance(logical.labels, list)
                else logical.labels
            )
            return SemanticClassify(
                input=batch,
                labels=labels_enum,
                model=self.session_state.get_language_model(logical.model_alias),
                temperature=logical.temperature,
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

            ).execute()

        return self._convert_expr(logical.expr).map_batches(
            sem_summarize_fn, return_dtype=pl.Utf8
        )

    @_convert_expr.register(ArrayJoinExpr)
    def _convert_array_join_expr(self, logical: ArrayJoinExpr) -> pl.Expr:
        return self._convert_expr(logical.expr).list.join(
            logical.delimiter
        )


    @_convert_expr.register(ContainsExpr)
    def _convert_contains_expr(self, logical: ContainsExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        substr_expr = (
            self._convert_expr(logical.substr)
            if isinstance(logical.substr, LogicalExpr)
            else pl.lit(logical.substr)
        )
        return physical_expr.str.contains(pattern=substr_expr, literal=True)


    @_convert_expr.register(ContainsAnyExpr)
    def _convert_contains_any_expr(self, logical: ContainsAnyExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        return physical_expr.str.contains_any(
            patterns=logical.substrs, ascii_case_insensitive=logical.case_insensitive
        )


    # Group like/regex expressions together
    def _handle_regex_like_expr(self, logical, pattern_field="pattern", literal: bool = False) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        pattern = getattr(logical, pattern_field)
        return physical_expr.str.contains(pattern=pattern, literal=literal)


    @_convert_expr.register(RLikeExpr)
    @_convert_expr.register(LikeExpr)
    @_convert_expr.register(ILikeExpr)
    def _convert_like_expr(self, logical) -> pl.Expr:
        return self._handle_regex_like_expr(logical)


    @_convert_expr.register(StartsWithExpr)
    def _convert_starts_with_expr(self, logical: StartsWithExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        substr_expr = (
            self._convert_expr(logical.substr)
            if isinstance(logical.substr, LogicalExpr)
            else pl.lit(logical.substr)
        )
        return physical_expr.str.starts_with(prefix=substr_expr)


    @_convert_expr.register(EndsWithExpr)
    def _convert_ends_with_expr(self, logical: EndsWithExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        substr_expr = (
            self._convert_expr(logical.substr)
            if isinstance(logical.substr, LogicalExpr)
            else pl.lit(logical.substr)
        )
        return physical_expr.str.ends_with(suffix=substr_expr)


    @_convert_expr.register(EmbeddingsExpr)
    def _convert_embeddings_expr(self, logical: EmbeddingsExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        if logical.dimensions is None:
            raise InternalError("Embedding dimensions not set for embeddings expression")

        def embeddings_fn(batch: pl.Series) -> pl.Series:
            embedding_model = self.session_state.get_embedding_model(logical.model_alias)
            return pl.from_arrow(embedding_model.get_embeddings(batch))

        return physical_expr.map_batches(embeddings_fn, return_dtype=pl.Array(pl.Float32, logical.dimensions))


    @_convert_expr.register(SplitPartExpr)
    def _convert_split_part_expr(self, logical: SplitPartExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        is_delimiter_column = isinstance(logical.delimiter, LogicalExpr)
        is_part_number_column = isinstance(logical.part_number, LogicalExpr)

        # If neither delimiter nor part_number is a column, use the standard split_part
        if not is_delimiter_column and not is_part_number_column:
            # arr.get is zero indexed, split_part is 1-based per the Spark spec
            if logical.part_number > 0:
                pl_index = logical.part_number - 1
            else:
                pl_index = logical.part_number

            return (
                physical_expr.str.split(logical.delimiter)
                .list.get(index=pl_index, null_on_oob=True)
                .fill_null(
                    ""
                )  # spark semantics expect an empty string if part number is out of range
            )

        # Convert expressions for delimiter and part number
        delimiter_expr = (
            self._convert_expr(logical.delimiter)
            if is_delimiter_column
            else pl.lit(logical.delimiter)
        )
        part_number_expr = (
            self._convert_expr(logical.part_number)
            if is_part_number_column
            else pl.lit(logical.part_number)
        )

        # If only delimiter is a column, we can pass it directly to split
        if is_delimiter_column and not is_part_number_column:
            # Convert part number from 1-based to 0-based for positive numbers
            pl_index = (
                logical.part_number - 1 if logical.part_number > 0 else logical.part_number
            )
            return (
                physical_expr.str.split(delimiter_expr)
                .list.get(index=pl_index, null_on_oob=True)
                .fill_null("")
            )

        # If part_number is a column, use over expressions
        # First split using the delimiter (either column or literal)
        split_expr = physical_expr.str.split(delimiter_expr)

        # Convert from 1-based to 0-based indexing for positive numbers
        part_expr = (
            pl.when(part_number_expr.first() > 0)
            .then(part_number_expr.first() - 1)
            .otherwise(part_number_expr.first())
        )

        # Get the part and handle out of range with empty string
        return (
            split_expr.list.get(part_expr, null_on_oob=True)
            .fill_null("")
            .over(part_number_expr)
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


    @_convert_expr.register(StripCharsExpr)
    def _convert_strip_chars_expr(self, logical: StripCharsExpr) -> pl.Expr:
        physical_expr = self._convert_expr(logical.expr)
        chars_expr = (
            self._convert_expr(logical.chars)
            if isinstance(logical.chars, LogicalExpr)
            else pl.lit(logical.chars)
        )

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
        is_search_column = isinstance(logical.search, LogicalExpr)
        is_replace_column = isinstance(logical.replacement, LogicalExpr)

        # If neither search nor replacement is a column, use the standard string replace
        if not is_search_column and not is_replace_column:
            if logical.replacement_count == -1:
                return physical_expr.str.replace_all(
                    pattern=logical.search,
                    value=logical.replacement,
                    literal=logical.literal,
                )
            else:
                return physical_expr.str.replace(
                    pattern=logical.search,
                    value=logical.replacement,
                    literal=logical.literal,
                    n=logical.replacement_count,
                )

        # Handle column-based replacement
        if is_search_column:
            search_expr = self._convert_expr(logical.search)
            pattern_expr = search_expr.first()
        else:
            search_expr = pl.lit(logical.search)
            pattern_expr = search_expr

        if is_replace_column:
            replacement_expr = self._convert_expr(logical.replacement)
        else:
            replacement_expr = pl.lit(logical.replacement)

        if logical.replacement_count == -1:
            result = physical_expr.str.replace_all(
                pattern=pattern_expr,
                value=replacement_expr,
                literal=logical.literal,
            )
        else:
            result = physical_expr.str.replace(
                pattern=pattern_expr,
                value=replacement_expr,
                literal=logical.literal,
                n=logical.replacement_count,
            )

        if is_search_column:
            return result.over(search_expr)
        else:
            return result


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
