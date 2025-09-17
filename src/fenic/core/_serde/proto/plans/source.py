"""Source plan serialization/deserialization."""

from io import BytesIO

import polars as pl

from fenic.core._logical_plan.plans.source import (
    DocSource,
    FileSource,
    InMemorySource,
    TableSource,
)
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    _serialize_logical_plan_helper,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    DocContentTypeProto,
    DocSourceProto,
    FileSourceProto,
    InMemorySourceProto,
    LogicalPlanProto,
    TableSourceProto,
)
from fenic.core.types.enums import DocContentType
from fenic.core.types.schema import Schema

# =============================================================================
# InMemorySource
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_in_memory_source(
    in_memory_source: InMemorySource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a logical plan in memory (wrapper)."""
    return LogicalPlanProto(
        in_memory_source=InMemorySourceProto(
            source=in_memory_source._source.serialize(format="binary"),
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_in_memory_source(
    in_memory_source: InMemorySourceProto, context: SerdeContext, schema: Schema
):
    """Deserialize an InMemorySource LogicalPlan Node."""
    buffered_bytes = BytesIO(in_memory_source.source)
    deserialized_dataframe: pl.DataFrame = pl.DataFrame.deserialize(buffered_bytes, format="binary")
    return InMemorySource.from_schema(
        source=deserialized_dataframe,
        schema=schema,
    )


# =============================================================================
# FileSource
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_file_source(
    file_source: FileSource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a file source (wrapper)."""
    if file_source._options:
        options_merge_schema = file_source._options.get("merge_schemas", None)
        options_schema = (
            context.serialize_fenic_schema(file_source._options.get("schema"))
            if file_source._options.get("schema", None) else None
        )
    else:
        options_merge_schema = None
        options_schema = None
    return LogicalPlanProto(
        file_source=FileSourceProto(
            paths=file_source._paths,
            file_format=file_source._file_format,
            options_merge_schema=options_merge_schema,
            options_schema=options_schema,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_file_source(
    file_source: FileSourceProto, context: SerdeContext, schema: Schema
) -> FileSource:
    """Deserialize a FileSource LogicalPlan Node."""
    options = {}
    if file_source.HasField("options_merge_schema"):
        options["merge_schemas"] = file_source.options_merge_schema
    if file_source.HasField("options_schema"):
        options["schema"] = context.deserialize_fenic_schema(file_source.options_schema)
    return FileSource.from_schema(
        paths=list(file_source.paths),
        file_format=file_source.file_format,
        options=options,
        schema=schema,
    )


# =============================================================================
# TableSource
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_table_source(
    table_source: TableSource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a table source (wrapper)."""
    return LogicalPlanProto(
        table_source=TableSourceProto(
            table_name=table_source._table_name,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_table_source(
    table_source: TableSourceProto, context: SerdeContext, schema: Schema
) -> TableSource:
    """Deserialize a TableSource LogicalPlan Node."""
    return TableSource.from_schema(
        table_name=table_source.table_name,
        schema=schema,
    )

# =============================================================================
# DocSource
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_doc_source(
    doc_source: DocSource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a doc source (wrapper)."""
    return LogicalPlanProto(
        doc_source=DocSourceProto(
            paths=doc_source._paths,
            content_type=context.serialize_python_literal(
                "content_type",
                doc_source._content_type,
                DocContentTypeProto,
            ),
            exclude=doc_source._exclude,
            recursive=doc_source._recursive,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_doc_source(
    doc_source: DocSourceProto, context: SerdeContext, schema: Schema
) -> DocSource:
    """Deserialize a DocSource LogicalPlan Node."""
    doc_source = DocSource.from_schema(
        paths=list(doc_source.paths),
        content_type=context.deserialize_python_literal(
            "content_type",
            doc_source.content_type,
            DocContentType,
            DocContentTypeProto,
        ),
        exclude=doc_source.exclude if doc_source.HasField("exclude") else None,
        recursive=doc_source.recursive,
        schema=schema,
    )

    return doc_source
