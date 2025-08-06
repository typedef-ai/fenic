"""Source plan serialization/deserialization."""

from io import BytesIO

import polars as pl

from fenic.core._logical_plan.plans.source import (
    FileSource,
    InMemorySource,
    TableSource,
)
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    serialize_logical_plan,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    FileSourceProto,
    InMemorySourceProto,
    LogicalPlanProto,
    TableSourceProto,
)

# =============================================================================
# InMemorySource
# =============================================================================


@serialize_logical_plan.register
def _serialize_in_memory_source(
    in_memory_source: InMemorySource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a logical plan in memory."""
    proto = InMemorySourceProto(
        source=in_memory_source._source.serialize(format="binary"),
        schema=context.serialize_fenic_schema(in_memory_source.schema()),
    )
    return LogicalPlanProto(in_memory_source=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_in_memory_source(
    in_memory_source: InMemorySourceProto, context: SerdeContext
):
    """Deserialize an InMemorySource LogicalPlan Node."""
    buffered_bytes = BytesIO(in_memory_source.source)
    deserialized_dataframe: pl.DataFrame = pl.DataFrame.deserialize(buffered_bytes, format="binary")
    return InMemorySource.from_schema(
        source=deserialized_dataframe,
        schema=context.deserialize_fenic_schema(in_memory_source.schema),
    )


# =============================================================================
# FileSource
# =============================================================================


@serialize_logical_plan.register
def _serialize_file_source(
    file_source: FileSource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a file source."""
    if file_source._options:
        options_merge_schema = file_source._options.get("merge_schemas", None)
        options_schema = (
            context.serialize_fenic_schema(file_source._options.get("schema"))
            if file_source._options.get("schema", None) else None
        )
    else:
        options_merge_schema = None
        options_schema = None
    proto = FileSourceProto(
        paths=file_source._paths,
        file_format=file_source._file_format,
        schema=context.serialize_fenic_schema(file_source.schema()),
        options_merge_schema=options_merge_schema,
        options_schema=options_schema,
    )
    return LogicalPlanProto(file_source=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_file_source(
    file_source: FileSourceProto, context: SerdeContext
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
        schema=context.deserialize_fenic_schema(file_source.schema),
    )


# =============================================================================
# TableSource
# =============================================================================


@serialize_logical_plan.register
def _serialize_table_source(
    table_source: TableSource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a table source."""
    proto = TableSourceProto(
        table_name=table_source._table_name,
        schema=context.serialize_fenic_schema(table_source.schema()),
    )
    return LogicalPlanProto(table_source=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_table_source(
    table_source: TableSourceProto, context: SerdeContext
) -> TableSource:
    """Deserialize a TableSource LogicalPlan Node."""
    return TableSource.from_schema(
        table_name=table_source.table_name,
        schema=context.deserialize_fenic_schema(table_source.schema),
    )
