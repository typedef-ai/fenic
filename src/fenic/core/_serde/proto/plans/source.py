"""Source plan serialization/deserialization."""

from io import BytesIO
from typing import Optional

import polars as pl

from fenic.core._interfaces.session_state import BaseSessionState
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
    source_dataframe = in_memory_source._source

    source_dataframe_bytes = source_dataframe.serialize(format="binary")
    proto = InMemorySourceProto(source=source_dataframe_bytes)
    return LogicalPlanProto(in_memory_source=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_in_memory_source(
    in_memory_source: InMemorySourceProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None
):
    """Deserialize an InMemorySource LogicalPlan Node."""
    buffered_bytes = BytesIO(in_memory_source.source)
    deserialized_dataframe: pl.DataFrame = pl.DataFrame.deserialize(buffered_bytes, format="binary")
    return InMemorySource(
        source=deserialized_dataframe, session_state=session_state
    )


# =============================================================================
# FileSource
# =============================================================================


@serialize_logical_plan.register
def _serialize_file_source(
    file_source: FileSource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a file source."""
    if file_source.schema():
        schema = context.serialize_fenic_schema(SerdeContext.SCHEMA, file_source.schema())
    else:
        schema = None
    proto = FileSourceProto(
        paths=file_source._paths,
        file_format=file_source._file_format,
        schema=schema,
        merge_schemas=file_source._options.get("merge_schemas", False),
    )
    return LogicalPlanProto(file_source=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_file_source(
    file_source: FileSourceProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None
):
    """Deserialize a FileSource LogicalPlan Node."""
    options = {}
    if file_source.merge_schemas:
        options["merge_schemas"] = file_source.merge_schemas
    if file_source.schema:
        options["schema"] = context.deserialize_fenic_schema(SerdeContext.SCHEMA, file_source.schema)
    return FileSource(
        paths=list(file_source.paths),
        file_format=file_source.file_format,
        session_state=session_state,
        options=options,
    )


# =============================================================================
# TableSource
# =============================================================================


@serialize_logical_plan.register
def _serialize_table_source(
    table_source: TableSource, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a table source."""
    proto = TableSourceProto(table_name=table_source._table_name)
    return LogicalPlanProto(table_source=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_table_source(
    table_source: TableSourceProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None
):
    """Deserialize a TableSource LogicalPlan Node."""
    return TableSource(table_name=table_source.table_name, session_state=session_state)
