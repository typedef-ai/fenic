"""Sink plan serialization/deserialization."""

from fenic.core._logical_plan.plans.sink import FileSink, TableSink
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    _serialize_logical_plan_helper,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    FileSinkProto,
    LogicalPlanProto,
    TableSinkProto,
)
from fenic.core.types.schema import Schema

# =============================================================================
# FileSink
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_file_sink(
    file_sink: FileSink, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a file sink (wrapper)."""
    return LogicalPlanProto(
        file_sink=FileSinkProto(
            child=context.serialize_logical_plan(SerdeContext.CHILD, file_sink.child),
            path=file_sink.path,
            sink_type=file_sink.sink_type,
            mode=file_sink.mode,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_file_sink(file_sink: FileSinkProto, context: SerdeContext, schema: Schema) -> FileSink:
    """Deserialize a FileSink LogicalPlan Node."""
    child = context.deserialize_logical_plan(SerdeContext.CHILD, file_sink.child)
    result = FileSink.from_schema(
        child=child,
        sink_type=file_sink.sink_type,
        path=file_sink.path,
        mode=file_sink.mode,
        schema=schema,
    )
    return result


# =============================================================================
# TableSink
# =============================================================================


@_serialize_logical_plan_helper.register
def _serialize_table_sink(
    table_sink: TableSink, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a table sink (wrapper)."""
    return LogicalPlanProto(
        table_sink=TableSinkProto(
            child=context.serialize_logical_plan(SerdeContext.CHILD, table_sink.child),
            table_name=table_sink.table_name,
            mode=table_sink.mode,
        )
    )


@_deserialize_logical_plan_helper.register
def _deserialize_table_sink(table_sink: TableSinkProto, context: SerdeContext, schema: Schema):
    """Deserialize a TableSink LogicalPlan Node."""
    child = context.deserialize_logical_plan(SerdeContext.CHILD, table_sink.child)
    return TableSink.from_schema(
        child=child,
        table_name=table_sink.table_name,
        mode=table_sink.mode,
        schema=schema,
    )
