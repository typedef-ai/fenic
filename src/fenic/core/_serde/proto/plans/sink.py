"""Sink plan serialization/deserialization."""

from typing import Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.sink import FileSink, TableSink
from fenic.core._serde.proto.plan_serde import (
    _deserialize_logical_plan_helper,
    serialize_logical_plan,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    FileSinkProto,
    LogicalPlanProto,
    TableSinkProto,
)

# =============================================================================
# FileSink
# =============================================================================


@serialize_logical_plan.register
def _serialize_file_sink(
    file_sink: FileSink, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a file sink."""
    input_proto = context.serialize_logical_plan(SerdeContext.INPUT, file_sink.child)
    proto = FileSinkProto(
        input=input_proto,
        path=file_sink.path,
        format=file_sink.sink_type,
        mode=file_sink.mode,
    )
    return LogicalPlanProto(file_sink=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_file_sink(file_sink: FileSinkProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a FileSink LogicalPlan Node."""
    child = context.deserialize_logical_plan(SerdeContext.INPUT, file_sink.input, session_state=session_state)
    result = FileSink(
        child=child,
        sink_type=file_sink.format,
        path=file_sink.path,
        mode=file_sink.mode,
    )
    result.session_state = session_state
    return result


# =============================================================================
# TableSink
# =============================================================================


@serialize_logical_plan.register
def _serialize_table_sink(
    table_sink: TableSink, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a table sink."""
    input_proto = context.serialize_logical_plan(SerdeContext.INPUT, table_sink.child)
    proto = TableSinkProto(
        input=input_proto, table_name=table_sink.table_name, mode=table_sink.mode
    )
    return LogicalPlanProto(table_sink=proto)


@_deserialize_logical_plan_helper.register
def _deserialize_table_sink(table_sink: TableSinkProto, context: SerdeContext, session_state: Optional[BaseSessionState] = None):
    """Deserialize a TableSink LogicalPlan Node."""
    child = context.deserialize_logical_plan(SerdeContext.INPUT, table_sink.input, session_state=session_state)
    result = TableSink(
        child=child, table_name=table_sink.table_name, mode=table_sink.mode
    )
    result.session_state = session_state
    return result
