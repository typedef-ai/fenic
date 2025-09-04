from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple
from urllib.parse import urlparse

import grpc
import polars as pl
import pyarrow as pa
import pyarrow.flight as flight  # noqa: F401 # Required to import flight.
from fenic_cloud.hasura_client.generated_graphql_client import (
    TypedefExecutionStatusReferenceEnum as QUERY_STATE,
)
from fenic_cloud.hasura_client.hasura_execution import (
    get_query_execution_by_id,
    query_execution_subscription,
)
from fenic_cloud.protos.engine.v1.engine_pb2 import (
    CollectExecutionRequest,
    CountExecutionRequest,
    GetExecutionResultRequest,
    GetExecutionResultResponse,
    InferSchemaFromCSVRequest,
    InferSchemaFromParquetRequest,
    InferSchemaRequest,
    InferSchemaResponse,
    SaveAsTableExecutionRequest,
    SaveToFileExecutionRequest,
    ShowExecutionRequest,
    StartExecutionRequest,
    TableIdentifier,
)
from fenic_cloud.protos.engine.v1.engine_pb2_grpc import EngineServiceStub

from fenic._backends.cloud.metrics import get_query_execution_metrics
from fenic._backends.schema_serde import deserialize_schema, serialize_schema
from fenic.core._interfaces import BaseExecution
from fenic.core._serde import LogicalPlanSerde
from fenic.core.error import (
    CloudExecutionError,
    CloudSessionError,
    ExecutionError,
    InternalError,
    ValidationError,
)
from fenic.core.metrics import QueryMetrics
from fenic.core.types import Schema

if TYPE_CHECKING:
    from fenic._backends.cloud.session_state import CloudSessionState
    from fenic.core._interfaces import BaseLineage
    from fenic.core._logical_plan.plans import LogicalPlan
    from fenic.core.metrics import QueryMetrics

import logging

logger = logging.getLogger(__name__)

CLOUD_SUPPORTED_SCHEMES = ["s3"]

class CloudExecution(BaseExecution):
    def __init__(
        self, session_state: CloudSessionState, engine_stub: EngineServiceStub
    ):
        """Initialize the cloud execution client.

        Args:
            channel: A gRPC channel to communicate with the engine service.
        """
        self.session_state = session_state
        self._engine_stub = engine_stub
        logger.debug(
            f"Initialized enginestub with channel: {session_state.engine_channel}"
        )

    def collect(
        self, plan: LogicalPlan, n: Optional[int] = None
    ) -> Tuple[pl.DataFrame, QueryMetrics]:
        """Execute a logical plan and return a Polars DataFrame and query metrics."""
        request = StartExecutionRequest(
            collect=CollectExecutionRequest(
                serialized_plan=LogicalPlanSerde.serialize(plan)
            )
        )

        future = asyncio.run_coroutine_threadsafe(
            self._send_execution_request_and_wait_for_execution_state(
                request, [QUERY_STATE.READY]
            ),
            self.session_state.asyncio_loop,
        )
        execution_id = future.result()
        df = self._get_execution_result_from_arrow(execution_id)

        return df, self._get_query_execution_metrics(execution_id)

    def show(self, plan: LogicalPlan, n: int = 10) -> Tuple[str, QueryMetrics]:
        """Execute a logical plan and return a string representation of the sample rows."""
        logger.debug(f"Sending show request: {plan}")
        request = StartExecutionRequest(
            show=ShowExecutionRequest(
                serialized_plan=LogicalPlanSerde.serialize(plan), row_limit=n
            )
        )
        future = asyncio.run_coroutine_threadsafe(
            self._send_execution_request_and_wait_for_execution_state(
                request, [QUERY_STATE.READY]
            ),
            self.session_state.asyncio_loop,
        )
        execution_id = future.result()

        result_future = asyncio.run_coroutine_threadsafe(
            self._get_execution_result(execution_id), self.session_state.asyncio_loop
        )
        result_response = result_future.result()

        if not result_response.HasField("show_result"):
            raise CloudExecutionError(
                f"Result of show execution '{execution_id}' did not include show string."
            )

        return result_response.show_result, self._get_query_execution_metrics(execution_id)

    def count(self, plan: LogicalPlan) -> Tuple[int, QueryMetrics]:
        """Execute a logical plan and return the number of rows."""
        request = StartExecutionRequest(
            count=CountExecutionRequest(
                serialized_plan=LogicalPlanSerde.serialize(plan)
            )
        )
        future = asyncio.run_coroutine_threadsafe(
            self._send_execution_request_and_wait_for_execution_state(
                request, [QUERY_STATE.READY]
            ),
            self.session_state.asyncio_loop,
        )
        execution_id = future.result()

        result_future = asyncio.run_coroutine_threadsafe(
            self._get_execution_result(execution_id), self.session_state.asyncio_loop
        )
        result_response = result_future.result()

        if not result_response.HasField("count_result"):
            raise CloudExecutionError(
                f"Result of count execution '{execution_id}' did not include count field."
            )

        return result_response.count_result, self._get_query_execution_metrics(execution_id)

    def build_lineage(self, plan: LogicalPlan) -> BaseLineage:
        """Build a lineage graph from a logical plan."""
        # TODO: Implement lineage building
        raise NotImplementedError(
            "Lineage building not implemented for cloud execution"
        )

    def save_as_table(
        self,
        logical_plan: LogicalPlan,
        table_name: str,
        mode: Literal["error", "append", "overwrite", "ignore"],
    ) -> QueryMetrics:
        """Execute the logical plan and save the result as a table."""
        logger.debug(f"Saving plan {logical_plan} as table: {table_name}")
        # TODO (DY): check that current catalog and schema (if specified in table_name) match session state
        table_identifier = TableIdentifier(
            catalog=self.session_state.catalog,
            schema=self.session_state.schema,
            table=table_name,
        )
        request = StartExecutionRequest(
            save_as_table=SaveAsTableExecutionRequest(
                serialized_plan=LogicalPlanSerde.serialize(logical_plan),
                table_identifier=table_identifier,
                mode=mode,
            )
        )
        # Engine can skip READY for save operations, as a future optimization
        future = asyncio.run_coroutine_threadsafe(
            self._send_execution_request_and_wait_for_execution_state(
                request, [QUERY_STATE.READY, QUERY_STATE.COMPLETED]
            ),
            self.session_state.asyncio_loop,
        )
        execution_id = future.result()
        return self._get_query_execution_metrics(execution_id)

    def save_as_view(
        self,
        logical_plan: LogicalPlan,
        view_name: str,
        view_description: Optional[str] = None,
    ) -> None:
        """Save the dataframe as a view."""
        # TODO: Implement saving dataframe view
        raise NotImplementedError("Saving a view of the dataframe not implemented for cloud execution")
    
    def save_to_file(
        self,
        logical_plan: LogicalPlan,
        file_path: str,
        mode: Literal["error", "overwrite", "ignore"] = "error",
    ) -> QueryMetrics:
        """Execute the logical plan and save the result as a CSV or parquet file."""
        if urlparse(file_path).scheme not in CLOUD_SUPPORTED_SCHEMES:
            raise ValidationError(
                "Cloud execution only supports writes to S3 buckets. "
                f"Got: {file_path}"
                "Make sure Typedef Cloud has a role with permissions to write to your bucket. "
                "Then include the s3 prefix and bucket in the file path. "
                "Example: df.write.csv('s3://{{bucket}}/{{key}}')."
            )
        logger.debug(f"Saving plan {logical_plan} to file: {file_path}")
        if not file_path.endswith(".csv") and not file_path.endswith(".parquet"):
            # should be enforced at the api level
            raise InternalError(f"Can only call save_to_file with .csv or .parquet file extensions. Got: {file_path}")
        request = StartExecutionRequest(
            save_to_file=SaveToFileExecutionRequest(
                serialized_plan=LogicalPlanSerde.serialize(logical_plan),
                file_path=file_path,
                mode=mode,
            )
        )
        # Engine can skip READY for save operations, as a future optimization
        future = asyncio.run_coroutine_threadsafe(
            self._send_execution_request_and_wait_for_execution_state(
                request, [QUERY_STATE.READY, QUERY_STATE.COMPLETED]
            ),
            self.session_state.asyncio_loop,
        )
        execution_id = future.result()
        return self._get_query_execution_metrics(execution_id)

    def infer_schema_from_csv(
        self, paths: list[str], **options: Dict[str, Any]
    ) -> Schema:
        """Infer the schema of a CSV file."""
        non_s3_paths = [path for path in paths if urlparse(path).scheme not in CLOUD_SUPPORTED_SCHEMES]
        if len(non_s3_paths) > 0:
            raise ValidationError(
                "Cloud execution only supports reads from S3 buckets. "
                f"Got: {non_s3_paths}"
                "Make sure Typedef Cloud has a role with permissions to read from your bucket. "
                "Then include the s3 prefix and bucket in the file path. "
                "Example: df.read.csv(['s3://{{bucket}}/file1.csv', 's3://{{bucket}}/file2.csv'], merge_schemas=True)."
            )
        request = InferSchemaRequest(
            infer_schema_from_csv=InferSchemaFromCSVRequest(
                file_list=",".join(paths),
                merge_schemas=options.get("merge_schemas", False),
            )
        )
        schema = options.get("schema", None)
        if schema:
            request.infer_schema_from_csv.schema = serialize_schema(schema)
        future = asyncio.run_coroutine_threadsafe(
            self._send_infer_request_and_get_response(request),
            self.session_state.asyncio_loop,
        )
        response = future.result()
        schema = deserialize_schema(response.schema)
        return schema

    def infer_schema_from_parquet(
        self, paths: list[str], **options: Dict[str, Any]
    ) -> Schema:
        """Infer the schema of a Parquet file."""
        non_s3_paths = [path for path in paths if not urlparse(path).scheme == "s3"]
        if len(non_s3_paths) > 0:
            raise ValidationError(
                "Cloud execution only supports reads from S3 buckets. "
                f"Got: {non_s3_paths}"
                "Make sure Typedef Cloud has a role with permissions to read from your bucket. "
                "Then include the s3 prefix and bucket in the file path. "
                "Example: df.read.parquet(['s3://{{bucket}}/file1.parquet', 's3://{{bucket}}/file2.parquet'], merge_schemas=True)."
            )
        request = InferSchemaRequest(
            infer_schema_from_parquet=InferSchemaFromParquetRequest(
                file_list=",".join(paths),
                merge_schemas=options.get("merge_schemas", False),
            )
        )
        response_future = asyncio.run_coroutine_threadsafe(
            self._send_infer_request_and_get_response(request),
            self.session_state.asyncio_loop,
        )
        response = response_future.result()
        schema = deserialize_schema(response.schema)
        return schema

    async def _send_infer_request_and_get_response(
        self, request: InferSchemaRequest
    ) -> InferSchemaResponse:
        """Send a infer request to the engine service and wait for the response."""
        try:
            response = await self._engine_stub.InferSchema(
                request, metadata=self.session_state.get_engine_grpc_metadata()
            )
        except grpc.RpcError as e:
            file_format = "csv" if request.HasField("infer_schema_from_csv") else "parquet"
            raise ExecutionError(f"Failed to infer schema from {file_format} files. Details: {e}") from e
        logger.debug(f"Infer schema response: {response}")
        return response

    async def _send_execution_request_and_wait_for_execution_state(
        self,
        request: StartExecutionRequest,
        states: list[QUERY_STATE],
    ) -> str:
        """Send a request to the engine service.  Wait for execution to process or fail and return the execution id and query metrics."""
        logger.debug(f"Sending execution request: {request}")

        try:
            response = await self._engine_stub.StartExecution(
                request, metadata=self.session_state.get_engine_grpc_metadata()
            )
        except grpc.RpcError as e:
            raise CloudSessionError("Failed to start execution") from e

        # Always end when we detect a failed state
        states.append(QUERY_STATE.FAILED)
        logger.debug(
            f"Execution start response: {response}.  Waiting for states:{states}"
        )
        execution_id = response.execution_id
        status = await query_execution_subscription(
            self.session_state.hasura_user_client,
            execution_id,
            states,
        )
        if status == QUERY_STATE.FAILED:
            logger.error(f"Execution {execution_id} failed")
            execution_details = await get_query_execution_by_id(
                self.session_state.hasura_user_client, execution_id
            )
            raise CloudExecutionError(
                f"Execution {execution_id} failed: '{execution_details.error_message}'"
            )

        logger.info(f"Execution {execution_id} completed")
        return execution_id

    async def _get_execution_result(
        self, execution_id: str
    ) -> GetExecutionResultResponse:
        """Get the result of an execution."""
        result_request = GetExecutionResultRequest(execution_uuid=execution_id)
        try:
            result_response = await self._engine_stub.GetExecutionResult(
                result_request, metadata=self.session_state.get_engine_grpc_metadata()
            )
        except grpc.RpcError as e:
            raise CloudSessionError(
                f"Failed to get result for execution '{execution_id}'"
            ) from e
        return result_response

    def _get_execution_result_from_arrow(self, execution_id: str) -> pl.DataFrame:
        """Get the result of an execution as a Polars DataFrame."""
        try:
            logger.debug(f"Connecting to arrow IPC: {self.session_state.arrow_ipc_uri}")
            arrow_client = pa.flight.connect(
                _get_arrow_grpc_uri(
                    self.session_state.arrow_ipc_uri,
                    self.session_state.arrow_ipc_uri_secure)
            )
            options = pa.flight.FlightCallOptions(
                headers=_get_arrow_ipc_headers(self.session_state.session_uuid))
            reader = arrow_client.do_get(
                pa.flight.Ticket(str(execution_id).encode("utf-8")),
                options,
            )
            table = reader.read_all()
            return pl.DataFrame(table)
        except pa.flight.FlightServerError as e:
            raise CloudSessionError(
                f"Failed to stream result for execution '{execution_id}'"
            ) from e
        except Exception as e:
            raise CloudSessionError(
                "Failed while connecting to arrow IPC"
            ) from e

    def _get_query_execution_metrics(self, execution_id: str) -> QueryMetrics:
        """Get query execution metrics from the cloud catalog."""
        future = asyncio.run_coroutine_threadsafe(
            get_query_execution_metrics(self.session_state.hasura_user_client, execution_id),
            self.session_state.asyncio_loop,
        )
        return future.result()

def _get_arrow_ipc_headers(
        session_id: str,
    ) -> list[Tuple[str, str]]:
    """Get the headers for the arrow IPC connection."""
    logger.debug(f"Getting arrow IPC headers for session: {session_id}")
    return [(b"session-id", session_id.encode("utf-8")),
            (b"content-type", b"application/grpc"),
            (b"endpoint", b"results")]

def _get_arrow_grpc_uri(
    uri: str,
    secure: bool) -> str:
    """Get the gRPC URI for the arrow IPC connection."""
    return f"grpc+tls://{uri}" if secure else f"grpc://{uri}"
