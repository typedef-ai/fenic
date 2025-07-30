import asyncio
import os
import threading
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import patch

import polars as pl
import pytest

from fenic.api.session.config import OpenAILanguageModel

pytest.importorskip("grpc")
pytest.importorskip("fenic_cloud.hasura_client")

from fenic_cloud.hasura_client.generated_graphql_client.list_query_execution_metric_by_query_execution_id import (
    ListQueryExecutionMetricByQueryExecutionId,
)
from fenic_cloud.hasura_client.generated_graphql_client.list_query_execution_metric_by_query_execution_id import (
    ListQueryExecutionMetricByQueryExecutionIdTypedefQueryExecutionByPk as QueryExecutionMetricByPk,
)
from fenic_cloud.protos.engine.v1.engine_pb2 import (
    ConfigSessionRequest,
    ConfigSessionResponse,
    GetExecutionResultRequest,
    GetExecutionResultResponse,
    InferSchemaRequest,
    InferSchemaResponse,
    StartExecutionRequest,
    StartExecutionResponse,
)
from fenic_cloud.protos.engine.v1.engine_pb2_grpc import EngineServiceServicer
from fenic_cloud.protos.omnitype.v1.common_pb2 import EngineRequestUris
from fenic_cloud.protos.omnitype.v1.entrypoint_pb2 import (
    GetOrCreateSessionResponse,
    RegisterAppResponse,
    TerminateSessionResponse,
)
from fenic_cloud.protos.omnitype.v1.entrypoint_pb2_grpc import (
    EntrypointServiceServicer,
)
from grpc import RpcError, StatusCode

from fenic import ColumnField, IntegerType, Schema, StringType, configure_logging
from fenic._backends.cloud.engine_config import CloudSessionConfig
from fenic._backends.cloud.manager import (
    CloudSessionManager,
    CloudSessionManagerConfigDependencies,
)
from fenic._backends.cloud.settings import CloudSettings
from fenic._backends.schema_serde import serialize_schema
from fenic.api.session import (
    CloudConfig,
    CloudExecutorSize,
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import (
    ConfigurationError,
    ExecutionError,
    SessionError,
    ValidationError,
)

pytestmark = pytest.mark.cloud


@pytest.fixture(scope="session")
def cloud_app_name():
    return "test-app"


@dataclass
class MockHasuraQueryExecution:
    status: str


@dataclass
class MockHasuraExecutionUpdate:
    typedef_query_execution_by_pk: Optional[MockHasuraQueryExecution]


class MockAsyncIterator:
    def __init__(self, updates):
        self.updates = updates
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.updates):
            raise StopAsyncIteration
        update = self.updates[self.index]
        self.index += 1
        return update


class MockHasuraUserClient:
    """Mocks the webql socket letting us subscribe to query execution status updates."""

    updates: List[MockHasuraExecutionUpdate]

    def __init__(self):
        self.updates = [
            MockHasuraExecutionUpdate(MockHasuraQueryExecution(status="RUNNING")),
            MockHasuraExecutionUpdate(MockHasuraQueryExecution(status="READY")),
            MockHasuraExecutionUpdate(MockHasuraQueryExecution(status="COMPLETED")),
        ]

    def query_execution_details(self, query_execution_id):
        return MockAsyncIterator(self.updates)

    async def list_query_execution_metric_by_query_execution_id(self, query_execution_id):
        return ListQueryExecutionMetricByQueryExecutionId(
            typedef_query_execution_by_pk=QueryExecutionMetricByPk(
                query_execution_metrics=[]
            )
        )


class MockHasuraClient:
    """Mock implementation of the HasuraClient."""

    def get_user_client(self, auth_token):
        return MockHasuraUserClient()


@pytest.fixture(scope="session")
def mock_hasura_client():
    return MockHasuraClient()


class MockEntrypointService(EntrypointServiceServicer):
    """Mock implementation of the EntrypointService."""

    async def RegisterApp(self, request, metadata):
        return RegisterAppResponse(uuid="test_uuid")

    async def GetOrCreateSession(self, request, metadata):
        return GetOrCreateSessionResponse(
            uuid="test_uuid",
            name="test-name",
            canonical_name="test-canonical-name",
            uris=EngineRequestUris(
                remote_actions_uri="http://test-cloud-actions-uri/",
                remote_results_uri_prefix="http://test-cloud-results-uri-prefix/",
            ),
            existing=False,
            ephemeral_catalog_id="test-ephemeral-catalog-id",
        )

    async def TerminateSession(self, request, metadata):
        return TerminateSessionResponse()


class MockChannel:
    async def close(self):
        await asyncio.sleep(0)


@pytest.fixture(scope="session")
def mock_entrypoint_service():
    return MockEntrypointService()


# Engine grpc mock
class MockEngineService(EngineServiceServicer):
    """Mock implementation of the EngineService.
    Methods are implemented at the test level.
    """

    async def StartExecution(self, request: StartExecutionRequest, metadata):
        return StartExecutionResponse(execution_id="test-execution-id")

    async def GetExecutionResult(self, request: GetExecutionResultRequest, metadata):
        pass

    async def InferSchema(self, request: InferSchemaRequest, metadata):
        return InferSchemaResponse(schema="test-schema")

    async def ConfigSession(self, request: ConfigSessionRequest, metadata):
        return ConfigSessionResponse()


@pytest.fixture(scope="session")
def mock_engine_service():
    return MockEngineService()


@pytest.fixture(scope="session")
def cloud_session_config(cloud_app_name):
    """Creates a test session config."""
    return SessionConfig(
        app_name=cloud_app_name,
        semantic=SemanticConfig(
            language_models={"nano" : OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=500, tpm=200_000)},
            embedding_models={"oai-small": OpenAILanguageModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)}
        ),
        cloud=CloudConfig(
            size=CloudExecutorSize.SMALL,
        ),
    )


# Necessary because pytest fixtures don't support session level monkeypatching out of the box yet. See https://github.com/pytest-dev/pytest/issues/363
@pytest.fixture(scope="session")
def monkeypatch_session():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def create_mock_session(cloud_session_config, mock_engine_service):
    # Mock engine channel and stub creation after entrypoint returns endpoints
    def mock_create_engine_channels(self):
        self.engine_channel = MockChannel()
        self.arrow_ipc_channel = MockChannel()
        self.engine_stub = mock_engine_service

    with patch(
        "fenic._backends.cloud.session_state.CloudSessionState._create_engine_channels",
        new=mock_create_engine_channels,
    ):
        session = Session.get_or_create(cloud_session_config)
    return session


@pytest.fixture(scope="session")
def cloud_settings_mock(monkeypatch_session):
    monkeypatch_session.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch_session.setenv("TYPEDEF_CLIENT_ID", "test_client_id")
    monkeypatch_session.setenv("TYPEDEF_CLIENT_SECRET", "test_client_secret")
    monkeypatch_session.setenv("HASURA_GRAPHQL_ADMIN_SECRET", "test_admin_secret")
    monkeypatch_session.setenv(
        "CLOUD_SESSION_AUTH_PROVIDER_URI", "http://test-auth-provider"
    )
    monkeypatch_session.setenv("CLOUD_SESSION_TYPEDEF_INSTANCE", "test-instance")
    monkeypatch_session.setenv("CLOUD_SESSION_TYPEDEF_ENVIRONMENT", "test")
    monkeypatch_session.setenv(
        "CLOUD_SESSION_HASURA_GRAPHQL_URI", "http://test-hasura-graphql"
    )
    monkeypatch_session.setenv(
        "CLOUD_SESSION_HASURA_GRAPHQL_WS_URI", "ws://test-hasura-graphql"
    )
    monkeypatch_session.setenv("CLOUD_SESSION_API_AUTH_URI", "http://test-api-auth")
    monkeypatch_session.setenv("CLOUD_SESSION_CLIENT_TOKEN", "test_token")
    monkeypatch_session.setenv("CLOUD_SESSION_ENTRYPOINT_URI", "test-entrypoint-uri")
    monkeypatch_session.setenv(
        "CLOUD_SESSION_TEST_ENGINE_GRPC_URL", "test-engine-grpc-uri"
    )
    monkeypatch_session.setenv(
        "CLOUD_SESSION_TEST_ENGINE_ARROW_URL", "test-engine-arrow-uri"
    )


@pytest.fixture(scope="session")
def cloud_session(
    cloud_settings_mock,
    cloud_session_config,
    mock_hasura_client,
    mock_entrypoint_service,
    mock_engine_service,
):
    """Creates a test cloud session with all environment settings updated."""
    configure_logging()
    # Create the event loop and background thread
    asyncio_loop = asyncio.new_event_loop()
    background_thread = threading.Thread(target=asyncio_loop.run_forever, daemon=True)
    background_thread.start()
    cloud_session_manager = CloudSessionManager()
    cloud_session_manager.configure(
        CloudSessionManagerConfigDependencies(
            settings=CloudSettings(),
            asyncio_loop=asyncio_loop,
            background_thread=background_thread,
            entrypoint_channel=MockChannel(),
            entrypoint_stub=mock_entrypoint_service,
            hasura_client=mock_hasura_client,
        )
    )

    # Mock engine channel and stub creation after entrypoint returns endpoints
    def mock_create_engine_channels(self):
        self.engine_channel = MockChannel()
        self.arrow_ipc_channel = MockChannel()
        self.engine_stub = mock_engine_service

    with patch(
        "fenic._backends.cloud.session_state.CloudSessionState._create_engine_channels",
        new=mock_create_engine_channels,
    ):
        session = Session.get_or_create(cloud_session_config)

    yield session
    session.stop()
    if os.path.exists(f"{cloud_session_config.app_name}.duckdb"):
        os.remove(f"{cloud_session_config.app_name}.duckdb")


def test_cloud_simple_count(cloud_session, mock_engine_service):
    async def _get_execution_result(execution_id, metadata):
        return GetExecutionResultResponse(count_result=3)

    mock_engine_service.GetExecutionResult = _get_execution_result

    df = cloud_session.create_dataframe({"a": [1, 2, 3]})
    count_result = df.count()
    assert count_result == 3


def test_cloud_simple_show(cloud_session, mock_engine_service):
    async def _get_execution_result(execution_id, metadata):
        return GetExecutionResultResponse(show_result="test-show-result")

    mock_engine_service.GetExecutionResult = _get_execution_result

    df = cloud_session.create_dataframe({"a": [1, 2, 3]})
    df.show()


def test_cloud_simple_collect(cloud_session, mock_engine_service):
    with patch(
        "fenic._backends.cloud.execution.CloudExecution._get_execution_result_from_arrow",
        return_value=pl.DataFrame({"a": [1, 2, 3]}),
    ):
        df = cloud_session.create_dataframe({"a": [1, 2, 3]})
        df_result = df.to_polars()
        assert df_result.equals(pl.DataFrame({"a": [1, 2, 3]}))


def test_cloud_save(cloud_session, mock_engine_service):
    async def _get_execution_result(execution_id, metadata):
        return GetExecutionResultResponse()

    mock_engine_service.GetExecutionResult = _get_execution_result

    df = cloud_session.create_dataframe({"a": [1, 2, 3]})
    # test csv
    df.write.csv("s3://test-bucket/test-file-list.csv", mode="overwrite")
    # test parquet
    df.write.parquet("s3://test-bucket/test-file-list.parquet", mode="error")

    # test that we only accept s3 paths
    with pytest.raises(ValidationError, match="Cloud execution only supports writes to S3 buckets."):
        df.write.csv("file://test-file-list.csv", mode="overwrite")
    with pytest.raises(ValidationError, match="Cloud execution only supports writes to S3 buckets."):
        df.write.parquet("test-file-list.parquet", mode="ignore")


TEST_SCHEMA = Schema(column_fields=[ColumnField(name="a", data_type=IntegerType)])
TEST_PASSED_IN_SCHEMA = Schema(
    column_fields=[ColumnField(name="b", data_type=StringType)]
)


def test_cloud_simple_infer_schema(cloud_session, mock_engine_service):
    async def _infer_schema(request: InferSchemaRequest, metadata):
        if (
            request.HasField("infer_schema_from_csv")
            and request.infer_schema_from_csv.schema
        ):
            return InferSchemaResponse(schema=request.infer_schema_from_csv.schema)
        else:
            return InferSchemaResponse(schema=serialize_schema(TEST_SCHEMA))

    mock_engine_service.InferSchema = _infer_schema

    schema = cloud_session._session_state.execution.infer_schema_from_parquet(
        paths=["s3://test-bucket/test-file-list.parquet"],
        merge_schemas=True,
    )
    assert schema == TEST_SCHEMA

    schema = cloud_session._session_state.execution.infer_schema_from_csv(
        paths=["s3://test-bucket/test-file-list.csv"],
    )
    assert schema == TEST_SCHEMA

    schema = cloud_session._session_state.execution.infer_schema_from_csv(
        paths=["s3://test-bucket/test-file-list.csv"],
        merge_schemas=True,
        schema=TEST_PASSED_IN_SCHEMA,
    )
    assert schema == TEST_PASSED_IN_SCHEMA

    # Test that we only accept s3 paths
    with pytest.raises(
        ValidationError, match="Cloud execution only supports reads from S3 buckets."
    ):
        cloud_session._session_state.execution.infer_schema_from_csv(
            paths=["file://test-file-list.csv"],
        )

    with pytest.raises(
        ValidationError, match="Cloud execution only supports reads from S3 buckets."
    ):
        cloud_session._session_state.execution.infer_schema_from_parquet(
            paths=["test-file-list.csv"],
        )

def test_session_configuration_errors(monkeypatch_session, cloud_session, cloud_session_config):
    """Test handling of configuration errors during session creation."""
    monkeypatch_session.delenv("OPENAI_API_KEY")
    with pytest.raises(
        ConfigurationError,
        match="OPENAI_API_KEY is not set. Please set it in your environment",
    ):
        CloudSessionConfig(cloud_session_config)

    monkeypatch_session.setenv("OPENAI_API_KEY", "test_api_key")

def test_cloud_errors(cloud_session, mock_engine_service):
    """Test handling of gRPC errors during execution."""

    class MockGrpcError(RpcError):
        def code(self):
            return StatusCode.INTERNAL

        def details(self):
            return "Test gRPC error"

    async def mock_grpc_error(request, metadata):
        raise MockGrpcError()

    def mock_arrow_grpc_error(request):
        raise MockGrpcError()

    # Test start execution returns an error
    mock_engine_service.StartExecution = mock_grpc_error

    df = cloud_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(SessionError, match="Failed to start execution. Please file a ticket with Typedef support."):
        df.count()

    # Test get execution result returns an error
    mock_engine_service.StartExecution = MockEngineService().StartExecution
    mock_engine_service.GetExecutionResult = mock_grpc_error
    df = cloud_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(SessionError, match="Failed to get result for execution"):
        df.count()

    # Test collect execution returns a gRPC error
    with patch("pyarrow.flight.connect", mock_arrow_grpc_error):
        df = cloud_session.create_dataframe({"a": [1, 2, 3]})
        with pytest.raises(SessionError, match="Failed while connecting to arrow IPC. Please file a ticket with Typedef support."):
            df.to_polars()

    # Test infer schema returns a gRPC error
    mock_engine_service.InferSchema = mock_grpc_error
    with pytest.raises(ExecutionError, match="Failed to infer schema from csv files."):
        cloud_session._session_state.execution.infer_schema_from_csv(
            paths=["s3://test-bucket/test-file-list.csv"],
        )
