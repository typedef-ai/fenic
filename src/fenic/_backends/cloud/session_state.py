import asyncio
import json
import logging
import os
from typing import Optional

import grpc
from fenic_cloud.protos.engine.v1.engine_pb2 import (
    ConfigSessionRequest,
)
from fenic_cloud.protos.engine.v1.engine_pb2_grpc import EngineServiceStub
from fenic_cloud.protos.omnitype.v1.common_pb2 import (
    EngineInstanceMetadata,
    InstanceSize,
)
from fenic_cloud.protos.omnitype.v1.entrypoint_pb2 import (
    GetOrCreateSessionRequest,
    RegisterAppRequest,
    TerminateSessionRequest,
)
from fenic_cloud.protos.omnitype.v1.entrypoint_pb2_grpc import EntrypointServiceStub

from fenic import SessionConfig
from fenic._backends.cloud.engine_config import CloudSessionConfig
from fenic._backends.cloud.execution import CloudExecution
from fenic._backends.cloud.settings import CloudSettings
from fenic.core._interfaces import BaseSessionState
from fenic.core._interfaces.catalog import BaseCatalog
from fenic.core._resolved_session_config import (
    CloudExecutorSize,
    ResolvedSessionConfig,
)
from fenic.core.error import ConfigurationError, SessionError

engine_instance_size_map = {
    CloudExecutorSize.SMALL: InstanceSize.INSTANCE_SIZE_S,
    CloudExecutorSize.MEDIUM: InstanceSize.INSTANCE_SIZE_M,
    CloudExecutorSize.LARGE: InstanceSize.INSTANCE_SIZE_L,
    CloudExecutorSize.XLARGE: InstanceSize.INSTANCE_SIZE_XL,
}

logger = logging.getLogger(__name__)

GRPC_MAX_MESSAGE_SIZE = 1024 * 1024 * 10  # 10MB

class CloudSessionState(BaseSessionState):
    """Maintains the state for a cloud session, including database connections and cached dataframes and indices."""
    app_name: str
    config: ResolvedSessionConfig
    unresolved_config: SessionConfig
    settings: CloudSettings
    asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
    entrypoint_channel: Optional[grpc.Channel] = None
    entrypoint_stub: Optional[EntrypointServiceStub] = None
    engine_channel: Optional[grpc.Channel] = None
    engine_stub: Optional[EngineServiceStub] = None
    arrow_ipc_channel: Optional[grpc.Channel] = None
    engine_uri: Optional[str] = None
    arrow_ipc_uri: Optional[str] = None
    arrow_ipc_uri_secure: bool = False
    app_uuid: Optional[str] = None
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    session_canonical_name: Optional[str] = None
    cloud_catalog: BaseCatalog = None

    def __init__(
        self,
        config: ResolvedSessionConfig,
        unresolved_config: SessionConfig,
        settings: CloudSettings,
        asyncio_loop: asyncio.AbstractEventLoop,
        entrypoint_channel: grpc.Channel,
        entrypoint_stub: EntrypointServiceStub,
    ):
        super().__init__(config)
        self.config = config
        self.app_name = config.app_name
        self.settings = settings
        self.asyncio_loop = asyncio_loop
        self.entrypoint_channel = entrypoint_channel
        self.entrypoint_stub = entrypoint_stub
        self.unresolved_config = unresolved_config

    async def configure_cloud_session(self):
        """Configure the cloud session."""
        self.authenticate_user_and_create_hasura_client()
        # Setup session engine service and connection to the engine
        await self._entrypoint_register_app()
        await self._entrypoint_get_or_create_session_engine()
        self._create_engine_channels()

        # Setup cloud interfaces and send session config to the engine service
        await self._send_config_session_request_to_engine()
        self._execution = CloudExecution(self, self.engine_stub)
        logger.debug("Initialized CloudSessionState")

    def stop(self, skip_usage_summary: bool):
        """Stop a cloud session and clean up resources.
        
        Note: There is currently no usage summary printed for cloud sessions, so skip_usage_summary is ignored."""
        from fenic._backends.cloud.manager import CloudSessionManager

        logger.info(f"Terminating session {self.app_name}:{self.session_uuid}")
        future = asyncio.run_coroutine_threadsafe(
            self._cleanup_session_state(), self.asyncio_loop
        )
        # block until termination is complete
        future.result()

        CloudSessionManager().remove_session(app_name=self.app_name)

    async def _cleanup_session_state(self):
        """Clean up resources for a cloud session."""
        await self.engine_channel.close()
        await self.arrow_ipc_channel.close()

        await self._entrypoint_terminate_session()

    @property
    def execution(self) -> CloudExecution:
        return self._execution

    @property
    def catalog(self):
        from fenic._backends.cloud.catalog import CloudCatalog
        from fenic._backends.cloud.manager import CloudSessionManager

        if self.cloud_catalog is None:
            self.cloud_catalog = CloudCatalog(
                ephemeral_catalog_id=self.ephemeral_catalog_id,
                asyncio_loop=self.asyncio_loop,
                cloud_session_manager=CloudSessionManager(),
            )
        return self.cloud_catalog

    # properties and methods referencing dynamic state managed by the CloudSessionManager
    @property
    def hasura_user_client(self):
        from fenic._backends.cloud.manager import CloudSessionManager

        return CloudSessionManager().hasura_user_client

    @property
    def client_token(self):
        from fenic._backends.cloud.manager import CloudSessionManager

        return CloudSessionManager().client_token

    def authenticate_user_and_create_hasura_client(self):
        from fenic._backends.cloud.manager import CloudSessionManager

        CloudSessionManager().authenticate_user_and_create_hasura_client()

    def get_engine_grpc_metadata(self):
        """Get metadata with authorization header for gRPC requests to the engine."""
        if not self.client_token:
            raise SessionError(
                "Could not authenticate user. "
                "Please make sure your TYPEDEF_CLIENT_ID and TYPEDEF_CLIENT_SECRET you received from Typedef "
                "are active and correct, and set in your environment. "
                "If authentication is still failing, please file a ticket with Typedef support."
            )
        return [("hasura-auth-token", self.client_token),
                ("session-id", self.session_uuid),
                ("content-type", "application/grpc"),
                ("endpoint", "actions")]

    def _get_entrypoint_grpc_metadata(self):
        """Get metadata with authorization header for gRPC requests."""
        if not self.client_token:
            raise SessionError(
                "Could not authenticate user. "
                "Please make sure your TYPEDEF_CLIENT_ID and TYPEDEF_CLIENT_SECRET you received from Typedef "
                "are active and correct, and set in your environment. "
                "If authentication is still failing, please file a ticket with Typedef support."
            )
        return [("authorization", f"Bearer {self.client_token}")]

    async def _entrypoint_register_app(self):
        """Register an application with the entrypoint service.

        Args:
            app_name: Name of the application to register

        Returns:
            str: The UUID of the registered application
        """
        if not self.app_name:
            raise ConfigurationError("App name is required to create a session")
        request = RegisterAppRequest(name=self.app_name)
        logger.debug(f"Registering app with request: {request}")
        response = await self.entrypoint_stub.RegisterApp(
            request, metadata=self._get_entrypoint_grpc_metadata()
        )
        logger.info(f"Registered app {self.app_name} with UUID {response.uuid}")
        self.app_uuid = response.uuid

    async def _entrypoint_get_or_create_session_engine(self):
        """Get or create a session for the application. Entrypoint service will trigger engine creation for the session.

        Args:
            app_uuid: UUID of the registered application
            environment_name: Optional environment name to create the session in

        Returns:
            tuple containing:
                - session_uuid: UUID of the session
                - app_name: Name of the application
                - canonical_name: Canonical name of the application
                - uris: Engine request URIs for the session
        """
        request = GetOrCreateSessionRequest(
            app_uuid=self.app_uuid,
            environment_name=(
                self.settings.typedef_environment
                if self.settings.typedef_environment
                else None
            ),
            engine_metadata=(
                EngineInstanceMetadata(
                    instance_size=(
                        engine_instance_size_map[self.config.cloud.size]
                        if self.config.cloud.size
                        else None
                    )
                )
                if self.config.cloud.size
                else None
            ),
        )

        logger.debug(f"Getting or creating session with request: {request}")
        response = await self.entrypoint_stub.GetOrCreateSession(
            request, metadata=self._get_entrypoint_grpc_metadata()
        )
        logger.debug(
            f"Created session {response.uuid} for app {self.app_name}:{self.app_uuid}.  response: {response}"
        )
        existing = response.existing

        self.session_uuid = response.uuid
        self.session_name = response.name
        self.session_canonical_name = response.canonical_name
        self.engine_uri = response.uris.remote_actions_uri
        self.arrow_ipc_uri = response.uris.remote_results_uri_prefix
        self.ephemeral_catalog_id = response.ephemeral_catalog_id
        logger.info(
            f"{'Found' if existing else 'Created'} Executor with session_id: {self.session_uuid}"
        )
        return

    def _create_engine_channels(self):
        """Create a channel to the engine grpc and arrow_ipc."""
        asyncio.set_event_loop(self.asyncio_loop)
        if self.settings.test_engine_grpc_url:
            logger.warn(
                f"Ignoring engine endpoint provided by entrypoint.  Using test engine URL: {self.settings.test_engine_grpc_url}"
            )
            self.engine_uri = self.settings.test_engine_grpc_url
            self.arrow_ipc_uri = self.settings.test_engine_arrow_url

        if "localhost" in self.engine_uri or "127.0.0.1" in self.engine_uri:
            secure = False
            self.engine_channel = grpc.aio.insecure_channel(self.engine_uri)
            self.arrow_ipc_channel = grpc.aio.insecure_channel(self.arrow_ipc_uri)
            self.arrow_ipc_uri_secure = False
        else:
            # For cloud engines, use secure channel
            secure = True
            credentials = grpc.ssl_channel_credentials()
            self.engine_uri = _add_port_to_cloud_uri(self.engine_uri)
            self.arrow_ipc_uri = _add_port_to_cloud_uri(self.arrow_ipc_uri)
            self.arrow_ipc_uri_secure = True
            self.engine_channel = grpc.aio.secure_channel(
                target=self.engine_uri,
                credentials=credentials,
                options=_get_grpc_retry_policy())
            self.arrow_ipc_channel = grpc.aio.secure_channel(target=self.arrow_ipc_uri, credentials=credentials)
        self.engine_stub = EngineServiceStub(self.engine_channel)
        logger.debug(
            f"Created {'secure' if secure else 'insecure'} gRPC channels to engine and arrow_ipc at {self.engine_uri} and {self.arrow_ipc_uri}"
        )

    async def _send_config_session_request_to_engine(self):
        """Configure the session with the engine service."""
        # copy the config and remove the cloud config
        config = CloudSessionConfig(self.unresolved_config)
        request = ConfigSessionRequest(session_config=config.serialize())

        logger.info(f"Sending config session request: {request}")
        try:
            await self.engine_stub.ConfigSession(
                request, metadata=self.get_engine_grpc_metadata()
            )
        except grpc.RpcError as e:
            if (
                e.code() == grpc.StatusCode.FAILED_PRECONDITION
                and "Session already configured" in e.details()
            ):
                logger.info("Session already configured, skipping")
            else:
                raise ConfigurationError(
                    "Failed to configure cloud session"
                ) from e

    async def _entrypoint_terminate_session(self) -> None:
        """Remove a session from the entrypoint service."""
        request = TerminateSessionRequest(uuid=self.session_uuid)
        await self.entrypoint_stub.TerminateSession(
            request, metadata=self._get_entrypoint_grpc_metadata()
        )
        logger.info(f"Terminated cloud session {self.session_uuid}")

def _add_port_to_cloud_uri(uri: str) -> str:
    """Add the port to the cloud URI.
       It assumes 443 as the default port for cloud URIs."""
    return uri + ":443"

def _get_grpc_retry_policy():
    # check out https://github.com/grpc/grpc/blob/master/examples/python/retry/async_retry_client.py for more details
    service_config_json = os.getenv(
        "TYPEDEF_CLOUD_GRPC_RETRY_POLICY",
        json.dumps(
            {
                "methodConfig": [
                    {
                        "name": [{}],
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "10s",
                            "maxBackoff": "120s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNKNOWN", "UNAVAILABLE"],
                        },
                    }
                ]
            }
        )
    )
    logger.debug(f"Using retry policy: {service_config_json}")

    options = [
        ("grpc.enable_retries", 1),
        ("grpc.service_config", service_config_json),
        ('grpc.max_send_message_length', os.getenv("TYPEDEF_CLOUD_GRPC_MAX_SEND_MESSAGE_LENGTH", GRPC_MAX_MESSAGE_SIZE)),
        ('grpc.max_receive_message_length', os.getenv("TYPEDEF_CLOUD_GRPC_MAX_RECEIVE_MESSAGE_LENGTH", GRPC_MAX_MESSAGE_SIZE)),
    ]

    return options
