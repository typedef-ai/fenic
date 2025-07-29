from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import grpc
from fenic_cloud.auth_client.client import authenticate_user, get_user_token
from fenic_cloud.hasura_client import (
    HasuraClient,
)
from fenic_cloud.hasura_client.generated_graphql_client import (
    Client as HasuraUserClient,
)
from fenic_cloud.protos.omnitype.v1.entrypoint_pb2_grpc import EntrypointServiceStub

from fenic import SessionConfig
from fenic._backends.cloud.session_state import CloudSessionState
from fenic._backends.cloud.settings import CloudSettings
from fenic.core.error import InternalError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class CloudSessionManagerConfigDependencies:
    settings: CloudSettings
    asyncio_loop: asyncio.AbstractEventLoop
    background_thread: threading.Thread
    entrypoint_channel: grpc.aio.Channel
    entrypoint_stub: EntrypointServiceStub
    hasura_client: HasuraClient

class ChannelHolder:
    """Holder for an object and an event to signal when the object is ready.
    This is used to avoid race conditions when creating objects in a background thread.
    """
    event: threading.Event
    future: asyncio.Future[grpc.aio.Channel]

    def __init__(self):
        self.event = threading.Event()
        self.future = None

    def set_channel_future(self, future: asyncio.Future[grpc.aio.Channel]):
        self.future = future
        self.event.set()

    def get_channel(self) -> grpc.aio.Channel:
        self.event.wait()
        return self.future.result()

class CloudSessionManager:
    """Cloud session manager for managing cloud sessions.
    This class is a singleton, and will only initialize once.
    """

    _class_instance: Optional["CloudSessionManager"] = None
    _lock = asyncio.Lock()
    _background_thread: Optional[threading.Thread] = None
    _live_session_states: Dict[str, CloudSessionState] = {}
    _asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
    _settings: Optional[CloudSettings] = None
    _entrypoint_stub: Optional[EntrypointServiceStub] = None
    _entrypoint_channel: Optional[grpc.aio.Channel] = None
    _hasura_client: Optional[HasuraClient] = None
    _user_id: Optional[str] = None
    _organization_id: Optional[str] = None
    initialized: bool = False
    hasura_user_client: Optional[HasuraUserClient] = None
    client_token: Optional[str] = None

    def __new__(cls, *args, **kwargs):
        if cls._class_instance is not None:
            return cls._class_instance
        manager = super().__new__(cls)
        manager.initialized = False
        cls._class_instance = manager
        return manager

    def configure(self, dependencies: CloudSessionManagerConfigDependencies):
        """Configure the cloud session manager."""
        if self.initialized:
            logger.warning(
                "CloudSessionManager already initialized, should not call configure again."
            )
            return
        self._settings = dependencies.settings
        self._asyncio_loop = dependencies.asyncio_loop
        self._background_thread = dependencies.background_thread
        self._entrypoint_channel = dependencies.entrypoint_channel
        self._entrypoint_stub = dependencies.entrypoint_stub
        self._hasura_client = dependencies.hasura_client

        if self._settings.client_token:
            self.client_token = self._settings.client_token
        self.initialized = True

    @staticmethod
    async def _create_secure_channel(
        entrypoint_uri: str,
        credentials: grpc.ChannelCredentials) -> grpc.aio.Channel:
        return grpc.aio.secure_channel(entrypoint_uri, credentials)

    @staticmethod
    def _start_loop(
        asyncio_loop: asyncio.AbstractEventLoop,
        entrypoint_uri: str,
        credentials: grpc.ChannelCredentials,
        channel_holder: ChannelHolder) -> None:
        asyncio.set_event_loop(asyncio_loop)
        result = asyncio.run_coroutine_threadsafe(
            CloudSessionManager._create_secure_channel(entrypoint_uri, credentials),
            asyncio_loop
        )
        channel_holder.set_channel_future(result)
        asyncio_loop.run_forever()


    @staticmethod
    def create_global_session_dependencies() -> (
        Optional[CloudSessionManagerConfigDependencies]
    ):
        """Create the dependencies for the cloud session manager."""
        if (
            CloudSessionManager._class_instance
            and CloudSessionManager._class_instance.initialized
        ):
            return None

        settings = CloudSettings()
        logger.debug(f"Cloud settings: {settings}")

        # Create the event loop and background thread
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)
        credentials = grpc.ssl_channel_credentials()
        channel_holder = ChannelHolder()

        # The secure channel will use the current event loop, if there is
        # already an event loop running, then the channel will use that event loop.
        # This will cause the calls to be scheduled on the wrong thread.
        background_thread = threading.Thread(
            target=CloudSessionManager._start_loop,
            args=(asyncio_loop, settings.entrypoint_uri, credentials, channel_holder),
            daemon=True
        )
        background_thread.start()
        entrypoint_channel = channel_holder.get_channel()
        logger.debug(f"Created secure gRPC channel to {settings.entrypoint_uri}")

        entrypoint_stub = EntrypointServiceStub(entrypoint_channel)
        logger.debug(f"Initialized entrypoint stub with channel: {entrypoint_channel}")
        hasura_client = HasuraClient(
            graphql_uri=settings.hasura_graphql_uri,
            graphql_ws_uri=settings.hasura_graphql_ws_uri,
        )
        return CloudSessionManagerConfigDependencies(
            settings=settings,
            asyncio_loop=asyncio_loop,
            background_thread=background_thread,
            entrypoint_channel=entrypoint_channel,
            entrypoint_stub=entrypoint_stub,
            hasura_client=hasura_client,
        )

    async def get_or_create_session_state(self, session_config: SessionConfig) -> CloudSessionState:
        """Get or create a cloud session."""
        if not self.initialized:
            raise InternalError(
                "CloudSessionManager not initialized. Call configure first."
            )

        async with self._lock:
            if session_config.app_name in self._live_session_states:
                session = self._live_session_states[session_config.app_name]
                self.check_session_active_and_refresh(session_config.app_name)
                return session

            # Create a new cloud session state
            logger.info("Starting cloud session")

            # Create the session state
            cloud_session_state = CloudSessionState(
                config=session_config._to_resolved_config(),
                unresolved_config=session_config,
                settings=self._settings,
                asyncio_loop=self._asyncio_loop,
                entrypoint_channel=self._entrypoint_channel,
                entrypoint_stub=self._entrypoint_stub,
            )
            # Configure the cloud session
            await cloud_session_state.configure_cloud_session()
            self._live_session_states[session_config.app_name] = cloud_session_state
            return cloud_session_state

    def check_session_active_and_refresh(self, session_id: str) -> bool:
        # TODO: check if session is active with entrypoint
        pass

    def authenticate_user_and_create_hasura_client(self):
        """Authenticate the user and save the user token
        If user token is new, create a new hasura user client.
        """
        if self.client_token:
            # TODO: check if token is valid and get a new one if not
            logger.debug(f"{self._settings.typedef_instance} Using saved user token")
            if self.hasura_user_client:
                return
        else:
            logger.info(
                f"{self._settings.typedef_instance} Authenticating user {self._settings.client_id} with secret size {len(self._settings.client_secret) if self._settings.client_secret else 0}"
            )
            logger.debug(f"Auth URI: {self._settings.auth_provider_uri}")
            client_token = get_user_token(
                self._settings.auth_provider_uri,
                self._settings.typedef_instance,
                self._settings.client_id,
                self._settings.client_secret,
            )
            self.client_token = client_token
            self._user_id, self._organization_id, _ = authenticate_user(
                typedef_instance=self._settings.typedef_instance,
                token=self.client_token,
            )

        logger.debug("Creating hasura user client")
        self.hasura_user_client = self._hasura_client.get_user_client(
            auth_token=self.client_token
        )

    def remove_session(self, app_name: str):
        """Remove a session from the manager, shuts down the asyncio loop if no sessions are left."""
        loop_to_shutdown = None
        thread_to_join = None

        # Create a coroutine to handle the async lock
        async def _remove_session():
            async with self._lock:
                self._live_session_states.pop(app_name)
                if len(self._live_session_states) == 0:
                    # close the entrypoint channel
                    await self._entrypoint_channel.close()
                    return self._asyncio_loop, self._background_thread
                return None, None

        # Run the async operation in the event loop
        future = asyncio.run_coroutine_threadsafe(_remove_session(), self._asyncio_loop)
        loop_to_shutdown, thread_to_join = future.result()

        if loop_to_shutdown:
            cancel_future = asyncio.run_coroutine_threadsafe(
                self._cancel_event_loop_tasks(loop_to_shutdown), loop_to_shutdown
            )
            cancel_future.result()
            loop_to_shutdown.call_soon_threadsafe(loop_to_shutdown.stop)
            if thread_to_join and thread_to_join.is_alive():
                thread_to_join.join()
            loop_to_shutdown.close()

    async def _cancel_event_loop_tasks(self, loop: asyncio.AbstractEventLoop):
        """Cancels all pending tasks in the given asyncio event loop, except the current task.
        Returns immediately after cancelling tasks, doesn't wait for them to complete.

        Args:
            loop: The event loop to cancel tasks for.
        """
        logger.debug(f"Cancelling tasks in event loop: {loop}")
        asyncio.set_event_loop(loop)
        tasks = [
            t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)
        ]

        if not tasks:
            logger.debug("No tasks to cancel")
            return

        logger.debug(f"Cancelling {len(tasks)} tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelled task: {task!r}")

        # Give tasks a short time to respond to cancellation
        try:
            # Wait for a short time for tasks to cancel
            await asyncio.wait(tasks, timeout=1.0)
        except asyncio.CancelledError:
            logger.debug("Task cancellation was interrupted")
        except Exception as e:
            logger.debug(f"Error during task cancellation: {e}")

        # Report final status
        completed = sum(1 for t in tasks if t.done())
        logger.debug(f"Cancelled tasks status: {completed}/{len(tasks)} completed")
