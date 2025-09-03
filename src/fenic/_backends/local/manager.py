"""Session manager singleton for managing Session instances."""
from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Dict

from fenic._backends.local.session_state import LocalSessionState
from fenic._constants import INDEX_DIR, VECTOR_INDEX_DIR
from fenic.core._resolved_session_config import ResolvedSessionConfig

logger = logging.getLogger(__name__)


class LocalSessionManager:
    """Singleton class responsible for managing LocalSessionState instances.
    Ensures only one LocalSessionState exists per app name.
    """

    _instance = None
    _sessions_lock = threading.Lock()

    # Maps app name to SessionState
    _live_session_states: Dict[str, LocalSessionState]

    def __new__(cls):
        """Ensure singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(LocalSessionManager, cls).__new__(cls)
            cls._instance._live_session_states = {}
        return cls._instance

    def check_session_active(self, session_state: LocalSessionState) -> bool:
        """Check if a specific SessionState instance is still active."""
        with self._sessions_lock:
            app_name = session_state.app_name
            return app_name in self._live_session_states and self._live_session_states[app_name] is session_state

    def remove_session(self, app_name: str) -> None:
        """Remove a session state by app name from the SessionManager.

        Args:
            app_name: The name of the application to remove
        """
        with self._sessions_lock:
            if app_name not in self._live_session_states:
                logger.info(f"Redundant stop request: session.stop() invoked for {app_name}, which is already stopped")
                return
            session_state: LocalSessionState = self._live_session_states[
                app_name
            ]

            session_state.intermediate_df_client.cleanup()
            session_state.duckdb_conn.close()

            session_state.shutdown_models()

            # Remove LanceDB index directory
            if os.path.exists(VECTOR_INDEX_DIR):
                try:
                    shutil.rmtree(VECTOR_INDEX_DIR)
                    shutil.rmtree(INDEX_DIR)
                except Exception as e:
                    logger.warning(
                        f"Failed to cleanup intermediate semantic search data: {e}"
                    )

            del self._live_session_states[app_name]


    def get_or_create_session_state(
        self,
        config: ResolvedSessionConfig,
    ) -> LocalSessionState:
        """Get an existing SessionState or create a new one with appropriate clients.

        Args:
            app_name: The name of the application
            db_path: Path to the database
            semantic_config: Semantic configuration options

        Returns:
            A Session instance

        Raises:
            RuntimeError: If client or session creation fails
        """
        with self._sessions_lock:
            app_name = config.app_name
            if app_name in self._live_session_states:
                logger.info(
                    "Session already exists for this app name. Returning existing session."
                )
                session_state = self._live_session_states[app_name]
            else:
                # Create and store the session
                session_state = LocalSessionState(config)
                self._live_session_states[app_name] = session_state
            logger.info(f"Session ID: {session_state.session_id}")
            return session_state

    def get_existing_session_state(self, app_name: str) -> LocalSessionState:
        """Get a Session instance by app name."""
        with self._sessions_lock:
            if app_name not in self._live_session_states:
                raise ValueError(f"No session found for app name: {app_name}")
            return self._live_session_states[app_name]
