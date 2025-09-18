"""Session state management for query execution."""

import logging
import uuid
from functools import cached_property
from pathlib import Path
from typing import Optional

import boto3
import duckdb

import fenic._backends.local.utils.io_utils
from fenic._backends.local.catalog import LocalCatalog
from fenic._backends.local.execution import LocalExecution
from fenic._backends.local.model_registry import SessionModelRegistry
from fenic._backends.local.temp_df_db_client import TempDFDBClient
from fenic._inference import EmbeddingModel, LanguageModel
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._resolved_session_config import (
    ResolvedSemanticConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import SessionError
from fenic.core.metrics import LMMetrics, RMMetrics

logger = logging.getLogger(__name__)


class LocalSessionState(BaseSessionState):
    """Maintains the state for a query session, including database connections and cached dataframes
    and indices.
    """

    duckdb_conn: duckdb.DuckDBPyConnection
    s3_session: Optional[boto3.Session] = None
    _model_registry: SessionModelRegistry

    def __init__(self, config: ResolvedSessionConfig):
        super().__init__(config)
        self.app_name = config.app_name
        self.session_id = str(uuid.uuid4())

        base_path = Path(config.db_path) if config.db_path else Path(".")
        db_path = base_path / f"{config.app_name}.duckdb"
        intermediate_db_path = base_path / f"_{config.app_name}_tmp_dfs.duckdb"

        self.duckdb_conn = fenic._backends.local.utils.io_utils.configure_duckdb_conn_for_path(db_path)
        self._model_registry = self._configure_models(config.semantic)
        self.intermediate_df_client = TempDFDBClient(intermediate_db_path)
        self.s3_session = boto3.Session()

    def _configure_models(
            self, semantic_config: ResolvedSemanticConfig
    ) -> SessionModelRegistry:
        """Configure semantic settings on the session.

        Args:
            semantic_config: Semantic configuration
        """
        return SessionModelRegistry(semantic_config)

    def get_language_model(self, alias: Optional[ResolvedModelAlias] = None) -> LanguageModel:
        return self._model_registry.get_language_model(alias)

    def get_embedding_model(self, alias: Optional[ResolvedModelAlias] = None) -> EmbeddingModel:
        return self._model_registry.get_embedding_model(alias)

    def get_model_metrics(self) -> tuple[LMMetrics, RMMetrics]:
        """Get the language model and retriever model metrics."""
        return self._model_registry.get_language_model_metrics(), self._model_registry.get_embedding_model_metrics()

    def reset_model_metrics(self):
        """Reset the language model and retriever model metrics."""
        self._model_registry.reset_language_model_metrics()
        self._model_registry.reset_embedding_model_metrics()

    def shutdown_models(self):
        """Shutdown all registered language and embedding models."""
        self._model_registry.shutdown_models()

    @property
    def execution(self) -> LocalExecution:
        """Get the execution object."""
        return LocalExecution(self)

    @cached_property
    def catalog(self) -> LocalCatalog:
        """Get the catalog object."""
        return LocalCatalog(self.duckdb_conn)

    def stop(self):
        """Clean up the session state."""
        # Print session usage summary before stopping
        self._print_session_usage_summary()

        from fenic._backends.local.manager import LocalSessionManager

        LocalSessionManager().remove_session(self.app_name)

    def _check_active(self):
        """Check if the session is active, raise an error if it's stopped.

        Raises:
            SessionError: If the session has been stopped
        """
        from fenic._backends.local.manager import LocalSessionManager

        if not LocalSessionManager().check_session_active(self):
            raise SessionError(
                f"This session '{self.app_name}' has been stopped. Create a new session using Session.get_or_create()."
            )

    def _print_session_usage_summary(self):
        """Print total usage summary for this session."""
        try:
            costs = self.catalog.get_metrics_for_session(self.session_id)
            if costs["query_count"] > 0:
                print("\nSession Usage Summary:")
                print(f"  App Name: {self.app_name}")
                print(f"  Session ID: {self.session_id}")
                print(f"  Total queries executed: {costs['query_count']}")
                print(f"  Total execution time: {costs['total_execution_time_ms']:.2f}ms")
                print(f"  Total rows processed: {costs['total_output_rows']:,}")
                print(f"  Total language model cost: ${costs['total_lm_cost']:.6f}")
                print(f"  Total embedding model cost: ${costs['total_rm_cost']:.6f}")
                total_cost = costs['total_lm_cost'] + costs['total_rm_cost']
                print(f"  Total cost: ${total_cost:.6f}")
        except Exception as e:
            # Don't fail session stop if metrics summary fails
            logger.warning(f"Failed to print session usage summary: {e}")


