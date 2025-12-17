"""Session state management for query execution."""

import logging
import uuid
from decimal import ROUND_DOWN, Decimal
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
from fenic._inference.cache.protocol import LLMResponseCache
from fenic._inference.cache.sqlite_cache import SQLiteLLMCache
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._resolved_session_config import (
    ResolvedSemanticConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import SessionError
from fenic.core.metrics import LMMetrics, RMMetrics
from fenic.core.types.enums import CacheBackend

logger = logging.getLogger(__name__)


class LocalSessionState(BaseSessionState):
    """Maintains the state for a query session, including database connections and cached dataframes
    and indices.
    """

    duckdb_conn: duckdb.DuckDBPyConnection
    s3_session: Optional[boto3.Session] = None
    _model_registry: SessionModelRegistry
    _llm_cache: Optional[LLMResponseCache] = None
    _models_shutdown: bool = False

    def __init__(self, config: ResolvedSessionConfig):
        super().__init__(config)
        self.app_name = config.app_name
        self.session_id = str(uuid.uuid4())

        base_path = Path(config.db_path) if config.db_path else Path(".")
        db_path = base_path / f"{config.app_name}.duckdb"
        intermediate_db_path = base_path / f"_{config.app_name}_tmp_dfs.duckdb"

        self.duckdb_conn = (
            fenic._backends.local.utils.io_utils.configure_duckdb_conn_for_path(db_path)
        )

        # Initialize LLM response cache if configured
        self._llm_cache = self._initialize_cache(config, base_path)

        self._model_registry = self._configure_models(config.semantic, self._llm_cache)
        self.intermediate_df_client = TempDFDBClient(intermediate_db_path)
        self.s3_session = boto3.Session()

    def _initialize_cache(self, config: ResolvedSessionConfig, base_path: Path):
        """Initialize LLM response cache if enabled in semantic config.

        Args:
            config: Resolved session configuration.
            base_path: Base directory for database files.

        Returns:
            Cache instance if enabled, None otherwise.
        """
        cache_config = config.semantic.llm_response_cache if config.semantic else None

        if not cache_config:
            return None

        try:

            # Compute cache path alongside DuckDB with _ prefix (system database)
            cache_db_path = base_path / f"_{config.app_name}_llm_cache.db"

            # Ensure parent directory exists
            cache_db_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Initializing LLM cache at {cache_db_path} with TTL {cache_config.ttl}"
            )
            if cache_config.backend == CacheBackend.LOCAL:
                return SQLiteLLMCache(
                    db_path=str(cache_db_path),
                    ttl_seconds=cache_config.ttl_seconds,
                    max_size_mb=cache_config.max_size_mb,
                    namespace=cache_config.namespace,
                )
        except Exception as e:
            raise SessionError(f"Failed to initialize LLM cache: {e}") from e

    def _configure_models(
        self, semantic_config: ResolvedSemanticConfig, llm_cache
    ) -> SessionModelRegistry:
        """Configure semantic settings on the session.

        Args:
            semantic_config: Semantic configuration.
            llm_cache: Optional LLM response cache instance.
        """
        return SessionModelRegistry(semantic_config, cache=llm_cache)

    def get_language_model(
        self, alias: Optional[ResolvedModelAlias] = None
    ) -> LanguageModel:
        return self._model_registry.get_language_model(alias)

    def get_embedding_model(
        self, alias: Optional[ResolvedModelAlias] = None
    ) -> EmbeddingModel:
        return self._model_registry.get_embedding_model(alias)

    def get_model_metrics(self) -> tuple[LMMetrics, RMMetrics]:
        """Get the language model and retriever model metrics."""
        return (
            self._model_registry.get_language_model_metrics(),
            self._model_registry.get_embedding_model_metrics(),
        )

    def reset_model_metrics(self):
        """Reset the language model and retriever model metrics."""
        self._model_registry.reset_language_model_metrics()
        self._model_registry.reset_embedding_model_metrics()

    def shutdown_models(self):
        """Shutdown all registered language and embedding models.

        This method is idempotent and can be called multiple times safely.
        """
        if self._models_shutdown:
            return

        self._model_registry.shutdown_models()
        self._models_shutdown = True

    @property
    def execution(self) -> LocalExecution:
        """Get the execution object."""
        return LocalExecution(self)

    @cached_property
    def catalog(self) -> LocalCatalog:
        """Get the catalog object."""
        return LocalCatalog(self.duckdb_conn)

    def stop(self, skip_usage_summary: bool):
        """Clean up the session state.

        Shutdown order (CRITICAL):
        1. Shutdown models (stop accepting requests, finish in-flight)
        2. Close cache (flush WAL, close connections)
        3. Print usage summary (if not skipped)
        4. Remove from session manager
        """
        # STEP 1: Shutdown models FIRST to prevent them from trying to cache after cache is closed
        logger.debug("Shutting down models...")
        self.shutdown_models()

        # STEP 2: THEN close cache (all model threads have finished)
        if self._llm_cache is not None:
            try:
                self._llm_cache.close()
                logger.debug("LLM cache closed and WAL flushed")
            except Exception as e:
                logger.error(f"Failed to close LLM cache: {e}")

        # STEP 3: Print session usage summary
        if not skip_usage_summary:
            self._print_session_usage_summary()

        # STEP 4: Remove from session manager
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
                print(
                    f"  Total execution time: {costs['total_execution_time_ms']:.2f}ms"
                )
                print(f"  Total rows processed: {costs['total_output_rows']:,}")
                print(
                    f"  Total language model cost: ${_format_float(costs['total_lm_cost'])}"
                )
                if costs["total_lm_requests"] > 0:
                    print(
                        f"  Total language model requests: {costs['total_lm_requests']}"
                    )
                    print(
                        f"  Total language model tokens: {costs['total_lm_uncached_input_tokens']:,} input tokens, {costs['total_lm_cached_input_tokens']:,} cached input tokens, {costs['total_lm_output_tokens']:,} output tokens"
                    )
                print(
                    f"  Total embedding model cost: ${_format_float(costs['total_rm_cost'])}"
                )
                if costs["total_rm_requests"] > 0:
                    print(
                        f"  Total embedding model requests: {costs['total_rm_requests']}"
                    )
                    print(
                        f"  Total embedding model tokens: {costs['total_rm_input_tokens']:,} input tokens"
                    )
                total_cost = costs["total_lm_cost"] + costs["total_rm_cost"]
                print(f"  Total cost: ${_format_float(total_cost)}")
        except Exception as e:
            # Don't fail session stop if metrics summary fails
            logger.warning(f"Failed to print session usage summary: {e}")


# Utility functions


def _format_float(value: float) -> str:
    """Format float up to 6 decimal places, but strip trailing zeros. Always keep at least 2 decimals."""
    d = Decimal(value).quantize(
        Decimal("0.000001"), rounding=ROUND_DOWN
    )  # 6 decimals max
    s = format(d.normalize(), "f")  # remove exponent notation
    integer_part, _, decimal_part = s.partition(".")
    # Remove trailing zeros from decimals, then ensure at least 2 digits
    decimal_part = (decimal_part.rstrip("0") or "0").ljust(2, "0")
    return f"{integer_part}.{decimal_part}"
