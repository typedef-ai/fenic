"""TypeSafe model provider implementation."""

import logging

from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)
_VALIDATION_TIMEOUT = 10

_IMPORT_ERROR_MESSAGE = (
    "To use TypeSafe models, please install the required dependencies by running: "
    "pip install fenic[typesafe]"
)


def _import_sdk():
    try:
        from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(_IMPORT_ERROR_MESSAGE) from err
    return TypeSafeClient, AsyncTypeSafeClient


class TypeSafeModelProvider(ModelProviderClass):
    """TypeSafe implementation of ModelProvider.

    Reads the API key from ``TYPESAFE_API_KEY`` via the SDK's own environment handling;
    fenic never touches the value.
    """

    def __init__(self, base_url: str | None = None):
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "typesafe"

    def create_client(self):
        """Create a synchronous TypeSafe client instance."""
        sync_client, _ = _import_sdk()
        from typesafe_sdk import RetryPolicy

        return sync_client(
            base_url=self._base_url,
            retry=RetryPolicy(max_retries=0),
            timeout=MAX_MODEL_CLIENT_TIMEOUT,
        )

    def create_aio_client(self):
        """Create an asynchronous TypeSafe client instance."""
        return self._create_aio_client(timeout=MAX_MODEL_CLIENT_TIMEOUT)

    def _create_aio_client(self, timeout: int):
        """Create an asynchronous SDK client with an explicit transport timeout."""
        _, async_client = _import_sdk()
        from typesafe_sdk import RetryPolicy

        # The shared fenic scheduler owns retries and counts each attempt.
        return async_client(
            base_url=self._base_url,
            retry=RetryPolicy(max_retries=0),
            timeout=timeout,
        )

    async def validate_api_key(self) -> None:
        """Validate the TypeSafe API key by listing models."""
        client = self._create_aio_client(timeout=_VALIDATION_TIMEOUT)
        try:
            _ = await client.models.list()
        finally:
            await client.aclose()
        logger.debug("TypeSafe API key validation successful")
