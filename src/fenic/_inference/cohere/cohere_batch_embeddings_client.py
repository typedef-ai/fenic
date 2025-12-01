import logging
from typing import List, Optional, Union

import cohere

from fenic._inference.cohere.cohere_profile_manager import (
    CohereEmbeddingsProfileConfiguration,
    CohereEmbeddingsProfileManager,
)
from fenic._inference.cohere.cohere_provider import CohereModelProvider
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.profile_hash_mixin import ProfileHashMixin
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicEmbeddingsRequest
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._resolved_session_config import ResolvedCohereModelProfile
from fenic.core.metrics import RMMetrics

logger = logging.getLogger(__name__)


class CohereBatchEmbeddingsClient(
    ProfileHashMixin, ModelClient[FenicEmbeddingsRequest, List[float]]
):
    """Client for making batch requests to Cohere's embeddings API."""

    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        profile_configurations: Optional[dict[str, ResolvedCohereModelProfile]] = None,
        default_profile_name: Optional[str] = None,
    ):
        """Initialize the Cohere batch embeddings client.
        
        Args:
            rate_limit_strategy: Strategy for handling rate limits
            model: The model to use
            queue_size: Size of the request queue
            max_backoffs: Maximum number of backoff attempts
            preset_configurations: Dictionary of preset configurations
            default_preset_name: Default preset to use when none specified
        """
        super().__init__(
            model=model,
            model_provider=ModelProvider.COHERE,
            model_provider_class=CohereModelProvider(),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(model_name=model),
        )
        
        self._client = self.model_provider_class.create_aio_client()
        self.model = model
        
        self._model_parameters = model_catalog.get_embedding_model_parameters(
            ModelProvider.COHERE, model
        )
        self._metrics = RMMetrics()
        
        self._profile_manager = CohereEmbeddingsProfileManager(
            model_parameters=self._model_parameters,
            profile_configurations=profile_configurations,
            default_profile_name=default_profile_name,
        )



    async def make_single_request(
        self, request: FenicEmbeddingsRequest
    ) -> Union[None, List[float], TransientException, FatalException]:
        """Make a single request to the Cohere embeddings API.
        
        Args:
            request: The embedding request to process
            
        Returns:
            List of embedding floats, or an exception wrapper
        """
        try:
            profile_config = self._profile_manager.get_profile_by_name(request.model_profile)
            
            # Prepare the request parameters
            embed_params = {
                "texts": [request.doc],
                "model": self.model,
                "input_type": profile_config.input_type,
                "embedding_types": ["float"], # We only support float embeddings
            }
            
            # Add output dimensionality if specified
            if profile_config.output_dimensionality:
                embed_params["output_dimension"] = profile_config.output_dimensionality
            
            # Make the API call
            response = await self._client.embed(**embed_params)
            embedding_values = response.embeddings.float[0]

            # Count tokens and update metrics
            if hasattr(response, 'meta') and hasattr(response.meta, 'billed_units'):
                # Use Cohere's billed token count if available
                total_tokens = response.meta.billed_units.input_tokens
            else:
                # Fall back to our token counter
                total_tokens = self.token_counter.count_tokens(request.doc)

            self._metrics.num_input_tokens += total_tokens
            self._metrics.num_requests += 1
            self._metrics.cost += model_catalog.calculate_embedding_model_cost(
                model_provider=ModelProvider.COHERE,
                model_name=self.model,
                billable_inputs=total_tokens,
            )

            return embedding_values

                
        except cohere.TooManyRequestsError as e:
            # Rate limit error - retryable
            return TransientException(e)
        except cohere.InternalServerError as e:
            # Server error - retryable
            return TransientException(e)
        except cohere.ServiceUnavailableError as e:
            # Service unavailable - retryable
            return TransientException(e)
        except (cohere.BadRequestError, cohere.UnauthorizedError, cohere.ForbiddenError) as e:
            # Client errors - not retryable
            return FatalException(e)
        except Exception as e:
            # Catch-all for other errors
            return TransientException(e)

    def estimate_tokens_for_request(self, request: FenicEmbeddingsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.
        
        Args:
            request: The request to estimate tokens for
            
        Returns:
            TokenEstimate: The estimated token usage
        """
        return TokenEstimate(
            input_tokens=self.token_counter.count_tokens(request.doc), 
            output_tokens=0
        )

    def _get_max_output_token_request_limit(self, request: FenicEmbeddingsRequest) -> int:
        """Get maximum output tokens (always 0 for embeddings).
        
        Returns:
            0 since embeddings don't produce text tokens
        """
        return 0

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._metrics = RMMetrics()

    def get_metrics(self) -> RMMetrics:
        """Get the current metrics.
        
        Returns:
            The current metrics
        """
        return self._metrics

    def _resolve_profile_for_hash(self, profile_name: Optional[str]) -> CohereEmbeddingsProfileConfiguration:
        return self._profile_manager.get_profile_by_name(profile_name)