import hashlib
import os
from typing import List, Optional, Union

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import ContentEmbedding

from fenic._inference.google.google_profile_manager import (
    GoogleEmbeddingsProfileManager,
)
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicEmbeddingsRequest
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._resolved_session_config import ResolvedGoogleModelProfile
from fenic.core.metrics import RMMetrics


class GoogleBatchEmbeddingsClient(ModelClient[FenicEmbeddingsRequest, List[float]]):
    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        model_provider: ModelProvider,
        model: str = "gemini-2.0-flash-lite",
        queue_size: int = 100,
        max_backoffs: int = 10,
        profiles: Optional[dict[str, ResolvedGoogleModelProfile]] = None,
        default_profile_name: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            model_provider=model_provider,
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(model_name=model),
        )
        self.model = model
        # Native gen-ai client. Passing `vertexai=True` automatically routes traffic
        # through Vertex-AI if the environment is configured for it.
        if model_provider == ModelProvider.GOOGLE_DEVELOPER:
            if "GEMINI_API_KEY" in os.environ:
                self._base_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            else:
                self._base_client = genai.Client()
        else:
            self._base_client = genai.Client(vertexai=True)
        self._client = self._base_client.aio
        self._model_parameters = model_catalog.get_embedding_model_parameters(
            model_provider, model
        )
        self._metrics = RMMetrics()
        self._profile_manager = GoogleEmbeddingsProfileManager(
            model_parameters=self._model_parameters,
            profiles=profiles,
            default_profile_name=default_profile_name,
        )

    async def make_single_request(
        self, request: FenicEmbeddingsRequest
    ) -> Union[None, List[float], TransientException, FatalException]:
        try:
            additional_config = self._profile_manager.get_profile_by_name(request.model_profile)
            contents = request.doc
            response = await self._client.models.embed_content(
                model=self.model, contents=contents, config=additional_config.additional_embedding_config
            )
            if response.embeddings:
                embedding_data: ContentEmbedding = response.embeddings[0]
                if embedding_data.statistics:
                    total_tokens = embedding_data.statistics.token_count
                else:  # gemini only provides token count for embeddings when using the Vertex API
                    total_tokens = self.token_counter.count_tokens(request.doc)
                self._metrics.num_input_tokens += total_tokens
                self._metrics.num_requests += 1
                self._metrics.cost += model_catalog.calculate_embedding_model_cost(
                    model_provider=self.model_provider,
                    model_name=self.model,
                    billable_inputs=total_tokens,
                )
                return embedding_data.values
            else:
                return FatalException(
                    ValueError(
                        f"[{self.model_provider}] No embeddings found in response"
                    )
                )
        except ServerError as e:
            # Treat quota/timeouts as transient (retryable)
            return TransientException(e)
        except ClientError as e:
            if e.code == 429:
                return TransientException(e)
            else:
                return FatalException(e)

    def get_request_key(self, request: FenicEmbeddingsRequest) -> str:
        return hashlib.sha256(request.doc.encode()).hexdigest()[:10]

    def estimate_tokens_for_request(self, request: FenicEmbeddingsRequest) -> TokenEstimate:
        return TokenEstimate(
            input_tokens=self.token_counter.count_tokens(request.doc), output_tokens=0
        )

    def _get_max_output_tokens(self, request: FenicEmbeddingsRequest) -> int:
        return 0

    def reset_metrics(self):
        self._metrics = RMMetrics()

    def get_metrics(self) -> RMMetrics:
        return self._metrics
