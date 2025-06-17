from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from fenic._inference.model_catalog import (
    model_catalog,
)
from fenic._inference.model_client import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ModelClient,
)
from fenic._inference.token_counter import Tokenizable
from fenic._inference.types import LMRequestMessages
from fenic.core.error import ConfigurationError
from fenic.core.metrics import LMMetrics


@dataclass
class InferenceConfiguration:
    max_output_tokens: int
    temperature: float
    top_logprobs: Optional[int] = None
    response_format: Optional[type[BaseModel]] = None

class LanguageModel:
    def __init__(self, client: ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
        self.provider = client.model_provider
        self.model = client.model
        self.model_parameters = model_catalog.get_completion_model_parameters(self.provider, self.model)
        if self.model_parameters is None:
            raise ConfigurationError(model_catalog.generate_unsupported_completion_model_error_message(self.provider, self.model))
        # TPM might limit us before being limited by the actual context window length.
        self.max_context_window_length =  min(client.context_tokens_per_minute, self.model_parameters.context_window_length)
        self.client = client

    def get_completions(
        self,
        messages: list[LMRequestMessages],
        max_tokens: int,
        temperature: float = 0,
        response_format: Optional[type[BaseModel]] = None,
        top_logprobs: Optional[int] = None,
        operation_name: Optional[str] = None,
    ) -> list[Optional[FenicCompletionsResponse]]:
        # Create batch requests
        requests = []
        for message_list in messages:
            # if there are no messages, set the request as None, so it can be skipped.
            if not message_list:
                requests.append(None)
                continue

            request = FenicCompletionsRequest(
                messages=message_list,
                max_completion_tokens=max_tokens,
                top_logprobs=top_logprobs,
                structured_output=response_format,
                temperature=temperature,
            )
            requests.append(request)

        # Process batch requests
        return self.client.make_batch_requests(
            requests,
            operation_name=operation_name,
        )

    def count_tokens(self, messages: Tokenizable) -> int:
        return self.client.count_tokens(messages)


    def reset_metrics(self):
        self.client.reset_metrics()

    def get_metrics(self) -> LMMetrics:
        return self.client.get_metrics()
