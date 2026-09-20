import json
import logging
from dataclasses import dataclass
from typing import Optional

from fenic._inference.model_client import (
    ModelClient,
)
from fenic._inference.token_counter import Tokenizable
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import (
    model_catalog,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core.error import ConfigurationError
from fenic.core.metrics import LMMetrics
from fenic.core.types.judge import JudgeQuestion, validate_questions

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfiguration:
    # If max_output_tokens is not provided, model_client will add a guardrail based on the estimated output tokens.
    max_output_tokens: Optional[int]
    temperature: float
    top_logprobs: Optional[int] = None
    response_format: Optional[ResolvedResponseFormat] = None  # Resolved JSON schema
    model_profile: Optional[str] = None
    request_timeout: Optional[float] = None  # Timeout in seconds for a single LLM request

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
        response_format: Optional[ResolvedResponseFormat] = None,
        top_logprobs: Optional[int] = None,
        model_profile: Optional[str] = None,
        operation_name: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> list[Optional[FenicCompletionsResponse]]:
        # Create batch requests
        requests = []
        # Check model specific requirements for request params.
        temperature_param = temperature if self.model_parameters.supports_custom_temperature else None
        if temperature and not temperature_param:
            logger.warning(f"Model {self.model} does not support custom temperature.  Ignoring temperature parameter.")

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
                temperature=temperature_param,
                model_profile=model_profile,
            )
            requests.append(request)

        # Process batch requests
        return self.client.make_batch_requests(
            requests,
            operation_name=operation_name,
            request_timeout=request_timeout,
        )

    def get_judgments(
        self,
        states: list[Optional[str]],
        questions: tuple[JudgeQuestion, ...],
        model_profile: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> list[Optional[FenicCompletionsResponse]]:
        """Evaluate typed questions using the shared scheduler, cache, and metrics."""
        if not self.model_parameters.supports_judge:
            raise ConfigurationError("This provider does not support semantic.judge")
        questions = validate_questions(questions)
        requests = []
        owners = []
        failed = set()
        for index, state in enumerate(states):
            if state is None:
                failed.add(index)
                continue
            try:
                groups = self.client.judge_partitions(state, questions)
            except ValueError:
                failed.add(index)
                continue
            for group in groups:
                owners.append(index)
                requests.append(
                    FenicCompletionsRequest(
                        messages=LMRequestMessages(system="", examples=[], user=state),
                        max_completion_tokens=None,
                        top_logprobs=None,
                        structured_output=None,
                        temperature=None,
                        model_profile=model_profile,
                        judge_questions=group,
                    )
                )
        responses = self.client.make_batch_requests(
            requests, operation_name="semantic.judge", request_timeout=request_timeout
        )
        merged = [{} for _ in states]
        for owner, response in zip(owners, responses, strict=True):
            decoded = json.loads(response.completion) if response is not None else None
            if decoded is None:
                failed.add(owner)
            else:
                merged[owner].update(decoded)
        # Per-request usage has already settled in the scheduler; do not settle again.
        return [
            None
            if index in failed
            else FenicCompletionsResponse(json.dumps(value), None)
            for index, value in enumerate(merged)
        ]

    def count_tokens(self, messages: Tokenizable) -> int:
        return self.client.count_tokens(messages)


    def reset_metrics(self):
        self.client.reset_metrics()

    def get_metrics(self) -> LMMetrics:
        return self.client.get_metrics()
