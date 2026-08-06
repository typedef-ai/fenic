from fenic._inference.language_model import LanguageModel
from fenic._inference.model_client import ModelClient
from fenic._inference.rate_limit_strategy import TokenEstimate, UnifiedTokenRateLimitStrategy
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core.metrics import LMMetrics


class _StubProviderClass:
    _base_url = None


class _AccountingCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Fake provider whose successful calls deliberately record LMMetrics."""

    def __init__(self):
        super().__init__(
            model="gpt-4.1-nano",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_StubProviderClass(),
            rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=1_000, tpm=1_000_000),
            token_counter=TiktokenTokenCounter(model_name="gpt-4.1-nano"),
        )
        self._metrics = LMMetrics()

    async def make_single_request(self, request):
        self._metrics.num_uncached_input_tokens += 7
        self._metrics.num_output_tokens += 3
        self._metrics.num_requests += 1
        self._metrics.cost += 0.0000019
        return FenicCompletionsResponse(
            completion="accounted",
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10,
            ),
        )

    def estimate_tokens_for_request(self, request):
        return TokenEstimate(input_tokens=7, output_tokens=3)

    def get_metrics(self):
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(self, request):
        return request.max_completion_tokens


def _message(label: str) -> LMRequestMessages:
    return LMRequestMessages(system="system", examples=[], user=label)


def _assert_metrics(model: LanguageModel, requests: int) -> None:
    metrics = model.get_metrics()
    assert metrics.num_requests == requests
    assert metrics.num_uncached_input_tokens == requests * 7
    assert metrics.num_output_tokens == requests * 3
    assert metrics.cost == requests * 0.0000019


def test_legacy_and_iterator_completion_paths_both_accumulate_metrics():
    legacy_client = _AccountingCompletionsClient()
    iterator_client = _AccountingCompletionsClient()
    legacy_model = LanguageModel(legacy_client)
    iterator_model = LanguageModel(iterator_client)
    try:
        legacy_model.get_completions(
            [_message("legacy-a"), _message("legacy-b")],
            max_tokens=16,
            operation_name="legacy-test",
        )
        list(
            iterator_model.iter_completions(
                iter([_message("iterator-a"), _message("iterator-b")]),
                max_tokens=16,
                operation_name="iterator-test",
                batch_size=1,
            )
        )

        _assert_metrics(legacy_model, requests=2)
        _assert_metrics(iterator_model, requests=2)
    finally:
        legacy_client.shutdown()
        iterator_client.shutdown()


def test_reading_a_different_model_client_returns_its_own_zero_metrics():
    active_client = _AccountingCompletionsClient()
    unrelated_client = _AccountingCompletionsClient()
    active_model = LanguageModel(active_client)
    unrelated_model = LanguageModel(unrelated_client)
    try:
        list(
            active_model.iter_completions(
                iter([_message("active")]),
                max_tokens=16,
                operation_name="iterator-test",
            )
        )

        _assert_metrics(active_model, requests=1)
        _assert_metrics(unrelated_model, requests=0)
    finally:
        active_client.shutdown()
        unrelated_client.shutdown()
