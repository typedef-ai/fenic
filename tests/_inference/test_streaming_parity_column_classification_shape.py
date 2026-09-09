"""Provider-free streaming-parity guard at the TD-4768 column-classification shape.

The TD-4729 regression investigation measured a reported +12.6% streaming
regression at this workload shape and attributed it to provider service-time
drift, not to the streaming client (see
specs/td-flow/td-4729-colclass-regression/evidence.md). These tests pin that
conclusion as a permanent guard: the shape's defining client-side features —
many structured-output requests admitted through the bounded streaming
iterator under caps that never bind, completions settling out of order —
must stay at wall-clock parity with the batch path.

The latency model is deterministic and identical across arms, so the
"provider" term cancels and the assertion isolates client-side serialization.
Latencies are scaled down from the real cell (seconds) to keep CI fast; the
parity margin is generous because a real admission-serialization defect is
an order-of-magnitude signal (wall ~= sum of latencies instead of ~= max),
which is exactly what the negative-control test proves this harness detects.
"""

import asyncio
import json
import random
import time
from typing import List, Literal, Optional, Union

import polars as pl
from pydantic import BaseModel, Field

from fenic._backends.local.semantic_operators.map import Map
from fenic._inference.language_model import LanguageModel
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._inference.model_provider import ModelProviderClass
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core.metrics import LMMetrics

PARITY_ROWS = 48
CONTROL_ROWS = 16
LATENCY_SEED = 7
LATENCY_LO_S = 0.05
LATENCY_HI_S = 0.15
# rpm mirrors the production OpenAI client config that defined the real shape:
# look-ahead basis = max(batch_size, rpm) = 15_000, so the streaming caps sit
# at (1_000, 50_000) and never bind at these row counts, exactly as measured.
RPM = 15_000
TPM = 30_000_000


class _ClassificationItem(BaseModel):
    """Scaled stand-in for the narrowed column-classification wire contract."""

    request_index: int = Field(description="0-based index of the classified column")
    semantic_role: Literal["metric", "attribute", "timestamp"] = Field(
        description="What business concept this column represents"
    )
    is_pii: bool = Field(default=False, description="Whether the column is PII")


class _ClassificationBatch(BaseModel):
    items: List[_ClassificationItem] = Field(
        default_factory=list, description="One classification per requested column"
    )


def _response_json(row_index: int) -> str:
    return json.dumps(
        {
            "items": [
                {"request_index": row_index, "semantic_role": "attribute", "is_pii": False}
            ]
        }
    )


class _ParityProvider(ModelProviderClass):
    @property
    def name(self) -> str:
        return "parity-sim"

    def create_client(self):
        return object()

    def create_aio_client(self):
        return object()

    async def validate_api_key(self) -> None:
        return


class LatencyProfiledCompletionClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Real ModelClient over a simulated provider with deterministic per-row
    latency, keyed by the row marker so dispatch/retry order cannot change a
    row's latency."""

    def __init__(self, latencies: List[float]):
        super().__init__(
            model="gpt-4.1-nano",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_ParityProvider(),
            rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=RPM, tpm=TPM),
            token_counter=_ZeroTokenCounter(),
        )
        self._latencies = latencies
        self._metrics = LMMetrics()

    @staticmethod
    def _row_index(request: FenicCompletionsRequest) -> int:
        user = request.messages.user or ""
        return int(user[7 : user.index(";")])

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        row = self._row_index(request)
        await asyncio.sleep(self._latencies[row])
        return FenicCompletionsResponse(
            completion=_response_json(row),
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=100, completion_tokens=20, total_tokens=120
            ),
        )

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        return TokenEstimate(input_tokens=100, output_tokens=20)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> int:
        return request.max_completion_tokens or 0


class _ZeroTokenCounter:
    def count_tokens(self, messages, ignore_file: bool = False) -> int:
        return 0

    def count_file_input_tokens(self, messages) -> int:
        return 0

    def count_file_output_tokens(self, messages) -> int:
        return 0


def _latencies(n_rows: int) -> List[float]:
    rng = random.Random(LATENCY_SEED)
    return [rng.uniform(LATENCY_LO_S, LATENCY_HI_S) for _ in range(n_rows)]


def _run_arm(
    n_rows: int,
    *,
    stream: bool,
    slot_caps: Optional[tuple] = None,
) -> tuple:
    """Run one arm; returns (wall_seconds, outputs_as_list)."""
    latencies = _latencies(n_rows)
    client = LatencyProfiledCompletionClient(latencies)
    if slot_caps is not None:
        client._streaming_slot_caps = lambda basis: slot_caps
    model = LanguageModel(client)
    resolved_format = ResolvedResponseFormat.from_pydantic_model(
        _ClassificationBatch, generate_struct_type=True
    )
    prompts = [f"ROWIDX={i};classify this column batch" for i in range(n_rows)]
    operator = Map(
        input=pl.Series("prompts", prompts),
        jinja_template="{{ prompts }}",
        model=model,
        max_tokens=512,
        temperature=0,
        response_format=resolved_format,
    )
    previous = Map.stream_requests
    Map.stream_requests = stream
    try:
        start = time.monotonic()
        series = operator.execute()
        wall = time.monotonic() - start
    finally:
        Map.stream_requests = previous
        client.shutdown()
    return wall, series.to_list()


def test_streaming_stays_at_parity_with_batch_at_the_column_classification_shape():
    wall_batch, out_batch = _run_arm(PARITY_ROWS, stream=False)
    wall_stream, out_stream = _run_arm(PARITY_ROWS, stream=True)

    # Both arms return the same rows in input order.
    assert len(out_batch) == PARITY_ROWS
    assert out_stream == out_batch
    assert [row["items"][0]["request_index"] for row in out_stream] == list(
        range(PARITY_ROWS)
    )

    # Deterministic identical latencies: both walls ~= max latency + overhead.
    # A genuine admission-serialization defect turns the streaming wall into
    # ~= sum(latencies), an order of magnitude past this margin (proven
    # detectable by the control test below).
    assert wall_stream <= wall_batch * 1.5 + 0.75, (
        f"streaming wall {wall_stream:.3f}s exceeds parity margin against "
        f"batch wall {wall_batch:.3f}s"
    )


def test_parity_harness_detects_a_deliberately_serialized_admission_control():
    wall_batch, _ = _run_arm(CONTROL_ROWS, stream=False)
    wall_serialized, out_serialized = _run_arm(
        CONTROL_ROWS, stream=True, slot_caps=(1, 1)
    )

    # The control still returns correct, ordered results...
    assert [row["items"][0]["request_index"] for row in out_serialized] == list(
        range(CONTROL_ROWS)
    )
    # ...but its wall collapses to ~sum(latencies): the measurement detects it.
    assert wall_serialized >= wall_batch * 3, (
        f"serialized control wall {wall_serialized:.3f}s was not detected "
        f"against batch wall {wall_batch:.3f}s -- the parity harness lost "
        f"its ability to see admission serialization"
    )
