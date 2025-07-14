"""Batch token prediction for optimizing rate limiting in model clients."""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Protocol, TypeVar

import numpy as np

from fenic._inference.types import FenicCompletionsRequest, FenicCompletionsResponse

logger = logging.getLogger(__name__)

# Constants
MINIMUM_PREDICTION_CONFIDENCE = 0.5

# Type variables for generic protocol
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class DistributionShape(Enum):
    """Enumeration of distribution shapes for token prediction."""
    SYMMETRIC = "symmetric"
    RIGHT_SKEWED = "right_skewed"
    LEFT_SKEWED = "left_skewed"


@dataclass
class BatchTokenPredictions:
    """Predicted token usage for remaining requests in a batch."""
    expected_output_tokens: int  # Total output tokens (completion + thinking)
    sample_size: int
    confidence: float  # How reliable these predictions are (0.0-1.0)


class BatchTokenPredictor(Protocol[RequestT, ResponseT]):
    """Protocol for batch token prediction functionality.

    This protocol allows for batch token prediction between any request/response types.
    Implementers should provide type-specific logic for extracting token information
    and determining sampling criteria.
    """

    def should_use_sampling(self, requests: List[RequestT]) -> bool:
        """Determine if within-batch sampling should be used."""
        ...

    def get_batch_predictions(self, batch_id: str) -> BatchTokenPredictions:
        """Get predictions for a batch, with safe defaults."""
        ...

    def cleanup_batch_predictions(self, batch_id: str) -> None:
        """Clean up predictions when batch is complete."""
        ...

    def compute_batch_predictions(
            self,
            batch_id: str,
            sample_responses: List[ResponseT]
    ) -> None:
        """Extract token predictions from sample responses."""
        ...

    def predict_output_tokens(
            self,
            request: RequestT,
            batch_id: str,
            conservative_estimate_func: Callable[[RequestT], int]
    ) -> int:
        """Predict total output tokens using batch predictions if available."""
        ...

    def update_transient_exception_time(self, exception_time: float) -> None:
        """Update the time of the last transient exception for rate limiting checks."""
        ...


class CompletionsBatchTokenPredictor(BatchTokenPredictor[FenicCompletionsRequest, FenicCompletionsResponse]):
    """Concrete implementation of batch token prediction for Fenic completion requests/responses.

    This implementation is specifically designed for FenicCompletionsRequest/FenicCompletionsResponse
    and provides statistical analysis of token usage patterns to optimize rate limiting decisions.

    Implements the BatchTokenPredictorProtocol[FenicCompletionsRequest, FenicCompletionsResponse].
    """

    def __init__(self, enable_batch_sampling: bool = True):
        """Initialize the batch token predictor.

        Args:
            enable_batch_sampling: Whether to enable sampling-based predictions
        """
        self.enable_batch_sampling = enable_batch_sampling
        self._batch_predictions: Dict[str, BatchTokenPredictions] = {}
        self.last_transient_exception_time: float = 0

    def should_use_sampling(self, requests: List[FenicCompletionsRequest]) -> bool:
        """Determine if within-batch sampling should be used.

        Sampling is beneficial when:
        - We have enough requests to make sampling worthwhile (50+)
        - The feature is enabled
        - We're not currently rate-limited
        - These are completion requests (not embedding requests)

        Args:
            requests: List of completion requests to evaluate

        Returns:
            True if sampling should be used, False otherwise
        """
        return (
                len(requests) >= 50 and  # Need enough requests to make sampling worthwhile
                self.enable_batch_sampling and  # Feature flag
                not self._is_rate_limited()  # Skip if already hitting rate limits
        )

    def _is_rate_limited(self) -> bool:
        """Check if we're currently experiencing rate limiting.

        Uses a simple heuristic based on recent transient exceptions.

        Returns:
            True if we've had recent rate limit backoffs
        """
        return (time.time() - self.last_transient_exception_time) < 30  # Within last 30 seconds

    def _is_completion_model(self, requests: List[FenicCompletionsRequest]) -> bool:
        """Check if these are completion requests that produce output tokens.

        Args:
            requests: List of requests to check

        Returns:
            True if these are completion requests, False otherwise
        """
        if not requests:
            return False

        # All requests in the list should be FenicCompletionsRequest since we're type-specific now
        sample_request = next((req for req in requests if req is not None), None)
        if sample_request is None:
            return False

        return isinstance(sample_request, FenicCompletionsRequest)

    def get_batch_predictions(self, batch_id: str) -> BatchTokenPredictions:
        """Get predictions for a batch, with safe defaults.

        Args:
            batch_id: Identifier for the batch

        Returns:
            BatchTokenPredictions with current predictions or safe defaults
        """
        return self._batch_predictions.get(batch_id, BatchTokenPredictions(
            expected_output_tokens=0,
            sample_size=0,
            confidence=0.0
        ))

    def cleanup_batch_predictions(self, batch_id: str) -> None:
        """Clean up predictions when batch is complete.

        Args:
            batch_id: Identifier for the batch to clean up
        """
        self._batch_predictions.pop(batch_id, None)

    def _sample_statistics(self, sample_tokens: list[int]) -> tuple[int, float]:
        """Calculate statistics with method-specific confidence.

        Args:
            sample_tokens: List of token counts to analyze

        Returns:
            Tuple of (predicted_tokens, confidence)
        """
        if len(sample_tokens) < 2:
            return 0, 0.0

        tokens = np.array(sample_tokens)

        # Calculate shared statistics
        sample_median = np.median(tokens)
        sample_mean = np.mean(tokens)
        sample_stddev = np.std(tokens)
        sample_max = np.max(tokens)
        sample_min = np.min(tokens)

        # Detect distribution shape using quantile-based skewness
        distribution_shape = self._detect_distribution_shape(tokens)
        print(
            f"Sample statistics: median={sample_median:.2f}, mean={sample_mean:.2f}, stddev={sample_stddev:.2f}, max={sample_max:.2f}, min={sample_min:.2f}, distribution_shape={distribution_shape}")
        if distribution_shape == DistributionShape.SYMMETRIC:
            predicted_tokens, confidence = self._predict_symmetric(
                tokens, sample_mean, sample_stddev
            )
        else:  # Skewed distribution
            predicted_tokens, confidence = self._predict_skewed(
                tokens, distribution_shape, sample_median
            )

        return predicted_tokens, confidence

    def _detect_distribution_shape(self, tokens: np.ndarray) -> DistributionShape:
        """Detect if distribution is symmetric or skewed using quantile-based skewness."""
        q25, q50, q75 = np.percentile(tokens, [25, 50, 75])

        if q75 == q25:  # All values the same
            return DistributionShape.SYMMETRIC

        # Bowley's skewness coefficient (ranges from -1 to +1)
        quantile_skew = (q75 + q25 - 2 * q50) / (q75 - q25)

        # Adaptive threshold based on sample size
        # Smaller samples: more lenient (larger threshold)
        # Larger samples: more strict (smaller threshold)
        n = len(tokens)
        if n < 10:
            threshold = 0.2  # Very lenient for small samples
        elif n < 20:
            threshold = 0.15  # Moderate for medium samples
        else:
            threshold = 0.1  # Strict for large samples

        if abs(quantile_skew) < threshold:
            return DistributionShape.SYMMETRIC
        elif quantile_skew > 0:
            return DistributionShape.RIGHT_SKEWED
        else:
            return DistributionShape.LEFT_SKEWED

    def _predict_symmetric(self, tokens: np.ndarray, mean: float, std_dev: float) -> tuple[int, float]:
        """Prediction and confidence for symmetric distributions."""
        predicted = int(mean + std_dev)

        # Confidence based on coefficient of variation + sample size
        cv = std_dev / mean if mean > 0 else float('inf')
        cv_confidence = math.exp(-cv)

        # Sample size bonus (more samples = higher confidence for mean estimation)
        sample_size_factor = min(1.0, len(tokens) / 15.0)

        confidence = cv_confidence * (0.7 + 0.3 * sample_size_factor)
        return predicted, max(0.0, min(1.0, confidence))

    def _predict_skewed(self, tokens: np.ndarray, distribution_shape: DistributionShape, median: float) -> tuple[
        int, float]:
        """Prediction and confidence for skewed distributions."""
        # Calculate percentiles only when needed
        q25, q75, q90 = np.percentile(tokens, [25, 75, 90])

        if distribution_shape == DistributionShape.RIGHT_SKEWED:
            predicted = int(np.percentile(tokens, 85))
        else:  # LEFT_SKEWED
            predicted = int(q75)

        # Base confidence: how stable is the tail behavior
        if q90 > 0:
            tail_stability = q75 / q90  # Higher ratio = more stable tail
            base_confidence = min(0.8, tail_stability * 1.2)  # Cap at 0.8 for skewed data
        else:
            base_confidence = 0.6  # Moderate confidence when no tail data

        # Adjustments (small bonuses/penalties, not multipliers)

        # Spread penalty: wide spread reduces confidence
        iqr = q75 - q25
        if median > 0:
            relative_spread = iqr / median
            spread_penalty = min(0.2, relative_spread * 0.1)  # Max 0.2 penalty
        else:
            spread_penalty = 0.1

        # Sample size bonus: more samples help percentile reliability
        sample_size_bonus = min(0.1, len(tokens) / 200.0)  # Max 0.1 bonus

        # Final confidence: base ± adjustments
        confidence = base_confidence + sample_size_bonus - spread_penalty
        return predicted, max(0.0, min(1.0, confidence))

    def compute_batch_predictions(
            self,
            batch_id: str,
            sample_responses: List[FenicCompletionsResponse]
    ) -> None:
        """Extract token predictions from sample responses.

        Analyzes the token usage patterns in sample responses to predict
        token usage for remaining requests in the batch.

        Args:
            batch_id: Identifier for the batch
            sample_responses: List of completion responses to analyze
        """

        # Extract output token counts
        sample_completion_tokens = []
        sample_thinking_tokens = []
        sample_total_output_tokens = []
        for resp in sample_responses:
            if resp.usage is not None:
                # Total output tokens = completion + thinking tokens
                sample_completion_tokens.append(resp.usage.completion_tokens)
                sample_thinking_tokens.append(resp.usage.thinking_tokens)
                sample_total_output_tokens.append(resp.usage.completion_tokens + resp.usage.thinking_tokens)

        if not sample_completion_tokens:
            # No valid tokens found, store empty predictions
            token_predictions = BatchTokenPredictions(
                expected_output_tokens=0,
                sample_size=0,
                confidence=0.0
            )
            self._batch_predictions[batch_id] = token_predictions
            return

        # Calculate statistical predictions
        expected_output, confidence = self._sample_statistics(sample_total_output_tokens)
        expected_completion, completion_confidence = self._sample_statistics(sample_completion_tokens)
        expected_thinking, thinking_confidence = self._sample_statistics(sample_thinking_tokens)
        token_predictions = BatchTokenPredictions(
            expected_output_tokens=expected_output,
            sample_size=len(sample_total_output_tokens),
            confidence=confidence
        )

        logger.info(
            f"Batch {batch_id} predictions: {token_predictions.expected_output_tokens} output tokens (confidence: {token_predictions.confidence:.2f})"
        )
        if token_predictions.confidence < MINIMUM_PREDICTION_CONFIDENCE:
            logger.warning(
                f"Batch {batch_id} predictions are low confidence ({token_predictions.confidence:.2f}, minimum is {MINIMUM_PREDICTION_CONFIDENCE}), using conservative estimates"
            )
        self._batch_predictions[batch_id] = token_predictions

    def _is_completion_response_list(self, responses: List[FenicCompletionsResponse]) -> bool:
        """Check if the response list contains valid completion responses.

        Args:
            responses: List of responses to validate

        Returns:
            True if responses are valid completion responses
        """
        if not responses:
            return False

        # Check if we have at least one valid response
        return any(
            isinstance(resp, FenicCompletionsResponse)
            for resp in responses
            if resp is not None
        )

    def _calculate_prediction_confidence(
            self,
            output_tokens_average: float,
            output_tokens_stddev: float,
            output_tokens: List[int]
    ) -> float:
        """Calculate confidence in predictions based on sample variance.

        Uses coefficient of variation, skewness, and sample size to determine
        how reliable the predictions are.

        Args:
            output_tokens_average: Mean of output token counts
            output_tokens_stddev: Standard deviation of output token counts
            output_tokens: List of all output token counts

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if len(output_tokens) < 2:
            return 0.0  # Low confidence with single sample

        # Handle edge case where all numbers are identical
        if output_tokens_stddev == 0:
            return 1.0

        # Calculate coefficient of variation (relative variability)
        cv = abs(output_tokens_stddev / output_tokens_average) if output_tokens_average != 0 else float('inf')

        # Calculate skewness (measure of asymmetry)
        n = len(output_tokens)
        skewness = sum((x - output_tokens_average) ** 3 for x in output_tokens) / (n * output_tokens_stddev ** 3)

        # Base confidence from coefficient of variation
        # Lower CV = higher confidence
        cv_confidence = math.exp(-cv)

        # Penalty for skewness (asymmetric distributions)
        # High skewness means mean is pulled away from center
        skew_penalty = math.exp(-abs(skewness) * 0.5)

        # Sample size bonus (larger samples = more confidence)
        sample_bonus = min(1.0, math.log(n) / math.log(30))  # Caps at n=30

        # Combine factors
        confidence = cv_confidence * skew_penalty * (0.7 + 0.3 * sample_bonus)

        return min(1.0, max(0.0, confidence))

    def predict_output_tokens(
            self,
            request: FenicCompletionsRequest,
            batch_id: str,
            conservative_estimate_func: Callable[[FenicCompletionsRequest], int]
    ) -> int:
        """Predict total output tokens using batch predictions if available.

        Uses batch predictions when confidence is high enough, otherwise falls back
        to a conservative estimate.

        Args:
            request: The completion request to predict tokens for
            batch_id: The batch ID for context-aware prediction
            conservative_estimate_func: Function to get conservative estimate when predictions unavailable

        Returns:
            Predicted output tokens
        """
        batch_predictions = self.get_batch_predictions(batch_id)
        if batch_predictions.confidence > MINIMUM_PREDICTION_CONFIDENCE:
            return batch_predictions.expected_output_tokens

        # Use provider-specific conservative estimate
        return conservative_estimate_func(request)

    def update_transient_exception_time(self, exception_time: float) -> None:
        """Update the time of the last transient exception for rate limiting checks.

        Args:
            exception_time: Timestamp of the transient exception
        """
        self.last_transient_exception_time = exception_time