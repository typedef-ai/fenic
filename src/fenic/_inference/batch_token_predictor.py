"""Batch token prediction for optimizing rate limiting in model clients."""

import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, TypeVar

logger = logging.getLogger(__name__)

# Constants
MINIMUM_PREDICTION_CONFIDENCE = 0.3

# Type variables
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


@dataclass
class BatchTokenPredictions:
    """Predicted token usage for remaining requests in a batch."""
    expected_output_tokens: int  # Total output tokens (completion + thinking)
    sample_size: int
    confidence: float  # How reliable these predictions are (0.0-1.0)


class BatchTokenPredictor(Generic[RequestT, ResponseT]):
    """Manages batch token predictions for optimizing rate limiting."""
    
    def __init__(self, enable_batch_sampling: bool = True):
        """Initialize the batch token predictor.
        
        Args:
            enable_batch_sampling: Whether to enable sampling-based predictions
        """
        self.enable_batch_sampling = enable_batch_sampling
        self._batch_predictions: Dict[str, BatchTokenPredictions] = {}
        self.last_transient_exception_time: float = 0
    
    def should_use_sampling(self, requests: List[RequestT]) -> bool:
        """Determine if within-batch sampling should be used."""
        return (
            len(requests) >= 50 and  # Need enough requests to make sampling worthwhile
            self.enable_batch_sampling and  # Feature flag
            not self._is_rate_limited() and  # Skip if already hitting rate limits
            self._is_completion_model(requests)  # Skip for embedding models (no output tokens)
        )
    
    def _is_rate_limited(self) -> bool:
        """Check if we're currently experiencing rate limiting."""
        # Simple heuristic: check if we've had recent rate limit backoffs
        return (time.time() - self.last_transient_exception_time) < 30  # Within last 30 seconds
    
    def _is_completion_model(self, requests: List[RequestT]) -> bool:
        """Check if this is a completion model (not embedding) that produces output tokens."""
        if not requests:
            return False

        # Check if requests have max_completion_tokens attribute (completion models do, embedding models don't)
        sample_request = next((req for req in requests if req is not None), None)
        if sample_request is None:
            return False

        return hasattr(sample_request, 'max_completion_tokens')
    
    def get_batch_predictions(self, batch_id: str) -> BatchTokenPredictions:
        """Get predictions for a batch, with safe defaults."""
        return self._batch_predictions.get(batch_id, BatchTokenPredictions(
            expected_output_tokens=0,
            sample_size=0,
            confidence=0.0
        ))
    
    def cleanup_batch_predictions(self, batch_id: str):
        """Clean up predictions when batch is complete."""
        self._batch_predictions.pop(batch_id, None)
    
    def generate_token_predictions(
        self,
        batch_id: str,
        sample_responses: List[ResponseT]
    ):
        """Extract token predictions from sample responses."""
        output_tokens = []

        for resp in sample_responses:
            if resp and hasattr(resp, 'usage') and resp.usage:
                # Total output tokens = completion + thinking tokens
                total_output = resp.usage.completion_tokens + resp.usage.thinking_tokens
                if total_output > 0:
                    output_tokens.append(total_output)

        if not output_tokens:
            return None
        
        output_tokens_average = int(statistics.mean(output_tokens))
        output_tokens_stddev = statistics.stdev(output_tokens)
        expected_output = output_tokens_average + int(1.5 * output_tokens_stddev)
        confidence = self._calculate_prediction_confidence(output_tokens_average, output_tokens_stddev, output_tokens)

        token_predictions = BatchTokenPredictions(
            expected_output_tokens=expected_output,
            sample_size=len(output_tokens),
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

    def _calculate_prediction_confidence(
        self,
        output_tokens_average: float,
        output_tokens_stddev: float,
        output_tokens: List[int]
    ) -> float:
        """Calculate confidence in predictions based on sample variance."""
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
        request: RequestT, 
        batch_id: str, 
        conservative_estimate_func: Callable[[RequestT], int]
    ) -> int:
        """Predict total output tokens using batch predictions if available.
        
        Args:
            request: The request to predict tokens for
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
    
    def update_transient_exception_time(self, exception_time: float):
        """Update the time of the last transient exception for rate limiting checks."""
        self.last_transient_exception_time = exception_time