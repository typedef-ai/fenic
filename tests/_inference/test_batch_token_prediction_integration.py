"""Integration tests for batch token prediction functionality."""

from unittest.mock import Mock

import pytest

from fenic._inference.batch_token_predictor import (
    MINIMUM_PREDICTION_CONFIDENCE,
    CompletionsBatchTokenPredictor,
)
from fenic._inference.model_client import ModelClient
from fenic._inference.rate_limit_strategy import RateLimitStrategy
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider


class MockModelClient(ModelClient):
    """Mock model client for testing batch token prediction integration."""
    
    def __init__(self):
        # Mock the required dependencies
        rate_limit_strategy = Mock(spec=RateLimitStrategy)
        rate_limit_strategy.context_tokens_per_minute.return_value = 100000
        token_counter = Mock(spec=TiktokenTokenCounter)
        
        # Initialize without calling parent __init__ to avoid event loop setup
        self.model = "test-model"
        self.model_provider = ModelProvider.OPENAI
        self.rate_limit_strategy = rate_limit_strategy
        self.token_counter = token_counter
        self.batch_token_predictor = CompletionsBatchTokenPredictor()
        
    async def make_single_request(self, request):
        """Mock implementation of single request."""
        # Return a mock response with random token usage
        return FenicCompletionsResponse(
            completion="Mock completion",
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=50,
                completion_tokens=100,
                total_tokens=150,
                thinking_tokens=0
            )
        )
    
    def estimate_tokens_for_request(self, request, batch_id=None):
        """Mock token estimation."""
        from fenic._inference.rate_limit_strategy import TokenEstimate
        return TokenEstimate(input_tokens=50, output_tokens=100)
    
    def get_request_key(self, request):
        """Mock request key generation."""
        return str(hash(str(request)))
    
    def get_metrics(self):
        """Mock metrics."""
        from fenic.core.metrics import LMMetrics
        return LMMetrics()
    
    def reset_metrics(self):
        """Mock metrics reset."""
        pass
    
    def _get_max_output_tokens(self, request):
        """Mock conservative estimate."""
        return 200


class TestBatchTokenPredictionIntegration:
    """Integration tests for batch token prediction."""
    
    def test_batch_prediction_workflow(self):
        """Test the full batch prediction workflow."""
        client = MockModelClient()
        predictor = client.batch_token_predictor
        
        # Create mock requests
        requests = []
        for i in range(60):  # Enough for sampling
            request = FenicCompletionsRequest(
                messages=LMRequestMessages(
                    system="Test system",
                    examples=[],
                    user=f"Test user message {i}"
                ),
                max_completion_tokens=100,
                top_logprobs=None,
                structured_output=None,
                temperature=0.0
            )
            requests.append(request)
        
        # Test that sampling is enabled
        assert predictor.should_use_sampling(requests) is True
        
        # Test batch prediction computation
        batch_id = "test-batch-integration"
        
        # Create mock responses with varying token usage
        responses = []
        for i in range(10):
            response = FenicCompletionsResponse(
                completion=f"Response {i}",
                logprobs=None,
                usage=ResponseUsage(
                    prompt_tokens=50,
                    completion_tokens=80 + i * 10,  # Varying completion tokens
                    total_tokens=130 + i * 10,
                    thinking_tokens=0
                )
            )
            responses.append(response)
        
        # Compute predictions
        predictor.compute_batch_predictions(batch_id, responses)
        
        # Verify predictions were created
        predictions = predictor.get_batch_predictions(batch_id)
        assert predictions.expected_output_tokens > 0
        assert predictions.sample_size == 10
        assert predictions.confidence > 0
        
        # Test prediction usage
        test_request = requests[0]
        predicted_tokens = predictor.predict_output_tokens(
            test_request, 
            batch_id, 
            lambda req: 200  # Conservative estimate
        )
        
        if predictions.confidence > MINIMUM_PREDICTION_CONFIDENCE:
            assert predicted_tokens == predictions.expected_output_tokens
        else:
            assert predicted_tokens == 200  # Conservative estimate
        
        # Test cleanup
        predictor.cleanup_batch_predictions(batch_id)
        empty_predictions = predictor.get_batch_predictions(batch_id)
        assert empty_predictions.expected_output_tokens == 0
        assert empty_predictions.sample_size == 0
        assert empty_predictions.confidence == 0.0
    
    def test_batch_prediction_with_thinking_tokens(self):
        """Test batch prediction with thinking tokens."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch-thinking"
        
        # Create responses with thinking tokens
        responses = []
        for i in range(5):
            response = FenicCompletionsResponse(
                completion=f"Response {i}",
                logprobs=None,
                usage=ResponseUsage(
                    prompt_tokens=50,
                    completion_tokens=100,
                    total_tokens=200 + i * 20,  # Varying total tokens
                    thinking_tokens=50 + i * 20  # Varying thinking tokens
                )
            )
            responses.append(response)
        
        predictor.compute_batch_predictions(batch_id, responses)
        predictions = predictor.get_batch_predictions(batch_id)
        
        # Should include both completion and thinking tokens
        assert predictions.expected_output_tokens > 150  # Should be sum of both
        assert predictions.sample_size == 5
    
    def test_batch_prediction_confidence_thresholds(self):
        """Test that confidence thresholds work correctly."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch-confidence"
        
        # Create responses with high variance (low confidence)
        responses = []
        token_counts = [10, 20, 30, 200, 300]  # High variance
        for i, tokens in enumerate(token_counts):
            response = FenicCompletionsResponse(
                completion=f"Response {i}",
                logprobs=None,
                usage=ResponseUsage(
                    prompt_tokens=50,
                    completion_tokens=tokens,
                    total_tokens=50 + tokens,
                    thinking_tokens=0
                )
            )
            responses.append(response)
        
        predictor.compute_batch_predictions(batch_id, responses)
        predictions = predictor.get_batch_predictions(batch_id)
        
        # With high variance, confidence should be low
        assert predictions.confidence < MINIMUM_PREDICTION_CONFIDENCE
        
        # Test prediction fallback to conservative estimate
        test_request = FenicCompletionsRequest(
            messages=LMRequestMessages(system="test", examples=[], user="test"),
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.0
        )
        
        predicted_tokens = predictor.predict_output_tokens(
            test_request,
            batch_id,
            lambda req: 500  # Conservative estimate
        )
        
        # Should use conservative estimate due to low confidence
        assert predicted_tokens == 500
    
    def test_batch_prediction_rate_limit_handling(self):
        """Test that batch prediction respects rate limiting."""
        predictor = CompletionsBatchTokenPredictor()
        
        # Simulate recent rate limiting
        import time
        predictor.update_transient_exception_time(time.time() - 10)  # 10 seconds ago
        
        # Create requests
        requests = []
        for i in range(60):
            request = FenicCompletionsRequest(
                messages=LMRequestMessages(
                    system="Test system",
                    examples=[],
                    user=f"Test user message {i}"
                ),
                max_completion_tokens=100,
                top_logprobs=None,
                structured_output=None,
                temperature=0.0
            )
            requests.append(request)
        
        # Should not use sampling due to recent rate limiting
        assert predictor.should_use_sampling(requests) is False
    
    def test_batch_prediction_insufficient_requests(self):
        """Test behavior with insufficient requests for sampling."""
        predictor = CompletionsBatchTokenPredictor()
        
        # Create too few requests
        requests = []
        for i in range(20):  # Below threshold of 50
            request = FenicCompletionsRequest(
                messages=LMRequestMessages(
                    system="Test system",
                    examples=[],
                    user=f"Test user message {i}"
                ),
                max_completion_tokens=100,
                top_logprobs=None,
                structured_output=None,
                temperature=0.0
            )
            requests.append(request)
        
        # Should not use sampling due to insufficient requests
        assert predictor.should_use_sampling(requests) is False
    
    def test_batch_prediction_empty_responses(self):
        """Test handling of empty or invalid responses."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch-empty"
        
        # Test with empty responses
        predictor.compute_batch_predictions(batch_id, [])
        predictions = predictor.get_batch_predictions(batch_id)
        
        assert predictions.expected_output_tokens == 0
        assert predictions.sample_size == 0
        assert predictions.confidence == 0.0
        
        # Test with responses that have no usage data
        responses_no_usage = [
            FenicCompletionsResponse(
                completion="test",
                logprobs=None,
                usage=None
            )
        ]
        
        predictor.compute_batch_predictions(batch_id, responses_no_usage)
        predictions = predictor.get_batch_predictions(batch_id)
        
        assert predictions.expected_output_tokens == 0
        assert predictions.sample_size == 0
        assert predictions.confidence == 0.0
    
    @pytest.mark.parametrize("distribution_type", ["symmetric", "right_skewed", "left_skewed"])
    def test_batch_prediction_distribution_handling(self, distribution_type):
        """Test that different token distributions are handled correctly."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = f"test-batch-{distribution_type}"
        
        # Create responses with different distributions
        responses = []
        if distribution_type == "symmetric":
            token_counts = [90, 95, 100, 105, 110]  # Symmetric around 100
        elif distribution_type == "right_skewed":
            token_counts = [50, 60, 70, 150, 200]  # Right tail
        else:  # left_skewed
            token_counts = [200, 150, 120, 110, 100]  # Left tail
        
        for i, tokens in enumerate(token_counts):
            response = FenicCompletionsResponse(
                completion=f"Response {i}",
                logprobs=None,
                usage=ResponseUsage(
                    prompt_tokens=50,
                    completion_tokens=tokens,
                    total_tokens=50 + tokens,
                    thinking_tokens=0
                )
            )
            responses.append(response)
        
        predictor.compute_batch_predictions(batch_id, responses)
        predictions = predictor.get_batch_predictions(batch_id)
        
        # Should have valid predictions regardless of distribution
        assert predictions.expected_output_tokens > 0
        assert predictions.sample_size == 5
        assert 0.0 <= predictions.confidence <= 1.0