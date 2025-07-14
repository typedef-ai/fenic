"""Tests for the batch token prediction functionality."""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

from fenic._inference.batch_token_predictor import (
    BatchTokenPredictions,
    CompletionsBatchTokenPredictor,
)
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)


@dataclass
class MockRequest:
    """Mock request for testing."""
    messages: LMRequestMessages
    max_completion_tokens: int


@dataclass
class MockResponse:
    """Mock response for testing."""
    usage: Optional[ResponseUsage] = None
    

def create_mock_request(tokens: int = 100) -> MockRequest:
    """Create a mock request for testing."""
    return MockRequest(
        messages=LMRequestMessages(
            system="Test system",
            examples=[],
            user="Test user message"
        ),
        max_completion_tokens=tokens
    )


def create_mock_response(completion_tokens: int, thinking_tokens: int = 0) -> MockResponse:
    """Create a mock response with specified token usage."""
    usage = ResponseUsage(
        prompt_tokens=50,
        completion_tokens=completion_tokens,
        total_tokens=50 + completion_tokens + thinking_tokens,
        thinking_tokens=thinking_tokens
    )
    return MockResponse(usage=usage)


class TestCompletionsBatchTokenPredictor:
    """Test the CompletionsBatchTokenPredictor class."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = CompletionsBatchTokenPredictor()
        
        assert predictor.enable_batch_sampling is True
        assert predictor._batch_predictions == {}
        assert predictor.last_transient_exception_time == 0
    
    def test_should_use_sampling_true(self):
        """Test sampling decision with sufficient requests."""
        predictor = CompletionsBatchTokenPredictor()
        requests = [create_mock_request() for _ in range(50)]
        
        # Convert to FenicCompletionsRequest for the actual test
        fenic_requests = []
        for req in requests:
            fenic_req = FenicCompletionsRequest(
                messages=req.messages,
                max_completion_tokens=req.max_completion_tokens,
                top_logprobs=None,
                structured_output=None,
                temperature=0.0
            )
            fenic_requests.append(fenic_req)
        
        assert predictor.should_use_sampling(fenic_requests) is True
    
    def test_should_use_sampling_false_insufficient_requests(self):
        """Test sampling decision with insufficient requests."""
        predictor = CompletionsBatchTokenPredictor()
        requests = [create_mock_request() for _ in range(10)]
        
        # Convert to FenicCompletionsRequest for the actual test
        fenic_requests = []
        for req in requests:
            fenic_req = FenicCompletionsRequest(
                messages=req.messages,
                max_completion_tokens=req.max_completion_tokens,
                top_logprobs=None,
                structured_output=None,
                temperature=0.0
            )
            fenic_requests.append(fenic_req)
        
        assert predictor.should_use_sampling(fenic_requests) is False
    
    def test_should_use_sampling_false_rate_limited(self):
        """Test sampling decision when rate limited."""
        predictor = CompletionsBatchTokenPredictor()
        predictor.last_transient_exception_time = 1000000  # Recent timestamp
        
        requests = [create_mock_request() for _ in range(60)]
        fenic_requests = []
        for req in requests:
            fenic_req = FenicCompletionsRequest(
                messages=req.messages,
                max_completion_tokens=req.max_completion_tokens,
                top_logprobs=None,
                structured_output=None,
                temperature=0.0
            )
            fenic_requests.append(fenic_req)
        
        with patch('time.time', return_value=1000020):  # 20 seconds later
            assert predictor.should_use_sampling(fenic_requests) is False
    
    def test_get_batch_predictions_empty(self):
        """Test getting predictions for non-existent batch."""
        predictor = CompletionsBatchTokenPredictor()
        
        predictions = predictor.get_batch_predictions("non-existent-batch")
        
        assert predictions.expected_output_tokens == 0
        assert predictions.sample_size == 0
        assert predictions.confidence == 0.0
    
    def test_compute_batch_predictions_symmetric_distribution(self):
        """Test computing predictions with symmetric token distribution."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch"
        
        # Create responses with symmetric distribution
        responses = [
            create_mock_response(completion_tokens=100, thinking_tokens=0),
            create_mock_response(completion_tokens=110, thinking_tokens=0),
            create_mock_response(completion_tokens=120, thinking_tokens=0),
            create_mock_response(completion_tokens=130, thinking_tokens=0),
            create_mock_response(completion_tokens=140, thinking_tokens=0),
        ]
        
        # Convert to FenicCompletionsResponse
        fenic_responses = []
        for resp in responses:
            fenic_resp = FenicCompletionsResponse(
                completion="test",
                logprobs=None,
                usage=resp.usage
            )
            fenic_responses.append(fenic_resp)
        
        predictor.compute_batch_predictions(batch_id, fenic_responses)
        
        predictions = predictor.get_batch_predictions(batch_id)
        assert predictions.expected_output_tokens > 0
        assert predictions.sample_size == 5
        assert predictions.confidence > 0
    
    def test_compute_batch_predictions_skewed_distribution(self):
        """Test computing predictions with skewed token distribution."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch"
        
        # Create responses with right-skewed distribution
        responses = [
            create_mock_response(completion_tokens=50, thinking_tokens=0),
            create_mock_response(completion_tokens=60, thinking_tokens=0),
            create_mock_response(completion_tokens=70, thinking_tokens=0),
            create_mock_response(completion_tokens=200, thinking_tokens=0),  # Outlier
            create_mock_response(completion_tokens=250, thinking_tokens=0),  # Outlier
        ]
        
        # Convert to FenicCompletionsResponse
        fenic_responses = []
        for resp in responses:
            fenic_resp = FenicCompletionsResponse(
                completion="test",
                logprobs=None,
                usage=resp.usage
            )
            fenic_responses.append(fenic_resp)
        
        predictor.compute_batch_predictions(batch_id, fenic_responses)
        
        predictions = predictor.get_batch_predictions(batch_id)
        assert predictions.expected_output_tokens > 0
        assert predictions.sample_size == 5
        assert predictions.confidence > 0
    
    def test_compute_batch_predictions_with_thinking_tokens(self):
        """Test computing predictions with thinking tokens."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch"
        
        # Create responses with thinking tokens
        responses = [
            create_mock_response(completion_tokens=100, thinking_tokens=50),
            create_mock_response(completion_tokens=110, thinking_tokens=60),
            create_mock_response(completion_tokens=120, thinking_tokens=70),
        ]
        
        # Convert to FenicCompletionsResponse
        fenic_responses = []
        for resp in responses:
            fenic_resp = FenicCompletionsResponse(
                completion="test",
                logprobs=None,
                usage=resp.usage
            )
            fenic_responses.append(fenic_resp)
        
        predictor.compute_batch_predictions(batch_id, fenic_responses)
        
        predictions = predictor.get_batch_predictions(batch_id)
        # Should include both completion and thinking tokens
        assert predictions.expected_output_tokens > 150  # Should be sum of both
        assert predictions.sample_size == 3
    
    def test_compute_batch_predictions_empty_responses(self):
        """Test computing predictions with empty responses."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch"
        
        predictor.compute_batch_predictions(batch_id, [])
        
        predictions = predictor.get_batch_predictions(batch_id)
        assert predictions.expected_output_tokens == 0
        assert predictions.sample_size == 0
        assert predictions.confidence == 0.0
    
    def test_cleanup_batch_predictions(self):
        """Test cleaning up batch predictions."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch"
        
        # Add some predictions
        predictor._batch_predictions[batch_id] = BatchTokenPredictions(
            expected_output_tokens=100,
            sample_size=5,
            confidence=0.8
        )
        
        predictor.cleanup_batch_predictions(batch_id)
        
        # Should return default values after cleanup
        predictions = predictor.get_batch_predictions(batch_id)
        assert predictions.expected_output_tokens == 0
        assert predictions.sample_size == 0
        assert predictions.confidence == 0.0
    
    def test_predict_output_tokens_high_confidence(self):
        """Test predicting output tokens with high confidence."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch"
        
        # Set high confidence predictions
        predictor._batch_predictions[batch_id] = BatchTokenPredictions(
            expected_output_tokens=150,
            sample_size=10,
            confidence=0.8
        )
        
        request = FenicCompletionsRequest(
            messages=LMRequestMessages(system="test", examples=[], user="test"),
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.0
        )
        
        def conservative_estimate(req):
            return 200
        
        result = predictor.predict_output_tokens(request, batch_id, conservative_estimate)
        
        # Should use prediction since confidence is high
        assert result == 150
    
    def test_predict_output_tokens_low_confidence(self):
        """Test predicting output tokens with low confidence."""
        predictor = CompletionsBatchTokenPredictor()
        batch_id = "test-batch"
        
        # Set low confidence predictions
        predictor._batch_predictions[batch_id] = BatchTokenPredictions(
            expected_output_tokens=150,
            sample_size=10,
            confidence=0.3  # Below minimum threshold
        )
        
        request = FenicCompletionsRequest(
            messages=LMRequestMessages(system="test", examples=[], user="test"),
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.0
        )
        
        def conservative_estimate(req):
            return 200
        
        result = predictor.predict_output_tokens(request, batch_id, conservative_estimate)
        
        # Should use conservative estimate since confidence is low
        assert result == 200
    
    def test_update_transient_exception_time(self):
        """Test updating transient exception time."""
        predictor = CompletionsBatchTokenPredictor()
        
        predictor.update_transient_exception_time(12345.0)
        
        assert predictor.last_transient_exception_time == 12345.0