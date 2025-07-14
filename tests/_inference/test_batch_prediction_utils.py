"""Test utilities for batch token prediction functionality."""

from typing import List

import numpy as np
import pytest

from fenic._inference.batch_token_predictor import (
    MINIMUM_PREDICTION_CONFIDENCE,
    BatchTokenPredictions,
    CompletionsBatchTokenPredictor,
)
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)


class BatchPredictionTestUtils:
    """Test utilities for batch token prediction functionality."""
    
    @staticmethod
    def create_mock_request(
        user_message: str = "Test message",
        max_tokens: int = 100,
        temperature: float = 0.0,
        model_preset: str = None
    ) -> FenicCompletionsRequest:
        """Create a mock FenicCompletionsRequest for testing.
        
        Args:
            user_message: The user message content
            max_tokens: Maximum completion tokens
            temperature: Temperature setting
            model_preset: Optional model preset
            
        Returns:
            A mock FenicCompletionsRequest
        """
        return FenicCompletionsRequest(
            messages=LMRequestMessages(
                system="Test system message",
                examples=[],
                user=user_message
            ),
            max_completion_tokens=max_tokens,
            top_logprobs=None,
            structured_output=None,
            temperature=temperature,
            model_preset=model_preset
        )
    
    @staticmethod
    def create_mock_response(
        completion: str = "Test completion",
        completion_tokens: int = 100,
        thinking_tokens: int = 0,
        prompt_tokens: int = 50
    ) -> FenicCompletionsResponse:
        """Create a mock FenicCompletionsResponse for testing.
        
        Args:
            completion: The completion text
            completion_tokens: Number of completion tokens
            thinking_tokens: Number of thinking tokens
            prompt_tokens: Number of prompt tokens
            
        Returns:
            A mock FenicCompletionsResponse
        """
        return FenicCompletionsResponse(
            completion=completion,
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens + thinking_tokens,
                thinking_tokens=thinking_tokens
            )
        )
    
    @staticmethod
    def create_request_batch(
        size: int,
        base_message: str = "Test message",
        vary_tokens: bool = False
    ) -> List[FenicCompletionsRequest]:
        """Create a batch of mock requests for testing.
        
        Args:
            size: Number of requests to create
            base_message: Base message to use (will be numbered)
            vary_tokens: Whether to vary max_completion_tokens
            
        Returns:
            List of mock FenicCompletionsRequest objects
        """
        requests = []
        for i in range(size):
            max_tokens = 100 + (i * 10) if vary_tokens else 100
            request = BatchPredictionTestUtils.create_mock_request(
                user_message=f"{base_message} {i}",
                max_tokens=max_tokens
            )
            requests.append(request)
        return requests
    
    @staticmethod
    def create_response_batch(
        size: int,
        token_distribution: str = "uniform",
        base_tokens: int = 100,
        variance: int = 20
    ) -> List[FenicCompletionsResponse]:
        """Create a batch of mock responses with specified token distribution.
        
        Args:
            size: Number of responses to create
            token_distribution: Type of distribution ('uniform', 'normal', 'skewed')
            base_tokens: Base number of tokens
            variance: Variance in token counts
            
        Returns:
            List of mock FenicCompletionsResponse objects
        """
        responses = []
        
        if token_distribution == "uniform":
            token_counts = [base_tokens + i * (variance // size) for i in range(size)]
        elif token_distribution == "normal":
            np.random.seed(42)  # For reproducible tests
            token_counts = np.random.normal(base_tokens, variance, size).astype(int)
            token_counts = np.clip(token_counts, 10, base_tokens * 3)  # Reasonable bounds
        elif token_distribution == "skewed":
            # Create right-skewed distribution
            token_counts = [base_tokens] * (size // 2)  # Most values at base
            token_counts.extend([base_tokens + variance * 2] * (size // 4))  # Some higher
            token_counts.extend([base_tokens + variance * 4] * (size // 4))  # Few very high
        else:
            raise ValueError(f"Unknown distribution type: {token_distribution}")
        
        for i, tokens in enumerate(token_counts):
            response = BatchPredictionTestUtils.create_mock_response(
                completion=f"Response {i}",
                completion_tokens=max(10, int(tokens)),  # Ensure positive tokens
                thinking_tokens=0
            )
            responses.append(response)
        
        return responses
    
    @staticmethod
    def create_thinking_response_batch(
        size: int,
        completion_tokens: int = 100,
        thinking_tokens: int = 50
    ) -> List[FenicCompletionsResponse]:
        """Create a batch of responses with thinking tokens.
        
        Args:
            size: Number of responses to create
            completion_tokens: Base completion tokens
            thinking_tokens: Base thinking tokens
            
        Returns:
            List of mock FenicCompletionsResponse objects with thinking tokens
        """
        responses = []
        for i in range(size):
            # Add some variation
            comp_tokens = completion_tokens + (i * 10)
            think_tokens = thinking_tokens + (i * 5)
            
            response = BatchPredictionTestUtils.create_mock_response(
                completion=f"Thinking response {i}",
                completion_tokens=comp_tokens,
                thinking_tokens=think_tokens
            )
            responses.append(response)
        
        return responses
    
    @staticmethod
    def assert_valid_predictions(
        predictions: BatchTokenPredictions,
        expected_sample_size: int,
        min_tokens: int = 0,
        max_tokens: int = 10000
    ):
        """Assert that predictions are valid.
        
        Args:
            predictions: The predictions to validate
            expected_sample_size: Expected sample size
            min_tokens: Minimum expected tokens
            max_tokens: Maximum expected tokens
        """
        assert predictions.sample_size == expected_sample_size
        assert min_tokens <= predictions.expected_output_tokens <= max_tokens
        assert 0.0 <= predictions.confidence <= 1.0
    
    @staticmethod
    def assert_prediction_confidence(
        predictions: BatchTokenPredictions,
        should_be_high: bool = True,
        threshold: float = MINIMUM_PREDICTION_CONFIDENCE
    ):
        """Assert prediction confidence level.
        
        Args:
            predictions: The predictions to check
            should_be_high: Whether confidence should be high or low
            threshold: Confidence threshold to use
        """
        if should_be_high:
            assert predictions.confidence >= threshold, f"Expected high confidence, got {predictions.confidence}"
        else:
            assert predictions.confidence < threshold, f"Expected low confidence, got {predictions.confidence}"
    
    @staticmethod
    def simulate_rate_limiting(predictor: CompletionsBatchTokenPredictor, seconds_ago: float = 10):
        """Simulate rate limiting by setting recent exception time.
        
        Args:
            predictor: The predictor to modify
            seconds_ago: How many seconds ago the exception occurred
        """
        import time
        predictor.update_transient_exception_time(time.time() - seconds_ago)
    
    @staticmethod
    def create_high_variance_responses(size: int) -> List[FenicCompletionsResponse]:
        """Create responses with high variance for testing low confidence scenarios.
        
        Args:
            size: Number of responses to create
            
        Returns:
            List of responses with high token variance
        """
        # Create responses with very different token counts
        token_counts = [10, 50, 100, 300, 500][:size]
        if len(token_counts) < size:
            # Fill remaining with random high variance values
            import random
            random.seed(42)
            while len(token_counts) < size:
                token_counts.append(random.randint(10, 1000))
        
        responses = []
        for i, tokens in enumerate(token_counts):
            response = BatchPredictionTestUtils.create_mock_response(
                completion=f"High variance response {i}",
                completion_tokens=tokens,
                thinking_tokens=0
            )
            responses.append(response)
        
        return responses
    
    @staticmethod
    def create_symmetric_distribution(size: int, center: int = 100, spread: int = 20) -> List[int]:
        """Create a symmetric token distribution.
        
        Args:
            size: Number of values to generate
            center: Center value
            spread: Spread around center
            
        Returns:
            List of token counts with symmetric distribution
        """
        if size % 2 == 0:
            # Even size: create symmetric pairs
            tokens = []
            for i in range(size // 2):
                offset = (i + 1) * (spread // (size // 2))
                tokens.extend([center - offset, center + offset])
        else:
            # Odd size: center value plus symmetric pairs
            tokens = [center]
            for i in range(size // 2):
                offset = (i + 1) * (spread // (size // 2))
                tokens.extend([center - offset, center + offset])
        
        return tokens[:size]
    
    @staticmethod
    def create_skewed_distribution(size: int, skew_type: str = "right") -> List[int]:
        """Create a skewed token distribution.
        
        Args:
            size: Number of values to generate
            skew_type: Type of skew ('right' or 'left')
            
        Returns:
            List of token counts with skewed distribution
        """
        if skew_type == "right":
            # Right skew: most values low, few high
            tokens = [50] * (size // 2)  # Many low values
            tokens.extend([100] * (size // 4))  # Some medium values
            tokens.extend([200] * (size // 4))  # Few high values
        elif skew_type == "left":
            # Left skew: most values high, few low
            tokens = [200] * (size // 2)  # Many high values
            tokens.extend([100] * (size // 4))  # Some medium values
            tokens.extend([50] * (size // 4))  # Few low values
        else:
            raise ValueError(f"Unknown skew type: {skew_type}")
        
        return tokens[:size]


class TestBatchPredictionUtils:
    """Test the batch prediction utilities themselves."""
    
    def test_create_mock_request(self):
        """Test mock request creation."""
        request = BatchPredictionTestUtils.create_mock_request(
            user_message="Test message",
            max_tokens=150,
            temperature=0.5
        )
        
        assert isinstance(request, FenicCompletionsRequest)
        assert request.messages.user == "Test message"
        assert request.max_completion_tokens == 150
        assert request.temperature == 0.5
        assert request.messages.system == "Test system message"
    
    def test_create_mock_response(self):
        """Test mock response creation."""
        response = BatchPredictionTestUtils.create_mock_response(
            completion="Test completion",
            completion_tokens=120,
            thinking_tokens=30,
            prompt_tokens=60
        )
        
        assert isinstance(response, FenicCompletionsResponse)
        assert response.completion == "Test completion"
        assert response.usage.completion_tokens == 120
        assert response.usage.thinking_tokens == 30
        assert response.usage.prompt_tokens == 60
        assert response.usage.total_tokens == 210  # 60 + 120 + 30
    
    def test_create_request_batch(self):
        """Test batch request creation."""
        requests = BatchPredictionTestUtils.create_request_batch(
            size=5,
            base_message="Batch test",
            vary_tokens=True
        )
        
        assert len(requests) == 5
        assert all(isinstance(req, FenicCompletionsRequest) for req in requests)
        assert requests[0].messages.user == "Batch test 0"
        assert requests[0].max_completion_tokens == 100
        assert requests[1].max_completion_tokens == 110  # Varied
    
    def test_create_response_batch_uniform(self):
        """Test uniform response batch creation."""
        responses = BatchPredictionTestUtils.create_response_batch(
            size=5,
            token_distribution="uniform",
            base_tokens=100,
            variance=20
        )
        
        assert len(responses) == 5
        assert all(isinstance(resp, FenicCompletionsResponse) for resp in responses)
        
        # Check token progression
        tokens = [resp.usage.completion_tokens for resp in responses]
        assert tokens == [100, 104, 108, 112, 116]  # Uniform distribution
    
    def test_create_response_batch_normal(self):
        """Test normal response batch creation."""
        responses = BatchPredictionTestUtils.create_response_batch(
            size=10,
            token_distribution="normal",
            base_tokens=100,
            variance=20
        )
        
        assert len(responses) == 10
        tokens = [resp.usage.completion_tokens for resp in responses]
        
        # With seed=42, should have reasonable distribution around 100
        assert 50 <= np.mean(tokens) <= 150
        assert all(10 <= token <= 300 for token in tokens)  # Within clipped bounds
    
    def test_create_response_batch_skewed(self):
        """Test skewed response batch creation."""
        responses = BatchPredictionTestUtils.create_response_batch(
            size=8,
            token_distribution="skewed",
            base_tokens=100,
            variance=20
        )
        
        assert len(responses) == 8
        tokens = [resp.usage.completion_tokens for resp in responses]
        
        # Should have many base values and fewer high values
        assert tokens.count(100) == 4  # Half at base
        assert tokens.count(140) == 2  # Quarter at base + 2*variance
        assert tokens.count(180) == 2  # Quarter at base + 4*variance
    
    def test_create_thinking_response_batch(self):
        """Test thinking response batch creation."""
        responses = BatchPredictionTestUtils.create_thinking_response_batch(
            size=3,
            completion_tokens=100,
            thinking_tokens=50
        )
        
        assert len(responses) == 3
        assert all(resp.usage.thinking_tokens > 0 for resp in responses)
        assert responses[0].usage.completion_tokens == 100
        assert responses[0].usage.thinking_tokens == 50
        assert responses[1].usage.completion_tokens == 110  # Varied
        assert responses[1].usage.thinking_tokens == 55  # Varied
    
    def test_assert_valid_predictions(self):
        """Test prediction validation."""
        predictions = BatchTokenPredictions(
            expected_output_tokens=150,
            sample_size=10,
            confidence=0.8
        )
        
        # Should not raise
        BatchPredictionTestUtils.assert_valid_predictions(
            predictions,
            expected_sample_size=10,
            min_tokens=100,
            max_tokens=200
        )
        
        # Should raise for invalid sample size
        with pytest.raises(AssertionError):
            BatchPredictionTestUtils.assert_valid_predictions(
                predictions,
                expected_sample_size=5  # Wrong size
            )
    
    def test_assert_prediction_confidence(self):
        """Test confidence assertion."""
        high_conf_predictions = BatchTokenPredictions(
            expected_output_tokens=100,
            sample_size=10,
            confidence=0.8
        )
        
        low_conf_predictions = BatchTokenPredictions(
            expected_output_tokens=100,
            sample_size=10,
            confidence=0.3
        )
        
        # Should not raise for high confidence
        BatchPredictionTestUtils.assert_prediction_confidence(
            high_conf_predictions,
            should_be_high=True
        )
        
        # Should not raise for low confidence
        BatchPredictionTestUtils.assert_prediction_confidence(
            low_conf_predictions,
            should_be_high=False
        )
        
        # Should raise for wrong confidence level
        with pytest.raises(AssertionError):
            BatchPredictionTestUtils.assert_prediction_confidence(
                low_conf_predictions,
                should_be_high=True  # Expecting high but got low
            )
    
    def test_simulate_rate_limiting(self):
        """Test rate limiting simulation."""
        predictor = CompletionsBatchTokenPredictor()
        
        # Initially should not be rate limited
        requests = BatchPredictionTestUtils.create_request_batch(60)
        assert predictor.should_use_sampling(requests) is True
        
        # Simulate recent rate limiting
        BatchPredictionTestUtils.simulate_rate_limiting(predictor, seconds_ago=10)
        
        # Should now be rate limited
        assert predictor.should_use_sampling(requests) is False
    
    def test_create_high_variance_responses(self):
        """Test high variance response creation."""
        responses = BatchPredictionTestUtils.create_high_variance_responses(5)
        
        assert len(responses) == 5
        tokens = [resp.usage.completion_tokens for resp in responses]
        
        # Should have the expected high variance pattern
        assert tokens == [10, 50, 100, 300, 500]
        
        # Verify high variance
        variance = np.var(tokens)
        assert variance > 20000  # High variance
    
    def test_create_symmetric_distribution(self):
        """Test symmetric distribution creation."""
        # Test even size
        tokens = BatchPredictionTestUtils.create_symmetric_distribution(
            size=6,
            center=100,
            spread=20
        )
        
        assert len(tokens) == 6
        assert tokens == [94, 106, 88, 112, 82, 118]  # Symmetric pairs around 100
        
        # Test odd size
        tokens = BatchPredictionTestUtils.create_symmetric_distribution(
            size=5,
            center=100,
            spread=20
        )
        
        assert len(tokens) == 5
        assert tokens[0] == 100  # Center value
        assert 90 in tokens and 110 in tokens  # Symmetric pair
    
    def test_create_skewed_distribution(self):
        """Test skewed distribution creation."""
        # Test right skew
        tokens = BatchPredictionTestUtils.create_skewed_distribution(
            size=8,
            skew_type="right"
        )
        
        assert len(tokens) == 8
        assert tokens.count(50) == 4  # Many low values
        assert tokens.count(100) == 2  # Some medium values
        assert tokens.count(200) == 2  # Few high values
        
        # Test left skew
        tokens = BatchPredictionTestUtils.create_skewed_distribution(
            size=8,
            skew_type="left"
        )
        
        assert len(tokens) == 8
        assert tokens.count(200) == 4  # Many high values
        assert tokens.count(100) == 2  # Some medium values
        assert tokens.count(50) == 2  # Few low values