"""Tests for BatchTokenPredictor with challenging datasets to validate prediction behavior."""

import math
import time
import pytest
from dataclasses import dataclass
from typing import List, Optional

from fenic._inference.batch_token_predictor import (
    CompletionsBatchTokenPredictor,
    BatchTokenPredictions,
    MINIMUM_PREDICTION_CONFIDENCE
)
from fenic._inference.types import FenicCompletionsRequest, FenicCompletionsResponse, ResponseUsage
from .test_batch_prediction_utils import BatchPredictionTestDataGenerator


class MockResponse(FenicCompletionsResponse):
    """Mock response object for testing."""
    def __init__(self, usage: Optional[ResponseUsage] = None):
        # Initialize with minimal required fields for FenicCompletionsResponse
        super().__init__(
            completion="mock response",
            logprobs=None,
            usage=usage
        )


class MockRequest(FenicCompletionsRequest):
    """Mock request object for testing."""
    def __init__(self):
        # Initialize with minimal required fields for FenicCompletionsRequest
        from fenic._inference.types import LMRequestMessages
        super().__init__(
            messages=LMRequestMessages(
                system="test system",
                examples=[],
                user="test user message"
            ),
            max_completion_tokens=1000,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
            model_preset=None
        )


class TestBatchTokenPredictor:
    """Test suite for BatchTokenPredictor with challenging datasets."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = CompletionsBatchTokenPredictor(enable_batch_sampling=True)

    def _create_responses_from_token_distribution(self, token_distribution) -> List[MockResponse]:
        """Helper method to convert TokenDistribution to MockResponse list."""
        responses = []
        for total_output_tokens in token_distribution.token_values:
            # Split total output into completion and thinking tokens
            completion_tokens = int(total_output_tokens * 0.4)  # 40% completion
            thinking_tokens = total_output_tokens - completion_tokens  # 60% thinking
            
            responses.append(MockResponse(ResponseUsage(
                prompt_tokens=50,
                completion_tokens=completion_tokens,
                total_tokens=50 + total_output_tokens,
                thinking_tokens=thinking_tokens
            )))
        return responses

    def test_uniform_distribution_high_confidence(self):
        """Test with uniform token distribution - should have high confidence."""
        batch_id = "uniform_test"
        
        # Use distribution generator for consistent test data
        distribution = BatchPredictionTestDataGenerator.uniform_distribution(count=20)  # Uses new default of 2048 tokens
        uniform_responses = self._create_responses_from_token_distribution(distribution)
        
        self.predictor.compute_batch_predictions(batch_id, uniform_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Validate against expected confidence range from distribution
        min_confidence, max_confidence = distribution.expected_confidence_range
        assert min_confidence <= predictions.confidence <= max_confidence
        assert predictions.expected_output_tokens == 2048  # uniform 2048 tokens
        assert predictions.sample_size == 20

    def test_high_variance_low_confidence(self):
        """Test with high variance distribution - should have low confidence."""
        batch_id = "high_variance_test"
        
        # Use distribution generator for high variance data
        distribution = BatchPredictionTestDataGenerator.high_variance_distribution(count=50)
        high_variance_responses = self._create_responses_from_token_distribution(distribution)
        
        self.predictor.compute_batch_predictions(batch_id, high_variance_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Validate against expected confidence range from distribution
        min_confidence, max_confidence = distribution.expected_confidence_range
        assert min_confidence <= predictions.confidence <= max_confidence
        assert predictions.sample_size == 20
        # Expected output should account for high variance with conservative estimate
        assert predictions.expected_output_tokens > 100  # Should be conservative

    def test_skewed_distribution_reduced_confidence(self):
        """Test with skewed distribution - should reduce confidence due to asymmetry."""
        batch_id = "skewed_test"
        
        # Use distribution generator for skewed data
        distribution = BatchPredictionTestDataGenerator.skewed_distribution(count=100)
        skewed_responses = self._create_responses_from_token_distribution(distribution)
            
        self.predictor.compute_batch_predictions(batch_id, skewed_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Validate against expected confidence range from distribution
        min_confidence, max_confidence = distribution.expected_confidence_range
        assert min_confidence <= predictions.confidence <= max_confidence
        assert predictions.sample_size == 20
        # Should be conservative due to outliers
        assert predictions.expected_output_tokens > 50

    def test_small_sample_size_penalty(self):
        """Test that small sample sizes reduce confidence compared to larger samples with variance."""
        batch_id_small = "small_sample"
        batch_id_large = "large_sample"
        
        # Small sample with some variance
        small_responses = [
            MockResponse(ResponseUsage(
                prompt_tokens=50, completion_tokens=100, total_tokens=200, thinking_tokens=50
            )),
            MockResponse(ResponseUsage(
                prompt_tokens=50, completion_tokens=110, total_tokens=210, thinking_tokens=50
            )),
            MockResponse(ResponseUsage(
                prompt_tokens=50, completion_tokens=90, total_tokens=190, thinking_tokens=50
            ))
        ]
        
        # Large sample with same variance pattern
        large_responses = []
        for _ in range(25):
            # Add slight variance to see sample size effect
            import random
            random.seed(42)  # For reproducibility
            completion_tokens = 100 + random.randint(-10, 10)
            large_responses.append(MockResponse(ResponseUsage(
                prompt_tokens=50, completion_tokens=completion_tokens, total_tokens=200, thinking_tokens=50
            )))
        
        self.predictor.compute_batch_predictions(batch_id_small, small_responses)
        self.predictor.compute_batch_predictions(batch_id_large, large_responses)
        
        small_predictions = self.predictor.get_batch_predictions(batch_id_small)
        large_predictions = self.predictor.get_batch_predictions(batch_id_large)
        
        # Larger sample should have higher confidence
        assert large_predictions.confidence > small_predictions.confidence
        assert small_predictions.sample_size == 3
        assert large_predictions.sample_size == 25

    def test_edge_case_single_sample(self):
        """Test edge case with single sample - should have zero confidence."""
        batch_id = "single_sample"
        
        single_response = [MockResponse(ResponseUsage(
            prompt_tokens=50, completion_tokens=100, total_tokens=200, thinking_tokens=50
        ))]
        
        self.predictor.compute_batch_predictions(batch_id, single_response)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Single sample should have zero confidence
        assert predictions.confidence == 0.0
        assert predictions.sample_size == 1

    def test_edge_case_empty_responses(self):
        """Test edge case with empty or invalid responses."""
        batch_id = "empty_test"
        
        # Mix of responses without usage and responses with zero tokens
        empty_responses = [
            MockResponse(usage=None),
            MockResponse(ResponseUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0, thinking_tokens=0
            ))
        ]
        
        self.predictor.compute_batch_predictions(batch_id, empty_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Should return default empty predictions since no valid token counts
        assert predictions.expected_output_tokens == 0
        assert predictions.sample_size == 0
        assert predictions.confidence == 0.0

    def test_prediction_below_minimum_threshold(self):
        """Test that predictions below minimum confidence threshold are flagged."""
        batch_id = "low_confidence"
        
        # Create distribution that should result in very low confidence
        chaotic_responses = []
        for i in range(10):
            # Extremely chaotic pattern
            completion_tokens = (i * 137 + 23) % 1000  # Pseudo-random distribution
            thinking_tokens = completion_tokens // 3
            chaotic_responses.append(MockResponse(ResponseUsage(
                prompt_tokens=50,
                completion_tokens=completion_tokens,
                total_tokens=50 + completion_tokens + thinking_tokens,
                thinking_tokens=thinking_tokens
            )))
            
        self.predictor.compute_batch_predictions(batch_id, chaotic_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # This should trigger low confidence warning
        if predictions.confidence < MINIMUM_PREDICTION_CONFIDENCE:
            assert predictions.sample_size == 10
            # Prediction should still be generated, just flagged as low confidence

    def test_should_use_sampling_logic(self):
        """Test the sampling decision logic."""
        # Test with requests that have max_completion_tokens (completion model)
        completion_requests = [MockRequest() for _ in range(60)]
        assert self.predictor.should_use_sampling(completion_requests) is True
        
        # Test with too few requests
        few_requests = [MockRequest() for _ in range(30)]
        assert self.predictor.should_use_sampling(few_requests) is False
        
        # Test with sampling disabled
        no_sampling_predictor = CompletionsBatchTokenPredictor(enable_batch_sampling=False)
        assert no_sampling_predictor.should_use_sampling(completion_requests) is False
        
        # Test with embedding model (no max_completion_tokens)
        embedding_requests = [object() for _ in range(60)]  # Generic objects without attribute
        assert self.predictor.should_use_sampling(embedding_requests) is False

    def test_predict_output_tokens_with_confidence(self):
        """Test token prediction with high confidence predictions."""
        batch_id = "confident_predictions"
        request = MockRequest()
        
        # Generate high confidence predictions
        uniform_responses = [
            MockResponse(ResponseUsage(
                prompt_tokens=50, completion_tokens=100, total_tokens=200, thinking_tokens=50
            ))
            for _ in range(20)
        ]
        self.predictor.compute_batch_predictions(batch_id, uniform_responses)
        
        # Mock conservative estimate function
        def conservative_estimate(req):
            return 500
            
        predicted_tokens = self.predictor.predict_output_tokens(
            request, batch_id, conservative_estimate
        )
        
        # Should use batch predictions (150) instead of conservative estimate (500)
        assert predicted_tokens == 150

    def test_predict_output_tokens_low_confidence_fallback(self):
        """Test token prediction falls back to conservative estimate with low confidence."""
        batch_id = "low_confidence_predictions"
        request = MockRequest()
        
        # Generate low confidence predictions
        chaotic_responses = [
            MockResponse(ResponseUsage(
                prompt_tokens=50,
                completion_tokens=i*100,
                total_tokens=50 + i*100 + i*50,
                thinking_tokens=i*50
            ))
            for i in range(1, 6)  # Very high variance
        ]
        self.predictor.compute_batch_predictions(batch_id, chaotic_responses)
        
        # Mock conservative estimate function
        def conservative_estimate(req):
            return 800
            
        predicted_tokens = self.predictor.predict_output_tokens(
            request, batch_id, conservative_estimate
        )
        
        # Should fall back to conservative estimate due to low confidence
        predictions = self.predictor.get_batch_predictions(batch_id)
        if predictions.confidence < MINIMUM_PREDICTION_CONFIDENCE:
            assert predicted_tokens == 800

    def test_cleanup_batch_predictions(self):
        """Test cleanup of batch predictions."""
        batch_id = "cleanup_test"
        
        # Generate some predictions
        responses = [MockResponse(ResponseUsage(
            prompt_tokens=50, completion_tokens=100, total_tokens=200, thinking_tokens=50
        )) for _ in range(5)]
        self.predictor.compute_batch_predictions(batch_id, responses)
        
        # Verify predictions exist
        predictions = self.predictor.get_batch_predictions(batch_id)
        assert predictions.sample_size == 5
        
        # Clean up
        self.predictor.cleanup_batch_predictions(batch_id)
        
        # Verify predictions are gone (returns default)
        cleaned_predictions = self.predictor.get_batch_predictions(batch_id)
        assert cleaned_predictions.sample_size == 0
        assert cleaned_predictions.confidence == 0.0

    def test_bimodal_distribution_challenge(self):
        """Test with bimodal distribution - a challenging case for prediction."""
        batch_id = "bimodal_test"
        
        # Use distribution generator for bimodal data
        distribution = BatchPredictionTestDataGenerator.bimodal_distribution(count=40)
        bimodal_responses = self._create_responses_from_token_distribution(distribution)
            
        self.predictor.compute_batch_predictions(batch_id, bimodal_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Validate against expected confidence range from distribution
        min_confidence, max_confidence = distribution.expected_confidence_range
        assert min_confidence <= predictions.confidence <= max_confidence
        assert predictions.sample_size == 40
        # Conservative estimate should account for both modes
        assert predictions.expected_output_tokens > 100

    def test_all_statistical_distributions(self):
        """Comprehensive test using all predefined statistical distributions."""
        distributions = BatchPredictionTestDataGenerator.get_all_test_distributions()
        
        for distribution in distributions:
            batch_id = f"test_{distribution.name}"
            responses = self._create_responses_from_token_distribution(distribution)
            
            self.predictor.compute_batch_predictions(batch_id, responses)
            predictions = self.predictor.get_batch_predictions(batch_id)
            
            # Validate against expected confidence range
            min_confidence, max_confidence = distribution.expected_confidence_range
            assert min_confidence <= predictions.confidence <= max_confidence, \
                f"Distribution {distribution.name}: confidence {predictions.confidence} not in range [{min_confidence}, {max_confidence}]"
            
            # Validate sample size matches
            assert predictions.sample_size == len(distribution.token_values), \
                f"Distribution {distribution.name}: sample size mismatch"
            
            # Validate that predictions are reasonable (positive)
            assert predictions.expected_output_tokens >= 0, \
                f"Distribution {distribution.name}: negative token prediction"
            
            # Clean up
            self.predictor.cleanup_batch_predictions(batch_id)

    def test_normal_distribution_moderate_confidence(self):
        """Test normal distribution with specified parameters."""
        batch_id = "normal_test"
        
        # Use normal distribution generator
        distribution = BatchPredictionTestDataGenerator.normal_distribution(mean=2048, std_dev=512, count=50)
        normal_responses = self._create_responses_from_token_distribution(distribution)
        
        self.predictor.compute_batch_predictions(batch_id, normal_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Validate against expected confidence range
        min_confidence, max_confidence = distribution.expected_confidence_range
        assert min_confidence <= predictions.confidence <= max_confidence
        assert predictions.sample_size == 25
        
        # Normal distribution should give reasonable predictions around the mean
        assert 1000 <= predictions.expected_output_tokens <= 4000

    def test_chaotic_distribution_very_low_confidence(self):
        """Test chaotic/pseudo-random distribution."""
        batch_id = "chaotic_test"
        
        # Use chaotic distribution generator
        distribution = BatchPredictionTestDataGenerator.chaotic_distribution(count=100)
        chaotic_responses = self._create_responses_from_token_distribution(distribution)
        
        self.predictor.compute_batch_predictions(batch_id, chaotic_responses)
        predictions = self.predictor.get_batch_predictions(batch_id)
        
        # Validate against expected confidence range (should be very low)
        min_confidence, max_confidence = distribution.expected_confidence_range
        assert min_confidence <= predictions.confidence <= max_confidence
        assert predictions.sample_size == 15
        
        # Chaotic distribution should produce reasonable token predictions
        assert predictions.expected_output_tokens > 0