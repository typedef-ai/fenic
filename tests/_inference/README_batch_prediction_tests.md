# Batch Token Prediction Tests

This directory contains comprehensive tests for the Batch Token Prediction system in fenic. The tests are designed to validate the performance and behavior of the prediction system under various challenging scenarios.

## Architecture

The Batch Token Prediction system now uses a clean protocol-based architecture:

- **`BatchTokenPredictorProtocol[RequestT, ResponseT]`**: Generic protocol that can work with any request/response types
- **`FenicBatchTokenPredictor`**: Concrete implementation specifically for `FenicCompletionsRequest`/`FenicCompletionsResponse`
- **`BatchTokenPredictor`**: Type alias for `FenicBatchTokenPredictor` for backward compatibility

This design allows for future implementations with different request/response types while keeping the current implementation simple and type-safe.

## Test Files

### 1. `test_batch_token_predictor.py` - Deterministic Unit Tests

**Purpose**: Test the core `FenicBatchTokenPredictor` logic with controlled, deterministic datasets.

**Key Test Scenarios**:

- **Uniform Distribution**: Tests high confidence predictions with consistent token counts
- **High Variance**: Tests low confidence with exponentially varying token counts
- **Skewed Distribution**: Tests handling of distributions with outliers
- **Bimodal Distribution**: Tests challenging dual-peak distributions
- **Small Sample Size**: Tests confidence penalties for insufficient data
- **Edge Cases**: Tests empty responses, single samples, and invalid data

**Run Command**:

```bash
# Run just the deterministic tests
uv run pytest tests/_inference/test_batch_token_predictor.py -v

# Run with detailed output
uv run pytest tests/_inference/test_batch_token_predictor.py -v -s
```

### 2. `test_batch_token_prediction_integration.py` - Real LLM Integration Tests

**Purpose**: Test batch prediction with actual semantic operations and LLM responses.

**Key Test Scenarios**:

- **Varied Complexity**: Tests prediction across simple math, analysis, and creative writing tasks (75 operations to trigger sampling)
- **Token Variance Analysis**: Tests different variance patterns (consistent, variable, bimodal scenarios)
- **Thinking Models**: Tests batch prediction with reasoning models that produce thinking tokens
- **Performance Comparison**: Measures execution time and validates prediction accuracy

**Run Command**:

```bash
# Run integration tests (requires cloud setup and API keys)
uv run pytest tests/_inference/test_batch_token_prediction_integration.py -m cloud -v

# Run specific integration test
uv run pytest tests/_inference/test_batch_token_prediction_integration.py::TestBatchTokenPredictionIntegration::test_semantic_map_batch_prediction_varied_complexity -m cloud -v
```

### 3. `test_batch_prediction_utils.py` - Test Utilities

**Purpose**: Provides utility functions and data generators for testing batch prediction scenarios.

**Key Components**:

- `TokenDistribution`: Data class for representing test token patterns
- `BatchPredictionTestDataGenerator`: Generates challenging statistical distributions
- `SemanticOperationTestCases`: Predefined semantic operation test cases
- `create_comprehensive_test_dataset()`: Creates full test datasets

## How to Use These Tests

### Prerequisites

1. **Environment Setup**: Ensure you have API keys configured:

   ```bash
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   export GEMINI_API_KEY="your-key-here"
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

### Running Tests

#### Quick Deterministic Tests (No API calls)

```bash
# Fast unit tests that don't require API access
uv run pytest tests/_inference/test_batch_token_predictor.py
```

#### Full Integration Tests (Requires API keys)

```bash
# Run all cloud integration tests
uv run pytest tests/_inference/test_batch_token_prediction_integration.py -m cloud

# Run with detailed prediction analysis output
uv run pytest tests/_inference/test_batch_token_prediction_integration.py -m cloud -s
```

#### Custom Test Scenarios

You can create custom test scenarios using the utilities:

```python
from tests._inference.test_batch_prediction_utils import (
    BatchPredictionTestDataGenerator,
    create_comprehensive_test_dataset
)

# Generate specific distribution for testing
uniform_dist = BatchPredictionTestDataGenerator.uniform_distribution(base_tokens=200, count=30)
print(f"Expected confidence range: {uniform_dist.expected_confidence_range}")

# Create comprehensive dataset
test_data = create_comprehensive_test_dataset(num_samples_per_type=15)
print(f"Total test samples: {len(test_data['operation_type'])}")
```

## What These Tests Validate

### Statistical Accuracy

- **Confidence Calculation**: Tests that confidence scores properly reflect prediction reliability
- **Variance Handling**: Validates conservative estimates for high-variance scenarios
- **Sample Size Impact**: Ensures larger samples increase confidence appropriately

### Performance Optimization

- **Sampling Threshold**: Validates that sampling only triggers for large batches (50+ requests)
- **Rate Limit Awareness**: Tests that sampling is disabled during rate limiting
- **Model Type Detection**: Ensures sampling only applies to completion models, not embeddings

### Real-World Scenarios

- **Token Variance**: Tests with realistic semantic operations of varying complexity
- **Thinking Tokens**: Validates proper handling of reasoning model outputs
- **Batch Processing**: Tests end-to-end batch processing with actual LLM calls

### Edge Cases

- **Empty Responses**: Handles missing or invalid API responses gracefully
- **Single Samples**: Provides appropriate confidence scores for insufficient data
- **Mixed Distributions**: Handles complex, multi-modal token distributions

## Expected Test Results

### Deterministic Tests

- **Uniform distributions**: Should achieve confidence ≥ 0.9
- **High variance distributions**: Should achieve confidence ≤ 0.4
- **Skewed distributions**: Should achieve confidence between 0.2-0.7
- **Single samples**: Should achieve confidence = 0.0

### Integration Tests

- **Large batches (75+ operations)**: Should trigger batch prediction sampling
- **Simple operations**: Should show consistent, low token outputs with high confidence
- **Complex operations**: Should show variable, high token outputs with lower confidence
- **Thinking models**: Should properly track both completion and thinking tokens

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If integration tests fail due to rate limits, reduce the test dataset size or add delays between tests.

2. **Missing API Keys**: Ensure environment variables are set correctly:

   ```bash
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```

3. **Import Errors**: Ensure you're running from the fenic project root:
   ```bash
   cd /path/to/fenic
   uv run pytest tests/_inference/test_batch_token_predictor.py
   ```

### Debugging

Enable detailed logging to see batch prediction behavior:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fenic._inference.batch_token_predictor")
logger.setLevel(logging.DEBUG)
```

Run tests with detailed output:

```bash
uv run pytest tests/_inference/ -v -s --log-cli-level=INFO
```
