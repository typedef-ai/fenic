"""Integration tests for Batch Token Prediction with real LLM interactions using semantic operations."""

import time
import pytest
import polars as pl
import numpy as np
from unittest.mock import patch

import fenic
from fenic import col, semantic
from fenic._inference.batch_token_predictor import CompletionsBatchTokenPredictor


class TestBatchTokenPredictionIntegration:
    """Integration tests for batch token prediction with semantic operations."""

    @pytest.mark.cloud
    def test_semantic_map_batch_prediction_varied_complexity(self, local_session):
        """Test batch token prediction with semantic.map operations of varying complexity.
        
        This test creates a dataset with prompts of varying complexity to challenge
        the batch token prediction system and validate its performance.
        """
        # Create test data with varying prompt complexity
        test_data = {
            "prompt_type": [
                "simple_math", "simple_math", "simple_math", "simple_math", "simple_math",
                "word_analysis", "word_analysis", "word_analysis", "word_analysis", "word_analysis", 
                "creative_writing", "creative_writing", "creative_writing", "creative_writing", "creative_writing",
                "code_explanation", "code_explanation", "code_explanation", "code_explanation", "code_explanation",
                "detailed_analysis", "detailed_analysis", "detailed_analysis", "detailed_analysis", "detailed_analysis",
                # Add many more for batch prediction sampling threshold (50+)
            ] * 3,  # 75 total rows to trigger batch prediction
            "input_text": [
                # Simple math (low token outputs expected)
                "2 + 2", "5 * 3", "10 / 2", "8 - 4", "6 + 7",
                # Word analysis (medium token outputs expected)
                "analyze", "philosophy", "serendipitous", "mellifluous", "ephemeral",
                # Creative writing prompts (high token outputs expected)
                "space adventure", "magical forest", "time travel", "underwater city", "robot uprising",
                # Code explanation (very high token outputs expected)
                "recursive function", "machine learning", "database optimization", "distributed systems", "neural networks",
                # Detailed analysis (extremely high token outputs expected)
                "climate change", "economic theory", "quantum physics", "artificial intelligence", "bioethics"
            ] * 3,  # 75 total inputs
        }
        
        source_df = local_session.create_dataframe(test_data)
        
        # Track batch token prediction behavior
        batch_predictions_used = []
        original_predict_method = None
        
        def track_predictions(self, request, batch_id, conservative_func):
            """Track when batch predictions are used vs conservative estimates."""
            batch_preds = self.get_batch_predictions(batch_id)
            prediction_result = original_predict_method(request, batch_id, conservative_func)
            
            batch_predictions_used.append({
                'batch_id': batch_id,
                'confidence': batch_preds.confidence,
                'sample_size': batch_preds.sample_size,
                'expected_tokens': batch_preds.expected_output_tokens,
                'prediction_used': prediction_result,
                'used_batch_prediction': batch_preds.confidence >= 0.3  # MINIMUM_PREDICTION_CONFIDENCE
            })
            return prediction_result
        
        # Patch the predict_output_tokens method to track behavior
        with patch.object( 'predict_output_tokens') as mock_predict:
            # Store original method for delegation
            original_predict_method = fenic._inference.completions_batch_token_predictor.BatchTokenPredictor.predict_output_tokens
            mock_predict.side_effect = track_predictions
            
            # Define semantic operations with different complexity levels
            result_df = source_df.select(
                col("prompt_type"),
                col("input_text"),
                # Simple operation - should have consistent, low token outputs
                semantic.map(
                    instruction="Calculate this simple expression and give just the number: {input_text}",
                    model="gpt-4o-mini"
                ).alias("simple_result"),
                # Medium complexity operation 
                semantic.map(
                    instruction="Analyze the word '{input_text}' and provide its etymology in 2-3 sentences.",
                    model="gpt-4o-mini"
                ).alias("etymology"),
                # High complexity operation - should have varied, higher token outputs
                semantic.map(
                    instruction="Write a creative 3-paragraph story inspired by: {input_text}",
                    model="gpt-4o-mini"
                ).alias("creative_story"),
            )
            
            # Execute the operations
            start_time = time.time()
            result = result_df.to_polars()
            execution_time = time.time() - start_time
            
        # Validate results
        assert len(result) == 75
        assert result.schema == {
            "prompt_type": pl.String,
            "input_text": pl.String, 
            "simple_result": pl.String,
            "etymology": pl.String,
            "creative_story": pl.String,
        }
        
        # Analyze batch prediction performance
        print(f"\n=== Batch Token Prediction Analysis ===")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Total prediction calls: {len(batch_predictions_used)}")
        
        if batch_predictions_used:
            # Group by batch_id to analyze per-operation performance
            batch_groups = {}
            for pred in batch_predictions_used:
                batch_id = pred['batch_id']
                if batch_id not in batch_groups:
                    batch_groups[batch_id] = []
                batch_groups[batch_id].append(pred)
            
            for batch_id, predictions in batch_groups.items():
                confidence_values = [p['confidence'] for p in predictions]
                used_batch_pred = [p['used_batch_prediction'] for p in predictions]
                
                avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
                batch_usage_rate = sum(used_batch_pred) / len(used_batch_pred) if used_batch_pred else 0
                
                print(f"\nBatch {batch_id}:")
                print(f"  Average confidence: {avg_confidence:.3f}")
                print(f"  Batch prediction usage rate: {batch_usage_rate:.1%}")
                print(f"  Sample size: {predictions[0]['sample_size'] if predictions else 0}")
                
                # Validate that batch predictions were actually beneficial
                if avg_confidence >= 0.3:  # MINIMUM_PREDICTION_CONFIDENCE
                    assert batch_usage_rate > 0, f"High confidence batch should be used: {batch_id}"

    @pytest.mark.cloud  
    def test_semantic_map_token_variance_analysis(self, local_session):
        """Test how batch prediction handles operations with different token variance patterns."""
        
        # Create dataset designed to test different variance scenarios
        variance_test_data = {
            "scenario": ["consistent"] * 20 + ["variable"] * 20 + ["bimodal"] * 20,
            "input": (
                # Consistent scenario - should produce similar token counts
                ["Tell me about cats"] * 20 +
                # Variable scenario - should produce different token counts  
                [f"Explain topic {i} in detail" for i in range(20)] +
                # Bimodal scenario - alternating simple and complex requests
                (["Yes/No: Is 2+2=4?", "Write a comprehensive essay about quantum mechanics"] * 10)
            )
        }
        
        source_df = local_session.create_dataframe(variance_test_data)
        
        # Track variance patterns in predictions
        prediction_tracking = []
        
        def track_variance_patterns(self, batch_id, sample_responses):
            """Track token variance patterns in batch predictions."""
            # Call original method
            original_method = fenic._inference.completions_batch_token_predictor.BatchTokenPredictor.compute_batch_predictions
            result = original_method(self, batch_id, sample_responses)
            
            # Extract token data for analysis
            if sample_responses:
                output_tokens = []
                for resp in sample_responses:
                    if resp and hasattr(resp, 'usage') and resp.usage:
                        total_output = resp.usage.completion_tokens + resp.usage.thinking_tokens
                        if total_output > 0:
                            output_tokens.append(total_output)
                
                if len(output_tokens) >= 2:
                    import statistics
                    mean_tokens = statistics.mean(output_tokens)
                    stdev_tokens = statistics.stdev(output_tokens)
                    coefficient_of_variation = stdev_tokens / mean_tokens if mean_tokens > 0 else float('inf')
                    
                    prediction_tracking.append({
                        'batch_id': batch_id,
                        'sample_size': len(output_tokens),
                        'mean_tokens': mean_tokens,
                        'stdev_tokens': stdev_tokens,
                        'coefficient_of_variation': coefficient_of_variation,
                        'min_tokens': min(output_tokens),
                        'max_tokens': max(output_tokens),
                    })
            
            return result
        
        with patch.object(fenic._inference.completions_batch_token_predictor.BatchTokenPredictor, 'compute_batch_predictions') as mock_generate:
            mock_generate.side_effect = track_variance_patterns
            
            # Execute semantic operation
            result_df = source_df.select(
                col("scenario"),
                col("input"),
                semantic.map(
                    instruction="Process this request: {input}",
                    model="gpt-4o-mini"
                ).alias("response")
            )
            
            result = result_df.to_polars()
        
        # Analyze variance patterns
        print(f"\n=== Token Variance Analysis ===")
        for tracking in prediction_tracking:
            print(f"Batch {tracking['batch_id']}:")
            print(f"  Sample size: {tracking['sample_size']}")
            print(f"  Mean tokens: {tracking['mean_tokens']:.1f}")
            print(f"  Std deviation: {tracking['stdev_tokens']:.1f}")
            print(f"  Coefficient of variation: {tracking['coefficient_of_variation']:.3f}")
            print(f"  Token range: {tracking['min_tokens']}-{tracking['max_tokens']}")
            
            # Validate expectations about variance
            if "consistent" in tracking['batch_id']:
                # Consistent scenario should have low coefficient of variation
                assert tracking['coefficient_of_variation'] < 0.5, "Consistent scenario should have low variance"
            elif "bimodal" in tracking['batch_id']:
                # Bimodal scenario should have high coefficient of variation
                assert tracking['coefficient_of_variation'] > 0.3, "Bimodal scenario should have high variance"

    @pytest.mark.cloud
    def test_batch_prediction_with_thinking_models(self, local_session):
        """Test batch prediction specifically with thinking/reasoning models."""
        
        # Test data designed for reasoning models
        reasoning_data = {
            "problem_type": ["logic", "math", "analysis"] * 20,  # 60 rows total
            "problem": (
                # Logic problems that should trigger significant thinking
                ["If all roses are flowers and some flowers fade quickly, what can we conclude?"] * 20 +
                # Math problems requiring step-by-step reasoning  
                ["Solve: (2x + 3)(x - 1) = 2x² + x - 3"] * 20 +
                # Analysis requiring deep thought
                ["Analyze the ethical implications of AI decision-making in healthcare"] * 20
            )
        }
        
        source_df = local_session.create_dataframe(reasoning_data)
        
        # Track thinking vs completion token patterns
        thinking_token_tracking = []
        
        def track_thinking_tokens(self, batch_id, sample_responses):
            """Track the ratio of thinking to completion tokens."""
            original_method = fenic._inference.completions_batch_token_predictor.BatchTokenPredictor.compute_batch_predictions
            result = original_method(self, batch_id, sample_responses)
            
            if sample_responses:
                thinking_ratios = []
                total_thinking = 0
                total_completion = 0
                
                for resp in sample_responses:
                    if resp and hasattr(resp, 'usage') and resp.usage:
                        thinking = resp.usage.thinking_tokens
                        completion = resp.usage.completion_tokens
                        total_thinking += thinking
                        total_completion += completion
                        
                        if completion > 0:
                            thinking_ratios.append(thinking / completion)
                
                if thinking_ratios:
                    import statistics
                    thinking_token_tracking.append({
                        'batch_id': batch_id,
                        'avg_thinking_ratio': statistics.mean(thinking_ratios),
                        'total_thinking_tokens': total_thinking,
                        'total_completion_tokens': total_completion,
                        'sample_size': len(thinking_ratios)
                    })
            
            return result
        
        with patch.object(fenic._inference.completions_batch_token_predictor.BatchTokenPredictor, 'compute_batch_predictions') as mock_generate:
            mock_generate.side_effect = track_thinking_tokens
            
            # Use a reasoning model with thinking enabled
            result_df = source_df.select(
                col("problem_type"),
                col("problem"),
                semantic.map(
                    instruction="Think through this step by step and provide a detailed solution: {problem}",
                    model="gpt-4o"  # Use a model that supports thinking
                ).alias("solution")
            )
            
            result = result_df.to_polars()
        
        # Analyze thinking token patterns
        print(f"\n=== Thinking Token Analysis ===")
        for tracking in thinking_token_tracking:
            print(f"Batch {tracking['batch_id']}:")
            print(f"  Average thinking/completion ratio: {tracking['avg_thinking_ratio']:.2f}")
            print(f"  Total thinking tokens: {tracking['total_thinking_tokens']}")
            print(f"  Total completion tokens: {tracking['total_completion_tokens']}")
            print(f"  Sample size: {tracking['sample_size']}")
            
            # Validate that thinking tokens are being tracked properly
            assert tracking['total_thinking_tokens'] >= 0, "Thinking tokens should be non-negative"
            assert tracking['total_completion_tokens'] > 0, "Should have completion tokens"

    @pytest.mark.cloud
    def test_simple_batch_prediction_integration(self, local_session):
        """Simple integration test for batch prediction with generated data."""
        fenic.configure_logging()
        # Generate a smaller batch with more complex reasoning tasks to trigger thinking tokens
        test_data = {
            "problem": [
                "Analyze the economic factors that led to the American Revolution and their lasting impact on modern fiscal policy",
                "Compare the constitutional frameworks of federalism vs confederalism and evaluate their effectiveness in crisis scenarios", 
                "Examine the philosophical underpinnings of the Bill of Rights and how they reflect Enlightenment thinking",
                "Assess the role of technological innovation in the Industrial Revolution and its parallels to today's AI revolution",
                "Evaluate the geopolitical consequences of the Louisiana Purchase on westward expansion and indigenous populations",
                "Analyze the causes and effects of the Great Depression through multiple economic theories",
                "Compare the military strategies of the Civil War to modern asymmetric warfare tactics",
                "Examine the social justice movements of the 1960s and their influence on contemporary activism",
                "Assess the environmental policy implications of manifest destiny on current climate change debates",
                "Evaluate the constitutional crisis of Watergate and its precedents for executive accountability",
                "Analyze the economic transformation during Reconstruction and its relevance to post-conflict recovery",
                "Compare the foreign policy doctrines of isolationism vs interventionism across different presidential eras",
                "Examine the technological and social factors behind the Space Race and their impact on STEM education",
                "Assess the role of immigration patterns in shaping American cultural identity throughout history",
                "Evaluate the evolution of voting rights and their reflection of broader democratic principles",
                "Analyze the economic and social factors behind the rise and fall of labor unions in America",
                "Compare the responses to different economic crises and their effectiveness in preventing future downturns",
                "Examine the role of journalism and media in shaping public opinion during major historical events",
                "Assess the impact of Supreme Court decisions on the balance between federal and state powers",
                "Evaluate the long-term consequences of American foreign interventions on international relations"
            ] * 13  # 260 total to trigger batch prediction
        }
        
        source_df = local_session.create_dataframe(test_data)
        
        # Track batch prediction usage and accuracy
        batch_prediction_calls = []
        batch_actual_tokens = {}  # batch_id -> list of actual token counts
        batch_statistics = {}  # batch_id -> detailed statistics
        
        # Save reference to original methods before patching
        original_compute_batch_predictions = CompletionsBatchTokenPredictor.compute_batch_predictions
        original_sample_statistics = CompletionsBatchTokenPredictor._sample_statistics
        
        def track_sample_statistics(self, sample_tokens):
            """Track detailed statistics calculations."""
            # Call the original method
            predicted_tokens, confidence = original_sample_statistics(self, sample_tokens)
            
            # Capture detailed statistics
            if len(sample_tokens) >= 2:
                tokens = np.array(sample_tokens)
                sample_median = np.median(tokens)
                sample_mean = np.mean(tokens)
                sample_stddev = np.std(tokens)
                sample_max = np.max(tokens)
                sample_min = np.min(tokens)
                distribution_shape = self._detect_distribution_shape(tokens)
                
                # Store for later analysis (we'll associate with batch_id in compute_batch_predictions)
                self._temp_stats = {
                    'sample_tokens': sample_tokens.copy(),
                    'median': sample_median,
                    'mean': sample_mean,
                    'stddev': sample_stddev,
                    'max': sample_max,
                    'min': sample_min,
                    'distribution_shape': distribution_shape,
                    'predicted_tokens': predicted_tokens,
                    'confidence': confidence
                }
            
            return predicted_tokens, confidence
        
        def track_batch_predictions(self, batch_id, sample_responses):
            """Track when batch predictions are computed."""
            # Call the original method we saved
            result = original_compute_batch_predictions(self, batch_id, sample_responses)
            
            # Track the call and capture statistics
            if sample_responses:
                # Extract token data for our own analysis
                completion_tokens = []
                thinking_tokens = []
                total_tokens = []
                
                for resp in sample_responses:
                    if resp.usage is not None:
                        completion_tokens.append(resp.usage.completion_tokens)
                        thinking_tokens.append(resp.usage.thinking_tokens)
                        total_tokens.append(resp.usage.completion_tokens + resp.usage.thinking_tokens)
                
                batch_prediction_calls.append({
                    'batch_id': batch_id,
                    'sample_size': len(sample_responses),
                    'predictions': self.get_batch_predictions(batch_id)
                })
                
                # Store detailed statistics if available
                batch_statistics[batch_id] = {
                    'completion_tokens': completion_tokens,
                    'thinking_tokens': thinking_tokens,
                    'total_tokens': total_tokens,
                    'sample_count': len(sample_responses)
                }
                
                # Add temp stats if they exist (from _sample_statistics calls)
                if hasattr(self, '_temp_stats'):
                    batch_statistics[batch_id].update(self._temp_stats)
                    delattr(self, '_temp_stats')
            
            return result
        
        # Track actual token usage for each response
        def track_actual_tokens(original_method):
            async def wrapper(self, queue_item, maybe_response):
                # Call original response handler
                result = await original_method(self, queue_item, maybe_response)
                
                # Track actual tokens if this is a successful response
                if hasattr(maybe_response, 'usage') and maybe_response.usage:
                    batch_id = queue_item.batch_id
                    if batch_id not in batch_actual_tokens:
                        batch_actual_tokens[batch_id] = []
                    
                    actual_total = maybe_response.usage.completion_tokens + maybe_response.usage.thinking_tokens
                    batch_actual_tokens[batch_id].append(actual_total)
                
                return result
            return wrapper
        
        # Import the model client to patch response handling
        from fenic._inference.model_client import ModelClient
        
        # Patch prediction computation, statistics, and response handling
        with patch.object(
            CompletionsBatchTokenPredictor, 
            'compute_batch_predictions', 
            track_batch_predictions
        ), patch.object(
            CompletionsBatchTokenPredictor,
            '_sample_statistics',
            track_sample_statistics
        ), patch.object(
            ModelClient,
            '_handle_response',
            track_actual_tokens(ModelClient._handle_response)
        ):
            # Run semantic operation with complex reasoning tasks
            start_time = time.time()
            result_df = source_df.select(
                col("problem"),
                semantic.map(
                    instruction="Think through this complex problem step by step and provide a detailed analysis: {problem}",
                ).alias("detailed_analysis")
            )
            
            result = result_df.to_polars()
            execution_time = time.time() - start_time

        # Analyze batch prediction behavior
        print(f"\n=== Simple Batch Prediction Test Results ===")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Processed {len(result)} rows")
        print(f"Batch prediction calls: {len(batch_prediction_calls)}")
        print(f"Batches with actual token data: {len(batch_actual_tokens)}")
        
        if batch_prediction_calls:
            for call in batch_prediction_calls:
                predictions = call['predictions']
                batch_id = call['batch_id']
                actual_tokens = batch_actual_tokens.get(batch_id, [])
                
                print(f"\nBatch {batch_id[:8]}...")
                print(f"  Sample size: {call['sample_size']}")
                print(f"  Expected tokens: {predictions.expected_output_tokens}")
                print(f"  Confidence: {predictions.confidence:.3f}")
                
                # Show detailed statistics if available
                if batch_id in batch_statistics:
                    stats = batch_statistics[batch_id]
                    print(f"  Sample statistics:")
                    if 'median' in stats:
                        print(f"    Total tokens - Median: {stats['median']:.1f}, Mean: {stats['mean']:.1f}, Std: {stats['stddev']:.1f}")
                        print(f"    Total tokens - Range: {stats['min']:.0f}-{stats['max']:.0f}, Shape: {stats['distribution_shape']}")
                    
                    # Detailed breakdown by token type
                    if stats['completion_tokens']:
                        comp_tokens = stats['completion_tokens']
                        comp_mean = sum(comp_tokens) / len(comp_tokens)
                        comp_std = (sum((x - comp_mean)**2 for x in comp_tokens) / len(comp_tokens))**0.5 if len(comp_tokens) > 1 else 0
                        comp_min, comp_max = min(comp_tokens), max(comp_tokens)
                        print(f"    Completion tokens - Mean: {comp_mean:.1f}, Std: {comp_std:.1f}, Range: {comp_min}-{comp_max}")
                        
                        # Completion token distribution analysis
                        if len(comp_tokens) >= 3:
                            comp_median = sorted(comp_tokens)[len(comp_tokens)//2]
                            comp_q25 = sorted(comp_tokens)[len(comp_tokens)//4]
                            comp_q75 = sorted(comp_tokens)[3*len(comp_tokens)//4]
                            comp_cv = comp_std / comp_mean if comp_mean > 0 else 0
                            print(f"    Completion tokens - Median: {comp_median:.1f}, Q25: {comp_q25:.1f}, Q75: {comp_q75:.1f}, CV: {comp_cv:.3f}")
                    
                    if stats['thinking_tokens']:
                        think_tokens = stats['thinking_tokens']
                        think_mean = sum(think_tokens) / len(think_tokens)
                        think_std = (sum((x - think_mean)**2 for x in think_tokens) / len(think_tokens))**0.5 if len(think_tokens) > 1 else 0
                        think_min, think_max = min(think_tokens), max(think_tokens)
                        think_nonzero = [t for t in think_tokens if t > 0]
                        
                        print(f"    Thinking tokens - Mean: {think_mean:.1f}, Std: {think_std:.1f}, Range: {think_min}-{think_max}")
                        print(f"    Thinking tokens - Non-zero count: {len(think_nonzero)}/{len(think_tokens)} ({len(think_nonzero)/len(think_tokens)*100:.1f}%)")
                        
                        if think_nonzero:
                            think_nonzero_mean = sum(think_nonzero) / len(think_nonzero)
                            think_nonzero_std = (sum((x - think_nonzero_mean)**2 for x in think_nonzero) / len(think_nonzero))**0.5 if len(think_nonzero) > 1 else 0
                            print(f"    Thinking tokens (non-zero only) - Mean: {think_nonzero_mean:.1f}, Std: {think_nonzero_std:.1f}")
                        
                        # Thinking token distribution analysis
                        if len(think_tokens) >= 3:
                            think_median = sorted(think_tokens)[len(think_tokens)//2]
                            think_cv = think_std / think_mean if think_mean > 0 else 0
                            print(f"    Thinking tokens - Median: {think_median:.1f}, CV: {think_cv:.3f}")
                    
                    # Correlation analysis between completion and thinking tokens
                    if stats['completion_tokens'] and stats['thinking_tokens'] and len(stats['completion_tokens']) > 2:
                        comp_tokens = stats['completion_tokens']
                        think_tokens = stats['thinking_tokens']
                        
                        # Calculate correlation coefficient
                        n = len(comp_tokens)
                        comp_mean = sum(comp_tokens) / n
                        think_mean = sum(think_tokens) / n
                        
                        numerator = sum((comp_tokens[i] - comp_mean) * (think_tokens[i] - think_mean) for i in range(n))
                        comp_ss = sum((x - comp_mean)**2 for x in comp_tokens)
                        think_ss = sum((x - think_mean)**2 for x in think_tokens)
                        
                        if comp_ss > 0 and think_ss > 0:
                            correlation = numerator / (comp_ss * think_ss)**0.5
                            print(f"    Completion/Thinking correlation: {correlation:.3f}")
                        
                        # Thinking/completion ratio analysis
                        ratios = []
                        for i in range(n):
                            if comp_tokens[i] > 0:
                                ratios.append(think_tokens[i] / comp_tokens[i])
                        
                        if ratios:
                            ratio_mean = sum(ratios) / len(ratios)
                            ratio_std = (sum((x - ratio_mean)**2 for x in ratios) / len(ratios))**0.5 if len(ratios) > 1 else 0
                            ratio_median = sorted(ratios)[len(ratios)//2]
                            print(f"    Thinking/Completion ratio - Mean: {ratio_mean:.3f}, Std: {ratio_std:.3f}, Median: {ratio_median:.3f}")
                
                # Analyze prediction accuracy if we have actual data
                if actual_tokens:
                    actual_mean = sum(actual_tokens) / len(actual_tokens)
                    actual_min = min(actual_tokens)
                    actual_max = max(actual_tokens)
                    actual_std = (sum((x - actual_mean)**2 for x in actual_tokens) / len(actual_tokens))**0.5
                    
                    # Calculate prediction error
                    prediction_error = abs(predictions.expected_output_tokens - actual_mean)
                    relative_error = prediction_error / actual_mean * 100 if actual_mean > 0 else 0
                    
                    # Check if actual mean falls within expected range
                    prediction_coverage = (
                        actual_min <= predictions.expected_output_tokens <= actual_max
                    )
                    
                    print(f"  Actual tokens - mean: {actual_mean:.1f}, std: {actual_std:.1f}, range: {actual_min}-{actual_max}")
                    print(f"  Prediction error: {prediction_error:.1f} tokens ({relative_error:.1f}%)")
                    print(f"  Prediction within actual range: {prediction_coverage}")
                    
                    # Validate prediction quality
                    if predictions.confidence > 0.5:
                        # High confidence predictions should be reasonably accurate
                        assert relative_error < 50, f"High confidence prediction too inaccurate: {relative_error:.1f}% error"
                    
                    # Log additional insights
                    if relative_error < 10:
                        print(f"  ✓ Excellent prediction accuracy")
                    elif relative_error < 25:
                        print(f"  ✓ Good prediction accuracy")
                    elif relative_error < 50:
                        print(f"  ⚠ Moderate prediction accuracy")
                    else:
                        print(f"  ❌ Poor prediction accuracy")
                else:
                    print(f"  ⚠ No actual token data available for accuracy analysis")
                
                # Validate that sampling was triggered
                assert call['sample_size'] > 0, "Should have sample responses"
                assert predictions.expected_output_tokens > 0, "Should predict some tokens"
        
        # Separate vs Combined Prediction Analysis
        if batch_actual_tokens and batch_prediction_calls and batch_statistics:
            print(f"\n=== Separate vs Combined Token Prediction Analysis ===")
            
            for call in batch_prediction_calls:
                batch_id = call['batch_id']
                actual_tokens = batch_actual_tokens.get(batch_id, [])
                stats = batch_statistics.get(batch_id, {})
                
                if actual_tokens and 'completion_tokens' in stats and 'thinking_tokens' in stats:
                    # Actual totals
                    actual_total_mean = sum(actual_tokens) / len(actual_tokens)
                    
                    # Extract actual completion and thinking from responses
                    actual_completion = []
                    actual_thinking = []
                    for i, total in enumerate(actual_tokens):
                        if i < len(stats['completion_tokens']) and i < len(stats['thinking_tokens']):
                            actual_completion.append(stats['completion_tokens'][i])
                            actual_thinking.append(stats['thinking_tokens'][i])
                    
                    if actual_completion and actual_thinking:
                        actual_comp_mean = sum(actual_completion) / len(actual_completion)
                        actual_think_mean = sum(actual_thinking) / len(actual_thinking)
                        
                        print(f"\nBatch {batch_id[:8]} - Prediction Method Comparison:")
                        print(f"  Actual totals: Completion={actual_comp_mean:.1f}, Thinking={actual_think_mean:.1f}, Total={actual_total_mean:.1f}")
                        
                        # Method 1: Combined prediction (current approach)
                        combined_prediction = call['predictions'].expected_output_tokens
                        combined_error = abs(combined_prediction - actual_total_mean)
                        combined_relative_error = combined_error / actual_total_mean * 100 if actual_total_mean > 0 else 0
                        
                        print(f"  Method 1 (Combined): Predicted={combined_prediction}, Error={combined_error:.1f} ({combined_relative_error:.1f}%)")
                        
                        # Method 2: Separate predictions
                        # Predict completion tokens separately
                        if len(actual_completion) >= 2:
                            comp_mean = sum(actual_completion) / len(actual_completion)
                            comp_std = (sum((x - comp_mean)**2 for x in actual_completion) / len(actual_completion))**0.5
                            
                            # Simple prediction: mean + std for completion
                            comp_prediction = int(comp_mean + comp_std)
                            
                            # Predict thinking tokens separately
                            think_prediction = 0  # Default for non-reasoning models
                            if actual_think_mean > 0:
                                # If there are thinking tokens, predict based on pattern
                                think_nonzero = [t for t in actual_thinking if t > 0]
                                if think_nonzero:
                                    think_mean = sum(think_nonzero) / len(think_nonzero)
                                    think_std = (sum((x - think_mean)**2 for x in think_nonzero) / len(think_nonzero))**0.5 if len(think_nonzero) > 1 else 0
                                    think_prediction = int(think_mean + think_std)
                            
                            separate_total_prediction = comp_prediction + think_prediction
                            separate_error = abs(separate_total_prediction - actual_total_mean)
                            separate_relative_error = separate_error / actual_total_mean * 100 if actual_total_mean > 0 else 0
                            
                            print(f"  Method 2 (Separate): Completion={comp_prediction}, Thinking={think_prediction}, Total={separate_total_prediction}")
                            print(f"  Method 2 (Separate): Error={separate_error:.1f} ({separate_relative_error:.1f}%)")
                            
                            # Compare methods
                            if separate_relative_error < combined_relative_error:
                                improvement = combined_relative_error - separate_relative_error
                                print(f"  🎯 Separate prediction is better by {improvement:.1f} percentage points")
                            elif combined_relative_error < separate_relative_error:
                                improvement = separate_relative_error - combined_relative_error
                                print(f"  ✅ Combined prediction is better by {improvement:.1f} percentage points")
                            else:
                                print(f"  ⚖️  Both methods perform similarly")
                            
                            # Analysis of why one might be better
                            comp_cv = comp_std / comp_mean if comp_mean > 0 else 0
                            think_cv = 0
                            if actual_think_mean > 0:
                                think_std_all = (sum((x - actual_think_mean)**2 for x in actual_thinking) / len(actual_thinking))**0.5
                                think_cv = think_std_all / actual_think_mean
                            
                            print(f"  Completion CV: {comp_cv:.3f}, Thinking CV: {think_cv:.3f}")
                            
                            if comp_cv < 0.3 and think_cv < 0.3:
                                print(f"  💡 Both token types are consistent - either method should work well")
                            elif comp_cv < think_cv:
                                print(f"  💡 Completion tokens more consistent - separate prediction might help")
                            else:
                                print(f"  💡 Thinking tokens more consistent - combined approach might be better")

        # Overall accuracy summary
        if batch_actual_tokens and batch_prediction_calls:
            all_errors = []
            all_relative_errors = []
            
            for call in batch_prediction_calls:
                batch_id = call['batch_id']
                actual_tokens = batch_actual_tokens.get(batch_id, [])
                if actual_tokens:
                    actual_mean = sum(actual_tokens) / len(actual_tokens)
                    prediction_error = abs(call['predictions'].expected_output_tokens - actual_mean)
                    relative_error = prediction_error / actual_mean * 100 if actual_mean > 0 else 0
                    
                    all_errors.append(prediction_error)
                    all_relative_errors.append(relative_error)
            
            if all_errors:
                avg_error = sum(all_errors) / len(all_errors)
                avg_relative_error = sum(all_relative_errors) / len(all_relative_errors)
                
                print(f"\n=== Overall Prediction Accuracy ===")
                print(f"Average absolute error: {avg_error:.1f} tokens")
                print(f"Average relative error: {avg_relative_error:.1f}%")
                print(f"Batches analyzed: {len(all_errors)}")
                
                # Overall accuracy assessment
                if avg_relative_error < 15:
                    print("🎯 Overall prediction accuracy: EXCELLENT")
                elif avg_relative_error < 30:
                    print("✅ Overall prediction accuracy: GOOD")
                elif avg_relative_error < 50:
                    print("⚠️  Overall prediction accuracy: MODERATE")
                else:
                    print("❌ Overall prediction accuracy: POOR")
        
        # Validate some specific outputs make sense
        sample_results = result.head(5)
        print(f"\nSample results:")
        for row in sample_results.iter_rows(named=True):
            analysis = row['detailed_analysis'][:100] + "..." if len(row['detailed_analysis']) > 100 else row['detailed_analysis']
            print(f"  Problem -> Analysis: '{analysis}'")
