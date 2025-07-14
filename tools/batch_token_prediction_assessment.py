#!/usr/bin/env python3
"""
Batch Token Prediction Assessment Tool

This tool provides comprehensive assessment of batch token prediction accuracy 
across multiple models, both thinking and non-thinking. It's designed to evaluate
the effectiveness of the batch token prediction system under various conditions
and model configurations.

Usage:
    python batch_token_prediction_assessment.py --models openai:o4-mini:medium anthropic:claude-sonnet-4-20250514:shallow
    python batch_token_prediction_assessment.py --all-models --output results.json
    python batch_token_prediction_assessment.py --thinking-models-only --verbose
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np
import polars as pl

from fenic.core.error import ConfigurationError

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import fenic
from fenic import col, semantic
from fenic._inference.batch_token_predictor import CompletionsBatchTokenPredictor
from fenic._inference.model_client import ModelClient
from fenic.api.session.config import (
    AnthropicModelConfig,
    AnthropicModelPreset,
    GoogleGLAModelConfig,
    GoogleModelPreset,
    GoogleVertexModelConfig,
    OpenAIModelConfig,
    OpenAIModelPreset,
    SemanticConfig,
    SessionConfig,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog


@dataclass
class ModelTestConfig:
    """Configuration for a model test."""
    provider: ModelProvider
    model_name: str
    preset_name: str
    description: str
    supports_thinking: bool


@dataclass
class BatchPredictionResult:
    """Results from a batch prediction test."""
    model_config: ModelTestConfig
    sample_size: int
    total_tokens_actual: float
    total_tokens_predicted: float
    completion_tokens_actual: float
    completion_tokens_predicted: float
    thinking_tokens_actual: float
    thinking_tokens_predicted: float
    combined_prediction_error_pct: float
    separate_prediction_error_pct: float
    combined_wins: bool
    advantage_pct: float
    confidence: float
    thinking_token_usage_pct: float
    thinking_completion_correlation: float
    thinking_cv: float
    completion_cv: float
    execution_time_seconds: float
    accuracy_grade: str


@dataclass
class AssessmentReport:
    """Complete assessment report."""
    timestamp: str
    test_parameters: Dict
    model_results: List[BatchPredictionResult]
    summary_statistics: Dict
    recommendations: Dict


class BatchTokenPredictionAssessment:
    """Main assessment class for batch token prediction evaluation."""

    def __init__(self, verbose: bool = False):
        """Initialize the assessment tool."""
        self.verbose = verbose
        self.results: List[BatchPredictionResult] = []
        self.model_catalog = model_catalog
    
    def create_model_config_from_spec(self, provider: str, model_name: str, preset_name: Optional[str] = None) -> ModelTestConfig:
        """Create a ModelTestConfig from provider and model name using the model catalog."""
        # Parse provider enum
        try:
            provider_enum = ModelProvider(provider)
        except ValueError as e:
            raise ConfigurationError(f"Unsupported provider: {provider}. Supported: {[p.value for p in ModelProvider]}") from e
        
        # Get model parameters from catalog
        model_params = self.model_catalog.get_completion_model_parameters(provider_enum, model_name)
        if model_params is None:
            raise ConfigurationError(f"Model '{model_name}' not found for provider '{provider}'. "
                           f"Available models: {self.model_catalog._get_supported_completions_models_by_provider_as_string(provider_enum)}")
        
        # Determine if model supports thinking
        supports_thinking = model_params.supports_reasoning
        
        # Create description
        description = f"{provider.title()} {model_name}"
        if preset_name:
            description += f" ({preset_name})"
        if supports_thinking:
            description += " [reasoning]"
        
        return ModelTestConfig(
            provider=provider_enum,
            model_name=model_name,
            preset_name=preset_name or "default",
            description=description,
            supports_thinking=supports_thinking
        )
        
    def create_session_config(self, model_config: ModelTestConfig) -> SessionConfig:
        """Create a session config for the given model."""
        app_name = "batch_prediction_test"
        
        # Create model configuration based on provider
        if model_config.provider == ModelProvider.OPENAI:
            if model_config.supports_thinking:
                presets = {
                    "low": OpenAIModelPreset(reasoning_effort="low"),
                    "medium": OpenAIModelPreset(reasoning_effort="medium"),
                    "high": OpenAIModelPreset(reasoning_effort="high"),
                    "default": OpenAIModelPreset(reasoning_effort="medium")
                }
                default_preset = model_config.preset_name
            else:
                presets = {"standard": OpenAIModelPreset(), "default": OpenAIModelPreset()}
                default_preset = "default"
                
            language_model = OpenAIModelConfig(
                model_name=model_config.model_name,
                rpm=500,
                tpm=100_000,
                presets=presets,
                default_preset=default_preset
            )
            
        elif model_config.provider == ModelProvider.ANTHROPIC:
            if model_config.supports_thinking:
                presets = {
                    "shallow": AnthropicModelPreset(thinking_token_budget=1024),
                    "deep": AnthropicModelPreset(thinking_token_budget=4096),
                    "default": AnthropicModelPreset(thinking_token_budget=1024)
                }
                default_preset = model_config.preset_name
            else:
                presets = {"standard": AnthropicModelPreset(), "default": AnthropicModelPreset()}
                default_preset = "default"
                
            language_model = AnthropicModelConfig(
                model_name=model_config.model_name,
                rpm=500,
                input_tpm=100_000,
                output_tpm=75_000,
                presets=presets,
                default_preset=default_preset
            )
            
        elif model_config.provider == ModelProvider.GOOGLE_GLA:
            if model_config.supports_thinking:
                presets = {
                    "thinking_disabled": GoogleModelPreset(),
                    "auto": GoogleModelPreset(thinking_token_budget=-1),
                    "default": GoogleModelPreset(thinking_token_budget=-1)
                }
                default_preset = model_config.preset_name
            else:
                presets = {"thinking_disabled": GoogleModelPreset(), "default": GoogleModelPreset()}
                default_preset = "default"
                
            language_model = GoogleGLAModelConfig(
                model_name=model_config.model_name,
                rpm=1000,
                tpm=500_000,
                presets=presets,
                default_preset=default_preset
            )
            
        elif model_config.provider == ModelProvider.GOOGLE_VERTEX:
            if model_config.supports_thinking:
                presets = {
                    "thinking_disabled": GoogleModelPreset(),
                    "auto": GoogleModelPreset(thinking_token_budget=-1),
                    "default": GoogleModelPreset(thinking_token_budget=-1)
                }
                default_preset = model_config.preset_name
            else:
                presets = {"thinking_disabled": GoogleModelPreset(), "default": GoogleModelPreset()}
                default_preset = "default"
                
            language_model = GoogleVertexModelConfig(
                model_name=model_config.model_name,
                rpm=1000,
                tpm=500_000,
                presets=presets,
                default_preset=default_preset
            )
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
            
        # Create embedding model (using OpenAI for all)
        embedding_model = OpenAIModelConfig(
            model_name="text-embedding-3-small",
            rpm=3000,
            tpm=1_000_000
        )
        
        return SessionConfig(
            app_name=app_name,
            semantic=SemanticConfig(
                language_models={"test_model": language_model},
                default_language_model="test_model",
                embedding_models={"embedding_model": embedding_model},
                default_embedding_model="embedding_model"
            )
        )

    def create_test_dataset(self, session, size: int = 100) -> pl.DataFrame:
        """Create test dataset with consistent 500-word children's stories about American history."""
        
        # Load the historical events dataset using fenic
        csv_path = Path(__file__).parent / "batch_prediction_test_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Test data file not found: {csv_path}")
        
        # Use fenic to load the CSV data
        history_df = session.read.csv(str(csv_path))
        history_data = history_df.to_polars()
        
        # Create story writing tasks
        tasks = []
        years = []
        
        # Repeat the dataset as needed to reach the target size
        num_repeats = (size // len(history_data)) + 1
        
        for _repeat in range(num_repeats):
            for row in history_data.iter_rows(named=True):
                if len(tasks) >= size:
                    break
                    
                # Create a rich, complex story writing task with plot planning
                task = (
                    f"Create a comprehensive children's story about {row['description']} in {row['year']}, "
                    f"set in the {row['region']} region with the theme of {row['theme']}. "
                    f"Your response must be 1000-2000 words and include both detailed planning and the complete story.\n\n"
                    f"**STEP 1: STORY OUTLINE & PLANNING**\n"
                    f"First, create a detailed plot outline that includes:\n"
                    f"• Character profiles (protagonist, supporting characters, antagonist if applicable)\n"
                    f"• Three-act structure with specific plot points and conflicts\n"
                    f"• Historical research notes about the event and its impact\n"
                    f"• Thematic elements and how they'll be woven throughout\n"
                    f"• Setting details specific to {row['region']} in {row['year']}\n"
                    f"• Educational goals and key historical facts to convey\n"
                    f"• Potential challenges in making complex historical events age-appropriate\n\n"
                    f"**STEP 2: COMPLETE STORY**\n"
                    f"Then write the full story (800-1500 words) incorporating your outline, featuring:\n"
                    f"• A relatable child protagonist (age 8-12) who experiences the historical event\n"
                    f"• Authentic dialogue reflecting the time period and regional speech patterns\n"
                    f"• Rich sensory details and atmospheric descriptions\n"
                    f"• Multiple perspectives showing how different groups experienced this event\n"
                    f"• Emotional depth that helps children understand the human cost and significance\n"
                    f"• Natural integration of historical facts without being didactic\n"
                    f"• A compelling narrative arc with rising action, climax, and resolution\n"
                    f"• Cultural sensitivity and historical accuracy\n\n"
                    f"The complete response should demonstrate sophisticated understanding of both storytelling "
                    f"craft and historical context, showing how {row['theme']} themes shaped people's experiences "
                    f"during this pivotal moment in American history."
                )
                
                tasks.append(task)
                years.append(row['year'])
                
                if len(tasks) >= size:
                    break
        
        # Create a DataFrame with the story writing tasks
        return session.create_dataframe({
            "year": years[:size],
            "story_prompt": tasks[:size]
        })

    def run_model_assessment(self, model_config: ModelTestConfig) -> BatchPredictionResult:
        """Run assessment for a single model."""
        if self.verbose:
            print(f"\\n🔬 Testing {model_config.description}...")
            
        # Create session and test data
        session_config = self.create_session_config(model_config)
        session = fenic.Session.get_or_create(session_config)
        source_df = self.create_test_dataset(session)
        
        # Tracking variables
        batch_prediction_calls = []
        batch_actual_tokens = {}
        batch_statistics = {}
        
        # Save original methods
        original_compute_batch_predictions = CompletionsBatchTokenPredictor.compute_batch_predictions
        original_sample_statistics = CompletionsBatchTokenPredictor._sample_statistics
        
        def track_sample_statistics(self, sample_tokens):
            """Track detailed statistics calculations."""
            predicted_tokens, confidence = original_sample_statistics(self, sample_tokens)
            
            if len(sample_tokens) >= 2:
                tokens = np.array(sample_tokens)
                self._temp_stats = {
                    'sample_tokens': sample_tokens.copy(),
                    'median': np.median(tokens),
                    'mean': np.mean(tokens),
                    'stddev': np.std(tokens),
                    'max': np.max(tokens),
                    'min': np.min(tokens),
                    'distribution_shape': self._detect_distribution_shape(tokens),
                    'predicted_tokens': predicted_tokens,
                    'confidence': confidence
                }
            
            return predicted_tokens, confidence
        
        def track_batch_predictions(self, batch_id, sample_responses):
            """Track when batch predictions are computed."""
            result = original_compute_batch_predictions(self, batch_id, sample_responses)
            
            if sample_responses:
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
                
                batch_statistics[batch_id] = {
                    'completion_tokens': completion_tokens,
                    'thinking_tokens': thinking_tokens,
                    'total_tokens': total_tokens,
                    'sample_count': len(sample_responses)
                }
                
                if hasattr(self, '_temp_stats'):
                    batch_statistics[batch_id].update(self._temp_stats)
                    delattr(self, '_temp_stats')
            
            return result
        
        def track_actual_tokens(original_method):
            async def wrapper(self, queue_item, maybe_response):
                result = await original_method(self, queue_item, maybe_response)
                
                if hasattr(maybe_response, 'usage') and maybe_response.usage:
                    batch_id = queue_item.batch_id
                    if batch_id not in batch_actual_tokens:
                        batch_actual_tokens[batch_id] = []
                    
                    actual_total = maybe_response.usage.completion_tokens + maybe_response.usage.thinking_tokens
                    batch_actual_tokens[batch_id].append({
                        'total': actual_total,
                        'completion': maybe_response.usage.completion_tokens,
                        'thinking': maybe_response.usage.thinking_tokens
                    })
                
                return result
            return wrapper
        
        try:
            # Patch tracking methods
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
                # Run semantic operation
                start_time = time.time()
                result_df = source_df.select(
                    col("year"),
                    semantic.map(
                        instruction="{story_prompt}",
                        max_output_tokens=3000  # Increased for 1000-2000 word stories
                    ).alias("story")
                )
                
                result = result_df.to_polars()
                execution_time = time.time() - start_time
                
        finally:
            session.stop()
            # Clean up database
            db_path = f"{session_config.app_name}.duckdb"
            if Path(db_path).exists():
                Path(db_path).unlink()
        
        # Analyze results
        if not batch_prediction_calls or not batch_actual_tokens:
            raise RuntimeError(f"No batch prediction data collected for {model_config.description}")
            
        return self._analyze_results(
            model_config, batch_prediction_calls, batch_actual_tokens, 
            batch_statistics, execution_time, len(result)
        )
    
    def _analyze_results(
        self, 
        model_config: ModelTestConfig,
        batch_calls: List[Dict],
        actual_tokens: Dict,
        statistics: Dict,
        execution_time: float,
        total_rows: int
    ) -> BatchPredictionResult:
        """Analyze the collected results and create a BatchPredictionResult."""
        
        # Get the main batch data (assuming one primary batch)
        main_call = batch_calls[0]  # Take first/primary batch
        batch_id = main_call['batch_id']
        predictions = main_call['predictions']
        stats = statistics.get(batch_id, {})
        actual = actual_tokens.get(batch_id, [])
        
        if not actual:
            raise RuntimeError("No actual token data collected")
        
        # Calculate actual token statistics
        total_actual = [item['total'] for item in actual]
        completion_actual = [item['completion'] for item in actual]
        thinking_actual = [item['thinking'] for item in actual]
        
        total_tokens_actual = sum(total_actual) / len(total_actual)
        completion_tokens_actual = sum(completion_actual) / len(completion_actual)
        thinking_tokens_actual = sum(thinking_actual) / len(thinking_actual)
        
        # Calculate prediction accuracy
        total_tokens_predicted = predictions.expected_output_tokens
        combined_error = abs(total_tokens_predicted - total_tokens_actual) / total_tokens_actual * 100
        
        # Calculate separate prediction error
        if completion_actual and len(completion_actual) >= 2:
            comp_mean = sum(completion_actual) / len(completion_actual)
            comp_std = np.std(completion_actual)
            comp_prediction = int(comp_mean + comp_std)
            
            think_prediction = 0
            if thinking_tokens_actual > 0:
                think_nonzero = [t for t in thinking_actual if t > 0]
                if think_nonzero:
                    think_mean = sum(think_nonzero) / len(think_nonzero)
                    think_std = np.std(think_nonzero) if len(think_nonzero) > 1 else 0
                    think_prediction = int(think_mean + think_std)
            
            separate_total_prediction = comp_prediction + think_prediction
            separate_error = abs(separate_total_prediction - total_tokens_actual) / total_tokens_actual * 100
        else:
            separate_error = 100.0  # Default high error if insufficient data
            
        # Determine winner and advantage
        combined_wins = combined_error < separate_error
        advantage_pct = abs(separate_error - combined_error)
        
        # Calculate additional metrics
        thinking_usage_pct = len([t for t in thinking_actual if t > 0]) / len(thinking_actual) * 100
        
        # Calculate correlation if both token types exist
        if thinking_tokens_actual > 0 and len(completion_actual) > 2:
            correlation = np.corrcoef(completion_actual, thinking_actual)[0, 1]
        else:
            correlation = 0.0
            
        # Calculate coefficient of variation
        thinking_cv = np.std(thinking_actual) / thinking_tokens_actual if thinking_tokens_actual > 0 else 0.0
        completion_cv = np.std(completion_actual) / completion_tokens_actual if completion_tokens_actual > 0 else 0.0
        
        # Determine accuracy grade
        if combined_error < 10:
            accuracy_grade = "EXCELLENT"
        elif combined_error < 20:
            accuracy_grade = "GOOD"
        elif combined_error < 35:
            accuracy_grade = "MODERATE"
        else:
            accuracy_grade = "POOR"
            
        return BatchPredictionResult(
            model_config=model_config,
            sample_size=main_call['sample_size'],
            total_tokens_actual=total_tokens_actual,
            total_tokens_predicted=total_tokens_predicted,
            completion_tokens_actual=completion_tokens_actual,
            completion_tokens_predicted=comp_prediction if 'comp_prediction' in locals() else 0,
            thinking_tokens_actual=thinking_tokens_actual,
            thinking_tokens_predicted=think_prediction if 'think_prediction' in locals() else 0,
            combined_prediction_error_pct=combined_error,
            separate_prediction_error_pct=separate_error,
            combined_wins=combined_wins,
            advantage_pct=advantage_pct,
            confidence=predictions.confidence,
            thinking_token_usage_pct=thinking_usage_pct,
            thinking_completion_correlation=correlation,
            thinking_cv=thinking_cv,
            completion_cv=completion_cv,
            execution_time_seconds=execution_time,
            accuracy_grade=accuracy_grade
        )

    def run_assessment(self, model_specs: List[str]) -> AssessmentReport:
        """Run assessment for specified model specifications.
        
        Args:
            model_specs: List of model specifications in format 'provider:model' or 'provider:model:preset'
        """
        print("🚀 Starting Batch Token Prediction Assessment")
        print(f"📊 Testing {len(model_specs)} model configurations")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        
        results = []
        for i, model_spec in enumerate(model_specs, 1):
            try:
                # Parse model specification
                parts = model_spec.split(':')
                if len(parts) < 2:
                    print(f"❌ Invalid model specification: {model_spec}. Expected format: provider:model or provider:model:preset")
                    continue
                
                provider = parts[0]
                model_name = parts[1]
                preset_name = parts[2] if len(parts) > 2 else None
                
                # Create model config from catalog
                model_config = self.create_model_config_from_spec(provider, model_name, preset_name)
                print(f"\\n[{i}/{len(model_specs)}] Testing: {model_config.description}")
            except Exception as e:
                print(f"❌ Failed to create model config for {model_spec}: {e}")
                continue
            
            try:
                result = self.run_model_assessment(model_config)
                results.append(result)
                
                if self.verbose:
                    self._print_result_summary(result)
                    
            except Exception as e:
                print(f"❌ Failed to test {model_config.description}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        
        total_time = time.time() - start_time
        
        # Generate summary statistics and recommendations
        summary_stats = self._generate_summary_statistics(results)
        recommendations = self._generate_recommendations(results)
        
        # Create final report
        report = AssessmentReport(
            timestamp=timestamp,
            test_parameters={
                "models_tested": len(results),
                "total_execution_time": total_time,
                "dataset_size": 100,
                "model_specifications": model_specs
            },
            model_results=results,
            summary_statistics=summary_stats,
            recommendations=recommendations
        )
        
        self._print_final_report(report)
        return report
    
    def _print_result_summary(self, result: BatchPredictionResult):
        """Print a summary of a single result."""
        print(f"  ✅ {result.accuracy_grade} accuracy ({result.combined_prediction_error_pct:.1f}% error)")
        print(f"  🎯 Combined prediction advantage: {result.advantage_pct:.1f}%")
        if result.model_config.supports_thinking:
            print(f"  🧠 Thinking tokens: {result.thinking_token_usage_pct:.0f}% usage, CV: {result.thinking_cv:.3f}")
    
    def _generate_summary_statistics(self, results: List[BatchPredictionResult]) -> Dict:
        """Generate summary statistics across all results."""
        if not results:
            return {}
            
        thinking_models = [r for r in results if r.model_config.supports_thinking]
        non_thinking_models = [r for r in results if not r.model_config.supports_thinking]
        
        return {
            "total_models_tested": len(results),
            "thinking_models_tested": len(thinking_models),
            "non_thinking_models_tested": len(non_thinking_models),
            "average_combined_error": sum(r.combined_prediction_error_pct for r in results) / len(results),
            "average_separate_error": sum(r.separate_prediction_error_pct for r in results) / len(results),
            "combined_wins_rate": sum(1 for r in results if r.combined_wins) / len(results) * 100,
            "best_accuracy_model": min(results, key=lambda r: r.combined_prediction_error_pct).model_config.description,
            "most_efficient_model": min(results, key=lambda r: r.total_tokens_actual).model_config.description,
            "average_thinking_usage": sum(r.thinking_token_usage_pct for r in thinking_models) / len(thinking_models) if thinking_models else 0,
        }
    
    def _generate_recommendations(self, results: List[BatchPredictionResult]) -> Dict:
        """Generate recommendations based on results."""
        if not results:
            return {}
            
        best_accuracy = min(results, key=lambda r: r.combined_prediction_error_pct)
        most_efficient = min(results, key=lambda r: r.total_tokens_actual)
        highest_advantage = max(results, key=lambda r: r.advantage_pct)
        
        return {
            "best_for_accuracy": {
                "model": best_accuracy.model_config.description,
                "error_rate": best_accuracy.combined_prediction_error_pct,
                "reasoning": "Lowest prediction error rate"
            },
            "best_for_efficiency": {
                "model": most_efficient.model_config.description,
                "avg_tokens": most_efficient.total_tokens_actual,
                "reasoning": "Lowest average token usage"
            },
            "prediction_strategy": {
                "approach": "Combined prediction",
                "success_rate": f"{sum(1 for r in results if r.combined_wins) / len(results) * 100:.0f}%",
                "reasoning": "Combined prediction outperforms separate prediction across all models"
            },
            "highest_combined_advantage": {
                "model": highest_advantage.model_config.description,
                "advantage": highest_advantage.advantage_pct,
                "reasoning": "Greatest improvement from using combined vs separate prediction"
            }
        }
    
    def _print_final_report(self, report: AssessmentReport):
        """Print the final assessment report."""
        print("\\n" + "="*80)
        print("📋 BATCH TOKEN PREDICTION ASSESSMENT REPORT")
        print("="*80)
        print(f"⏰ Completed: {report.timestamp}")
        print(f"📊 Models tested: {report.summary_statistics['total_models_tested']}")
        print(f"🧠 Thinking models: {report.summary_statistics['thinking_models_tested']}")
        print(f"💭 Non-thinking models: {report.summary_statistics['non_thinking_models_tested']}")
        
        print("\\n🎯 KEY FINDINGS:")
        print(f"  • Combined prediction wins: {report.summary_statistics['combined_wins_rate']:.0f}% of cases")
        print(f"  • Average combined error: {report.summary_statistics['average_combined_error']:.1f}%")
        print(f"  • Average separate error: {report.summary_statistics['average_separate_error']:.1f}%")
        
        print("\\n🏆 RECOMMENDATIONS:")
        print(f"  • Best accuracy: {report.recommendations['best_for_accuracy']['model']}")
        print(f"    ({report.recommendations['best_for_accuracy']['error_rate']:.1f}% error)")
        print(f"  • Most efficient: {report.recommendations['best_for_efficiency']['model']}")
        print(f"    ({report.recommendations['best_for_efficiency']['avg_tokens']:.0f} avg tokens)")
        print(f"  • Strategy: {report.recommendations['prediction_strategy']['approach']}")
        print(f"    ({report.recommendations['prediction_strategy']['success_rate']} success rate)")
        
        print("\\n📈 DETAILED RESULTS:")
        for result in sorted(report.model_results, key=lambda r: r.combined_prediction_error_pct):
            thinking_info = ""
            if result.model_config.supports_thinking:
                thinking_info = f", Thinking: {result.thinking_token_usage_pct:.0f}%"
            
            print(f"  {result.model_config.description:45} | "
                  f"{result.accuracy_grade:9} | "
                  f"{result.combined_prediction_error_pct:5.1f}% error | "
                  f"{result.advantage_pct:4.1f}% advantage{thinking_info}")


def main():
    """Main entry point for the assessment tool."""
    parser = argparse.ArgumentParser(
        description="Assess batch token prediction accuracy across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific models
  python batch_token_prediction_assessment.py --models openai:o4-mini:medium anthropic:claude-sonnet-4-20250514:shallow
  
  # Test all thinking models
  python batch_token_prediction_assessment.py --thinking-models-only
  
  # Test all models and save results
  python batch_token_prediction_assessment.py --all-models --output results.json --verbose
        """
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific model configurations to test (format: provider:model or provider:model:preset, e.g., openai:o4-mini:medium, anthropic:claude-sonnet-4-20250514:shallow)"
    )
    parser.add_argument(
        "--all-models", 
        action="store_true", 
        help="Test all available model configurations"
    )
    parser.add_argument(
        "--thinking-models-only", 
        action="store_true", 
        help="Test only models that support thinking tokens"
    )
    parser.add_argument(
        "--non-thinking-models-only", 
        action="store_true", 
        help="Test only models that do not support thinking tokens"
    )
    parser.add_argument(
        "--output", 
        help="Output file path for results (JSON format)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine which models to test
    assessment = BatchTokenPredictionAssessment(verbose=args.verbose)
    
    if args.models:
        model_specs = args.models
    elif args.all_models:
        # Generate specs for all available models
        model_specs = []
        for provider in ModelProvider:
            models = assessment.model_catalog._get_supported_completions_models_by_provider(provider)
            for model_name, params in models.items():
                if params.supports_reasoning:
                    # Add reasoning presets for thinking models
                    if provider == ModelProvider.OPENAI:
                        model_specs.extend([
                            f"{provider.value}:{model_name}:low",
                            f"{provider.value}:{model_name}:medium"
                        ])
                    elif provider == ModelProvider.ANTHROPIC:
                        model_specs.extend([
                            f"{provider.value}:{model_name}:shallow",
                            f"{provider.value}:{model_name}:deep"
                        ])
                    elif provider in [ModelProvider.GOOGLE_GLA, ModelProvider.GOOGLE_VERTEX]:
                        model_specs.append(f"{provider.value}:{model_name}:auto")
                else:
                    # Non-reasoning models
                    model_specs.append(f"{provider.value}:{model_name}")
    elif args.thinking_models_only:
        # Generate specs for thinking models only
        model_specs = []
        for provider in ModelProvider:
            models = assessment.model_catalog._get_supported_completions_models_by_provider(provider)
            for model_name, params in models.items():
                if params.supports_reasoning:
                    if provider == ModelProvider.OPENAI:
                        model_specs.append(f"{provider.value}:{model_name}:medium")
                    elif provider == ModelProvider.ANTHROPIC:
                        model_specs.append(f"{provider.value}:{model_name}:shallow")
                    elif provider in [ModelProvider.GOOGLE_GLA, ModelProvider.GOOGLE_VERTEX]:
                        model_specs.append(f"{provider.value}:{model_name}:auto")
    elif args.non_thinking_models_only:
        # Generate specs for non-thinking models only
        model_specs = []
        for provider in ModelProvider:
            models = assessment.model_catalog._get_supported_completions_models_by_provider(provider)
            for model_name, params in models.items():
                if not params.supports_reasoning:
                    model_specs.append(f"{provider.value}:{model_name}")
    else:
        # Default to a representative subset
        model_specs = [
            "openai:o4-mini:low",
            "google-gla:gemini-2.5-flash:auto",
            "google-gla:gemini-2.5-pro:auto",
            "openai:o4-mini:medium",
            "openai:o4-mini:high",
            "anthropic:claude-opus-4-0:shallow",
            "anthropic:claude-sonnet-4-0:deep",

        ]
        print("No models specified, testing default subset. Use --help for options.")
    
    # Run assessment
    try:
        report = assessment.run_assessment(model_specs)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_data = {
                "report": asdict(report),
                "raw_results": [asdict(result) for result in report.model_results]
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"\\n💾 Results saved to: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n🛑 Assessment interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Assessment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())