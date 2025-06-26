"""Utility functions for testing batch token prediction scenarios."""

import random
import math
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TokenDistribution:
    """Represents a token distribution pattern for testing."""
    name: str
    description: str
    token_values: List[int]
    expected_confidence_range: Tuple[float, float]  # (min, max) expected confidence


class BatchPredictionTestDataGenerator:
    """Generates test data for challenging batch prediction scenarios."""
    
    @staticmethod
    def uniform_distribution(base_tokens: int = 2048, count: int = 20) -> TokenDistribution:
        """Generate uniform token distribution (high confidence expected).
        
        Simulates a batch where all responses use roughly the same token count,
        typical for similar prompt complexity.
        """
        return TokenDistribution(
            name="uniform",
            description="Consistent token counts with minimal variance",
            token_values=[base_tokens] * count,
            expected_confidence_range=(0.9, 1.0)
        )
    
    @staticmethod
    def high_variance_distribution(count: int = 20) -> TokenDistribution:
        """Generate high variance distribution (low confidence expected).
        
        Simulates mixed workloads: simple questions, medium analysis, and complex reasoning tasks.
        """
        token_values = []
        for i in range(count):
            # Mix of short responses (100-500), medium (1000-4000), and long responses (8000-32000)
            if i % 3 == 0:  # Short responses
                tokens = 100 + (i * 50) % 400  # 100-500 range
            elif i % 3 == 1:  # Medium responses  
                tokens = 1000 + (i * 200) % 3000  # 1000-4000 range
            else:  # Long responses
                tokens = 8000 + (i * 1000) % 24000  # 8000-32000 range
            token_values.append(tokens)
        
        return TokenDistribution(
            name="high_variance",
            description="Mixed workload with short, medium, and long responses",
            token_values=token_values,
            expected_confidence_range=(0.0, 0.4)
        )
    
    @staticmethod
    def skewed_distribution(count: int = 20) -> TokenDistribution:
        """Generate right-skewed distribution (moderate confidence expected).
        
        Simulates typical LLM usage: many short responses with occasional complex tasks.
        """
        # Many small values (typical quick responses), few large outliers (complex reasoning)
        small_values = [200 + random.randint(0, 300) for _ in range(int(count * 0.65))]  # 200-500 tokens
        large_values = [8000 + random.randint(0, 16000) for _ in range(int(count * 0.35))]  # 8000-24000 tokens
        token_values = small_values + large_values
        random.shuffle(token_values)
        
        return TokenDistribution(
            name="skewed",
            description="Right-skewed distribution with outliers (typical LLM usage)",
            token_values=token_values,
            expected_confidence_range=(0.05, 0.7)  # Lowered due to realistic variance
        )
    
    @staticmethod
    def bimodal_distribution(count: int = 20) -> TokenDistribution:
        """Generate bimodal distribution (challenging for prediction).
        
        Simulates two distinct usage patterns: quick Q&A vs detailed analysis.
        """
        # Two distinct peaks: quick responses and detailed analysis
        mode1 = [400 + random.randint(-100, 100) for _ in range(count // 2)]   # Quick responses: 300-500 tokens
        mode2 = [4000 + random.randint(-500, 500) for _ in range(count // 2)]  # Detailed analysis: 3500-4500 tokens
        token_values = mode1 + mode2
        random.shuffle(token_values)
        
        return TokenDistribution(
            name="bimodal",
            description="Two distinct peaks: quick responses vs detailed analysis",
            token_values=token_values,
            expected_confidence_range=(0.1, 0.8)
        )
    
    @staticmethod
    def normal_distribution(mean: int = 2048, std_dev: int = 512, count: int = 20) -> TokenDistribution:
        """Generate normal distribution (moderate confidence expected).
        
        Simulates consistent workload with natural variation around a mean.
        """
        token_values = []
        for _ in range(count):
            # Box-Muller transform for normal distribution
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            value = max(100, int(mean + std_dev * z))  # Ensure positive values, min 100 tokens
            token_values.append(value)
        
        return TokenDistribution(
            name="normal",
            description=f"Normal distribution (μ={mean}, σ={std_dev})",
            token_values=token_values,
            expected_confidence_range=(0.6, 0.9)
        )
    
    @staticmethod
    def chaotic_distribution(count: int = 20) -> TokenDistribution:
        """Generate chaotic/pseudo-random distribution (very low confidence expected).
        
        Simulates completely unpredictable workload with wide variation.
        """
        random.seed(42)  # For reproducibility
        token_values = []
        for i in range(count):
            # Pseudo-random chaotic sequence with realistic token ranges
            base_value = (i * 1373 + 239) % 32000 + 100  # 100-32100 tokens
            # Add some clustering to make it more realistic but still chaotic
            cluster_factor = (i * 7 + 13) % 3
            if cluster_factor == 0:
                value = base_value % 1000 + 100  # Small responses: 100-1100
            elif cluster_factor == 1:
                value = (base_value % 8000) + 1000  # Medium responses: 1000-9000
            else:
                value = (base_value % 32000) + 4000  # Large responses: 4000-36000
            token_values.append(value)
        
        return TokenDistribution(
            name="chaotic",
            description="Pseudo-random chaotic sequence with realistic token ranges",
            token_values=token_values,
            expected_confidence_range=(0.0, 0.6)
        )
    
    @staticmethod
    def get_all_test_distributions() -> List[TokenDistribution]:
        """Get all predefined test distributions for comprehensive testing."""
        return [
            BatchPredictionTestDataGenerator.uniform_distribution(),
            BatchPredictionTestDataGenerator.high_variance_distribution(),
            BatchPredictionTestDataGenerator.skewed_distribution(),
            BatchPredictionTestDataGenerator.bimodal_distribution(),
            BatchPredictionTestDataGenerator.normal_distribution(),
            BatchPredictionTestDataGenerator.chaotic_distribution(),
        ]


class SemanticOperationTestCases:
    """Predefined test cases for semantic operations with different complexity patterns."""
    
    @staticmethod
    def simple_operations() -> List[dict]:
        """Operations expected to produce consistent, low token outputs."""
        return [
            {
                "type": "simple_math",
                "instruction": "Calculate this simple expression and give just the number: {input}",
                "inputs": ["2 + 2", "5 * 3", "10 / 2", "8 - 4", "6 + 7"] * 10,
                "expected_tokens_range": (5, 15),
                "expected_variance": "low"
            },
            {
                "type": "yes_no_questions",
                "instruction": "Answer yes or no only: {input}",
                "inputs": ["Is 2+2=4?", "Is water wet?", "Is the sky blue?", "Is fire cold?", "Is ice hot?"] * 10,
                "expected_tokens_range": (1, 5),
                "expected_variance": "very_low"
            }
        ]
    
    @staticmethod
    def variable_operations() -> List[dict]:
        """Operations expected to produce varied token outputs."""
        return [
            {
                "type": "explanations",
                "instruction": "Explain this concept in 1-2 sentences: {input}",
                "inputs": [
                    "photosynthesis", "democracy", "gravity", "evolution", "quantum mechanics",
                    "blockchain", "artificial intelligence", "climate change", "psychology", "economics"
                ] * 5,
                "expected_tokens_range": (20, 80),
                "expected_variance": "medium"
            },
            {
                "type": "creative_tasks",
                "instruction": "Write a haiku about: {input}",
                "inputs": [
                    "spring rain", "mountain peak", "ocean waves", "city lights", "desert wind",
                    "forest path", "starry night", "morning dew", "autumn leaves", "winter snow"
                ] * 5,
                "expected_tokens_range": (15, 30),
                "expected_variance": "low_to_medium"
            }
        ]
    
    @staticmethod
    def complex_operations() -> List[dict]:
        """Operations expected to produce high, varied token outputs."""
        return [
            {
                "type": "detailed_analysis",
                "instruction": "Provide a detailed analysis of: {input}",
                "inputs": [
                    "the impact of social media on society",
                    "renewable energy adoption challenges",
                    "the future of remote work",
                    "ethical implications of AI",
                    "global economic inequality"
                ] * 10,
                "expected_tokens_range": (100, 300),
                "expected_variance": "high"
            },
            {
                "type": "creative_writing",
                "instruction": "Write a short story (3 paragraphs) about: {input}",
                "inputs": [
                    "time travel", "alien contact", "underwater city", "magical forest", "robot companion",
                    "lost civilization", "parallel universe", "dream world", "space exploration", "future society"
                ] * 5,
                "expected_tokens_range": (150, 400),
                "expected_variance": "very_high"
            }
        ]
    
    @staticmethod
    def reasoning_operations() -> List[dict]:
        """Operations that should trigger thinking tokens in reasoning models."""
        return [
            {
                "type": "logic_puzzles",
                "instruction": "Solve this logic puzzle step by step: {input}",
                "inputs": [
                    "If all cats are animals and some animals are pets, what can we conclude about cats?",
                    "A farmer has chickens and rabbits. There are 20 heads and 56 legs total. How many of each?",
                    "Three boxes contain apples, oranges, and mixed fruit. All labels are wrong. How do you relabel them?",
                    "A man lives on the 20th floor. He takes the elevator down every morning but only up to the 10th floor when returning. Why?"
                ] * 12,
                "expected_tokens_range": (80, 250),
                "expected_variance": "medium_to_high",
                "triggers_thinking": True
            },
            {
                "type": "mathematical_proofs",
                "instruction": "Provide a mathematical proof or detailed solution: {input}",
                "inputs": [
                    "Prove that the square root of 2 is irrational",
                    "Solve the quadratic equation: x² + 5x + 6 = 0",
                    "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
                    "Prove that the sum of interior angles in a triangle is 180°"
                ] * 12,
                "expected_tokens_range": (100, 300),
                "expected_variance": "high",
                "triggers_thinking": True
            }
        ]


def create_comprehensive_test_dataset(num_samples_per_type: int = 20) -> dict:
    """Create a comprehensive test dataset covering all complexity levels."""
    all_operations = (
        SemanticOperationTestCases.simple_operations() +
        SemanticOperationTestCases.variable_operations() +
        SemanticOperationTestCases.complex_operations() +
        SemanticOperationTestCases.reasoning_operations()
    )
    
    test_data = {
        "operation_type": [],
        "instruction": [],
        "input_text": [],
        "expected_complexity": [],
        "triggers_thinking": []
    }
    
    for operation in all_operations:
        # Take only the requested number of samples per type
        inputs = operation["inputs"][:num_samples_per_type]
        
        for input_text in inputs:
            test_data["operation_type"].append(operation["type"])
            test_data["instruction"].append(operation["instruction"])
            test_data["input_text"].append(input_text)
            test_data["expected_complexity"].append(operation["expected_variance"])
            test_data["triggers_thinking"].append(operation.get("triggers_thinking", False))
    
    return test_data