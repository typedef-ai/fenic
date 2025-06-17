"""Metrics tracking for query execution and model usage.

This module defines classes for tracking various metrics during query execution,
including language model usage, embedding model usage, operator performance,
and overall query statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LMMetrics:
    """Tracks language model usage metrics including token counts and costs.

    Attributes:
        num_uncached_input_tokens: Number of uncached tokens in the prompt/input
        num_cached_input_tokens: Number of cached tokens in the prompt/input,
        num_output_tokens: Number of tokens in the completion/output
        cost: Total cost in USD for the LM API call
    """

    num_uncached_input_tokens: int = 0
    num_cached_input_tokens: int = 0
    num_output_tokens: int = 0
    cost: float = 0.0
    num_requests: int = 0

    def __add__(self, other: "LMMetrics") -> "LMMetrics":
        """Add two LMMetrics instances together.

        Args:
            other: Another LMMetrics instance to add.

        Returns:
            A new LMMetrics instance with combined metrics.
        """
        return LMMetrics(
            num_uncached_input_tokens=self.num_uncached_input_tokens + other.num_uncached_input_tokens,
            num_cached_input_tokens=self.num_cached_input_tokens
            + other.num_cached_input_tokens,
            num_output_tokens=self.num_output_tokens + other.num_output_tokens,
            cost=self.cost + other.cost,
            num_requests=self.num_requests + other.num_requests,
        )


@dataclass
class RMMetrics:
    """Tracks embedding model usage metrics including token counts and costs.

    Attributes:
        num_input_tokens: Number of tokens to embed
        cost: Total cost in USD to embed the tokens
    """

    num_input_tokens: int = 0
    num_requests: int = 0
    cost: float = 0.0

    def __add__(self, other: "RMMetrics") -> "RMMetrics":
        """Add two RMMetrics instances together.

        Args:
            other: Another RMMetrics instance to add.

        Returns:
            A new RMMetrics instance with combined metrics.
        """
        return RMMetrics(
            num_input_tokens=self.num_input_tokens + other.num_input_tokens,
            num_requests=self.num_requests + other.num_requests,
            cost=self.cost + other.cost,
        )


@dataclass
class OperatorMetrics:
    """Metrics for a single operator in the query execution plan.

    Attributes:
        operator_id: Unique identifier for the operator
        num_output_rows: Number of rows output by this operator
        execution_time_ms: Execution time in milliseconds
        lm_metrics: Language model usage metrics for this operator
        is_cache_hit: Whether results were retrieved from cache
    """

    operator_id: str
    num_output_rows: int = 0
    execution_time_ms: float = 0.0
    lm_metrics: LMMetrics = field(default_factory=LMMetrics)
    rm_metrics: RMMetrics = field(default_factory=RMMetrics)
    is_cache_hit: bool = False


@dataclass
class PhysicalPlanRepr:
    """Tree node representing the physical execution plan, used for pretty printing execution plan."""

    operator_id: str
    children: List["PhysicalPlanRepr"] = field(default_factory=list)


@dataclass
class QueryMetrics:
    """Comprehensive metrics for an executed query.

    Includes overall statistics and detailed metrics for each operator
    in the execution plan.

    Attributes:
        execution_time_ms: Total query execution time in milliseconds
        num_output_rows: Total number of rows returned by the query
        total_lm_metrics: Aggregated language model metrics across all operators
    """

    execution_time_ms: float = 0.0
    num_output_rows: int = 0
    total_lm_metrics: LMMetrics = field(default_factory=LMMetrics)
    total_rm_metrics: RMMetrics = field(default_factory=RMMetrics)
    _operator_metrics: Dict[str, OperatorMetrics] = field(default_factory=dict)
    _plan_repr: PhysicalPlanRepr = field(
        default_factory=lambda: PhysicalPlanRepr(operator_id="empty")
    )

    def get_summary(self) -> str:
        """Summarize the query metrics in a single line.

        Returns:
            str: A concise summary of execution time, row count, and LM cost.
        """
        return (
            f"Query executed in {self.execution_time_ms:.2f}ms, "
            f"returned {self.num_output_rows:,} rows, "
            f"language model cost: ${self.total_lm_metrics.cost:.6f}, "
            f"embedding model cost: ${self.total_rm_metrics.cost:.6f}"
        )

    def get_execution_plan_details(self) -> str:
        """Generate a formatted execution plan with detailed metrics.

        Produces a hierarchical representation of the query execution plan,
        including performance metrics and language model usage for each operator.

        Returns:
            str: A formatted string showing the execution plan with metrics.
        """

        def _format_node(node: PhysicalPlanRepr, indent: int = 1) -> str:
            op = self._operator_metrics[node.operator_id]
            indent_str = "  " * indent

            details = [
                f"{indent_str}{op.operator_id}",
                f"{indent_str}  Output Rows: {op.num_output_rows:,}",
                f"{indent_str}  Execution Time: {op.execution_time_ms:.2f}ms",
                f"{indent_str}  Cached: {op.is_cache_hit}",
            ]

            if op.lm_metrics.cost > 0:
                details.extend(
                    [
                        f"{indent_str}  Language Model Usage: {op.lm_metrics.num_uncached_input_tokens:,} input tokens, {op.lm_metrics.num_cached_input_tokens:,} cached input tokens, {op.lm_metrics.num_output_tokens:,} output tokens",
                        f"{indent_str}  Language Model Cost: ${op.lm_metrics.cost:.6f}",
                    ]
                )

            if op.rm_metrics.cost > 0:
                details.extend(
                    [
                        f"{indent_str}  Embedding Model Usage: {op.rm_metrics.num_input_tokens:,} input tokens",
                        f"{indent_str}  Embedding Model Cost: ${op.rm_metrics.cost:.6f}",
                    ]
                )
            return (
                "\n".join(details)
                + "\n"
                + "".join(_format_node(child, indent + 1) for child in node.children)
            )

        return f"Execution Plan\n{_format_node(self._plan_repr)}"

    def __str__(self) -> str:
        """Generate a detailed string representation of the query metrics.

        Returns:
            str: A multi-line string containing execution time, row counts,
                language model and embedding model costs and token usage,
                and the execution plan details.
        """
        return (
            f"Execution time: {self.execution_time_ms:.2f}ms\n"
            f"Num Output Rows: {self.num_output_rows:,}\n"
            f"Language Model Cost: ${self.total_lm_metrics.cost:.6f}\n"
            f"Language Model Tokens: {self.total_lm_metrics.num_uncached_input_tokens:,} input tokens, {self.total_lm_metrics.num_cached_input_tokens:,} cached input tokens, {self.total_lm_metrics.num_output_tokens:,} output tokens\n"
            f"Language Model Requests: {self.total_lm_metrics.num_requests}\n"
            f"Embedding Model Cost: ${self.total_rm_metrics.cost:.6f}\n"
            f"Embedding Model Tokens: {self.total_rm_metrics.num_input_tokens:,} input tokens\n\n"
            f"Embedding Model Requests: {self.total_rm_metrics.num_requests}\n"
            f"{self.get_execution_plan_details()}"
        )
