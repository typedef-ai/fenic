"""Metrics tracking for query execution and model usage.

This module defines classes for tracking various metrics during query execution,
including language model usage, embedding model usage, operator performance,
and overall query statistics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


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
    """

    operator_id: str
    num_output_rows: int = 0
    execution_time_ms: float = 0.0
    lm_metrics: LMMetrics = field(default_factory=LMMetrics)
    rm_metrics: RMMetrics = field(default_factory=RMMetrics)


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
        execution_id: Unique identifier for this query execution
        session_id: Identifier for the session this query belongs to
        execution_time_ms: Total query execution time in milliseconds
        num_output_rows: Total number of rows returned by the query
        total_lm_metrics: Aggregated language model metrics across all operators
        end_ts: Timestamp when query execution completed
    """

    execution_id: str
    session_id: str
    execution_time_ms: float = 0.0
    num_output_rows: int = 0
    total_lm_metrics: LMMetrics = field(default_factory=LMMetrics)
    total_rm_metrics: RMMetrics = field(default_factory=RMMetrics)
    end_ts: datetime = field(default_factory=datetime.now)
    _operator_metrics: Dict[str, OperatorMetrics] = field(default_factory=dict)
    _plan_repr: PhysicalPlanRepr = field(
        default_factory=lambda: PhysicalPlanRepr(operator_id="empty")
    )

    @property
    def start_ts(self) -> datetime:
        """Calculate start timestamp from end timestamp and execution time."""
        from datetime import timedelta
        return self.end_ts - timedelta(milliseconds=self.execution_time_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert QueryMetrics to a dictionary for table storage.

        Returns:
            Dict containing all metrics fields suitable for database storage.
        """
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "execution_time_ms": self.execution_time_ms,
            "num_output_rows": self.num_output_rows,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "total_lm_cost": self.total_lm_metrics.cost,
            "total_lm_uncached_input_tokens": self.total_lm_metrics.num_uncached_input_tokens,
            "total_lm_cached_input_tokens": self.total_lm_metrics.num_cached_input_tokens,
            "total_lm_output_tokens": self.total_lm_metrics.num_output_tokens,
            "total_lm_requests": self.total_lm_metrics.num_requests,
            "total_rm_cost": self.total_rm_metrics.cost,
            "total_rm_input_tokens": self.total_rm_metrics.num_input_tokens,
            "total_rm_requests": self.total_rm_metrics.num_requests,
        }

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
