# fenic.core.metrics

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/metrics/

Metrics tracking for query execution and model usage.

This module defines classes for tracking various metrics during query execution,
including language model usage, embedding model usage, operator performance,
and overall query statistics.

Classes:

- **`LMMetrics`**
  –

  Tracks language model usage metrics including token counts and costs.
- **`OperatorMetrics`**
  –

  Metrics for a single operator in the query execution plan.
- **`PhysicalPlanRepr`**
  –

  Tree node representing the physical execution plan, used for pretty printing execution plan.
- **`QueryMetrics`**
  –

  Comprehensive metrics for an executed query.
- **`RMMetrics`**
  –

  Tracks embedding model usage metrics including token counts and costs.

## LMMetrics

```
LMMetrics(num_uncached_input_tokens: int = 0, num_cached_input_tokens: int = 0, num_output_tokens: int = 0, cost: float = 0.0, num_requests: int = 0, num_reserved_output_tokens: int = 0)
```

Tracks language model usage metrics including token counts and costs.

Attributes:

- **`num_uncached_input_tokens`**
  (`int`)
  –

  Number of uncached tokens in the prompt/input.
- **`num_cached_input_tokens`**
  (`int`)
  –

  Number of cached tokens in the prompt/input.
- **`num_output_tokens`**
  (`int`)
  –

  Number of tokens in the completion/output (actual usage).
- **`cost`**
  (`float`)
  –

  Total cost in USD for the LM API call.
- **`num_requests`**
  (`int`)
  –

  Total number of LM API requests made.
- **`num_reserved_output_tokens`**
  (`int`)
  –

  Output tokens debited from the TPM bucket at
  reservation time. Compare against num_output_tokens to measure
  reservation efficiency (actual / reserved → 1 is tight).

## OperatorMetrics

```
OperatorMetrics(operator_id: str, num_output_rows: int = 0, execution_time_ms: float = 0.0, lm_metrics: LMMetrics = LMMetrics(), rm_metrics: RMMetrics = RMMetrics())
```

Metrics for a single operator in the query execution plan.

Attributes:

- **`operator_id`**
  (`str`)
  –

  Unique identifier for the operator
- **`num_output_rows`**
  (`int`)
  –

  Number of rows output by this operator
- **`execution_time_ms`**
  (`float`)
  –

  Execution time in milliseconds
- **`lm_metrics`**
  (`LMMetrics`)
  –

  Language model usage metrics for this operator

## PhysicalPlanRepr

```
PhysicalPlanRepr(operator_id: str, children: List[PhysicalPlanRepr] = list())
```

Tree node representing the physical execution plan, used for pretty printing execution plan.

## QueryMetrics

```
QueryMetrics(execution_id: str, session_id: str, execution_time_ms: float = 0.0, num_output_rows: int = 0, total_lm_metrics: LMMetrics = LMMetrics(), total_rm_metrics: RMMetrics = RMMetrics(), end_ts: datetime = datetime.now(), _operator_metrics: Dict[str, OperatorMetrics] = dict(), _plan_repr: PhysicalPlanRepr = (lambda: PhysicalPlanRepr(operator_id='empty'))())
```

Comprehensive metrics for an executed query.

Includes overall statistics and detailed metrics for each operator
in the execution plan.

Attributes:

- **`execution_id`**
  (`str`)
  –

  Unique identifier for this query execution
- **`session_id`**
  (`str`)
  –

  Identifier for the session this query belongs to
- **`execution_time_ms`**
  (`float`)
  –

  Total query execution time in milliseconds
- **`num_output_rows`**
  (`int`)
  –

  Total number of rows returned by the query
- **`total_lm_metrics`**
  (`LMMetrics`)
  –

  Aggregated language model metrics across all operators
- **`end_ts`**
  (`datetime`)
  –

  Timestamp when query execution completed

Methods:

- **`get_execution_plan_details`**
  –

  Generate a formatted execution plan with detailed metrics.
- **`get_summary`**
  –

  Summarize the query metrics in a single line.
- **`to_dict`**
  –

  Convert QueryMetrics to a dictionary for table storage.

### start_ts

```
start_ts: datetime
```

Calculate start timestamp from end timestamp and execution time.

### get_execution_plan_details

```
get_execution_plan_details() -> str
```

Generate a formatted execution plan with detailed metrics.

Produces a hierarchical representation of the query execution plan,
including performance metrics and language model usage for each operator.

Returns:

- **`str`** ( `str`
  ) –

  A formatted string showing the execution plan with metrics.

Source code in `src/fenic/core/metrics.py`

```
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
                    f"{indent_str}  Language Model Usage: {op.lm_metrics.num_uncached_input_tokens:,} input tokens, {op.lm_metrics.num_cached_input_tokens:,} cached input tokens, {op.lm_metrics.num_output_tokens:,} output tokens ({op.lm_metrics.num_reserved_output_tokens:,} reserved)",
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
```

### get_summary

```
get_summary() -> str
```

Summarize the query metrics in a single line.

Returns:

- **`str`** ( `str`
  ) –

  A concise summary of execution time, row count, and LM cost.

Source code in `src/fenic/core/metrics.py`

```
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
```

### to_dict

```
to_dict() -> Dict[str, Any]
```

Convert QueryMetrics to a dictionary for table storage.

Returns:

- `Dict[str, Any]`
  –

  Dict containing all metrics fields suitable for database storage.

Source code in `src/fenic/core/metrics.py`

```
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
```

## RMMetrics

```
RMMetrics(num_input_tokens: int = 0, num_requests: int = 0, cost: float = 0.0)
```

Tracks embedding model usage metrics including token counts and costs.

Attributes:

- **`num_input_tokens`**
  (`int`)
  –

  Number of tokens to embed
- **`cost`**
  (`float`)
  –

  Total cost in USD to embed the tokens
