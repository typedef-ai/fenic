# CLAUDE.md - Fenic Technical Guide

## Overview

Fenic is a PySpark-inspired DataFrame framework designed for curating data for AI and agentic applications. It transforms structured and unstructured data using familiar DataFrame operations enhanced with semantic intelligence, providing first-class support for markdown, transcripts, and LLM-powered semantic operators. Future roadmap includes indexing and serving capabilities for human and agent consumption.

## Core Architecture

### Architecture Pattern

```
User API → Logical Plan → Validation → Optimization → Physical Plan → Execution
```

### Key Architectural Decisions

1. **Eager Validation**: Logical plans are validated during construction (not at execution time) to provide immediate feedback in interactive environments like notebooks.

2. **Dual Backend Design**:

   - **Local Backend**: DuckDB for storage + Polars for execution ("DuckDB for AI")
   - **Cloud Backend**: Client-side plan construction with remote execution via gRPC

3. **Custom Type System**: Fenic-specific types that wrap or extend Polars/Arrow types to support AI workloads (embeddings, markdown, JSON).

## Type System and Schema Management

### Type Architecture

```python
# Fenic types hierarchy
- Primitive types: Wrap Polars types with sugar (e.g., IntegerType unifies int8/16/32/64)
- AI-specific types: New types for AI workloads
  - EmbeddingType: Fixed-size float arrays for vector operations
  - MarkdownType: String with markdown-specific operations
  - JsonType: String with JSON validation and JQ operations
```

### Type Bridging

- **Cast Function**: Polars plugin implements casting between Fenic and Polars types
- **Persistence Layer**: System metadata table stores logical types alongside physical data
- **Schema Recovery**: Types are reinterpreted when loading from storage

## Plan Construction and Validation

### Logical Plan Construction

```python
# When users write df.select(...).filter(...):
1. Each operation creates a new LogicalPlan node
2. Expressions are bound to the plan context
3. Type checking happens immediately
4. Schema is computed and validated
```

### Plan Validation Rules

- Column references must exist in the schema
- Type compatibility for operations
- Plan specific expression validity (Aggregation Plan requires Aggregate Expressions)

## Expression Execution Patterns

### 1. IO-Bound Operations (Semantic Operators)

```python
# Located in _backends/local/semantic_operators/
# Pattern: Use Polars map_batches with async execution
# In transpiler:
polars_expr = pl.col("text").map_batches(execute_semantic_operation)
```

### 2. CPU-Bound Operations (Polars Plugins)

```python
# Located in rust/src/ and registered in _polars_plugins.py
# Pattern: Implement in Rust, expose as Polars expressions
```

## Transpilation Process

### Transpiler Architecture

```python
# _backends/local/transpiler/transpiler.py

class LocalTranspiler:
    def transpile(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        # 1. Apply optimizations
        optimized = self.optimizer.optimize(logical_plan)

        # 2. Convert to physical plan
        physical = PlanConverter(session_state).convert(optimized)

        return physical

class PlanConverter:
    def convert(self, logical_node: LogicalPlan) -> PhysicalPlan:
        # Recursively convert each node type
        # Filter → FilterExec
        # Projection → ProjectExec
        # etc.
```

### Expression Conversion

```python
# _backends/local/transpiler/expr_converter.py

class ExprConverter:
    def convert(self, expr: Expression) -> pl.Expr:
        # Convert Fenic expressions to Polars expressions
        # Handle special cases for semantic operators
        # Register plugins for custom operations
```

## Backend Implementation Details

### Local Backend

- **Storage**: DuckDB files (portable, shareable)
- **Execution**: Polars DataFrames (currently eager, not lazy)
- **Temp Storage**: TempDFDBClient manages intermediate results
- **Catalog**: LocalCatalog with metadata in system tables

### Cloud Backend

- **Serialization**: Currently pickle (TODO: migrate to protobuf/substrait)
- **Communication**: gRPC for plan submission
- **Data Transfer**: Apache Arrow Flight for results
- **Async Operations**: Subscription-based status monitoring

## Code Patterns and Best Practices

### 1. Type Hints Everything

### 2. Explicit Over Clever

### 3. Test Patterns

```python
def test_new_feature():
    # 1. Setup
    session = local_session()
    df = session.create_dataframe(test_data)

    # 2. Execute
    result = df.your_operation().collect()

    # 3. Validate schema first
    assert result.schema == expected_schema

    # 4. Validate data
    assert result.to_pylist() == expected_data
```

### 4. Semantic Operator Pattern

```python
class YourSemanticOperator(BaseSingleColumnInputOperator):
    def build_request_messages(self, batch: list[str]) -> list[Messages]:
        # Build prompts for batch

    def post_process_response(self, responses: list[str]) -> pl.Series:
        # Parse and validate responses
```

## Key Implementation Files

### Core Components

- `src/fenic/api/session/session.py` - Session management
- `src/fenic/api/dataframe/dataframe.py` - DataFrame API
- `src/fenic/core/_logical_plan/` - Logical plan definitions
- `src/fenic/_backends/local/transpiler/` - Plan transpilation

### Semantic Operations

- `src/fenic/_backends/local/semantic_operators/base.py` - Base patterns
- `src/fenic/_backends/local/semantic_operators/map.py` - Example implementation
- `src/fenic/_inference/model_client.py` - Async LLM client

### Type System

- `src/fenic/core/types/datatypes` - Type definitions
- `rust/src/dtypes/` - Rust type implementations

## Development Workflow

### Adding New Features

1. **New DataFrame Operation**:

   - Add logical plan node in `core/_logical_plan/plans/`
   - Add DataFrame method in `api/dataframe/dataframe.py`
   - Implement physical plan in `_backends/local/physical_plan/`
   - Add transpiler conversion in `transpiler/plan_converter.py`

2. **New Semantic Operator**:

   - Create operator in `_backends/local/semantic_operators/`
   - Inherit from appropriate base class
   - Add to semantic API in `api/functions/semantic.py`
   - Write tests focusing on prompt construction

3. **New Polars Plugin**:
   - Implement in Rust under `rust/src/`
   - Register in `_backends/local/polars_plugins/`
   - Add Python API in `api/functions/`

### Testing Guidelines

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test full pipelines
- **Semantic Tests**: Focus on prompt construction, not LLM outputs
- **Use Fixtures**: Leverage conftest.py fixtures for consistency

## Architecture TODOs

1. **Serialization**: Migrate from pickle to protobuf/substrait for cloud backend
2. **Lazy Evaluation**: Convert Polars execution from eager to lazy
3. **Feature Completeness**: Expose more Polars operations through Fenic API
4. **Serving Layer**: Implement data indexing and serving capabilities

## Key Invariants

1. **Schema Consistency**: Operations must preserve or explicitly transform schemas
2. **Type Safety**: All operations validate types at plan construction time
3. **Null Handling**: Semantic operators gracefully handle null inputs
4. **Error Propagation**: Failed LLM calls return None, not exceptions
