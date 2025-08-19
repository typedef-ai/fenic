# Fenic Protobuf Serialization/Deserialization (Serde) System

## Overview

The Fenic serde system provides robust bidirectional serialization between Python objects and Protocol Buffer messages for data types, logical expressions, and logical plans. This enables efficient storage, transmission, and cross-language compatibility of complex query plans and expression trees.

## Key Features

- **🎯 Centralized Context System**: All serialization operations go through a unified `SerdeContext` with built-in error handling and field path tracking
- **📦 Type-Safe Dispatch**: Uses Python's `@singledispatch` for automatic type-based serialization routing
- **🔍 Rich Error Reporting**: Automatic field path tracking provides precise error locations during serialization/deserialization
- **🧩 Modular Design**: Clean separation of concerns with dedicated modules for expressions, plans, and data types

## Architecture

### Directory Structure

```text
src/fenic/core/_serde/proto/
├── proto_serde.py          # Main API entry point (ProtoSerde class)
├── serde_context.py        # Centralized SerdeContext with all serde operations
├── expression_serde.py     # LogicalExpr dispatch functions
├── plan_serde.py          # LogicalPlan dispatch functions
├── datatype_serde.py      # DataType dispatch functions
├── enum_serde.py          # Enum serialization utilities
├── errors.py              # Error classes and exceptions
├── types.py               # Proto type imports with "Proto" suffix
├── expressions/           # Expression-specific serde implementations
│   ├── basic.py          # Column, Literal, Alias, Array expressions
│   ├── binary.py         # Arithmetic, comparison, boolean expressions
│   ├── semantic.py       # LLM/AI expressions (map, extract, classify)
│   ├── text.py          # Text processing expressions
│   ├── embedding.py     # Vector/embedding operations
│   ├── aggregate.py     # Aggregation expressions (sum, avg, count)
│   ├── case.py          # Case/when conditional expressions
│   ├── json.py          # JSON manipulation expressions
│   └── markdown.py      # Markdown processing expressions
└── plans/               # Plan-specific serde implementations
    ├── source.py        # Source plans (InMemory, File, Table)
    ├── transform.py     # Transform plans (Project, Filter, Sort, etc.)
    ├── join.py          # Join plans (Inner, Left, Semantic)
    ├── aggregate.py     # Aggregate plans (GroupBy)
    └── sink.py          # Sink plans (File, Table)
```

## Core Components

### 1. SerdeContext - The Central Hub

`SerdeContext` is the core of the system, providing unified access to all serialization operations with built-in error handling:

```python
from fenic.core._serde.proto.serde_context import create_serde_context

# Create a context for serde operations
context = create_serde_context()

# All serde operations go through context methods
expr_proto = context.serialize_logical_expr("expr", my_expression)
deserialized_expr = context.deserialize_logical_expr("expr", expr_proto)
```

#### Key Context Methods

**Logical Expressions:**

```python
context.serialize_logical_expr(field_name, expr) -> LogicalExprProto
context.deserialize_logical_expr(field_name, expr_proto) -> LogicalExpr
context.serialize_logical_expr_list(field_name, expr_list) -> List[LogicalExprProto]
context.deserialize_logical_expr_list(field_name, expr_proto_list) -> List[LogicalExpr]
```

**Logical Plans:**

```python
context.serialize_logical_plan(field_name, plan) -> LogicalPlanProto
context.deserialize_logical_plan(field_name, plan_proto, session_state) -> LogicalPlan
context.serialize_logical_plan_list(field_name, plan_list) -> List[LogicalPlanProto]
context.deserialize_logical_plan_list(field_name, plan_proto_list, session_state) -> List[LogicalPlan]
```

**Data Types:**

```python
context.serialize_data_type(field_name, data_type) -> DataTypeProto
context.deserialize_data_type(field_name, data_type_proto) -> DataType
```

**Utilities:**

```python
context.serialize_scalar_value(field_name, value) -> ScalarValueProto
context.deserialize_scalar_value(field_name, scalar_proto) -> Any
context.serialize_enum_value(field_name, enum_val, proto_enum) -> int
context.deserialize_enum_value(field_name, enum_type, proto_enum, value) -> Enum
```

#### Field Name Constants

SerdeContext provides constants for common field names to ensure consistency:

```python
context.serialize_logical_expr(SerdeContext.EXPR, expr)      # "expr"
context.serialize_logical_expr_list(SerdeContext.EXPRS, exprs) # "exprs"
context.serialize_logical_plan(SerdeContext.INPUT, plan)     # "input"
# Also: LEFT, RIGHT, VALUE, VALUES, DATA_TYPE, SCHEMA, etc.
```

### 2. Type Dispatchers

The system uses `@singledispatch` to automatically route serialization based on Python types:

```python
# Expression dispatcher (expression_serde.py)
@singledispatch
def serialize_logical_expr(expr: LogicalExpr, context: SerdeContext) -> LogicalExprProto:
    raise SerializationError(f"No serializer for {type(expr)}")

# Individual expressions register their handlers
@serialize_logical_expr.register
def _serialize_column_expr(expr: ColumnExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(column=ColumnExprProto(name=expr.name))
```

### 3. High-Level API

The `ProtoSerde` class provides the main entry point:

```python
from fenic.core._serde.proto.proto_serde import ProtoSerde

# Serialize a logical plan to bytes
plan_bytes = ProtoSerde.serialize(logical_plan)

# Deserialize bytes back to a logical plan
logical_plan = ProtoSerde.deserialize(plan_bytes, session_state)
```

## How It Works

### Serialization Flow

1. **Entry Point**: `ProtoSerde.serialize()` creates a `SerdeContext`
2. **Context Method**: Calls `context.serialize_logical_plan("root", plan)`
3. **Dispatcher**: Routes to appropriate handler based on plan type using `@singledispatch`
4. **Handler**: Individual plan serde function serializes the specific plan type
5. **Nested Objects**: Handler uses context methods for nested expressions/plans
6. **Error Tracking**: Context automatically tracks field paths for debugging
7. **Result**: Returns protobuf message, which is serialized to bytes

### Deserialization Flow

1. **Entry Point**: `ProtoSerde.deserialize()` parses bytes to protobuf message
2. **Context Method**: Calls `context.deserialize_logical_plan("root", proto, session_state)`
3. **Type Detection**: Extracts specific plan type from protobuf oneof field
4. **Dispatcher**: Routes to appropriate handler based on protobuf message type
5. **Handler**: Individual plan serde function deserializes the specific type
6. **Nested Objects**: Handler uses context methods for nested expressions/plans
7. **Constructor**: Creates Python object with exact parameter matching
8. **Result**: Returns fully constructed logical plan

### Error Handling

The context system provides automatic error handling with precise field paths:

```python
# If an error occurs deep in a nested structure like:
# Plan -> Filter -> BinaryExpr -> Left -> ColumnExpr
#
# Error message will be:
# "Serialization failed at root.filter.predicate.left.name in ColumnExpr"
```

## Adding New Expressions

### Step 0: Determine if expression is serializable

Some expressions, like UDFExpr are not currently safely serializable to protobuf spec, as they involve serde of untrusted
user provided code. These should be added to `unserializable.py` -- so that when plans that include these expressions are serialized,
they fail with an error that explains exactly why the expression is not serializable.

### Step 1: Update Protocol Buffer Schema

Add your expression to `protos/logical_plan/v1/expressions.proto`:

```protobuf
message MyNewExprProto {
  LogicalExprProto input_expr = 1;
  string parameter = 2;
  int32 config_value = 3;
}

// Add to LogicalExprProto oneof
message LogicalExprProto {
  oneof expr_type {
    // ... existing expressions ...
    MyNewExprProto my_new_expr = 999;  // Use next available number
  }
}
```

Regenerate Python types:

```bash
just generate-protos-py
```

### Step 2: Implement Serde Functions

Add to the appropriate module in `expressions/` (create new module if needed):

```python
# In expressions/my_category.py

from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import LogicalExprProto, MyNewExprProto

@serialize_logical_expr.register
def _serialize_my_new_expr(expr: MyNewExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize MyNewExpr using context methods."""
    return LogicalExprProto(
        my_new_expr=MyNewExprProto(
            input_expr=context.serialize_logical_expr(SerdeContext.EXPR, expr.input_expr),
            parameter=expr.parameter,
            config_value=expr.config_value,
        )
    )

@_deserialize_logical_expr_helper.register
def _deserialize_my_new_expr(proto: MyNewExprProto, context: SerdeContext) -> MyNewExpr:
    """Deserialize MyNewExpr using context methods."""
    return MyNewExpr(
        input_expr=context.deserialize_logical_expr(SerdeContext.EXPR, proto.input_expr),
        parameter=proto.parameter,
        config_value=proto.config_value,
    )
```

### Step 3: Register Your Module

Ensure your module is imported so the `@register` decorators execute:

```python
# In expressions/__init__.py or main module
from .my_category import _serialize_my_new_expr, _deserialize_my_new_expr
```

### Step 4: Add Tests

```python
def test_my_new_expr_serde():
    """Test MyNewExpr serialization round-trip."""
    from fenic.core._serde.proto.serde_context import create_serde_context

    context = create_serde_context()
    original_expr = MyNewExpr(
        input_expr=ColumnExpr("test_col"),
        parameter="test_param",
        config_value=42
    )

    # Test round-trip
    proto = context.serialize_logical_expr("test", original_expr)
    deserialized = context.deserialize_logical_expr("test", proto)

    assert isinstance(deserialized, MyNewExpr)
    assert deserialized.parameter == original_expr.parameter
    assert deserialized.config_value == original_expr.config_value
```

## Adding New Plans

The process is identical to expressions, but use:

- `plan_serde.py` dispatchers: `serialize_logical_plan` and `_deserialize_logical_plan_helper`
- Protocol buffer file: `protos/logical_plan/v1/plans.proto`
- Module location: `plans/` directory
- Session state parameter: Plans support `session_state` parameter in deserialization

```python
@serialize_logical_plan.register
def _serialize_my_new_plan(plan: MyNewPlan, context: SerdeContext) -> LogicalPlanProto:
    return LogicalPlanProto(
        my_new_plan=MyNewPlanProto(
            input=context.serialize_logical_plan(SerdeContext.INPUT, plan.input),
            parameter=plan.parameter,
        )
    )

@_deserialize_logical_plan_helper.register
def _deserialize_my_new_plan(
    proto: MyNewPlanProto,
    context: SerdeContext,
    session_state: Optional[BaseSessionState] = None
) -> MyNewPlan:
    return MyNewPlan(
        input=context.deserialize_logical_plan(SerdeContext.INPUT, proto.input, session_state),
        parameter=proto.parameter,
        session_state=session_state,
    )
```

## Adding New Data Types

Similar process using `datatype_serde.py`:

```python
@serialize_data_type.register
def _serialize_my_new_type(data_type: MyNewType, context: SerdeContext) -> DataTypeProto:
    return DataTypeProto(
        my_new_type=MyNewTypeProto(
            parameter=data_type.parameter,
        )
    )

@_deserialize_data_type_helper.register
def _deserialize_my_new_type(proto: MyNewTypeProto, context: SerdeContext) -> MyNewType:
    return MyNewType(parameter=proto.parameter)
```

## Best Practices

### ✅ Do

- **Use Context Methods**: Always use `context.serialize_*()` and `context.deserialize_*()` methods
- **Use Field Constants**: Use `SerdeContext.EXPR`, `SerdeContext.INPUT`, etc. for consistency in commonly used field names.
- **Match Parameters Exactly**: Constructor parameters must match serde function arguments exactly
- **Handle Optional Fields**: Check `proto.HasField()` for optional protobuf fields
- **Add Comprehensive Tests**: Test round-trip serialization with various inputs

### ❌ Don't

- **Call Dispatchers Directly**: Don't call `serialize_logical_expr()` directly, use context methods
- **Skip Error Handling**: Context methods handle errors automatically, don't bypass them
- **Ignore Parameter Mismatches**: Mismatched constructor parameters will cause runtime errors
- **Forget Module Registration**: Unregistered modules won't have their serde functions available

## Common Patterns

### Complex Nested Structures

```python
@serialize_logical_expr.register
def _serialize_case_expr(expr: CaseExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        case=CaseExprProto(
            when_exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, expr.when_clauses),
            else_expr=context.serialize_logical_expr("else_expr", expr.else_clause) if expr.else_clause else None,
        )
    )
```

### Enum Handling

```python
@serialize_logical_expr.register
def _serialize_binary_expr(expr: BinaryExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        binary=BinaryExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, expr.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, expr.right),
            operator=context.serialize_enum_value(SerdeContext.OPERATOR, expr.op, OperatorProto),
        )
    )
```

### Optional Field Handling

```python
@_deserialize_logical_expr_helper.register
def _deserialize_optional_expr(proto: OptionalExprProto, context: SerdeContext) -> OptionalExpr:
    return OptionalExpr(
        required_field=context.deserialize_logical_expr("required", proto.required_field),
        optional_field=context.deserialize_logical_expr("optional", proto.optional_field)
                      if proto.HasField("optional_field") else None,
    )
```

## Testing Strategy

The system includes comprehensive testing across multiple levels:

- **Unit Tests**: Individual expression/plan serialization
- **Integration Tests**: Complex nested structures
- **Round-Trip Tests**: Serialize → Deserialize → Compare
- **Error Tests**: Invalid inputs and error message validation
- **Performance Tests**: Large expression trees and benchmarking

Use the provided test utilities for consistent testing:

```python
from fenic.core._serde.proto.serde_context import create_serde_context

def verify_round_trip(expr: LogicalExpr):
    """Test that an expression survives round-trip serialization."""
    context = create_serde_context()
    proto = context.serialize_logical_expr("test", expr)
    deserialized = context.deserialize_logical_expr("test", proto)

    assert type(deserialized) == type(expr)
    # Add expression-specific assertions
    return deserialized
```
