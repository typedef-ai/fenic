# Expression Registration Approaches

This document compares different approaches for registering LogicalExpr serialization handlers.

## Current Manual Approach

**Example:**

```python
SerializationRegistry.register_with_custom(
    SortExpr, "sort", "ProtoSortExpr",
    make_typed_param_serializer(
        ParamSpec("ascending", "ascending", ParamType.BOOL),
        ParamSpec("nulls_last", "nulls_last", ParamType.BOOL)
    )
)
```

**Pros:**

- Explicit control over every aspect
- Clear parameter type specifications
- Easy to handle edge cases

**Cons:**

- Extremely verbose (6+ lines per expression)
- Error-prone (easy to make typos)
- Must maintain 73+ registrations manually
- Parameter names repeated 3 times
- High maintenance burden

## Introspection-Based Auto-Registration

**Example:**

```python
@auto_serializable
class SortExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, ascending: bool, nulls_last: bool):
        self.expr = expr
        self.ascending = ascending
        self.nulls_last = nulls_last
```

**How it works:**

- Analyzes `__init__` signature using Python's `inspect` module
- Infers parameter types from type annotations
- Automatically generates ParamSpecs
- Handles optional parameters (those with defaults)
- Skips LogicalExpr parameters (treats as children)

**Pros:**

- Single decorator per expression
- Zero boilerplate for simple cases
- Automatically handles new parameters
- Type-safe (uses existing annotations)
- Self-documenting

**Cons:**

- Less control over edge cases
- Requires good type annotations
- May need fallback for complex cases
- Magic behavior (less explicit)

**Success Rate:** ~80% of expressions can be auto-registered

## Declarative Registration with Decorators

**Example:**

```python
@serializable_expr(
    "sort", "ProtoSortExpr",
    ascending=ParamType.BOOL,
    nulls_last=ParamType.BOOL
)
class SortExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, ascending: bool, nulls_last: bool):
        ...
```

**Pros:**

- Concise but explicit
- Full control over parameter types
- Clear metadata at class definition
- Easy to read and maintain
- Works with complex parameter types

**Cons:**

- Still requires manual specification
- Parameter names duplicated (once in decorator, once in **init**)
- Slightly more verbose than auto-registration

## Pattern-Based Bulk Registration

**Example:**

```python
simple_expressions = [
    (NotExpr, "not", "ProtoNotExpr"),
    (ArrayExpr, "array", "ProtoArrayExpr"),
    (CoalesceExpr, "coalesce", "ProtoCoalesceExpr"),
]
RegistrationPatterns.register_simple_expressions(simple_expressions)
```

**Pros:**

- Compact for similar expressions
- Reduces duplication
- Easy to see related expressions together
- Good for bulk operations

**Cons:**

- Only works for similar expression types
- Still requires manual listing
- Limited to specific patterns

## Recommended Hybrid Approach

Use different approaches based on expression complexity:

### 1. Auto-registration for Simple Cases (60-70% of expressions)

```python
@auto_serializable
class SimpleExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, name: str, count: int):
        ...
```

### 2. Declarative for Complex Parameters (20-30% of expressions)

```python
@serializable_expr(
    "semantic_map", "ProtoSemanticMapExpr",
    instruction=ParamType.STRING,
    examples=ParamType.EXAMPLE_COLLECTION,
    model_config=ParamType.ENUM
)
class SemanticMapExpr(LogicalExpr):
    ...
```

### 3. Pattern-based for Bulk Simple Expressions (5-10% of expressions)

```python
RegistrationPatterns.register_simple_expressions([
    (NotExpr, "not", "ProtoNotExpr"),
    (ArrayExpr, "array", "ProtoArrayExpr"),
    # ... more simple expressions
])
```

### 4. Manual for Edge Cases (1-5% of expressions)

```python
# For expressions that need completely custom serialization logic
SerializationRegistry.register_with_custom(
    ComplexExpr, "complex", "ProtoComplexExpr",
    custom_serialization_function
)
```

## Migration Strategy

1. **Phase 1:** Implement auto-registration infrastructure
2. **Phase 2:** Convert simple expressions to `@auto_serializable`
3. **Phase 3:** Convert parameter-based expressions to `@serializable_expr`
4. **Phase 4:** Use bulk patterns for remaining simple cases
5. **Phase 5:** Manual registration only for true edge cases

## Performance Comparison

| Approach      | Registration Time | Runtime Overhead | Lines of Code | Maintenance |
| ------------- | ----------------- | ---------------- | ------------- | ----------- |
| Manual        | N/A               | None             | ~8 per expr   | High        |
| Auto          | ~1ms per expr     | Minimal          | ~1 per expr   | Low         |
| Declarative   | N/A               | None             | ~3 per expr   | Medium      |
| Pattern-based | N/A               | None             | ~0.3 per expr | Low         |

## Error Handling

Each approach provides different error detection:

- **Manual:** Errors at registration time
- **Auto:** Errors at registration time with detailed analysis
- **Declarative:** Errors at registration time with parameter validation
- **Pattern-based:** Errors at registration time for bulk operations

## Code Examples Impact

**Before (Manual - 73 expressions × 8 lines):** ~584 lines of boilerplate
**After (Hybrid approach):** ~150 lines total

**Reduction:** 74% fewer lines of registration code
